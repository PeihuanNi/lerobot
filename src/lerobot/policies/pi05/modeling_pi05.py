#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.token_selection_utils import (
    TokenSelectionState,
    apply_partial_update,
    apply_min_max_constraints,
    build_overlay,
    build_overlay_labels,
    compute_region_embeddings,
    compute_region_scores,
    compute_spatial_mask,
    compute_temporal_mask,
    detect_static_background,
    expand_region_mask,
    prune_image_embeddings,
    select_regions_by_mass,
    select_regions_by_topk_ratio,
    split_patch_tokens,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        output_attentions: bool = False,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        attentions = None
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                output_attentions=output_attentions,
            )
            prefix_past_key_values = prefix_output.past_key_values
            if output_attentions:
                attentions = prefix_output.attentions
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attentions = suffix_output.attentions
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            if output_attentions:
                raise ValueError("output_attentions is only supported for prefix-only or suffix-only forward.")
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        if output_attentions:
            return [prefix_output, suffix_output], prefix_past_key_values, attentions
        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self._token_selection_state = TokenSelectionState()

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def reset_token_selection_state(self):
        if self._token_selection_state is None:
            self._token_selection_state = TokenSelectionState()
        else:
            self._token_selection_state.reset()

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        image_embs: list[torch.Tensor] | None = None,
        image_pad_masks: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            if image_embs is None:

                def image_embed_func(img):
                    return self.paligemma_with_expert.embed_image(img)

                img_emb = self._apply_checkpoint(image_embed_func, img)
            else:
                img_emb = image_embs[idx]
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            if image_pad_masks is None:
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            else:
                pad_masks.append(image_pad_masks[idx])
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if self.config.token_selection_enabled:
            return self._sample_actions_with_token_selection(
                images,
                img_masks,
                tokens,
                masks,
                noise=noise,
                num_steps=num_steps,
                **kwargs,
            )
        with torch.no_grad():
            if num_steps is None:
                num_steps = self.config.num_inference_steps

            bsize = tokens.shape[0]
            device = tokens.device

            if noise is None:
                # Sample noise with padded dimension as expected by action_in_proj
                actions_shape = (
                    bsize,
                    self.config.chunk_size,
                    self.config.max_action_dim,
                )  # Use config max_action_dim for internal processing
                noise = self.sample_noise(actions_shape, device)

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            dt = -1.0 / num_steps

            x_t = noise
            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

                def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                    return self.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        x_t=input_x_t,
                        timestep=current_timestep,
                    )

                if self._rtc_enabled():
                    inference_delay = kwargs.get("inference_delay")
                    prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                    execution_horizon = kwargs.get("execution_horizon")

                    v_t = self.rtc_processor.denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=time,
                        original_denoise_step_partial=denoise_step_partial_call,
                        execution_horizon=execution_horizon,
                    )
                else:
                    v_t = denoise_step_partial_call(x_t)

                x_t = x_t + dt * v_t

                if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                    self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

            return x_t

    def _sample_actions_with_token_selection(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        cfg = self.config
        if num_steps is None:
            num_steps = cfg.num_inference_steps

        if not cfg.use_diffusion:
            raise ValueError("PI05 token selection only supports diffusion decoding.")

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (
                bsize,
                cfg.chunk_size,
                cfg.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        token_state = self._token_selection_state
        eval_interval = cfg.region_eval_interval if cfg.region_eval_interval > 0 else 1
        if cfg.dynamic_eval_enabled:
            frames_since_eval = token_state.frame_idx - token_state.last_eval_frame_idx
            if token_state.frame_idx == 0:
                eval_frame = True
            elif frames_since_eval >= cfg.max_eval_interval:
                eval_frame = True  # hard-cap fallback
            elif token_state.last_entropy is not None and token_state.last_entropy > cfg.eval_entropy_threshold:
                eval_frame = True  # model confused (high entropy) -> re-evaluate full image
            else:
                eval_frame = False  # model focused -> keep pruning
        else:
            eval_frame = token_state.frame_idx % eval_interval == 0

        with torch.no_grad():
            image_embs = [self.paligemma_with_expert.embed_image(img) for img in images]

        patch_embs = []
        metas = []
        for emb in image_embs:
            patch_emb, meta = split_patch_tokens(emb)
            patch_embs.append(patch_emb)
            metas.append(meta)

        if token_state.last_score_token is not None:
            if len(token_state.last_score_token) != len(patch_embs):
                token_state.reset()
            else:
                for idx, score in enumerate(token_state.last_score_token):
                    if score.shape[0] != bsize or score.shape[1] != patch_embs[idx].shape[1]:
                        token_state.reset()
                        break

        region_embs = []
        background_regions = []
        valid_masks = []
        for idx, patch_emb in enumerate(patch_embs):
            meta = metas[idx]
            region_emb = compute_region_embeddings(patch_emb, meta.patches_per_side, cfg.region_patch_size)
            region_embs.append(region_emb)
            valid_mask = img_masks[idx][:, None].expand(bsize, region_emb.shape[1])
            valid_masks.append(valid_mask)
            prev_region = None
            if token_state.last_region_embs is not None and len(token_state.last_region_embs) == len(patch_embs):
                prev_region = token_state.last_region_embs[idx]
                if prev_region.shape != region_emb.shape:
                    prev_region = None
            temporal = compute_temporal_mask(region_emb, prev_region, cfg.token_temporal_threshold, valid_mask)
            spatial = compute_spatial_mask(
                region_emb,
                meta.patches_per_side,
                cfg.region_patch_size,
                cfg.token_spatial_radius,
                cfg.token_spatial_threshold,
            )
            background_regions.append(temporal & spatial & valid_mask)

        # ── Static background: capture reference on the very first frame ─────
        _bg_first_frame = False
        if cfg.static_bg_enabled and token_state.reference_region_embs is None:
            token_state.reference_region_embs = [emb.detach().clone() for emb in region_embs]
            _bg_first_frame = True

        token_norms = [torch.linalg.vector_norm(patch.float(), dim=-1) for patch in patch_embs]

        def compute_gate_weights(values: Tensor) -> tuple[Tensor, Tensor]:
            grip = values[..., -1]
            if grip.shape[1] > 1:
                m = (grip[:, 1:] - grip[:, :-1]).abs().mean(dim=1)
            else:
                m = grip.abs().mean(dim=1)
            if cfg.grad_tau <= 0:
                gate = torch.zeros_like(m)
            else:
                gate = (m / cfg.grad_tau).clamp(0.0, 1.0)
            lambda_grip = 1.0 + cfg.grad_alpha * gate
            lambda_pos = 1.0 + cfg.grad_beta * (1.0 - gate)
            return lambda_pos, lambda_grip

        def build_prefix(image_embs, image_pad_masks=None):
            return self.embed_prefix(
                images,
                img_masks,
                tokens,
                masks,
                image_embs=image_embs,
                image_pad_masks=image_pad_masks,
            )

        def compute_prefix_cache(prefix_embs, prefix_pad_masks, prefix_att_masks):
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            return past_key_values

        def run_diffusion(
            prefix_pad_masks,
            past_key_values,
            capture_attn: bool = False,
        ):
            dt = -1.0 / num_steps
            x_t = noise
            last_attn = None
            last_v_t = None
            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
                _attn_start = num_steps - max(1, min(cfg.attn_num_denoise_steps, num_steps))
                want_attn = capture_attn and step >= _attn_start

                def denoise_step_call(input_x_t, current_timestep=time_tensor):
                    return self.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        x_t=input_x_t,
                        timestep=current_timestep,
                        output_attentions=want_attn,
                    )

                if want_attn:
                    v_t_raw, _, attn = denoise_step_call(x_t)
                else:
                    v_t_raw = denoise_step_call(x_t)
                    attn = None

                if self._rtc_enabled():
                    inference_delay = kwargs.get("inference_delay")
                    prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                    execution_horizon = kwargs.get("execution_horizon")
                    v_t = self.rtc_processor.denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=time,
                        original_denoise_step_partial=lambda x: self.denoise_step(
                            prefix_pad_masks=prefix_pad_masks,
                            past_key_values=past_key_values,
                            x_t=x,
                            timestep=time_tensor,
                        ),
                        execution_horizon=execution_horizon,
                    )
                else:
                    v_t = v_t_raw

                x_t = x_t + dt * v_t

                if want_attn:
                    if last_attn is None:
                        last_attn = list(attn)  # list of per-layer tensors
                        _attn_count = 1
                    else:
                        for _li in range(len(attn)):
                            last_attn[_li] = last_attn[_li] + attn[_li]
                        _attn_count += 1
                    last_v_t = v_t_raw  # always use the final step's v_t
                if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                    self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
            # Average accumulated attention over collected denoise steps
            if last_attn is not None and _attn_count > 1:
                last_attn = tuple(a / _attn_count for a in last_attn)
            elif last_attn is not None:
                last_attn = tuple(last_attn)
            return x_t, last_v_t, last_attn

        score_tokens = None
        actions = None
        prefix_pad_masks = None
        last_attn = None
        last_v_t = None

        if eval_frame:
            if cfg.grad_score_method == "full_grad":
                image_embs = [emb.detach().requires_grad_(True) for emb in image_embs]
                prefix_embs, prefix_pad_masks, prefix_att_masks = build_prefix(image_embs)
                past_key_values = compute_prefix_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
                with torch.enable_grad():
                    grad_steps = max(1, min(cfg.grad_denoise_steps, num_steps))
                    start_step = num_steps - grad_steps
                    dt = -1.0 / num_steps
                    x_t = noise
                    objective = torch.zeros((), device=device)
                    for step in range(num_steps):
                        time = 1.0 + step * dt
                        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
                        v_t_raw, suffix_out, _ = self.denoise_step(
                            prefix_pad_masks=prefix_pad_masks,
                            past_key_values=past_key_values,
                            x_t=x_t,
                            timestep=time_tensor,
                            return_suffix_out=True,
                        )
                        proxy = suffix_out[..., : v_t_raw.shape[-1]] + (
                            v_t_raw - suffix_out[..., : v_t_raw.shape[-1]]
                        ).detach()
                        if step >= start_step:
                            lambda_pos, lambda_grip = compute_gate_weights(proxy)
                            pos_sq = (proxy[..., :-1].float() ** 2).sum(dim=-1)
                            grip_sq = (proxy[..., -1].float() ** 2)
                            objective = objective + (lambda_pos[:, None] * pos_sq + lambda_grip[:, None] * grip_sq).sum()
                        v_t = v_t_raw
                        x_t = x_t + dt * v_t
                        if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                            self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
                    grads = torch.autograd.grad(objective, image_embs, retain_graph=False, allow_unused=True)
                score_tokens = []
                for grad, emb, meta in zip(grads, image_embs, metas, strict=True):
                    if grad is None:
                        score_tokens.append(torch.zeros(emb.shape[0], meta.num_patches, device=emb.device))
                        continue
                    grad_patch = grad[:, meta.num_extra_tokens :, :]
                    emb_patch = emb[:, meta.num_extra_tokens :, :]
                    grad_norm = torch.linalg.vector_norm(grad_patch.float(), dim=-1)
                    emb_norm = torch.linalg.vector_norm(emb_patch.float(), dim=-1)
                    score_tokens.append(grad_norm * emb_norm)
                actions = x_t.detach()
            else:
                prefix_embs, prefix_pad_masks, prefix_att_masks = build_prefix(image_embs)
                past_key_values = compute_prefix_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
                x_t, last_v_t, last_attn = run_diffusion(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    capture_attn=True,
                )
                if last_attn is None or last_v_t is None:
                    raise ValueError("Missing attention output for attention-based scoring.")
                # Aggregate attention over last N Expert layers (mean pooling)
                _n_layers = max(1, min(cfg.attn_num_layers, len(last_attn)))
                if _n_layers == 1:
                    attn_weights = last_attn[-1]
                else:
                    attn_weights = torch.stack(last_attn[-_n_layers:], dim=0).mean(dim=0)
                action_len = cfg.chunk_size
                query_len = attn_weights.shape[2]
                action_end = query_len
                action_start = max(0, action_end - action_len)
                vision_indices = []
                offset = 0
                for meta in metas:
                    offset += meta.num_extra_tokens
                    vision_indices.extend(range(offset, offset + meta.num_patches))
                    offset += meta.num_patches
                vision_indices = torch.tensor(vision_indices, device=attn_weights.device)
                key_len = attn_weights.shape[3]
                if vision_indices.numel() > 0 and key_len > 0:
                    vision_indices = vision_indices[vision_indices < key_len]
                if vision_indices.numel() == 0 or action_start >= action_end:
                    score_tokens = token_norms
                else:
                    action_attn = attn_weights[:, :, action_start:action_end, vision_indices]
                    if action_attn.shape[2] == 0:
                        score_tokens = token_norms
                    else:
                        alpha = action_attn.float().clamp_min(1e-8).pow(cfg.attn_score_beta)
                        if cfg.grad_score_method == "attn_only":
                            if cfg.grad_action_agg == "max":
                                score = alpha.max(dim=2).values.sum(dim=1)
                            else:
                                score = alpha.sum(dim=(1, 2))
                        else:
                            pos = last_v_t[..., :-1]
                            grip = last_v_t[..., -1:]
                            if cfg.partial_grad_phi == "l1":
                                pos_grad = pos.sign()
                                grip_grad = grip.sign()
                            else:
                                pos_grad = 2 * pos
                                grip_grad = 2 * grip
                            grad_action = torch.cat(
                                [
                                    pos_grad * cfg.partial_grad_pos_weight,
                                    grip_grad * cfg.partial_grad_grip_weight,
                                ],
                                dim=-1,
                            )
                            last_layer = self.paligemma_with_expert.gemma_expert.model.layers[-1]
                            wo = last_layer.self_attn.o_proj.weight
                            hidden_size = wo.shape[0]
                            action_dim = grad_action.shape[-1]
                            if action_dim >= hidden_size:
                                grad_hidden = grad_action[..., :hidden_size]
                            else:
                                grad_hidden = torch.zeros(
                                    bsize, action_len, hidden_size, device=grad_action.device, dtype=grad_action.dtype
                                )
                                grad_hidden[..., :action_dim] = grad_action
                            grad_hidden = grad_hidden.to(dtype=wo.dtype)
                            grad_attn = torch.matmul(grad_hidden.reshape(-1, hidden_size), wo)
                            head_dim = last_layer.self_attn.head_dim
                            num_heads = grad_attn.shape[1] // head_dim
                            grad_attn = grad_attn.view(bsize, action_len, num_heads, head_dim)
                            g_head_norm = torch.linalg.vector_norm(grad_attn.float(), dim=-1)
                            if cfg.grad_head_beta != 1.0:
                                g_head_norm = g_head_norm.clamp_min(1e-8).pow(cfg.grad_head_beta)
                            if cfg.grad_head_norm == "max":
                                denom = g_head_norm.max(dim=-1, keepdim=True).values
                            else:
                                denom = g_head_norm.sum(dim=-1, keepdim=True)
                            head_count = g_head_norm.shape[-1]
                            g_head_norm = torch.where(
                                denom > 0,
                                g_head_norm / denom,
                                torch.full_like(g_head_norm, 1.0 / head_count),
                            )
                            g_weight = g_head_norm.permute(0, 2, 1).unsqueeze(-1)
                            if g_weight.shape[2] != alpha.shape[2]:
                                g_weight = g_weight[:, :, -alpha.shape[2] :, :]
                            if cfg.grad_action_agg == "max":
                                score = (alpha * g_weight).max(dim=2).values.sum(dim=1)
                            else:
                                score = (alpha * g_weight).sum(dim=(1, 2))
                        score_tokens = []
                        start = 0
                        for meta in metas:
                            end = start + meta.num_patches
                            score_tokens.append(score[:, start:end])
                            start = end
                actions = x_t.detach()
        else:
            if token_state.last_score_token is not None:
                score_tokens = token_state.last_score_token
            else:
                score_tokens = token_norms

        score_regions = []
        important_regions = []
        keep_regions = []
        clipped_regions = []
        static_bg_regions = []
        raw_cosine_static_regions = []
        keep_token_masks = []
        important_token_masks = []
        effective_important_token_masks = []
        overlay_labels = []
        overlay_grids = []
        for idx, score_token in enumerate(score_tokens):
            meta = metas[idx]
            score_region = compute_region_scores(score_token, meta.patches_per_side, cfg.region_patch_size)
            if eval_frame and cfg.grad_region_ema > 0 and token_state.last_score_region is not None:
                score_region = (
                    cfg.grad_region_ema * token_state.last_score_region[idx]
                    + (1.0 - cfg.grad_region_ema) * score_region
                )
            # ── Dynamic pruning: scale aggressiveness by region-score entropy ──
            # Normalised entropy h ∈ [0,1] of region score distribution:
            #   h=0 → attention fully focused → aggressive pruning
            #   h=1 → attention fully diffuse → conservative (keep more)
            if cfg.dynamic_mass_enabled:
                import math as _math
                _r = score_region.float()  # [B, R]
                _r_total = _r.sum(dim=-1, keepdim=True)
                _r_norm = torch.where(
                    _r_total > 0, _r / _r_total,
                    torch.full_like(_r, 1.0 / _r.shape[-1]),
                )
                _r_ent = -(_r_norm * torch.log(_r_norm.clamp_min(1e-8))).sum(dim=-1)
                _h = (_r_ent / _math.log(_r.shape[-1])).mean().item()  # ∈ [0,1]
                if cfg.pruning_method == "topk_ratio":
                    _effective_param = cfg.keep_ratio_low + _h * (cfg.keep_ratio_high - cfg.keep_ratio_low)
                else:  # "mass"
                    _effective_param = cfg.mass_low + _h * (cfg.mass_high - cfg.mass_low)
            else:
                _effective_param = cfg.grad_region_mass  # static fallback (mass mode)

            def _select_important(sr):
                if cfg.pruning_method == "topk_ratio" and cfg.dynamic_mass_enabled:
                    return select_regions_by_topk_ratio(sr, _effective_param)
                else:
                    return select_regions_by_mass(sr, _effective_param)

            if eval_frame:
                important_region = _select_important(score_region)
                if cfg.grad_keep_prev and token_state.last_important_region is not None:
                    important_region = important_region | token_state.last_important_region[idx]
            else:
                if token_state.last_important_region is not None:
                    important_region = token_state.last_important_region[idx]
                else:
                    important_region = _select_important(score_region)
            # ── Static background detection (cosine sim to first-frame ref) ────
            static_bg_region = None
            static_bg_token = None
            raw_cosine_static_region = None
            raw_cosine_static_token = None
            if (cfg.static_bg_enabled
                    and idx == cfg.static_bg_camera_idx
                    and not _bg_first_frame
                    and token_state.reference_region_embs is not None):
                static_bg_region, raw_cosine_static_region = detect_static_background(
                    region_embs[idx],
                    token_state.reference_region_embs[idx],
                    cfg.static_bg_threshold,
                    valid_masks[idx],
                    score_region=score_region,
                    score_gate=cfg.static_bg_score_gate,
                )
                static_bg_token = expand_region_mask(
                    static_bg_region, meta.patches_per_side, cfg.region_patch_size,
                )
                raw_cosine_static_token = expand_region_mask(
                    raw_cosine_static_region, meta.patches_per_side, cfg.region_patch_size,
                )
            keep_pre = important_region & valid_masks[idx]
            if static_bg_region is not None:
                keep_pre = keep_pre & ~static_bg_region
            # When dynamic_mass is enabled, mass already controls how many
            # regions to keep — skip the hard min/max clamp so the adaptive
            # mass has full effect.  Otherwise use fixed min/max counts.
            if cfg.dynamic_mass_enabled:
                # trust mass-based selection; no hard count clamp
                keep_region = keep_pre
                clipped_region = torch.zeros_like(keep_pre, dtype=torch.bool)
            else:
                region_area = cfg.region_patch_size * cfg.region_patch_size
                min_regions = math.ceil(cfg.min_kept_tokens / region_area) if cfg.min_kept_tokens > 0 else 0
                max_regions = (
                    math.floor(cfg.max_kept_tokens / region_area)
                    if cfg.max_kept_tokens > 0
                    else score_region.shape[1]
                )
                keep_region, clipped_region = apply_min_max_constraints(
                    keep_pre, score_region, min_regions, max_regions
                )
            keep_token = expand_region_mask(keep_region, meta.patches_per_side, cfg.region_patch_size)
            important_token = expand_region_mask(important_region, meta.patches_per_side, cfg.region_patch_size)
            effective_important_token = important_token & keep_token
            clipped_token = expand_region_mask(clipped_region, meta.patches_per_side, cfg.region_patch_size)
            overlay_label = build_overlay_labels(keep_token, important_token, clipped_token, static_bg=static_bg_token, raw_cosine_static=raw_cosine_static_token)
            overlay_labels.append(overlay_label)
            overlay_grids.append(overlay_label.view(bsize, meta.patches_per_side, meta.patches_per_side))
            score_regions.append(score_region)
            important_regions.append(important_region)
            keep_regions.append(keep_region)
            clipped_regions.append(clipped_region)
            static_bg_regions.append(static_bg_region)
            raw_cosine_static_regions.append(raw_cosine_static_region)
            keep_token_masks.append(keep_token)
            important_token_masks.append(important_token)
            effective_important_token_masks.append(effective_important_token)

        # Build mask that removes ONLY static-bg tokens (for eval-frame pruning)
        _has_static_bg = any(s is not None for s in static_bg_regions)
        if _has_static_bg:
            static_bg_keep_masks = []
            for idx, sbg in enumerate(static_bg_regions):
                if sbg is not None:
                    sbg_token = expand_region_mask(
                        sbg, metas[idx].patches_per_side, cfg.region_patch_size,
                    )
                    static_bg_keep_masks.append(~sbg_token)  # keep = NOT background
                else:
                    # No bg detection for this camera — keep all
                    static_bg_keep_masks.append(
                        torch.ones(bsize, metas[idx].num_patches, dtype=torch.bool,
                                   device=region_embs[idx].device)
                    )
        else:
            static_bg_keep_masks = None

        if eval_frame and (cfg.overlay_show_scores or cfg.overlay_show_ids):
            base_dir = cfg.local_log_dir or cfg.rollout_dir
            if base_dir:
                out_dir = Path(base_dir)
                if cfg.run_id_note:
                    out_dir = out_dir / cfg.run_id_note
                out_dir.mkdir(parents=True, exist_ok=True)
                for img_idx, meta in enumerate(metas):
                    region_h = meta.patches_per_side // cfg.region_patch_size
                    region_w = region_h
                    for batch_idx in range(bsize):
                        img = images[img_idx][batch_idx]
                        if img.shape[0] == 3:
                            img = img.permute(1, 2, 0)
                        img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        _sbg = (static_bg_regions[img_idx][batch_idx].cpu()
                               if static_bg_regions[img_idx] is not None else None)
                        _rcs = (raw_cosine_static_regions[img_idx][batch_idx].cpu()
                                if raw_cosine_static_regions[img_idx] is not None else None)
                        overlay = build_overlay(
                            img,
                            region_h,
                            region_w,
                            cfg.region_patch_size,
                            important_regions[img_idx][batch_idx].cpu(),
                            keep_regions[img_idx][batch_idx].cpu(),
                            clipped_regions[img_idx][batch_idx].cpu(),
                            cfg.overlay_show_scores,
                            cfg.overlay_show_ids,
                            scores=score_regions[img_idx][batch_idx].detach().cpu(),
                            static_bg=_sbg,
                            raw_cosine_static=_rcs,
                        )
                        try:
                            from PIL import Image
                        except ImportError:
                            continue
                        Image.fromarray(overlay.numpy()).save(
                            out_dir / f"overlay_f{token_state.frame_idx:06d}_i{img_idx}_b{batch_idx}.png"
                        )

        if not eval_frame and cfg.vision_partial_update_enabled:
            image_embs = apply_partial_update(
                image_embs, token_state.last_image_embs, effective_important_token_masks, metas
            )

        cache_image_embs = image_embs

        image_pad_masks = None
        if not eval_frame and cfg.token_prune_enabled:
            image_embs, image_pad_masks = prune_image_embeddings(
                image_embs, keep_token_masks, img_masks, metas
            )
        elif eval_frame and cfg.token_prune_enabled and _has_static_bg and static_bg_keep_masks is not None:
            # Even on eval frames, prune static-background tokens
            image_embs, image_pad_masks = prune_image_embeddings(
                image_embs, static_bg_keep_masks, img_masks, metas
            )

        if actions is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = build_prefix(image_embs, image_pad_masks)
            past_key_values = compute_prefix_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
            with torch.no_grad():
                actions, _, _ = run_diffusion(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                )
                actions = actions.detach()

        token_state.last_score_token = [score.detach() for score in score_tokens]
        token_state.last_score_region = [score.detach() for score in score_regions]
        token_state.last_important_region = [mask.detach() for mask in important_regions]
        token_state.last_region_embs = [emb.detach() for emb in region_embs]
        token_state.last_image_embs = [emb.detach() for emb in cache_image_embs]
        token_state.last_overlay_labels = [label.detach() for label in overlay_labels]
        token_state.last_overlay_grid = [grid.detach() for grid in overlay_grids]

        # Compute entropy of score distribution for dynamic eval interval
        if eval_frame and score_tokens:
            token_state.eval_frame_count += 1
            token_state.last_eval_frame_idx = token_state.frame_idx
            _entropy_sum = 0.0
            for _st in score_tokens:
                # z-score normalise before softmax so that tiny differences between
                # summed attention weights (≈ 0.31 ± tiny) are amplified into a
                # meaningful distribution (otherwise entropy is always ≈ ln(N_tokens))
                _s = _st.float()
                _mean = _s.mean(dim=-1, keepdim=True)
                _std  = _s.std(dim=-1, keepdim=True).clamp_min(1e-8)
                _probs = torch.nn.functional.softmax((_s - _mean) / _std, dim=-1)
                _ent = -(_probs * torch.log(_probs + 1e-8)).sum(dim=-1)
                _entropy_sum += _ent.mean().item()
            token_state.last_entropy = _entropy_sum / len(score_tokens)

        if cfg.token_selection_enabled and token_state.frame_idx > 0:
            _eval_rate = token_state.eval_frame_count / (token_state.frame_idx + 1)
            _avg_interval = 1.0 / _eval_rate if _eval_rate > 0 else float("inf")
            _entropy_str = f"{token_state.last_entropy:.3f}" if token_state.last_entropy is not None else "N/A"
            # Store stats in token_state so the eval loop can show them in tqdm postfix
            # Only count tokens from *valid* images (img_masks=True); placeholder
            # images (missing cameras) have img_mask=False and should not inflate
            # the total count or appear as "pruned" tokens.
            _valid_masks = [
                m for m, im in zip(keep_token_masks, img_masks)
                if im.any()
            ] if keep_token_masks else []
            _marked_n = sum(int(m.sum().item()) for m in _valid_masks)
            _total_n = sum(m.shape[-1] for m in _valid_masks)
            # Count actually-kept tokens this frame:
            #  - non-eval + prune_enabled: standard keep_token_masks
            #  - eval + static_bg: only bg tokens are pruned
            #  - otherwise: full tokens (no pruning)
            if not cfg.token_prune_enabled:
                _actual_kept = _total_n
            elif not eval_frame:
                _actual_kept = _marked_n
            elif _has_static_bg and static_bg_keep_masks is not None:
                _sbg_valid = [
                    m for m, im in zip(static_bg_keep_masks, img_masks)
                    if im.any()
                ]
                _actual_kept = sum(int(m.sum().item()) for m in _sbg_valid)
            else:
                _actual_kept = _total_n
            _prune_pct = (1.0 - _actual_kept / _total_n) * 100.0 if _total_n > 0 else 0.0
            # Accumulate for per-episode average pruning ratio
            token_state.total_kept += _actual_kept
            token_state.total_possible += _total_n
            token_state.last_stats = {
                "entropy": _entropy_str,
                "avg_ivl": f"{_avg_interval:.1f}f ({_eval_rate:.0%})",
                "kept": f"{_actual_kept}/{_total_n}",
                "prune": f"{_prune_pct:.1f}%",
                "eval": "Y" if eval_frame else "N",
            }

        token_state.frame_idx += 1

        return actions

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        output_attentions: bool = False,
        return_suffix_out: bool = False,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        attentions = None
        if output_attentions:
            outputs_embeds, _, attentions = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_attentions=True,
            )
        else:
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if output_attentions or return_suffix_out:
            return v_t, suffix_out, attentions
        return v_t


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Handle tied weights: embed_tokens.weight == lm_head.weight in Gemma
            # Checkpoints usually only save lm_head.weight; add embed_tokens alias if absent
            _missing_embed_keys = [
                name for name, _ in model.named_parameters()
                if name.endswith("embed_tokens.weight") and name not in remapped_state_dict
            ]
            for _ek in _missing_embed_keys:
                # Find the corresponding lm_head.weight in the same model scope
                _lm_head_key = _ek.replace("embed_tokens.weight", "lm_head.weight")
                # Also try language_model.lm_head.weight path variants
                _alt_key = _ek.rsplit("embed_tokens.weight", 1)[0] + "lm_head.weight"
                _src_key = _lm_head_key if _lm_head_key in remapped_state_dict else (
                    _alt_key if _alt_key in remapped_state_dict else None
                )
                if _src_key:
                    remapped_state_dict[_ek] = remapped_state_dict[_src_key]
                    print(f"Tied weight: {_ek} <- {_src_key}")
                else:
                    print(f"[WARN] Could not resolve tied weight for {_ek}")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if hasattr(self, "model"):
            self.model.reset_token_selection_state()

    def get_token_selection_state(self) -> dict[str, Tensor] | None:
        if not hasattr(self, "model") or not hasattr(self.model, "_token_selection_state"):
            return None
        state = self.model._token_selection_state
        if state is None:
            return None

        overlay_grid = None
        if state.last_overlay_grid:
            overlay_grid = torch.stack(state.last_overlay_grid, dim=0)

        token_scores = None
        if state.last_score_token:
            token_scores = torch.stack(state.last_score_token, dim=0)

        region_scores = None
        if state.last_score_region:
            region_scores = torch.stack(state.last_score_region, dim=0)

        return {
            "last_overlay_grid": overlay_grid,
            "last_token_scores": token_scores,
            "last_region_scores": region_scores,
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
