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
    apply_min_max_constraints,
    build_overlay,
    build_overlay_labels,
    compute_relative_deviation,
    compute_region_scores,
    discard_top_scoring_regions,
    expand_region_mask,
    prune_image_embeddings,
    ratio_to_keep_tokens,
    sigmoid_ratio,
    split_patch_tokens,
    update_ema_mean_and_var,
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
            if (self.config.interp_plot_actions_l1
                    and self._token_selection_state.actions_l1_history):
                if not hasattr(self, '_plot_episode_idx'):
                    self._plot_episode_idx = 0
                self._save_actions_l1_plot(self._token_selection_state, self._plot_episode_idx)
                self._plot_episode_idx += 1
            self._token_selection_state.reset()

    def _save_actions_l1_plot(self, state, episode_idx):
        """Save a line chart of the predicted-action L1 norm at each inference frame."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        history = state.actions_l1_history
        if not history:
            return
        cfg = self.config
        base_dir = cfg.local_log_dir or cfg.rollout_dir or "outputs/eval/actions_l1"
        out_dir = Path(base_dir)
        if cfg.run_id_note:
            out_dir = out_dir / cfg.run_id_note
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = [h[0] for h in history]
        l1_norms = [h[1] for h in history]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, l1_norms, marker="o", markersize=3,
                linewidth=1.2, color="tab:blue", label="Action L1 Norm")
        ax.set_xlabel("Inference Frame Index")
        ax.set_ylabel("Action L1 Norm")
        ax.set_title("Predicted Action L1 Norm over Time")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"actions_l1_episode_{episode_idx}.png", dpi=120)
        plt.close(fig)

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
        if cfg.score_debug_heatmap:
            eval_frame = True  # debug heatmap: evaluate every frame
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

        valid_masks = []
        for idx, meta in enumerate(metas):
            num_regions = (meta.patches_per_side // cfg.region_patch_size) ** 2
            valid_masks.append(img_masks[idx][:, None].expand(bsize, num_regions))

        token_norms = [torch.linalg.vector_norm(patch.float(), dim=-1) for patch in patch_embs]

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

        if eval_frame:
            if cfg.grad_score_method == "transformer_interpretability":
                # ── Gradient-Weighted Attention (GradxAttn) ───────────────
                # For each scored denoise step:
                #   1. Forward with x_t.requires_grad_(True).
                #   2. Objective: ||v_t||_F^2.
                #   3. grad_x = dobj/dx_t -> per-action importance w[a]=||grad_x[a]||
                #   4. Aggregate attention over last N Expert layers.
                #   5. score_j = sum_a w[a] * sum_h A[h,a,j]
                #   6. Accumulate across scored denoise steps, then average.
                prefix_embs, prefix_pad_masks, prefix_att_masks = build_prefix(image_embs)
                past_key_values = compute_prefix_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

                _interp_step = cfg.interp_denoise_step  # -1 = all steps, >=0 = specific
                _dt = -1.0 / num_steps
                x_t = noise.clone()
                _score_accum = None  # [B, total_vis_tokens]
                _score_count = 0
                _n_layers = max(1, min(
                    cfg.attn_num_layers,
                    len(self.paligemma_with_expert.gemma_expert.model.layers),
                ))

                for _step in range(num_steps):
                    _time = 1.0 + _step * _dt
                    _time_t = torch.tensor(_time, dtype=torch.float32, device=device).expand(bsize)
                    _need_score = (_interp_step < 0 or _interp_step == _step)

                    if _need_score:
                        _x_in = x_t.detach().requires_grad_(True)
                        with torch.enable_grad():
                            _v_raw, _, _step_attns = self.denoise_step(
                                prefix_pad_masks=prefix_pad_masks,
                                past_key_values=past_key_values,
                                x_t=_x_in,
                                timestep=_time_t,
                                output_attentions=True,
                                return_suffix_out=True,
                            )
                            # ── Action-step subset for objective ──
                            _a_lo = max(0, cfg.interp_action_start)
                            _a_hi = cfg.chunk_size if cfg.interp_action_end < 0 else min(cfg.interp_action_end, cfg.chunk_size)
                            _attn_list = list(_step_attns)

                            # ── Compute objective target ──
                            if cfg.interp_objective == "action_sample_L1":
                                # x_0_hat = x_t - t * v_theta
                                _obj_target = (_x_in[:, _a_lo:_a_hi, :] - _time * _v_raw[:, _a_lo:_a_hi, :]).float()
                            else:
                                # vector_field_L2 (default): use v_raw directly
                                _obj_target = _v_raw[:, _a_lo:_a_hi, :].float()

                            if cfg.interp_variant == "dimension_independent":
                                # Per-dimension backprop: A_bar = (1/D) sum_d mean_h(A * |dv_d/dA|)
                                _v_sel = _obj_target[:, :, :7]  # 只用前7维
                                _action_dim = 7
                                if cfg.interp_use_residual:
                                    _dim_A_bars = None
                                    for _d in range(_action_dim):
                                        _obj_d = _v_sel[:, :, _d].sum()
                                        _gs = torch.autograd.grad(
                                            _obj_d, _attn_list,
                                            retain_graph=(_d < _action_dim - 1),
                                        )
                                        if _dim_A_bars is None:
                                            _dim_A_bars = [
                                                (_attn_list[l].detach().float() * _gs[l].float().abs()).mean(dim=1)
                                                for l in range(len(_attn_list))
                                            ]
                                        else:
                                            for l in range(len(_attn_list)):
                                                _dim_A_bars[l] = _dim_A_bars[l] + (
                                                    _attn_list[l].detach().float() * _gs[l].float().abs()
                                                ).mean(dim=1)
                                    for l in range(len(_attn_list)):
                                        _dim_A_bars[l] = _dim_A_bars[l] / _action_dim
                                else:
                                    _last_attn = _attn_list[-1]
                                    _A_last_d = _last_attn.detach().float()
                                    _dim_A_bar = None
                                    for _d in range(_action_dim):
                                        _obj_d = _v_sel[:, :, _d].sum()
                                        _gd = torch.autograd.grad(
                                            _obj_d, _last_attn,
                                            retain_graph=(_d < _action_dim - 1),
                                        )[0]
                                        _c = (_A_last_d * _gd.float().abs()).mean(dim=1)
                                        _dim_A_bar = _c if _dim_A_bar is None else _dim_A_bar + _c
                                    _dim_A_bar = _dim_A_bar / _action_dim
                            else:
                                if cfg.interp_objective == "action_sample_L1":
                                    _obj = _obj_target.abs().sum()
                                else:
                                    _obj = (_obj_target ** 2).sum()
                                if cfg.interp_use_residual:
                                    _grads = torch.autograd.grad(
                                        _obj, _attn_list, retain_graph=False,
                                    )
                                else:
                                    _last_attn = _attn_list[-1]
                                    _grad_last = torch.autograd.grad(
                                        _obj, _last_attn, retain_graph=False,
                                    )[0]

                        # -- build vision-token indices (shared by both paths) --
                        _ref_A = _step_attns[-1]
                        _suffix_len = _ref_A.shape[2]
                        _total_len = _ref_A.shape[3]
                        _prefix_len = _total_len - _suffix_len
                        _action_len = cfg.chunk_size
                        _action_end = _suffix_len
                        _action_start = max(0, _action_end - _action_len)
                        # Narrow to user-selected action-step subset
                        _q_a_start = _action_start + _a_lo
                        _q_a_end = _action_start + _a_hi
                        _sel_alen = _a_hi - _a_lo

                        _vis_idx = []
                        _off = 0
                        for meta in metas:
                            _off += meta.num_extra_tokens
                            _vis_idx.extend(range(_off, _off + meta.num_patches))
                            _off += meta.num_patches
                        _vis_idx_t = torch.tensor(_vis_idx, device=device)
                        if _vis_idx_t.numel() > 0 and _total_len > 0:
                            _vis_idx_t = _vis_idx_t[_vis_idx_t < _total_len]

                        if cfg.interp_use_residual:
                            # ── Residual propagation: R^l = R^{l-1} @ Â^l ──
                            _num_expert_layers = len(_attn_list)
                            _diag = torch.arange(_suffix_len, device=device)
                            # R: [B, suffix_len, total_len] — identity init
                            _R = torch.zeros(bsize, _suffix_len, _total_len,
                                             device=device, dtype=torch.float32)
                            _R[:, _diag, _prefix_len + _diag] = 1.0

                            for _ell in range(_num_expert_layers):
                                if cfg.interp_variant == "dimension_independent":
                                    _A_bar = _dim_A_bars[_ell]
                                elif cfg.interp_variant == "abs_heads":
                                    _A_ell = _attn_list[_ell].detach().float()
                                    _g_ell = _grads[_ell].float()
                                    _A_bar = (_g_ell * _A_ell).abs().mean(dim=1)  # [B, Qs, K]
                                else:  # "original"
                                    _A_ell = _attn_list[_ell].detach().float()
                                    _g_ell = _grads[_ell].float()
                                    _ga = (_g_ell * _A_ell).mean(dim=1)  # [B, Qs, K]
                                    _A_bar = torch.relu(_ga)
                                _A_hat = _A_bar.clone()
                                _A_hat[:, _diag, _prefix_len + _diag] += 1.0
                                # Row-normalize
                                _row_sum = _A_hat.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                                _A_hat = _A_hat / _row_sum
                                # R_new = R @ Â_full  (prefix rows = identity)
                                _R_s = _R[:, :, _prefix_len:]   # [B, Qs, Qs]
                                _R_p = _R[:, :, :_prefix_len] + torch.bmm(_R_s, _A_hat[:, :, :_prefix_len])
                                _R_sf = torch.bmm(_R_s, _A_hat[:, :, _prefix_len:])
                                _R = torch.cat([_R_p, _R_sf], dim=-1)

                            # Extract action→vision relevance (selected action subset)
                            if _vis_idx_t.numel() > 0 and _sel_alen > 0:
                                _step_score = _R[:, _q_a_start:_q_a_end, :][:, :, _vis_idx_t]
                                _step_score = _step_score.mean(dim=1)   # [B, num_vis]
                                _step_score = torch.relu(_step_score) if cfg.interp_variant == "original" else torch.abs(_step_score)
                            else:
                                _step_score = None
                        else:
                            # -- Last-layer scoring --
                            if cfg.interp_variant == "dimension_independent":
                                _A_bar = _dim_A_bar
                            elif cfg.interp_variant == "abs_heads":
                                _A_last = _last_attn.detach().float()
                                _A_bar = (_grad_last.float() * _A_last).abs().mean(dim=1)
                            else:  # "original"
                                _A_last = _last_attn.detach().float()
                                _ga = (_grad_last.float() * _A_last).mean(dim=1)
                                _A_bar = torch.relu(_ga)

                            if _vis_idx_t.numel() > 0 and _sel_alen > 0:
                                _step_score = _A_bar[:, _q_a_start:_q_a_end, :][:, :, _vis_idx_t]
                                _step_score = _step_score.mean(dim=1)  # [B, num_vis]
                                _step_score = torch.relu(_step_score) if cfg.interp_variant == "original" else torch.abs(_step_score)
                            else:
                                _step_score = None

                        if _step_score is not None:
                            if _score_accum is None:
                                _score_accum = _step_score
                            else:
                                _score_accum = _score_accum + _step_score
                            _score_count += 1

                        _v_t = _v_raw.detach()
                    else:
                        with torch.no_grad():
                            _v_t = self.denoise_step(
                                prefix_pad_masks=prefix_pad_masks,
                                past_key_values=past_key_values,
                                x_t=x_t,
                                timestep=_time_t,
                            )

                    x_t = x_t.detach() + _dt * _v_t.detach()

                # Average across scored steps
                if _score_count > 1:
                    _score_accum = _score_accum / _score_count

                if _score_accum is None:
                    score_tokens = token_norms
                else:
                    score_tokens = []
                    _start = 0
                    for meta in metas:
                        _end = _start + meta.num_patches
                        score_tokens.append(_score_accum[:, _start:_end])
                        _start = _end
                actions = x_t.detach()


            elif cfg.grad_score_method == "attn_only":
                prefix_embs, prefix_pad_masks, prefix_att_masks = build_prefix(image_embs)
                past_key_values = compute_prefix_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
                x_t, _, last_attn = run_diffusion(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    capture_attn=True,
                )
                if last_attn is None:
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
                        if cfg.grad_action_agg == "max":
                            score = alpha.max(dim=2).values.sum(dim=1)
                        else:
                            score = alpha.sum(dim=(1, 2))
                        score_tokens = []
                        start = 0
                        for meta in metas:
                            end = start + meta.num_patches
                            score_tokens.append(score[:, start:end])
                            start = end
                actions = x_t.detach()
            else:
                raise ValueError(f"Unsupported grad_score_method: {cfg.grad_score_method}")
        else:
            if token_state.last_score_token is not None:
                score_tokens = token_state.last_score_token
            else:
                score_tokens = token_norms

        score_regions = []
        important_regions = []
        keep_regions = []
        clipped_regions = []
        keep_token_masks = []
        overlay_labels = []
        overlay_grids = []
        if token_state.global_active_region_masks is None:
            token_state.global_active_region_masks = [torch.ones_like(vm) for vm in valid_masks]
        _restore_global_pool_now = (
            eval_frame
            and cfg.reset_discard_pool_on_gripper_close
            and token_state.pending_global_pool_restore
        )
        if _restore_global_pool_now:
            token_state.global_active_region_masks = [torch.ones_like(vm) for vm in valid_masks]
            token_state.pending_global_pool_restore = False
        for idx, score_token in enumerate(score_tokens):
            meta = metas[idx]
            score_region = compute_region_scores(score_token, meta.patches_per_side, cfg.region_patch_size)
            if eval_frame and cfg.grad_region_ema > 0 and token_state.last_score_region is not None:
                score_region = (
                    cfg.grad_region_ema * token_state.last_score_region[idx]
                    + (1.0 - cfg.grad_region_ema) * score_region
                )

            # --- Global Active Token Pool: Discard previous top-scoring regions ---
            if eval_frame and not _restore_global_pool_now and cfg.discard_prev_kept_ratio > 0.0:
                if (token_state.last_score_region is not None and token_state.last_keep_grid is not None
                        and len(token_state.last_score_region) > idx and len(token_state.last_keep_grid) > idx):
                    _prev_scores = token_state.last_score_region[idx]
                    
                    # keep_grid is stored as a boolean grid, potentially [H, W] or [B, H, W]
                    _prev_kept_tokens = token_state.last_keep_grid[idx]
                    
                    # Store original shape to revert later if needed, though we just need it to match _prev_scores shape
                    # which is [NumRegions] when unbatched, or [B, NumRegions] when batched.
                    _is_batched = (_prev_scores.dim() == 2)
                    
                    # Force it to [B, NumPatches] internally for compute_region_scores
                    _prev_kept_flat = _prev_kept_tokens.reshape(-1) # [B * NumPatches] or [NumPatches]
                    _num_patches = metas[idx].num_patches
                    if len(_prev_kept_flat) == _num_patches:
                        # It was unbatched
                        _prev_kept_flat = _prev_kept_flat.unsqueeze(0) # [1, NumPatches]
                    else:
                        # It was batched
                        _b = len(_prev_kept_flat) // _num_patches
                        _prev_kept_flat = _prev_kept_flat.view(_b, _num_patches) # [B, NumPatches]
                        
                    _pps = metas[idx].patches_per_side
                    _rps = cfg.region_patch_size
                    
                    # compute_region_scores expects [B, NumPatches]
                    _prev_kept_regions = compute_region_scores(_prev_kept_flat.float(), _pps, _rps) > 0.5
                    
                    # Revert to original batching structure
                    if not _is_batched:
                        _prev_kept_regions = _prev_kept_regions.squeeze(0)
                    
                    # Safeguard: Do not deplete global pool below min_kept_tokens
                    _current_active = token_state.global_active_region_masks[idx].sum(dim=-1)
                    _max_discard = torch.clamp_min(_current_active - cfg.min_kept_tokens, 0)
                    
                    _discard_mask = discard_top_scoring_regions(
                        _prev_scores, _prev_kept_regions, cfg.discard_prev_kept_ratio, _max_discard,
                        mode=cfg.discard_mode
                    )
                    # Permanently remove from global active pool
                    token_state.global_active_region_masks[idx] &= ~_discard_mask
            
            # Sub-select valid masks by what is left in the global pool
            _global_mask = token_state.global_active_region_masks[idx]
            valid_masks[idx] = valid_masks[idx] & _global_mask
            
            # Mask out discarded tokens from being selected (score=-inf)
            _eff_score_region = torch.where(valid_masks[idx], score_region, torch.full_like(score_region, -float('inf')))
            # ── Debug heatmap: skip all pruning / bg logic, just collect scores ──
            if cfg.score_debug_heatmap:
                num_r = score_region.shape[1]
                num_p = meta.num_patches
                pps = meta.patches_per_side
                score_regions.append(score_region)
                important_regions.append(torch.ones(bsize, num_r, dtype=torch.bool, device=device))
                keep_regions.append(torch.ones(bsize, num_r, dtype=torch.bool, device=device))
                clipped_regions.append(torch.zeros(bsize, num_r, dtype=torch.bool, device=device))
                keep_token_masks.append(torch.ones(bsize, num_p, dtype=torch.bool, device=device))
                overlay_labels.append(torch.zeros(bsize, num_p, dtype=torch.long, device=device))
                overlay_grids.append(torch.zeros(bsize, pps, pps, dtype=torch.long, device=device))
                continue

            # ── 3. Pruning Ratio: select "important" regions ─────────────────
            def _select_important():
                # Fixed-count TopK is hardcoded: start from all valid regions,
                # then apply min/max keep-token constraints below.
                return valid_masks[idx].clone()

            if eval_frame:
                important_region = _select_important()
                if cfg.grad_keep_prev and token_state.last_important_region is not None:
                    # Also prune the previous kept memory against the global mask so we 
                    # don't accidentally resurrect permanently discarded tokens
                    _prev_imp = token_state.last_important_region[idx] & valid_masks[idx]
                    important_region = important_region | _prev_imp
            else:
                if token_state.last_important_region is not None:
                    important_region = token_state.last_important_region[idx] & valid_masks[idx]
                else:
                    important_region = _select_important()
            keep_pre = important_region & valid_masks[idx]
            # min/max guardrails: always applied in all pruning modes
            # ── L1 Dynamic Prune Ratio: select effective min/max based on
            #    previous frame's action L1 norm ──
            if cfg.dynamic_prune_mode == "l1_threshold":
                _prev_l1 = token_state.last_actions_l1
                if _prev_l1 is not None and _prev_l1 > cfg.l1_dynamic_prune_threshold:
                    # High L1 → aggressive pruning (fewer tokens kept)
                    _eff_min = cfg.min_kept_tokens
                    _eff_max = cfg.min_kept_tokens
                else:
                    # Low L1 or first frame → conservative (more tokens kept)
                    _eff_min = cfg.max_kept_tokens
                    _eff_max = cfg.max_kept_tokens
            elif cfg.dynamic_prune_mode == "ema":
                _ema = token_state.ema_l1
                _raw_l1 = token_state.last_actions_l1
                if _ema is not None and _raw_l1 is not None and _ema > 1e-8:
                    _dev = compute_relative_deviation(_raw_l1, _ema)
                    assert _dev is not None
                    _ratio = sigmoid_ratio(_dev)
                    _eff = ratio_to_keep_tokens(
                        _ratio,
                        cfg.min_kept_tokens,
                        cfg.max_kept_tokens,
                        cfg.ema_direction,
                    )
                    _eff_min = _eff
                    _eff_max = _eff
                else:
                    _eff_min = cfg.max_kept_tokens
                    _eff_max = cfg.max_kept_tokens
            elif cfg.dynamic_prune_mode == "accel":
                # Acceleration-based: map xyz / rotation independently into half
                # of the token span so translation and rotation have equal weight.
                _ema_xyz = token_state.ema_accel_xyz
                _raw_xyz = token_state.last_accel_xyz
                _ema_rot = token_state.ema_accel_rot
                _raw_rot = token_state.last_accel_rot
                if (
                    _ema_xyz is not None
                    and _raw_xyz is not None
                    and _ema_xyz > 1e-8
                    and _ema_rot is not None
                    and _raw_rot is not None
                    and _ema_rot > 1e-8
                ):
                    _xyz_dev = compute_relative_deviation(_raw_xyz, _ema_xyz)
                    _rot_dev = compute_relative_deviation(_raw_rot, _ema_rot)
                    assert _xyz_dev is not None and _rot_dev is not None
                    _xyz_ratio = sigmoid_ratio(_xyz_dev)
                    _rot_ratio = sigmoid_ratio(_rot_dev)
                    _half_span = (cfg.max_kept_tokens - cfg.min_kept_tokens) / 2.0
                    _xyz_delta = _xyz_ratio * _half_span
                    _rot_delta = _rot_ratio * _half_span
                    if cfg.ema_direction == "reverse":
                        _eff = int(round(cfg.min_kept_tokens + _xyz_delta + _rot_delta))
                    else:
                        _eff = int(round(cfg.max_kept_tokens - _xyz_delta - _rot_delta))
                    _eff = max(min(_eff, cfg.max_kept_tokens), cfg.min_kept_tokens)
                    _eff_min = _eff
                    _eff_max = _eff
                else:
                    _eff_min = cfg.max_kept_tokens
                    _eff_max = cfg.max_kept_tokens
            else:  # "none"
                # Fixed prune ratio mode: use prune_ratio for both min and max
                _eff_min = cfg.prune_ratio
                _eff_max = cfg.prune_ratio
            region_area = cfg.region_patch_size * cfg.region_patch_size
            min_regions = math.ceil(_eff_min / region_area) if _eff_min > 0 else 0
            max_regions = (
                math.floor(_eff_max / region_area)
                if _eff_max > 0
                else score_region.shape[1]
            )
            keep_region, clipped_region = apply_min_max_constraints(
                keep_pre, _eff_score_region, min_regions, max_regions
            )
            keep_token = expand_region_mask(keep_region, meta.patches_per_side, cfg.region_patch_size)
            important_token = expand_region_mask(important_region, meta.patches_per_side, cfg.region_patch_size)
            clipped_token = expand_region_mask(clipped_region, meta.patches_per_side, cfg.region_patch_size)
            overlay_label = build_overlay_labels(keep_token, important_token, clipped_token)
            overlay_labels.append(overlay_label)
            overlay_grids.append(overlay_label.view(bsize, meta.patches_per_side, meta.patches_per_side))
            score_regions.append(score_region)
            important_regions.append(important_region)
            keep_regions.append(keep_region)
            clipped_regions.append(clipped_region)
            keep_token_masks.append(keep_token)

        # ── Build normalised heatmap grids for debug visualisation ──────────
        heatmap_grids = []
        if cfg.score_debug_heatmap or cfg.overlay_mode == "heatmap":
            for idx, sr in enumerate(score_regions):
                meta = metas[idx]
                rps = meta.patches_per_side // cfg.region_patch_size
                # sr: [B, num_regions]  — normalise to [0, 1] per batch element
                sr_f = sr.float()
                sr_min = sr_f.min(dim=-1, keepdim=True).values
                sr_max = sr_f.max(dim=-1, keepdim=True).values
                sr_norm = torch.where(
                    sr_max > sr_min,
                    (sr_f - sr_min) / (sr_max - sr_min + 1e-8),
                    torch.zeros_like(sr_f),
                )
                heatmap_grids.append(sr_norm.view(-1, rps, rps))

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
                        )
                        try:
                            from PIL import Image
                        except ImportError:
                            continue
                        Image.fromarray(overlay.numpy()).save(
                            out_dir / f"overlay_f{token_state.frame_idx:06d}_i{img_idx}_b{batch_idx}.png"
                        )

        image_pad_masks = None
        if not cfg.score_debug_heatmap and not eval_frame and cfg.token_prune_enabled:
            image_embs, image_pad_masks = prune_image_embeddings(
                image_embs, keep_token_masks, img_masks, metas
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
        token_state.last_overlay_labels = [label.detach() for label in overlay_labels]
        token_state.last_overlay_grid = [grid.detach() for grid in overlay_grids]
        token_state.last_heatmap_grid = [g.detach() for g in heatmap_grids] if heatmap_grids else None
        # Store binary keep mask for heatmap pruning visualization
        if keep_token_masks:
            token_state.last_keep_grid = [
                m.detach().view(bsize, metas[i].patches_per_side, metas[i].patches_per_side)
                for i, m in enumerate(keep_token_masks)
            ]
        else:
            token_state.last_keep_grid = None

        if eval_frame:
            token_state.eval_frame_count += 1

        if cfg.token_selection_enabled and token_state.frame_idx > 0:
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
            if cfg.token_prune_enabled and not eval_frame:
                _actual_kept = _marked_n
            else:
                _actual_kept = _total_n
            _prune_pct = (1.0 - _actual_kept / _total_n) * 100.0 if _total_n > 0 else 0.0
            # Accumulate for per-episode average pruning ratio
            token_state.total_kept += _actual_kept
            token_state.total_possible += _total_n
            token_state.last_stats = {
                "interval": f"{eval_interval}f",
                "kept": f"{_actual_kept}/{_total_n}",
                "prune": f"{_prune_pct:.1f}%",
                "eval": "Y" if eval_frame else "N",
            }

        _need_l1 = (cfg.interp_plot_actions_l1
                     or cfg.dynamic_prune_mode in ("l1_threshold", "ema", "accel"))
        if _need_l1 and actions is not None:
            _l1 = actions.detach().float().abs().sum(dim=-1).mean().item()
            token_state.last_actions_l1 = _l1
            if cfg.interp_plot_actions_l1:
                token_state.actions_l1_history.append(
                    (token_state.frame_idx, _l1)
                )
            if cfg.dynamic_prune_mode in ("ema", "accel"):
                # Update L1 EMA (used by both ema and accel modes for reference)
                if token_state.ema_l1 is None:
                    token_state.ema_l1 = _l1
                else:
                    token_state.ema_l1 = (cfg.l1_ema_alpha * token_state.ema_l1
                                          + (1.0 - cfg.l1_ema_alpha) * _l1)
                token_state.ema_l1_mean, token_state.ema_l1_var = update_ema_mean_and_var(
                    _l1,
                    token_state.ema_l1_mean,
                    token_state.ema_l1_var,
                    cfg.l1_ema_alpha,
                )
            if cfg.dynamic_prune_mode == "accel":
                # Compute chunk-internal acceleration:
                # A_xyz / A_rot = sum_i( |v_{i+1} - v_i| ) over the corresponding dims.
                _act = actions.detach().float()  # [B, ChunkSize, ActionDim]
                if _act.dim() == 3 and _act.shape[1] > 1:
                    _xyz_dims = min(3, _act.shape[-1])
                    _xyz = _act[:, :, :_xyz_dims]
                    _xyz_dv = (_xyz[:, 1:, :] - _xyz[:, :-1, :]).abs()
                    _accel_xyz = _xyz_dv.sum().item() / _act.shape[0]
                    _rot_start = min(3, _act.shape[-1])
                    _rot_end = min(6, _act.shape[-1])
                    if _rot_end > _rot_start:
                        _rot = _act[:, :, _rot_start:_rot_end]
                        _rot_dv = (_rot[:, 1:, :] - _rot[:, :-1, :]).abs()
                        _accel_rot = _rot_dv.sum().item() / _act.shape[0]
                    else:
                        _accel_rot = 0.0
                else:
                    _accel_xyz = 0.0
                    _accel_rot = 0.0
                token_state.last_accel_xyz = _accel_xyz
                token_state.last_accel_rot = _accel_rot
                token_state.last_accel = 0.5 * (_accel_xyz + _accel_rot)
                _prev_ema_xyz = token_state.ema_accel_xyz
                _prev_ema_rot = token_state.ema_accel_rot
                if token_state.ema_accel_xyz is None:
                    token_state.ema_accel_xyz = _accel_xyz
                else:
                    token_state.ema_accel_xyz = (
                        cfg.l1_ema_alpha * token_state.ema_accel_xyz
                        + (1.0 - cfg.l1_ema_alpha) * _accel_xyz
                    )
                if token_state.ema_accel_rot is None:
                    token_state.ema_accel_rot = _accel_rot
                else:
                    token_state.ema_accel_rot = (
                        cfg.l1_ema_alpha * token_state.ema_accel_rot
                        + (1.0 - cfg.l1_ema_alpha) * _accel_rot
                    )
                token_state.ema_accel = 0.5 * (
                    token_state.ema_accel_xyz + token_state.ema_accel_rot
                )
                _, token_state.ema_accel_xyz_var = update_ema_mean_and_var(
                    _accel_xyz,
                    _prev_ema_xyz,
                    token_state.ema_accel_xyz_var,
                    cfg.l1_ema_alpha,
                )
                _, token_state.ema_accel_rot_var = update_ema_mean_and_var(
                    _accel_rot,
                    _prev_ema_rot,
                    token_state.ema_accel_rot_var,
                    cfg.l1_ema_alpha,
                )
        if cfg.reset_discard_pool_on_gripper_close and actions is not None:
            _act = actions.detach().float()
            if _act.numel() > 0 and (_act[..., -1] > cfg.gripper_close_threshold).any().item():
                token_state.pending_global_pool_restore = True

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

        heatmap_grid = None
        if state.last_heatmap_grid:
            heatmap_grid = torch.stack(state.last_heatmap_grid, dim=0)

        keep_grid = None
        if state.last_keep_grid:
            keep_grid = torch.stack(state.last_keep_grid, dim=0)

        return {
            "last_overlay_grid": overlay_grid,
            "last_token_scores": token_scores,
            "last_region_scores": region_scores,
            "last_heatmap_grid": heatmap_grid,
            "last_keep_grid": keep_grid,
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
