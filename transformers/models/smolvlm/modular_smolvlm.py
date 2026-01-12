# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
# Written by Orr Zohar
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
from typing import Callable, Optional, Union

import math
import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..idefics3.configuration_idefics3 import Idefics3Config, Idefics3VisionConfig
from ..idefics3.image_processing_idefics3 import Idefics3ImageProcessor
from ..idefics3.image_processing_idefics3_fast import Idefics3ImageProcessorFast
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
    Idefics3VisionTransformer,
)


logger = logging.get_logger(__name__)


class SmolVLMVisionConfig(Idefics3VisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`SmolVLMVisionModel`]. It is used to instantiate a
    SmolVLM vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SigLIP checkpoint
    [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) used in SmolVLM
    [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer
    >>> from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig

    >>> # Initializing a SmolVLMVisionConfig with google/siglip-so400m-patch14-384 style configuration
    >>> configuration = SmolVLMVisionConfig()

    >>> # Initializing a SmolVLMVisionTransformer (with random weights) from the google/siglip-so400m-patch14-384 style configuration
    >>> model = SmolVLMVisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "smolvlm_vision"
    pass


class SmolVLMPreTrainedModel(Idefics3PreTrainedModel):
    pass


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def merge_with_cache(
    new_states: torch.Tensor,
    cached_states: Optional[torch.Tensor],
    update_mask: Optional[torch.Tensor],
    use_ste: bool,
) -> torch.Tensor:
    if cached_states is None or update_mask is None:
        return new_states
    mask = update_mask.unsqueeze(-1)
    if use_ste:
        return new_states + (~mask) * (cached_states - new_states).detach()
    return torch.where(mask, new_states, cached_states)


def merge_kv_with_cache(
    new_states: torch.Tensor,
    cached_states: Optional[torch.Tensor],
    update_mask: Optional[torch.Tensor],
    use_ste: bool,
) -> torch.Tensor:
    if cached_states is None or update_mask is None:
        return new_states
    mask = update_mask[:, None, :, None]
    if use_ste:
        return new_states + (~mask) * (cached_states - new_states).detach()
    return torch.where(mask, new_states, cached_states)


class SmolVLMVisionEmbeddings(nn.Module):
    """
    Modified version of SigLIP embeddings to enable variable resolution inputs.
    """

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def _get_position_ids(self, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, max_nb_patches_h, max_nb_patches_w = patch_attention_mask.shape

        boundaries = torch.arange(
            1 / self.num_patches_per_side,
            1.0,
            1 / self.num_patches_per_side,
            device=patch_attention_mask.device,
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w),
            fill_value=0,
            dtype=torch.long,
            device=patch_attention_mask.device,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            h_indices = torch.arange(nb_patches_h, device=position_ids.device, dtype=torch.float32)
            w_indices = torch.arange(nb_patches_w, device=position_ids.device, dtype=torch.float32)

            fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
            fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

        return position_ids

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        position_ids = self._get_position_ids(patch_attention_mask)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class SmolVLMVisionAttention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SmolVLMVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SmolVLMEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SmolVLMVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SmolVLMVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward_with_kv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        batch_size, seq_length, _ = hidden_states.shape
        num_heads = self.self_attn.num_heads
        head_dim = self.self_attn.head_dim

        queries = self.self_attn.q_proj(hidden_states)
        keys = self.self_attn.k_proj(hidden_states)
        values = self.self_attn.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.self_attn.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.self_attn.config._attn_implementation]

        attn_output, _ = attention_interface(
            self.self_attn,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.self_attn.is_causal,
            scaling=self.self_attn.scale,
            dropout=0.0 if not self.training else self.self_attn.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim).contiguous()
        attn_output = self.self_attn.out_proj(attn_output)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, keys, values

    def forward_partial(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        update_indices: torch.Tensor,
        attention_mask_updated: Optional[torch.Tensor],
        cached_output: torch.Tensor,
        cached_key_states: torch.Tensor,
        cached_value_states: torch.Tensor,
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]:
        if update_indices.numel() == 0:
            return cached_output, cached_key_states, cached_value_states

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states[:, update_indices, :])

        batch_size, seq_length, _ = residual.shape
        num_heads = self.self_attn.num_heads
        head_dim = self.self_attn.head_dim

        queries = self.self_attn.q_proj(hidden_states)
        keys = self.self_attn.k_proj(hidden_states)
        values = self.self_attn.v_proj(hidden_states)

        queries = queries.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        key_states = cached_key_states.clone()
        value_states = cached_value_states.clone()
        key_states[:, :, update_indices, :] = keys
        value_states[:, :, update_indices, :] = values

        attn_mask = attention_mask_updated if attention_mask is not None else None
        attn_output, _ = eager_attention_forward(
            self.self_attn,
            queries,
            key_states,
            value_states,
            attn_mask,
            scaling=self.self_attn.scale,
            dropout=0.0 if not self.training else self.self_attn.dropout,
        )

        attn_output = attn_output.reshape(batch_size, update_indices.numel(), self.embed_dim).contiguous()
        attn_output = self.self_attn.out_proj(attn_output)

        updated = residual[:, update_indices, :] + attn_output
        updated = updated + self.mlp(self.layer_norm2(updated))

        outputs = cached_output.clone()
        outputs[:, update_indices, :] = updated
        return outputs, key_states, value_states


class SmolVLMEncoder(nn.Module):
    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SmolVLMEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
        use_ste: bool = False,
        update_indices: Optional[torch.Tensor] = None,
        attention_mask_updated: Optional[torch.Tensor] = None,
        return_layer_outputs: bool = False,
        return_kv_cache: bool = False,
    ) -> Union[tuple, BaseModelOutput]:
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input embeddings to the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                Full attention mask for all tokens.
            update_mask (`torch.BoolTensor`, *optional*):
                Boolean mask of tokens that should be updated in partial-update mode.
            cache (`dict`, *optional*):
                Cached per-layer outputs and KV tensors from the previous frame.
            use_ste (`bool`, *optional*, defaults to `False`):
                Whether to use straight-through estimation when merging cached states.
            update_indices (`torch.LongTensor`, *optional*):
                Indices of tokens that should be updated in partial-update mode.
            attention_mask_updated (`torch.Tensor`, *optional*):
                Attention mask sliced to the updated tokens (query side) when doing partial updates.
            return_layer_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return per-layer hidden states for caching.
            return_kv_cache (`bool`, *optional*, defaults to `False`):
                Whether to return per-layer key/value tensors for caching.
        """
        hidden_states = inputs_embeds
        layer_outputs = [] if return_layer_outputs else None
        kv_cache = [] if return_kv_cache else None

        if update_mask is None or cache is None:
            for encoder_layer in self.layers:
                if return_kv_cache:
                    hidden_states, key_states, value_states = encoder_layer.forward_with_kv(
                        hidden_states,
                        attention_mask,
                    )
                    kv_cache.append((key_states, value_states))
                else:
                    hidden_states = encoder_layer(hidden_states, attention_mask)
                if return_layer_outputs:
                    layer_outputs.append(hidden_states)
            output = BaseModelOutput(last_hidden_state=hidden_states)
            if return_layer_outputs or return_kv_cache:
                return output, layer_outputs, kv_cache
            return output

        if use_ste:
            for layer_idx, encoder_layer in enumerate(self.layers):
                if return_kv_cache:
                    new_hidden_states, key_states, value_states = encoder_layer.forward_with_kv(
                        hidden_states,
                        attention_mask,
                    )
                else:
                    new_hidden_states = encoder_layer(hidden_states, attention_mask)
                cached_output = cache["layer_outputs"][layer_idx]
                hidden_states = merge_with_cache(new_hidden_states, cached_output, update_mask, use_ste=True)
                if return_layer_outputs:
                    layer_outputs.append(hidden_states)
                if return_kv_cache:
                    cached_key_states = cache["key_states"][layer_idx]
                    cached_value_states = cache["value_states"][layer_idx]
                    key_states = merge_kv_with_cache(key_states, cached_key_states, update_mask, use_ste=True)
                    value_states = merge_kv_with_cache(value_states, cached_value_states, update_mask, use_ste=True)
                    kv_cache.append((key_states, value_states))
            output = BaseModelOutput(last_hidden_state=hidden_states)
            if return_layer_outputs or return_kv_cache:
                return output, layer_outputs, kv_cache
            return output

        for layer_idx, encoder_layer in enumerate(self.layers):
            cached_output = cache["layer_outputs"][layer_idx]
            cached_key_states = cache["key_states"][layer_idx]
            cached_value_states = cache["value_states"][layer_idx]
            hidden_states, key_states, value_states = encoder_layer.forward_partial(
                hidden_states,
                attention_mask,
                update_indices,
                attention_mask_updated,
                cached_output,
                cached_key_states,
                cached_value_states,
            )
            if return_layer_outputs:
                layer_outputs.append(hidden_states)
            if return_kv_cache:
                kv_cache.append((key_states, value_states))

        output = BaseModelOutput(last_hidden_state=hidden_states)
        if return_layer_outputs or return_kv_cache:
            return output, layer_outputs, kv_cache
        return output


@auto_docstring(
    custom_intro="""
    The SmolVLM Vision Transformer Model outputting raw image embedding.
    """
)
class SmolVLMVisionTransformer(SmolVLMPreTrainedModel):
    config: SmolVLMVisionConfig
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "hidden_states": SmolVLMEncoderLayer,
        "attentions": SmolVLMVisionAttention,
    }

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.embeddings = SmolVLMVisionEmbeddings(config)
        self.encoder = SmolVLMEncoder(config)
        self.patch_size = config.patch_size
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._partial_update_cache = {}

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def reset_partial_update_cache(self):
        self._partial_update_cache = {}

    def get_last_update_mask(self, cache_key: int = 0) -> Optional[torch.Tensor]:
        cache = self._partial_update_cache.get(cache_key)
        if cache is None:
            return None
        return cache.get("last_update_mask_grid")

    def _compute_center_update_mask(
        self, patch_attention_mask: torch.BoolTensor, center_patch_ratio: float
    ) -> tuple[torch.BoolTensor, bool]:
        batch_size, max_h, max_w = patch_attention_mask.shape
        ratio = max(0.0, min(center_patch_ratio, 1.0))
        update_masks = []
        for batch_idx in range(batch_size):
            p_mask = patch_attention_mask[batch_idx]
            valid_h = int(p_mask[:, 0].sum().item())
            valid_w = int(p_mask[0].sum().item())
            if valid_h == 0 or valid_w == 0:
                update_masks.append(
                    torch.zeros(max_h * max_w, dtype=torch.bool, device=patch_attention_mask.device)
                )
                continue

            center_scale = math.sqrt(ratio) if ratio > 0 else 0.0
            center_h = max(1, int(round(valid_h * center_scale)))
            center_w = max(1, int(round(valid_w * center_scale)))
            center_h = min(center_h, valid_h)
            center_w = min(center_w, valid_w)

            offset_h = (valid_h - center_h) // 2
            offset_w = (valid_w - center_w) // 2

            mask_grid = torch.zeros((max_h, max_w), dtype=torch.bool, device=patch_attention_mask.device)
            mask_grid[offset_h : offset_h + center_h, offset_w : offset_w + center_w] = True
            mask_grid = mask_grid & p_mask
            update_masks.append(mask_grid.view(-1))

        update_mask = torch.stack(update_masks, dim=0)
        same_mask = bool(torch.all(update_mask == update_mask[:1]).item())
        return update_mask, same_mask

    def _compute_center_patch_bounds(
        self, patch_attention_mask: torch.BoolTensor, center_patch_ratio: float
    ) -> tuple[int, int, int, int]:
        p_mask = patch_attention_mask[0]
        valid_h = int(p_mask[:, 0].sum().item())
        valid_w = int(p_mask[0].sum().item())
        if valid_h == 0 or valid_w == 0:
            return 0, 0, 0, 0

        ratio = max(0.0, min(center_patch_ratio, 1.0))
        center_scale = math.sqrt(ratio) if ratio > 0 else 0.0
        center_h = max(1, int(round(valid_h * center_scale)))
        center_w = max(1, int(round(valid_w * center_scale)))
        center_h = min(center_h, valid_h)
        center_w = min(center_w, valid_w)

        offset_h = (valid_h - center_h) // 2
        offset_w = (valid_w - center_w) // 2
        return offset_h, offset_w, center_h, center_w

    @check_model_inputs(tie_last_hidden_states=False)
    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutput]:
        center_patch_ratio = kwargs.pop("center_patch_ratio", None)
        full_update_interval = kwargs.pop("full_update_interval", None)
        force_full_update = kwargs.pop("force_full_update", False)
        enable_partial_update = kwargs.pop("enable_partial_update", True)
        reuse_log_interval = kwargs.pop("reuse_log_interval", 10)
        cache_key = kwargs.pop("cache_key", 0)
        if cache_key is None:
            cache_key = 0
        cache_name = kwargs.pop("cache_name", None)
        if cache_name is None:
            cache_name = f"cache{cache_key}"

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        patch_attention_mask_grid = patch_attention_mask
        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        if center_patch_ratio is None:
            center_patch_ratio = 1.0

        use_partial_update = enable_partial_update and center_patch_ratio < 1.0
        update_mask = None
        update_indices = None
        center_bounds = None
        disable_reason = None

        if use_partial_update:
            update_mask, same_mask = self._compute_center_update_mask(
                patch_attention_mask_grid,
                center_patch_ratio,
            )
            if not same_mask:
                use_partial_update = False
                disable_reason = "batch masks differ"
            else:
                update_indices = torch.nonzero(update_mask[0], as_tuple=False).squeeze(-1)
                if update_indices.numel() == patch_attention_mask.shape[1]:
                    use_partial_update = False
                    disable_reason = "update covers all tokens"
                else:
                    update_indices = update_indices.to(pixel_values.device)
                    center_bounds = self._compute_center_patch_bounds(
                        patch_attention_mask_grid,
                        center_patch_ratio,
                    )
                    if center_bounds[2] == 0 or center_bounds[3] == 0:
                        use_partial_update = False
                        disable_reason = "empty center bounds"
                    elif update_indices.numel() != center_bounds[2] * center_bounds[3]:
                        use_partial_update = False
                        disable_reason = "center bounds mismatch"

        use_ste = torch.is_grad_enabled() and self.training

        cache = self._partial_update_cache.get(cache_key) if use_partial_update else None
        frame_counter = cache.get("frame_counter", 0) if cache is not None else 0
        if use_partial_update:
            frame_counter += 1
        scheduled_full_update = (
            full_update_interval is not None
            and full_update_interval > 0
            and frame_counter % full_update_interval == 0
        )

        num_patches = patch_attention_mask.shape[1]
        embed_dim = self.embeddings.embed_dim
        num_heads = self.encoder.layers[0].self_attn.num_heads
        head_dim = self.encoder.layers[0].self_attn.head_dim

        if use_partial_update:
            cache_valid = (
                cache is not None
                and cache.get("patch_embeddings") is not None
                and cache["patch_embeddings"].shape == (batch_size, num_patches, embed_dim)
                and cache["patch_embeddings"].device == pixel_values.device
                and len(cache.get("layer_outputs", [])) == len(self.encoder.layers)
                and cache["layer_outputs"][0].shape == (batch_size, num_patches, embed_dim)
                and len(cache.get("key_states", [])) == len(self.encoder.layers)
                and len(cache.get("value_states", [])) == len(self.encoder.layers)
                and cache["key_states"][0].shape == (batch_size, num_heads, num_patches, head_dim)
                and cache["value_states"][0].shape == (batch_size, num_heads, num_patches, head_dim)
            )
        else:
            cache_valid = False

        full_update = force_full_update or scheduled_full_update or not use_partial_update or not cache_valid

        if not use_partial_update and disable_reason is not None and enable_partial_update:
            cache_entry = self._partial_update_cache.setdefault(cache_key, {})
            if not cache_entry.get("disable_logged", False):
                logger.info(
                    "SmolVLMVisionTransformer partial-update disabled: %s cam=%s",
                    disable_reason,
                    cache_name,
                )
                cache_entry["disable_logged"] = True

        if not use_partial_update or full_update:
            hidden_states = self.embeddings(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask_grid,
            )
        elif use_ste:
            hidden_states = self.embeddings(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask_grid,
            )
            hidden_states = merge_with_cache(hidden_states, cache["patch_embeddings"], update_mask, use_ste=True)
        else:
            position_ids = self.embeddings._get_position_ids(patch_attention_mask_grid)
            pos_embeds = self.embeddings.position_embedding(position_ids)
            offset_h, offset_w, center_h, center_w = center_bounds
            h0 = offset_h * self.patch_size
            h1 = (offset_h + center_h) * self.patch_size
            w0 = offset_w * self.patch_size
            w1 = (offset_w + center_w) * self.patch_size
            patch_embeds = self.embeddings.patch_embedding(pixel_values[:, :, h0:h1, w0:w1])
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            patch_embeds = patch_embeds + pos_embeds[:, update_indices, :]
            hidden_states = cache["patch_embeddings"].clone()
            hidden_states[:, update_indices, :] = patch_embeds

        should_log = False
        last_log_frame = cache.get("last_log_frame") if cache is not None else None
        last_mode = cache.get("last_mode") if cache is not None else None
        reuse_sum = cache.get("reuse_sum", 0.0) if cache is not None else 0.0
        reuse_count = cache.get("reuse_count", 0) if cache is not None else 0
        if use_partial_update:
            mode = "full" if full_update else ("partial_ste" if use_ste else "partial_cache")
            if last_mode is None or last_mode != mode:
                should_log = True
            elif reuse_log_interval and reuse_log_interval > 0 and frame_counter % reuse_log_interval == 0:
                should_log = True
            if last_log_frame is not None and last_log_frame == frame_counter:
                should_log = False
            if should_log:
                update_tokens = int(update_indices.numel()) if update_indices is not None else 0
                effective_update_tokens = num_patches if full_update else update_tokens
                reuse_ratio = 1.0 - (effective_update_tokens / float(num_patches)) if num_patches > 0 else 0.0
                message = (
                    "SmolVLMVisionTransformer partial-update: cam=%s mode=%s ratio=%.4f tokens=%d/%d reuse=%.2f "
                    "frame=%s full_interval=%s cache_valid=%s scheduled_full=%s force_full=%s ste=%s"
                )
                logger.info(
                    message,
                    cache_name,
                    mode,
                    float(center_patch_ratio),
                    effective_update_tokens,
                    num_patches,
                    reuse_ratio,
                    frame_counter,
                    full_update_interval,
                    cache_valid,
                    scheduled_full_update,
                    force_full_update,
                    use_ste,
                )
            if reuse_log_interval and reuse_log_interval > 0 and num_patches > 0:
                update_tokens = int(update_indices.numel()) if update_indices is not None else 0
                effective_update_tokens = num_patches if full_update else update_tokens
                reuse_ratio = 1.0 - (effective_update_tokens / float(num_patches))
                reuse_sum += reuse_ratio
                reuse_count += 1
                if frame_counter % reuse_log_interval == 0:
                    reuse_avg = reuse_sum / float(reuse_count)
                    avg_msg = (
                        "SmolVLMVisionTransformer reuse avg: cam=%s %.2f over %d frames "
                        "(interval=%d frame=%s)"
                    )
                    logger.info(
                        avg_msg,
                        cache_name,
                        reuse_avg,
                        reuse_count,
                        reuse_log_interval,
                        frame_counter,
                    )

        if self.config._attn_implementation != "flash_attention_2":
            attention_mask_full = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
        elif not torch.any(~patch_attention_mask):
            attention_mask_full = None
        else:
            attention_mask_full = patch_attention_mask

        render_update_mask_grid = None
        if use_partial_update:
            if full_update:
                render_update_mask_grid = torch.ones_like(patch_attention_mask_grid, dtype=torch.bool)
                encoder_outputs, layer_outputs, kv_cache = self.encoder(
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask_full,
                    return_layer_outputs=True,
                    return_kv_cache=True,
                )
            else:
                render_update_mask_grid = update_mask.view(
                    batch_size, patch_attention_mask_grid.shape[1], patch_attention_mask_grid.shape[2]
                )
                if use_ste:
                    encoder_outputs, layer_outputs, kv_cache = self.encoder(
                        inputs_embeds=hidden_states,
                        attention_mask=attention_mask_full,
                        update_mask=update_mask,
                        cache=cache,
                        use_ste=True,
                        return_layer_outputs=True,
                        return_kv_cache=True,
                    )
                else:
                    attention_mask_updated = (
                        attention_mask_full[:, :, update_indices, :] if attention_mask_full is not None else None
                    )
                    encoder_outputs, layer_outputs, kv_cache = self.encoder(
                        inputs_embeds=hidden_states,
                        attention_mask=attention_mask_full,
                        update_mask=update_mask,
                        cache=cache,
                        use_ste=False,
                        update_indices=update_indices,
                        attention_mask_updated=attention_mask_updated,
                        return_layer_outputs=True,
                        return_kv_cache=True,
                    )

            self._partial_update_cache[cache_key] = {
                "patch_embeddings": hidden_states.detach(),
                "layer_outputs": [layer.detach() for layer in layer_outputs],
                "key_states": [layer_k.detach() for layer_k, _ in kv_cache],
                "value_states": [layer_v.detach() for _, layer_v in kv_cache],
                "batch_size": batch_size,
                "device": hidden_states.device,
                "frame_counter": frame_counter,
                "last_log_frame": frame_counter if should_log else last_log_frame,
                "last_mode": mode if use_partial_update else last_mode,
                "reuse_sum": reuse_sum,
                "reuse_count": reuse_count,
                "last_update_mask_grid": (
                    render_update_mask_grid.detach().to("cpu") if render_update_mask_grid is not None else None
                ),
            }
        else:
            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask_full,
            )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
        )


class SmolVLMConfig(Idefics3Config):
    r"""
    This is the configuration class to store the configuration of a [`SmolVLMModel`]. It is used to instantiate a
    SmolVLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the SmolVLM
    [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism. Only
            relevant if `config.is_decoder=True`.
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*, defaults to `IdeficsVisionConfig`):
            Custom vision config or dict for the vision tower
        text_config (`PretrainedConfig` or `dict`, *optional*, defaults to `LlamaConfig`):
            Custom text config or dict for the text model
        scale_factor (`int`, *optional*, defaults to 2):
            The scale factor for the image encoder.
        pad_token_id (`int`, *optional*, defaults to 128002):
            The id of the padding token.

    Example:
    ```python
    >>> from transformers import SmolVLMModel, SmolVLMConfig
    >>> # Initializing configuration
    >>> configuration = SmolVLMConfig()
    >>> # Initializing a model from the configuration
    >>> model = SmolVLMModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "smolvlm"
    pass


class SmolVLMImageProcessor(Idefics3ImageProcessor):
    pass


class SmolVLMImageProcessorFast(Idefics3ImageProcessorFast):
    pass


class SmolVLMBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    pass


class SmolVLMModel(Idefics3Model):
    """
    A subclass of Idefics3Model. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """

    def inputs_merger(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ):
        _, patch_size, _ = image_hidden_states.shape

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask[..., 0]  # slice off the hidden dim
        else:
            image_mask = input_ids == self.config.image_token_id

        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    def get_image_features(
        self, pixel_values: torch.FloatTensor, pixel_attention_mask: Optional[torch.LongTensor] = None
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        if not any(real_images_inds):
            # no images, leave one empty image.
            real_images_inds[0] = True

        pixel_values = pixel_values[real_images_inds].contiguous()
        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        image_hidden_states = image_hidden_states.last_hidden_state

        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, SmolVLMBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")

        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask).to(inputs_embeds.device)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=inputs_embeds.device)

        if image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return SmolVLMBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = SmolVLMModel(config)
        self.model.text_model.generation_config = GenerationConfig.from_model_config(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(self, **super_kwargs):
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        >>> model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=torch.bfloat16, device_map="auto")

        >>> # Create inputs
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "video", "path": path/to/video},
        ...             {"type": "text", "text": "What is happening in this video?"},
        ...         ]
        ...     }
        ... ]

        >>> inputs = processor.apply_chat_template([messages], add_generation_prompt=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts)
        ```"""
        super().forward(**super_kwargs)


__all__ = [
    "SmolVLMVisionConfig",
    "SmolVLMConfig",
    "SmolVLMImageProcessor",
    "SmolVLMImageProcessorFast",
    "SmolVLMForConditionalGeneration",
    "SmolVLMPreTrainedModel",
    "SmolVLMModel",
    "SmolVLMVisionTransformer",
]
