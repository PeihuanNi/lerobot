#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import logging
import math
import time
from collections import defaultdict, deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype

logger = logging.getLogger(__name__)

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
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
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
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
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # Reset model caches for partial-frame update
        if hasattr(self, 'model') and hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def get_last_token_overlay(self, image_index: int = 0) -> dict[str, object] | None:
        if not hasattr(self, "model"):
            return None
        getter = getattr(self.model, "get_last_token_overlay", None)
        if getter is None:
            return None
        return getter(image_index=image_index)

    def reset_timing_stats(self) -> None:
        if not hasattr(self, "model"):
            return
        resetter = getattr(self.model, "reset_timing_stats", None)
        if resetter is not None:
            resetter()

    def get_last_timing(self) -> dict[str, float] | None:
        if not hasattr(self, "model"):
            return None
        getter = getattr(self.model, "get_last_timing", None)
        if getter is None:
            return None
        return getter()

    def get_timing_summary(self) -> dict[str, object] | None:
        if not hasattr(self, "model"):
            return None
        getter = getattr(self.model, "get_timing_summary", None)
        if getter is None:
            return None
        return getter()

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

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

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        img_keys = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)
            img_keys.append(key.split(".")[-1])

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
            img_keys.append(missing_img_keys[num_empty_cameras].split(".")[-1])

        if hasattr(self, "model"):
            self.model._last_image_keys = img_keys
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Frame counter for token selection scheduling.
        self._frame_counter = 0
        self._prev_image_tokens: dict[str, torch.Tensor] = {}
        self._token_keep_masks: dict[str, torch.Tensor] = {}
        self._token_cache_batch_size: int | None = None
        self._logged_rtc_skip = False
        self._last_background_masks: dict[str, torch.Tensor] = {}
        self._last_important_masks: dict[str, torch.Tensor] = {}
        self._last_token_masks: dict[str, torch.Tensor] = {}
        self._last_token_meta: dict[str, dict[str, object]] = {}
        self._last_overlay_keys: list[str] = []
        self._timing_totals: dict[str, float] = defaultdict(float)
        self._timing_counts: int = 0
        self._timing_totals_eval: dict[str, float] = defaultdict(float)
        self._timing_totals_noeval: dict[str, float] = defaultdict(float)
        self._timing_counts_eval: int = 0
        self._timing_counts_noeval: int = 0
        self._last_timing: dict[str, float] = {}
        self._llm_pruned_total: float = 0.0
        self._llm_pruned_count: int = 0
        self._llm_unpruned_total: float = 0.0
        self._llm_unpruned_count: int = 0
        self._pruned_token_total: float = 0.0
        self._pruned_token_counts: int = 0
        self._pruned_token_denom_total: float = 0.0
        self._token_stats_totals: dict[str, int] = {
            "total": 0,
            "important": 0,
            "prunable_bg": 0,
            "other": 0,
        }
        self._token_stats_counts: int = 0

    def reset_cache(self):
        """Reset cached embeddings. Should be called when the environment resets."""
        self._frame_counter = 0
        self._prev_image_tokens = {}
        self._token_keep_masks = {}
        self._token_cache_batch_size = None
        self._logged_rtc_skip = False
        self._last_background_masks = {}
        self._last_important_masks = {}
        self._last_token_masks = {}
        self._last_token_meta = {}
        self._last_overlay_keys = []
        if hasattr(self.vlm_with_expert, "reset_vision_cache"):
            self.vlm_with_expert.reset_vision_cache()
        # Also reset VLM KV cache
        self.vlm_with_expert._prefix_kv_cache = {
            "past_key_values": None,
            "prefix_pad_masks": None,
            "position_ids": None,
            "batch_size": None,
            "device": None,
            "frame_counter": 0,
        }

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def get_last_token_overlay(self, image_index: int = 0) -> dict[str, object] | None:
        if not self._last_overlay_keys:
            return None
        if image_index < 0 or image_index >= len(self._last_overlay_keys):
            return None
        key = self._last_overlay_keys[image_index]
        meta = self._last_token_meta.get(key)
        if meta is None:
            return None
        return {
            "key": key,
            "grid_h": meta.get("grid_h", 0),
            "grid_w": meta.get("grid_w", 0),
            "has_cls": bool(meta.get("has_cls", False)),
            "background_mask": self._last_background_masks.get(key),
            "important_mask": self._last_important_masks.get(key),
            "token_mask": self._last_token_masks.get(key),
        }

    def _sync_if_cuda(self, device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def reset_timing_stats(self) -> None:
        self._timing_totals = defaultdict(float)
        self._timing_counts = 0
        self._timing_totals_eval = defaultdict(float)
        self._timing_totals_noeval = defaultdict(float)
        self._timing_counts_eval = 0
        self._timing_counts_noeval = 0
        self._last_timing = {}
        self._llm_pruned_total = 0.0
        self._llm_pruned_count = 0
        self._llm_unpruned_total = 0.0
        self._llm_unpruned_count = 0
        self._pruned_token_total = 0.0
        self._pruned_token_counts = 0
        self._pruned_token_denom_total = 0.0
        self._token_stats_totals = {
            "total": 0,
            "important": 0,
            "prunable_bg": 0,
            "other": 0,
        }
        self._token_stats_counts = 0

    def get_last_timing(self) -> dict[str, float]:
        return dict(self._last_timing)

    def get_timing_summary(self) -> dict[str, object]:
        summary: dict[str, object] = {
            "num_calls": self._timing_counts,
            "num_eval_calls": self._timing_counts_eval,
            "num_noeval_calls": self._timing_counts_noeval,
            "num_llm_pruned_calls": self._llm_pruned_count,
            "num_llm_unpruned_calls": self._llm_unpruned_count,
        }
        if self._timing_counts > 0:
            avg = {k: v / self._timing_counts for k, v in self._timing_totals.items()}
            summary["avg_s"] = avg
        if self._timing_counts_eval > 0:
            avg_eval = {k: v / self._timing_counts_eval for k, v in self._timing_totals_eval.items()}
            summary["avg_eval_s"] = avg_eval
        if self._timing_counts_noeval > 0:
            avg_noeval = {k: v / self._timing_counts_noeval for k, v in self._timing_totals_noeval.items()}
            summary["avg_noeval_s"] = avg_noeval
        if self._llm_pruned_count > 0:
            summary["avg_llm_pruned_s"] = self._llm_pruned_total / self._llm_pruned_count
        else:
            summary["avg_llm_pruned_s"] = 0.0
        if self._llm_unpruned_count > 0:
            summary["avg_llm_unpruned_s"] = self._llm_unpruned_total / self._llm_unpruned_count
        else:
            summary["avg_llm_unpruned_s"] = 0.0
        if self._pruned_token_counts > 0:
            summary["avg_pruned_tokens"] = self._pruned_token_total / self._pruned_token_counts
        else:
            summary["avg_pruned_tokens"] = 0.0
        if self._pruned_token_denom_total > 0:
            summary["avg_pruned_ratio"] = self._pruned_token_total / self._pruned_token_denom_total
        else:
            summary["avg_pruned_ratio"] = 0.0
        token_totals = dict(self._token_stats_totals)
        summary["token_counts"] = token_totals
        total_tokens = token_totals.get("total", 0)
        if self._token_stats_counts > 0:
            summary["token_avg_counts"] = {
                key: value / self._token_stats_counts for key, value in token_totals.items()
            }
        if total_tokens > 0:
            summary["token_ratios"] = {
                "important": token_totals.get("important", 0) / total_tokens,
                "prunable_bg": token_totals.get("prunable_bg", 0) / total_tokens,
                "other": token_totals.get("other", 0) / total_tokens,
            }
        return summary

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def _merge_center_outer_patches(
        self, new_emb: "torch.Tensor", cached_emb: "torch.Tensor", center_ratio: float
    ) -> "torch.Tensor":
        """Merge center patches from new_emb with outer patches from cached_emb.
        
        Uses Straight-Through Estimator (STE): forward pass uses cached values for outer
        patches, but gradients flow through to new_emb for all patches.
        
        Args:
            new_emb: New embeddings from current frame, shape (B, num_patches, hidden_dim)
            cached_emb: Cached embeddings from previous frame, same shape
            center_ratio: Fraction of patches (by area) to take from center. 
                         E.g., 0.5 means center sqrt(0.5) ≈ 0.707 of each side.
        
        Returns:
            Merged embeddings with center from new, outer from cached (with STE gradient).
        """
        bsize, num_patches, hidden_dim = new_emb.shape
        
        # Calculate patch grid dimensions (assuming square grid)
        patches_per_side = int(round(num_patches ** 0.5))
        if patches_per_side * patches_per_side != num_patches:
            # Non-square patch grid, fall back to full update
            return new_emb
        
        # Calculate center region size based on center_ratio (area ratio)
        # center_ratio = (center_side / patches_per_side)^2
        # So center_side = patches_per_side * sqrt(center_ratio)
        center_side = int(patches_per_side * (center_ratio ** 0.5))
        center_side = max(1, min(center_side, patches_per_side))  # Clamp to valid range
        
        # Calculate offset to center the region
        offset = (patches_per_side - center_side) // 2
        
        # Build mask: True for center patches, False for outer patches
        center_mask = torch.zeros(num_patches, dtype=torch.bool, device=new_emb.device)
        for row in range(offset, offset + center_side):
            for col in range(offset, offset + center_side):
                idx = row * patches_per_side + col
                center_mask[idx] = True
        
        # Expand mask to match embedding shape: (1, num_patches, 1)
        center_mask = center_mask.view(1, num_patches, 1)
        
        # STE: forward uses cached for outer, but gradient flows through new_emb
        # merged = center_mask * new_emb + ~center_mask * cached_emb  (no gradient to outer)
        # With STE: merged = new_emb + (~center_mask * (cached_emb - new_emb)).detach()
        # This way: forward = center uses new, outer uses cached
        #           backward = gradient flows entirely to new_emb
        outer_diff = (cached_emb - new_emb).detach()  # detach the difference
        merged = new_emb + (~center_mask) * outer_diff
        
        return merged

    def _ensure_token_cache_state(self, batch_size: int) -> None:
        if self._token_cache_batch_size is None:
            self._token_cache_batch_size = batch_size
            return
        if self._token_cache_batch_size != batch_size:
            self._token_cache_batch_size = batch_size
            self._prev_image_tokens = {}
            self._token_keep_masks = {}

    def _record_timing(self, timing: dict[str, float], is_eval: bool, pruned: bool) -> None:
        self._timing_counts += 1
        timing["call_idx"] = float(self._timing_counts)
        self._last_timing = dict(timing)
        for key, value in timing.items():
            if key == "call_idx":
                continue
            self._timing_totals[key] += float(value)
            if is_eval:
                self._timing_totals_eval[key] += float(value)
            else:
                self._timing_totals_noeval[key] += float(value)
        llm_time = float(timing.get("diffusion_prefix_s", 0.0))
        if pruned:
            self._llm_pruned_total += llm_time
            self._llm_pruned_count += 1
        else:
            self._llm_unpruned_total += llm_time
            self._llm_unpruned_count += 1
        if is_eval:
            self._timing_counts_eval += 1
        else:
            self._timing_counts_noeval += 1

    def _accumulate_token_stats(
        self,
        background_masks: dict[str, torch.Tensor],
        image_token_masks: list[torch.Tensor],
        image_meta: list[dict[str, object]],
    ) -> None:
        total_tokens = 0
        important_tokens = 0
        prunable_tokens = 0
        other_tokens = 0

        for token_mask, meta in zip(image_token_masks, image_meta, strict=False):
            key = str(meta["key"])
            if token_mask.ndim != 2:
                continue
            valid = token_mask.bool()
            bg_mask = background_masks.get(key)
            if bg_mask is None or bg_mask.shape != valid.shape:
                bg_mask = torch.zeros_like(valid)
            important_mask = self._last_important_masks.get(key)
            if important_mask is None or important_mask.shape != valid.shape:
                important_mask = torch.zeros_like(valid)

            important = important_mask & valid
            prunable = bg_mask & (~important) & valid
            other = valid & (~important) & (~bg_mask)

            total_tokens += int(valid.sum().item())
            important_tokens += int(important.sum().item())
            prunable_tokens += int(prunable.sum().item())
            other_tokens += int(other.sum().item())

        if total_tokens == 0:
            return
        self._token_stats_counts += 1
        self._token_stats_totals["total"] += total_tokens
        self._token_stats_totals["important"] += important_tokens
        self._token_stats_totals["prunable_bg"] += prunable_tokens
        self._token_stats_totals["other"] += other_tokens

    def _infer_patch_grid(self, image: torch.Tensor, num_tokens: int) -> tuple[int, int, bool]:
        grid_h = 0
        grid_w = 0
        has_cls = False
        vision_model = self.vlm_with_expert.get_vlm_model().vision_model
        patch_size = getattr(vision_model, "patch_size", None)
        if patch_size is not None and image is not None:
            if isinstance(patch_size, (tuple, list)):
                patch_h = int(patch_size[0])
                patch_w = int(patch_size[1])
            else:
                patch_h = int(patch_size)
                patch_w = int(patch_size)
            if patch_h > 0 and patch_w > 0:
                grid_h = int(image.shape[-2] // patch_h)
                grid_w = int(image.shape[-1] // patch_w)
                num_patches = grid_h * grid_w
                if num_tokens == num_patches + 1:
                    return grid_h, grid_w, True
                if num_tokens == num_patches:
                    return grid_h, grid_w, False
        side = int(round(num_tokens**0.5))
        if side * side == num_tokens:
            return side, side, False
        if side * side + 1 == num_tokens:
            return side, side, True
        return 0, 0, False

    def _encode_images(self, images, img_masks):
        image_embs: list[torch.Tensor] = []
        image_token_masks: list[torch.Tensor] = []
        image_meta: list[dict[str, object]] = []
        image_keys = getattr(self, "_last_image_keys", None)
        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            if image_keys and _img_idx < len(image_keys):
                image_key = image_keys[_img_idx]
            else:
                image_key = f"image{_img_idx}"

            img_emb = self.vlm_with_expert.embed_image(
                img,
                cache_name=image_key,
                cache_key=_img_idx,
            )

            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            if img_mask is None:
                img_mask = torch.ones(bsize, dtype=torch.bool, device=img_emb.device)
            token_mask = img_mask[:, None].expand(bsize, num_img_embs)

            grid_h, grid_w, has_cls = self._infer_patch_grid(img, num_img_embs)
            image_embs.append(img_emb)
            image_token_masks.append(token_mask)
            image_meta.append({"key": image_key, "grid_h": grid_h, "grid_w": grid_w, "has_cls": has_cls})

        return image_embs, image_token_masks, image_meta

    def _build_prefix_from_tokens(
        self, image_embs, image_token_masks, lang_tokens, lang_masks, state: "torch.Tensor" = None
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        embs = []
        pad_masks = []
        att_masks = []

        for img_emb, img_token_mask in zip(image_embs, image_token_masks, strict=False):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img_emb.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
                att_masks += [0] * image_start_mask.shape[1]

            embs.append(img_emb)
            pad_masks.append(img_token_mask)
            att_masks += [0] * img_emb.shape[1]

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img_emb.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * image_end_mask.shape[1]

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def _compute_background_mask(
        self,
        tokens: torch.Tensor,
        prev_tokens: torch.Tensor | None,
        grid_h: int,
        grid_w: int,
        has_cls: bool,
    ) -> torch.Tensor:
        bsize, num_tokens = tokens.shape[:2]
        temporal_thresh = self.config.token_temporal_threshold
        spatial_thresh = self.config.token_spatial_threshold
        if temporal_thresh is None and spatial_thresh is None:
            return torch.zeros((bsize, num_tokens), dtype=torch.bool, device=tokens.device)

        tokens_f = tokens.float()
        temporal_ok = None
        if temporal_thresh is not None:
            if prev_tokens is None or prev_tokens.shape != tokens.shape:
                temporal_ok = torch.zeros((bsize, num_tokens), dtype=torch.bool, device=tokens.device)
            else:
                prev_f = prev_tokens.float()
                temporal_sim = (F.normalize(tokens_f, dim=-1) * F.normalize(prev_f, dim=-1)).sum(dim=-1)
                temporal_ok = temporal_sim >= temporal_thresh

        spatial_ok = None
        if spatial_thresh is not None:
            radius = max(0, int(self.config.token_spatial_radius))
            if grid_h > 0 and grid_w > 0 and radius > 0:
                patch_tokens = tokens_f[:, 1:] if has_cls else tokens_f
                if patch_tokens.shape[1] == grid_h * grid_w:
                    patch_tokens = patch_tokens.view(bsize, grid_h, grid_w, -1)
                    x = patch_tokens.permute(0, 3, 1, 2)
                    kernel = torch.ones(
                        (x.shape[1], 1, 2 * radius + 1, 2 * radius + 1),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    neighbor_sum = F.conv2d(x, kernel, padding=radius, groups=x.shape[1])
                    neighbor_sum = neighbor_sum - x
                    count_kernel = torch.ones(
                        (1, 1, 2 * radius + 1, 2 * radius + 1),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    ones = torch.ones((1, 1, grid_h, grid_w), dtype=x.dtype, device=x.device)
                    neighbor_count = F.conv2d(ones, count_kernel, padding=radius) - 1.0
                    neighbor_count = neighbor_count.clamp(min=1.0)
                    neighbor_mean = neighbor_sum / neighbor_count
                    spatial_sim = (x * neighbor_mean).sum(dim=1).view(bsize, grid_h * grid_w)
                    if has_cls:
                        spatial_sim = torch.cat(
                            [torch.zeros((bsize, 1), dtype=spatial_sim.dtype, device=spatial_sim.device), spatial_sim],
                            dim=1,
                        )
                    spatial_ok = spatial_sim >= spatial_thresh
            if spatial_ok is None:
                spatial_ok = torch.zeros((bsize, num_tokens), dtype=torch.bool, device=tokens.device)

        if temporal_ok is None:
            background_mask = spatial_ok
        elif spatial_ok is None:
            background_mask = temporal_ok
        else:
            background_mask = temporal_ok & spatial_ok

        if has_cls and background_mask.numel() > 0:
            background_mask[:, 0] = False
        return background_mask

    def _apply_background_fill(self, tokens: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        if background_mask is None or not background_mask.any():
            return tokens
        mode = self.config.background_fill
        if mode == "none":
            return tokens
        tokens_out = tokens.clone()
        if mode == "mean":
            fill = tokens_out.mean(dim=1, keepdim=True).expand_as(tokens_out)
        else:
            fill = torch.zeros_like(tokens_out)
        tokens_out = torch.where(background_mask.unsqueeze(-1), fill, tokens_out)
        return tokens_out

    def _apply_background_padding(
        self, token_mask: torch.Tensor, background_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if background_mask is None:
            return token_mask
        if background_mask.shape != token_mask.shape:
            return token_mask
        return token_mask & (~background_mask)

    def _build_region_masks(
        self, grid_h: int, grid_w: int, has_cls: bool, device: torch.device
    ) -> torch.Tensor | None:
        if grid_h <= 0 or grid_w <= 0:
            return None
        region_size = max(1, int(self.config.region_patch_size))
        num_regions_h = (grid_h + region_size - 1) // region_size
        num_regions_w = (grid_w + region_size - 1) // region_size
        num_regions = num_regions_h * num_regions_w
        num_patches = grid_h * grid_w
        total_tokens = num_patches + (1 if has_cls else 0)

        region_masks = torch.zeros((num_regions, total_tokens), dtype=torch.bool, device=device)
        patch_idx = torch.arange(num_patches, device=device).view(grid_h, grid_w)
        region_idx = 0
        for rh in range(num_regions_h):
            for rw in range(num_regions_w):
                r0 = rh * region_size
                r1 = min((rh + 1) * region_size, grid_h)
                c0 = rw * region_size
                c1 = min((rw + 1) * region_size, grid_w)
                indices = patch_idx[r0:r1, c0:c1].reshape(-1)
                if has_cls:
                    indices = indices + 1
                region_masks[region_idx, indices] = True
                region_idx += 1
        return region_masks

    def _prune_tokens_for_image(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        background_mask: torch.Tensor | None,
        keep_mask: torch.Tensor,
        timing: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsize, num_tokens, hidden = tokens.shape
        device = tokens.device
        if timing is not None:
            self._sync_if_cuda(device)
            mask_start = time.perf_counter()
        if keep_mask.ndim == 1:
            keep_mask = keep_mask.unsqueeze(0).expand(bsize, -1)
        if keep_mask.shape[:2] != (bsize, num_tokens):
            return tokens, token_mask
        effective_keep = keep_mask & token_mask
        if background_mask is not None:
            effective_keep = effective_keep & (~background_mask)

        min_keep = max(1, int(self.config.min_kept_tokens))
        for b in range(bsize):
            if int(effective_keep[b].sum().item()) < min_keep:
                effective_keep[b] = token_mask[b]

        keep_counts = effective_keep.sum(dim=1)
        max_keep = int(keep_counts.max().item()) if keep_counts.numel() > 0 else 0
        max_keep = max(1, max_keep)
        if timing is not None:
            self._sync_if_cuda(device)
            mask_end = time.perf_counter()
            timing["prune_mask_s"] = timing.get("prune_mask_s", 0.0) + (mask_end - mask_start)
            self._sync_if_cuda(device)
            pack_start = time.perf_counter()
        pruned_tokens = tokens.new_zeros((bsize, max_keep, hidden))
        pruned_masks = torch.zeros((bsize, max_keep), dtype=torch.bool, device=tokens.device)

        for b in range(bsize):
            idx = torch.nonzero(effective_keep[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            pruned_tokens[b, : idx.numel()] = tokens[b, idx]
            pruned_masks[b, : idx.numel()] = True
        if timing is not None:
            self._sync_if_cuda(device)
            pack_end = time.perf_counter()
            timing["prune_pack_s"] = timing.get("prune_pack_s", 0.0) + (pack_end - pack_start)

        return pruned_tokens, pruned_masks

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: "torch.Tensor" = None
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        image_embs, image_token_masks, _ = self._encode_images(images, img_masks)
        return self._build_prefix_from_tokens(image_embs, image_token_masks, lang_tokens, lang_masks, state=state)

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def _run_diffusion(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        noise: torch.Tensor,
        use_rtc: bool = True,
        timing: dict[str, float] | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> torch.Tensor:
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        if timing is not None:
            self._sync_if_cuda(device)
        diffusion_start = time.perf_counter()
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if timing is not None:
            self._sync_if_cuda(device)
        prefix_start = time.perf_counter()
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        if timing is not None:
            self._sync_if_cuda(device)
        prefix_end = time.perf_counter()

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        if timing is not None:
            self._sync_if_cuda(device)
        denoise_start = time.perf_counter()
        for step in range(num_steps):
            t_val = 1.0 + step * dt
            time_tensor = torch.tensor(t_val, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if use_rtc and self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=t_val,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if use_rtc and self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=t_val, x_t=x_t, v_t=v_t)
        if timing is not None:
            self._sync_if_cuda(device)
        denoise_end = time.perf_counter()
        diffusion_end = denoise_end

        if timing is not None:
            timing["diffusion_prefix_s"] = prefix_end - prefix_start
            timing["diffusion_denoise_s"] = denoise_end - denoise_start
            timing["diffusion_total_s"] = diffusion_end - diffusion_start
        return x_t

    def _evaluate_region_importance(
        self,
        image_embs: list[torch.Tensor],
        image_token_masks: list[torch.Tensor],
        image_meta: list[dict[str, object]],
        background_masks: dict[str, torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        baseline_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if self._rtc_enabled():
            if not self._logged_rtc_skip:
                logger.info("Region importance evaluation skipped because RTC is enabled.")
                self._logged_rtc_skip = True
            return None

        bsize = baseline_actions.shape[0]
        region_specs: list[tuple[int, int]] = []
        region_masks_by_image: list[torch.Tensor | None] = []

        for image_idx, meta in enumerate(image_meta):
            grid_h = int(meta["grid_h"])
            grid_w = int(meta["grid_w"])
            has_cls = bool(meta["has_cls"])
            region_masks = self._build_region_masks(grid_h, grid_w, has_cls, image_embs[image_idx].device)
            region_masks_by_image.append(region_masks)
            if region_masks is None:
                continue
            for region_idx in range(region_masks.shape[0]):
                region_mask = region_masks[region_idx]
                active = bool(region_mask.any().item())
                if active:
                    region_specs.append((image_idx, region_idx))

        if not region_specs:
            return None

        num_variants = len(region_specs)
        batched_image_embs = [emb.repeat(num_variants, 1, 1) for emb in image_embs]
        batched_image_masks = [mask.repeat(num_variants, 1) for mask in image_token_masks]

        perturbation = self.config.region_perturbation
        noise_std = float(self.config.region_noise_std)
        mean_tokens_by_image: dict[int, torch.Tensor] = {}
        prev_tokens_by_image: dict[int, torch.Tensor] = {}

        if perturbation in {"mean", "prev"}:
            for image_idx, meta in enumerate(image_meta):
                mean_tokens_by_image[image_idx] = image_embs[image_idx].mean(dim=1, keepdim=True).expand_as(
                    image_embs[image_idx]
                )
                prev_tokens = self._prev_image_tokens.get(str(meta["key"]))
                if prev_tokens is not None and prev_tokens.shape == image_embs[image_idx].shape:
                    prev_tokens_by_image[image_idx] = prev_tokens

        for variant_idx, (image_idx, region_idx) in enumerate(region_specs):
            meta = image_meta[image_idx]
            region_masks = region_masks_by_image[image_idx]
            if region_masks is None:
                continue
            region_mask = region_masks[region_idx]
            block = slice(variant_idx * bsize, (variant_idx + 1) * bsize)
            tokens_block = batched_image_embs[image_idx][block]
            region_token_mask = region_mask[None, :, None]
            if not region_token_mask.any():
                continue

            if perturbation == "mean":
                replacement = mean_tokens_by_image[image_idx]
            elif perturbation == "prev":
                replacement = prev_tokens_by_image.get(image_idx)
                if replacement is None:
                    replacement = mean_tokens_by_image[image_idx]
            elif perturbation == "noise":
                replacement = torch.randn_like(tokens_block) * noise_std
            else:
                replacement = torch.zeros_like(tokens_block)

            perturbed = torch.where(region_token_mask, replacement, tokens_block)
            batched_image_embs[image_idx][block] = perturbed

        lang_tokens_rep = lang_tokens.repeat(num_variants, 1)
        lang_masks_rep = lang_masks.repeat(num_variants, 1)
        if state.ndim == 2:
            state_rep = state.repeat(num_variants, 1)
        else:
            state_rep = state.repeat(num_variants, 1, 1)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._build_prefix_from_tokens(
            batched_image_embs, batched_image_masks, lang_tokens_rep, lang_masks_rep, state=state_rep
        )
        noise_rep = noise.repeat(num_variants, 1, 1)
        actions_variants = self._run_diffusion(
            prefix_embs, prefix_pad_masks, prefix_att_masks, noise_rep, use_rtc=False
        )

        actions_variants = actions_variants.view(num_variants, bsize, *actions_variants.shape[1:])
        deltas = torch.norm(actions_variants - baseline_actions[None, ...], dim=(2, 3))
        important_flags = deltas >= float(self.config.region_importance_threshold)

        keep_masks: dict[str, torch.Tensor] = {}
        for image_idx, meta in enumerate(image_meta):
            region_masks = region_masks_by_image[image_idx]
            if region_masks is None:
                continue
            importance = torch.zeros(
                (bsize, region_masks.shape[0]), dtype=torch.bool, device=baseline_actions.device
            )
            for variant_idx, (spec_image_idx, region_idx) in enumerate(region_specs):
                if spec_image_idx != image_idx:
                    continue
                importance[:, region_idx] |= important_flags[variant_idx]

            important_mask = torch.einsum("br,rn->bn", importance.float(), region_masks.float()) > 0
            if bool(meta["has_cls"]) and important_mask.shape[1] > 0:
                important_mask[:, 0] = True

            self._last_important_masks[str(meta["key"])] = important_mask.detach()

            interval_keep_mask = important_mask
            bg_mask = background_masks.get(str(meta["key"]))
            if bg_mask is not None and bg_mask.shape == important_mask.shape:
                interval_keep_mask = ~(bg_mask & (~important_mask))
            keep_masks[str(meta["key"])] = interval_keep_mask

            total_tokens = image_embs[image_idx].shape[1] * bsize
            if total_tokens > 0:
                bg_count = int(bg_mask.sum().item()) if bg_mask is not None else 0
                important_count = int(important_mask.sum().item())
                if self.config.token_selection_log_frames:
                    logger.info(
                        "Token selection frame=%d image=%s total=%d bg=%d (%.3f) important=%d (%.3f) regions=%d eval=%d",
                        self._frame_counter,
                        meta["key"],
                        total_tokens,
                        bg_count,
                        bg_count / total_tokens,
                        important_count,
                        important_count / total_tokens,
                        region_masks.shape[0],
                        sum(1 for spec in region_specs if spec[0] == image_idx),
                    )

        return keep_masks

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        timing: dict[str, float] = {}
        device = state.device
        self._sync_if_cuda(device)
        total_start = time.perf_counter()
        bsize = state.shape[0]

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        if not self.config.token_selection_enabled:
            self._sync_if_cuda(device)
            encode_start = time.perf_counter()
            image_embs, image_token_masks, _ = self._encode_images(images, img_masks)
            self._sync_if_cuda(device)
            timing["vision_encode_s"] = time.perf_counter() - encode_start

            self._sync_if_cuda(device)
            prefix_start = time.perf_counter()
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._build_prefix_from_tokens(
                image_embs, image_token_masks, lang_tokens, lang_masks, state=state
            )
            self._sync_if_cuda(device)
            timing["prefix_build_s"] = time.perf_counter() - prefix_start

            x_t = self._run_diffusion(
                prefix_embs, prefix_pad_masks, prefix_att_masks, noise, use_rtc=True, timing=timing, **kwargs
            )
            timing["background_s"] = 0.0
            timing["prune_s"] = 0.0
            timing["region_eval_s"] = 0.0
            timing["token_selection_s"] = 0.0
            self._sync_if_cuda(device)
            timing["total_s"] = time.perf_counter() - total_start
            self._record_timing(timing, is_eval=False, pruned=False)
            self._frame_counter += 1
            return x_t

        self._ensure_token_cache_state(bsize)
        self._sync_if_cuda(device)
        encode_start = time.perf_counter()
        image_embs, image_token_masks, image_meta = self._encode_images(images, img_masks)
        self._sync_if_cuda(device)
        timing["vision_encode_s"] = time.perf_counter() - encode_start
        raw_image_embs = [emb.detach() for emb in image_embs]

        background_masks: dict[str, torch.Tensor] = {}
        self._last_overlay_keys = []
        self._sync_if_cuda(device)
        background_start = time.perf_counter()
        for emb, token_mask, meta in zip(image_embs, image_token_masks, image_meta, strict=False):
            prev_tokens = self._prev_image_tokens.get(str(meta["key"]))
            bg_mask = self._compute_background_mask(
                emb, prev_tokens, int(meta["grid_h"]), int(meta["grid_w"]), bool(meta["has_cls"])
            )
            background_masks[str(meta["key"])] = bg_mask
            key = str(meta["key"])
            self._last_background_masks[key] = bg_mask.detach()
            self._last_token_masks[key] = token_mask.detach()
            self._last_token_meta[key] = {
                "grid_h": int(meta["grid_h"]),
                "grid_w": int(meta["grid_w"]),
                "has_cls": bool(meta["has_cls"]),
            }
            self._last_overlay_keys.append(key)
        self._sync_if_cuda(device)
        timing["background_s"] = time.perf_counter() - background_start

        eval_interval = int(self.config.region_eval_interval)
        do_region_eval = eval_interval > 0 and (self._frame_counter % eval_interval == 0)
        pruned_forward = False
        removed_tokens = 0.0
        total_tokens = 0.0
        total_tokens_computed = False

        if self.config.token_prune_enabled and not do_region_eval and self._token_keep_masks:
            pruned_embs = []
            pruned_masks = []
            self._sync_if_cuda(device)
            prune_start = time.perf_counter()
            for emb, mask, meta in zip(image_embs, image_token_masks, image_meta, strict=False):
                mask_total = int(mask.sum().item())
                total_tokens += mask_total
                total_tokens_computed = True
                keep_mask = self._token_keep_masks.get(str(meta["key"]))
                if keep_mask is None:
                    pruned_embs.append(emb)
                    pruned_masks.append(mask)
                    continue
                pruned_emb, pruned_mask = self._prune_tokens_for_image(
                    emb, mask, None, keep_mask, timing=timing
                )
                if int(pruned_mask.sum().item()) < int(mask.sum().item()):
                    pruned_forward = True
                removed_tokens += max(0, mask_total - int(pruned_mask.sum().item()))
                pruned_embs.append(pruned_emb)
                pruned_masks.append(pruned_mask)
            self._sync_if_cuda(device)
            timing["prune_s"] = time.perf_counter() - prune_start
            self._sync_if_cuda(device)
            prefix_start = time.perf_counter()
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._build_prefix_from_tokens(
                pruned_embs, pruned_masks, lang_tokens, lang_masks, state=state
            )
            self._sync_if_cuda(device)
            timing["prefix_build_s"] = time.perf_counter() - prefix_start
        else:
            timing["prune_s"] = 0.0
            self._sync_if_cuda(device)
            prefix_start = time.perf_counter()
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._build_prefix_from_tokens(
                image_embs, image_token_masks, lang_tokens, lang_masks, state=state
            )
            self._sync_if_cuda(device)
            timing["prefix_build_s"] = time.perf_counter() - prefix_start

        x_t = self._run_diffusion(
            prefix_embs, prefix_pad_masks, prefix_att_masks, noise, use_rtc=True, timing=timing, **kwargs
        )

        if not do_region_eval and not total_tokens_computed:
            total_tokens = sum(int(mask.sum().item()) for mask in image_token_masks)
            total_tokens_computed = True

        if do_region_eval:
            self._sync_if_cuda(device)
            region_start = time.perf_counter()
            keep_masks = self._evaluate_region_importance(
                image_embs,
                image_token_masks,
                image_meta,
                background_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                x_t,
            )
            self._sync_if_cuda(device)
            timing["region_eval_s"] = time.perf_counter() - region_start
            if keep_masks:
                self._token_keep_masks.update(keep_masks)
        else:
            timing["region_eval_s"] = 0.0

        for emb, meta in zip(raw_image_embs, image_meta, strict=False):
            self._prev_image_tokens[str(meta["key"])] = emb

        self._accumulate_token_stats(background_masks, image_token_masks, image_meta)
        timing["token_selection_s"] = (
            timing.get("background_s", 0.0)
            + timing.get("region_eval_s", 0.0)
            + timing.get("prune_s", 0.0)
        )
        self._sync_if_cuda(device)
        timing["total_s"] = time.perf_counter() - total_start
        self._record_timing(timing, is_eval=do_region_eval, pruned=pruned_forward)
        if not do_region_eval:
            self._pruned_token_total += removed_tokens
            self._pruned_token_counts += 1
            self._pruned_token_denom_total += total_tokens
        self._frame_counter += 1
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
