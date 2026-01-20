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
import time as time_module
from collections import deque
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
        if hasattr(self.model, "reset_token_cache"):
            self.model.reset_token_cache()
        if hasattr(self.model, "reset_vision_cache"):
            self.model.reset_vision_cache()

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

    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

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

    def requires_grad_for_action(self) -> bool:
        return (
            self.config.token_selection_enabled
            and self.config.token_importance_method == "grad"
        )

    def get_last_token_overlay(self, image_index: int = 0) -> dict[str, object] | None:
        if hasattr(self.model, "get_last_token_overlay"):
            return self.model.get_last_token_overlay(image_index=image_index)
        return None

    def get_timing_summary(self) -> dict[str, object] | None:
        if hasattr(self.model, "get_timing_summary"):
            return self.model.get_timing_summary()
        return None

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

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

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
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
        self.reset_token_cache()

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def reset_token_cache(self) -> None:
        self._frame_counter = 0
        self._last_image_embs: dict[int, torch.Tensor] = {}
        self._last_bg_masks: dict[int, torch.Tensor] = {}
        self._last_important_masks: dict[int, torch.Tensor] = {}
        self._last_effective_important_masks: dict[int, torch.Tensor] = {}
        self._last_keep_masks: dict[int, torch.Tensor] = {}
        self._last_clipped_masks: dict[int, torch.Tensor] = {}
        self._last_token_scores: dict[int, torch.Tensor] = {}
        self._last_region_scores: dict[int, torch.Tensor] = {}
        self._last_token_masks: dict[int, torch.Tensor] = {}
        self._last_grid_info: dict[int, tuple[bool, int, int]] = {}
        self._last_token_overlay: dict[int, dict[str, object]] = {}
        self._timing = self._init_timing()

    def reset_vision_cache(self) -> None:
        if hasattr(self.vlm_with_expert, "reset_vision_cache"):
            self.vlm_with_expert.reset_vision_cache()

    def get_last_token_overlay(self, image_index: int = 0) -> dict[str, object] | None:
        return self._last_token_overlay.get(image_index)

    def _init_timing(self) -> dict[str, object]:
        return {
            "num_calls": 0,
            "num_eval_calls": 0,
            "num_noeval_calls": 0,
            "sum_s": {},
            "sum_eval_s": {},
            "sum_noeval_s": {},
            "llm_pruned_s": 0.0,
            "llm_pruned_calls": 0,
            "llm_unpruned_s": 0.0,
            "llm_unpruned_calls": 0,
            "pruned_tokens": 0.0,
            "pruned_ratio": 0.0,
            "pruned_calls": 0,
            "token_counts": {"total": 0.0, "prunable_bg": 0.0, "important": 0.0},
            "token_count_calls": 0,
        }

    def _accumulate_timing(self, key: str, value: float, bucket: str = "sum_s") -> None:
        bucket_dict = self._timing.setdefault(bucket, {})
        bucket_dict[key] = bucket_dict.get(key, 0.0) + value

    def get_timing_summary(self) -> dict[str, object]:
        timing = self._timing
        num_calls = int(timing.get("num_calls", 0))
        num_eval_calls = int(timing.get("num_eval_calls", 0))
        num_noeval_calls = int(timing.get("num_noeval_calls", 0))
        if num_calls <= 0:
            return {}

        def _avg(bucket: str, denom: int) -> dict[str, float]:
            data = timing.get(bucket, {})
            if denom <= 0 or not isinstance(data, dict):
                return {}
            return {k: v / denom for k, v in data.items()}

        avg_s = _avg("sum_s", num_calls)
        avg_eval_s = _avg("sum_eval_s", num_eval_calls)
        avg_noeval_s = _avg("sum_noeval_s", num_noeval_calls)

        llm_pruned_calls = int(timing.get("llm_pruned_calls", 0))
        llm_unpruned_calls = int(timing.get("llm_unpruned_calls", 0))
        avg_llm_pruned_s = (
            float(timing.get("llm_pruned_s", 0.0)) / llm_pruned_calls if llm_pruned_calls > 0 else 0.0
        )
        avg_llm_unpruned_s = (
            float(timing.get("llm_unpruned_s", 0.0)) / llm_unpruned_calls if llm_unpruned_calls > 0 else 0.0
        )

        pruned_calls = int(timing.get("pruned_calls", 0))
        avg_pruned_tokens = (
            float(timing.get("pruned_tokens", 0.0)) / pruned_calls if pruned_calls > 0 else 0.0
        )
        avg_pruned_ratio = (
            float(timing.get("pruned_ratio", 0.0)) / pruned_calls if pruned_calls > 0 else 0.0
        )

        token_count_calls = int(timing.get("token_count_calls", 0))
        token_counts = timing.get("token_counts", {})
        token_avg_counts = {}
        if token_count_calls > 0 and isinstance(token_counts, dict):
            token_avg_counts = {k: v / token_count_calls for k, v in token_counts.items()}

        return {
            "num_calls": num_calls,
            "num_eval_calls": num_eval_calls,
            "num_noeval_calls": num_noeval_calls,
            "avg_s": avg_s,
            "avg_eval_s": avg_eval_s,
            "avg_noeval_s": avg_noeval_s,
            "avg_llm_pruned_s": avg_llm_pruned_s,
            "avg_llm_unpruned_s": avg_llm_unpruned_s,
            "num_llm_pruned_calls": llm_pruned_calls,
            "num_llm_unpruned_calls": llm_unpruned_calls,
            "avg_pruned_tokens": avg_pruned_tokens,
            "avg_pruned_ratio": avg_pruned_ratio,
            "token_avg_counts": token_avg_counts,
        }

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

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        image_embs: list[torch.Tensor] | None = None,
        return_image_spans: bool = False,
    ):
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        image_spans: list[tuple[int, int]] = []
        current_len = 0
        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
                current_len += image_start_mask.shape[-1]

            if image_embs is None:
                img_emb = self.vlm_with_expert.embed_image(img)
            else:
                img_emb = image_embs[_img_idx]

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            image_spans.append((current_len, current_len + num_img_embs))
            current_len += num_img_embs
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
                current_len += image_end_mask.shape[1]
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
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

        # Set attention masks so that image and language inputs do not attend to state or actions
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

        if return_image_spans:
            return embs, pad_masks, att_masks, image_spans
        return embs, pad_masks, att_masks

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

    def _get_grid_info(self, num_tokens: int) -> tuple[bool, int, int]:
        if num_tokens <= 0:
            return False, 0, 0
        has_cls = False
        grid_tokens = num_tokens
        side = int(round(math.sqrt(max(num_tokens - 1, 1))))
        if num_tokens > 1 and side * side == num_tokens - 1:
            has_cls = True
            grid_tokens = num_tokens - 1
        side = int(round(math.sqrt(max(grid_tokens, 1))))
        if side * side == grid_tokens and side > 0:
            grid_h = side
            grid_w = side
        else:
            grid_h = 1
            grid_w = grid_tokens
        return has_cls, grid_h, grid_w

    def _build_token_mask(self, img_mask: torch.Tensor | None, num_tokens: int, bsize: int) -> torch.Tensor:
        if img_mask is None:
            return torch.ones((bsize, num_tokens), dtype=torch.bool)
        if img_mask.ndim != 1:
            img_mask = img_mask.view(-1)
        bsize = img_mask.shape[0]
        return img_mask[:, None].expand(bsize, num_tokens).clone()

    def _compute_spatial_mask(
        self, tokens: torch.Tensor, grid_h: int, grid_w: int, radius: int
    ) -> torch.Tensor:
        bsize, num_tokens, _ = tokens.shape
        if grid_h * grid_w != num_tokens or radius <= 0:
            return torch.ones((bsize, num_tokens), dtype=torch.bool, device=tokens.device)
        tokens_grid = tokens.view(bsize, grid_h, grid_w, -1)
        accum = torch.zeros((bsize, grid_h, grid_w), dtype=tokens.dtype, device=tokens.device)
        counts = torch.zeros((bsize, grid_h, grid_w), dtype=tokens.dtype, device=tokens.device)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                y0 = max(0, -dy)
                y1 = min(grid_h, grid_h - dy)
                x0 = max(0, -dx)
                x1 = min(grid_w, grid_w - dx)
                if y1 <= y0 or x1 <= x0:
                    continue
                src = tokens_grid[:, y0:y1, x0:x1]
                nbr = tokens_grid[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                sim = (src * nbr).sum(dim=-1)
                accum[:, y0:y1, x0:x1] += sim
                counts[:, y0:y1, x0:x1] += 1.0
        avg = accum / counts.clamp(min=1.0)
        return (avg >= self.config.token_spatial_threshold).view(bsize, -1)

    def _compute_background_mask(
        self,
        image_index: int,
        image_embs: torch.Tensor,
        token_mask: torch.Tensor,
        grid_info: tuple[bool, int, int],
    ) -> torch.Tensor:
        has_cls, grid_h, grid_w = grid_info
        prev = self._last_image_embs.get(image_index)
        if prev is None or prev.shape != image_embs.shape:
            bg = torch.zeros_like(token_mask, dtype=torch.bool, device=image_embs.device)
            self._last_image_embs[image_index] = image_embs.detach()
            return bg

        if has_cls:
            cur_tokens = image_embs[:, 1:]
            prev_tokens = prev[:, 1:]
            token_mask_nocls = token_mask[:, 1:]
        else:
            cur_tokens = image_embs
            prev_tokens = prev
            token_mask_nocls = token_mask

        cur_norm = F.normalize(cur_tokens.float(), dim=-1)
        prev_norm = F.normalize(prev_tokens.float(), dim=-1)
        temporal = (cur_norm * prev_norm).sum(dim=-1)
        temporal_mask = temporal >= self.config.token_temporal_threshold

        spatial_mask = self._compute_spatial_mask(cur_norm, grid_h, grid_w, self.config.token_spatial_radius)
        spatial_mask = spatial_mask.to(dtype=torch.bool)
        bg_nocls = temporal_mask & spatial_mask & token_mask_nocls

        bg = torch.zeros_like(token_mask, dtype=torch.bool, device=image_embs.device)
        if has_cls:
            bg[:, 1:] = bg_nocls
        else:
            bg = bg_nocls
        self._last_image_embs[image_index] = image_embs.detach()
        return bg

    def _aggregate_region_scores(
        self,
        token_scores: torch.Tensor,
        token_mask: torch.Tensor,
        grid_info: tuple[bool, int, int],
    ) -> torch.Tensor:
        has_cls, grid_h, grid_w = grid_info
        if has_cls:
            scores = token_scores[:, 1:]
            mask = token_mask[:, 1:]
        else:
            scores = token_scores
            mask = token_mask
        bsize, num_tokens = scores.shape[:2]
        if grid_h * grid_w != num_tokens:
            return scores
        scores_grid = scores.view(bsize, grid_h, grid_w)
        mask_grid = mask.view(bsize, grid_h, grid_w)
        r = max(1, int(self.config.region_patch_size))
        num_regions_h = (grid_h + r - 1) // r
        num_regions_w = (grid_w + r - 1) // r
        region_scores = torch.zeros(
            (bsize, num_regions_h * num_regions_w), dtype=scores.dtype, device=scores.device
        )
        for rh in range(num_regions_h):
            r0 = rh * r
            r1 = min((rh + 1) * r, grid_h)
            for rw in range(num_regions_w):
                c0 = rw * r
                c1 = min((rw + 1) * r, grid_w)
                region = scores_grid[:, r0:r1, c0:c1]
                region_mask = mask_grid[:, r0:r1, c0:c1]
                region_sum = (region * region_mask).sum(dim=(1, 2))
                if self.config.grad_region_reduce == "mean":
                    denom = region_mask.sum(dim=(1, 2)).clamp(min=1)
                    region_val = region_sum / denom
                else:
                    region_val = region_sum
                region_scores[:, rh * num_regions_w + rw] = region_val
        return region_scores

    def _select_regions(self, region_scores: torch.Tensor) -> torch.Tensor:
        bsize, num_regions = region_scores.shape
        if num_regions == 0:
            return torch.zeros_like(region_scores, dtype=torch.bool)
        total = region_scores.sum(dim=1, keepdim=True)
        scores = region_scores
        if total.max().item() > 0:
            scores = region_scores / total.clamp(min=1e-6)
        mass = float(self.config.grad_region_mass)
        if mass >= 1.0 or total.max().item() <= 0:
            return torch.ones((bsize, num_regions), dtype=torch.bool, device=region_scores.device)

        sorted_scores, sorted_idx = torch.sort(scores, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=1)
        keep_sorted = cumsum <= mass
        keep_sorted[:, 0] = True
        keep = torch.zeros_like(keep_sorted, dtype=torch.bool)
        keep.scatter_(1, sorted_idx, keep_sorted)
        return keep

    def _regions_to_token_mask(
        self, region_keep: torch.Tensor, grid_info: tuple[bool, int, int], num_tokens: int
    ) -> torch.Tensor:
        has_cls, grid_h, grid_w = grid_info
        bsize = region_keep.shape[0]
        r = max(1, int(self.config.region_patch_size))
        num_regions_h = (grid_h + r - 1) // r
        num_regions_w = (grid_w + r - 1) // r
        if num_regions_h * num_regions_w != region_keep.shape[1]:
            mask = torch.zeros((bsize, num_tokens), dtype=torch.bool, device=region_keep.device)
            return mask
        grid_mask = torch.zeros((bsize, grid_h, grid_w), dtype=torch.bool, device=region_keep.device)
        for rh in range(num_regions_h):
            r0 = rh * r
            r1 = min((rh + 1) * r, grid_h)
            for rw in range(num_regions_w):
                c0 = rw * r
                c1 = min((rw + 1) * r, grid_w)
                region_idx = rh * num_regions_w + rw
                region_flag = region_keep[:, region_idx][:, None, None]
                grid_mask[:, r0:r1, c0:c1] |= region_flag
        flat = grid_mask.view(bsize, grid_h * grid_w)
        if has_cls:
            out = torch.zeros((bsize, num_tokens), dtype=torch.bool, device=region_keep.device)
            if num_tokens == flat.shape[1] + 1:
                out[:, 1:] = flat
            else:
                out = flat
            return out
        return flat

    def _apply_keep_constraints(
        self,
        keep_mask: torch.Tensor,
        token_scores: torch.Tensor | None,
        token_mask: torch.Tensor,
        has_cls: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep_mask = keep_mask & token_mask
        count_mask = token_mask.clone()
        if has_cls and count_mask.shape[1] > 0:
            count_mask[:, 0] = False
            keep_mask[:, 0] = True

        scores = token_scores
        if scores is None:
            scores = torch.zeros_like(keep_mask, dtype=torch.float32)
        else:
            scores = scores.float()

        scores = scores.clone()
        scores[~count_mask] = float("-inf")

        clipped_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
        max_keep = int(self.config.max_kept_tokens)
        if max_keep > 0:
            keep_count = keep_mask & count_mask
            num_keep = keep_count.sum(dim=1)
            count_sum = count_mask.sum(dim=1)
            for b in range(keep_mask.shape[0]):
                if int(num_keep[b]) > max_keep and int(count_sum[b]) > 0:
                    k = min(max_keep, int(count_sum[b]))
                    topk = torch.topk(scores[b], k=k, dim=0).indices
                    new_keep = torch.zeros_like(keep_mask[b], dtype=torch.bool)
                    new_keep[topk] = True
                    if has_cls and keep_mask.shape[1] > 0:
                        new_keep[0] = True
                    clipped_mask[b] = (keep_mask[b] & count_mask[b]) & (~new_keep)
                    keep_mask[b] = (keep_mask[b] & ~count_mask[b]) | new_keep

        min_keep = int(self.config.min_kept_tokens)
        if min_keep > 0:
            keep_count = (keep_mask & count_mask).sum(dim=1)
            for b in range(keep_mask.shape[0]):
                need = min_keep - int(keep_count[b])
                if need <= 0:
                    continue
                missing_scores = scores[b].clone()
                missing_scores[~count_mask[b]] = float("-inf")
                missing_scores[keep_mask[b]] = float("-inf")
                k = min(need, int(count_mask[b].sum().item()))
                if k <= 0:
                    continue
                topk = torch.topk(missing_scores, k=k, dim=0).indices
                keep_mask[b, topk] = True

        return keep_mask, clipped_mask

    def _build_update_mask_grid(
        self,
        token_mask: torch.Tensor,
        grid_info: tuple[bool, int, int],
        num_tokens: int,
    ) -> torch.Tensor | None:
        has_cls, grid_h, grid_w = grid_info
        if grid_h <= 0 or grid_w <= 0:
            return None
        if has_cls and token_mask.shape[1] == num_tokens:
            token_mask = token_mask[:, 1:]
        if token_mask.shape[1] != grid_h * grid_w:
            return None
        return token_mask.view(token_mask.shape[0], grid_h, grid_w)

    def _apply_keep_mask_to_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        image_spans: list[tuple[int, int]],
        keep_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize, seq_len = prefix_embs.shape[:2]
        keep_prefix = torch.ones((bsize, seq_len), dtype=torch.bool, device=prefix_embs.device)
        for idx, (start, end) in enumerate(image_spans):
            keep_mask = keep_masks[idx]
            if keep_mask.shape[1] != (end - start):
                continue
            keep_prefix[:, start:end] = keep_mask

        keep_lens = keep_prefix.sum(dim=1)
        max_len = int(keep_lens.max().item()) if keep_lens.numel() > 0 else 0
        new_embs = torch.zeros((bsize, max_len, prefix_embs.shape[-1]), dtype=prefix_embs.dtype, device=prefix_embs.device)
        new_pad = torch.zeros((bsize, max_len), dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device)
        new_att = torch.zeros((bsize, max_len), dtype=prefix_att_masks.dtype, device=prefix_att_masks.device)
        for b in range(bsize):
            idxs = torch.nonzero(keep_prefix[b], as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            new_embs[b, : idxs.numel()] = prefix_embs[b, idxs]
            new_pad[b, : idxs.numel()] = prefix_pad_masks[b, idxs]
            new_att[b, : idxs.numel()] = prefix_att_masks[b, idxs]
        return new_embs, new_pad, new_att

    def _compute_grad_objective(self, denoise_outputs: list[torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] <= 0:
            return torch.tensor(0.0, device=actions.device)
        grip = actions[..., -1]
        if grip.shape[1] > 1:
            m = (grip[:, 1:] - grip[:, :-1]).abs().mean(dim=1)
        else:
            m = torch.zeros(actions.shape[0], device=actions.device)
        gate = torch.clamp(m / float(self.config.grad_tau), 0.0, 1.0)
        lambda_grip = 1.0 + float(self.config.grad_alpha) * gate
        lambda_pos = 1.0 + float(self.config.grad_beta) * (1.0 - gate)

        objective = 0.0
        for v_t in denoise_outputs:
            if v_t.shape[-1] <= 1:
                v_grip = v_t
                v_pos = None
            else:
                v_pos = v_t[..., :-1]
                v_grip = v_t[..., -1:]
            if v_pos is not None and v_pos.numel() > 0:
                objective = objective + (lambda_pos[:, None, None] * (v_pos**2)).mean()
            objective = objective + (lambda_grip[:, None, None] * (v_grip**2)).mean()
        return objective

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
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        total_start = time_module.perf_counter()
        token_sel_enabled = bool(self.config.token_selection_enabled)
        token_prune_enabled = bool(self.config.token_prune_enabled)
        method = self.config.token_importance_method
        do_region_eval = token_sel_enabled and (
            self._frame_counter % int(self.config.region_eval_interval) == 0
        )
        self._frame_counter += 1

        vision_start = time_module.perf_counter()
        image_embs: list[torch.Tensor] = []
        token_masks: list[torch.Tensor] = []
        grid_infos: list[tuple[bool, int, int]] = []
        for idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            update_mask_grid = None
            enable_partial_update = False
            force_full_update = False
            if token_sel_enabled and self.config.vision_partial_update_enabled and not do_region_eval:
                last_imp = self._last_effective_important_masks.get(idx)
                if last_imp is not None:
                    grid_prev = self._last_grid_info.get(idx)
                    if grid_prev is None:
                        grid_prev = self._get_grid_info(last_imp.shape[1])
                    update_mask_grid = self._build_update_mask_grid(
                        last_imp.to(device=img.device),
                        grid_prev,
                        last_imp.shape[1],
                    )
                    enable_partial_update = update_mask_grid is not None
            if do_region_eval:
                force_full_update = True
                enable_partial_update = False
                update_mask_grid = None
            img_emb = self.vlm_with_expert.embed_image(
                img,
                cache_name=f"image{idx}",
                cache_key=idx,
                update_mask_grid=update_mask_grid,
                enable_partial_update=enable_partial_update,
                force_full_update=force_full_update,
            )
            if token_sel_enabled and method == "grad" and do_region_eval:
                img_emb = img_emb.detach().requires_grad_(True)
            image_embs.append(img_emb)
            grid_info = self._get_grid_info(img_emb.shape[1])
            grid_infos.append(grid_info)
            token_mask = self._build_token_mask(img_mask, img_emb.shape[1], img_emb.shape[0]).to(
                img_emb.device
            )
            token_masks.append(token_mask)
            self._last_grid_info[idx] = grid_info
        vision_s = time_module.perf_counter() - vision_start

        background_start = time_module.perf_counter()
        background_masks: list[torch.Tensor] = []
        if token_sel_enabled:
            for idx, img_emb in enumerate(image_embs):
                bg = self._compute_background_mask(idx, img_emb, token_masks[idx], grid_infos[idx])
                background_masks.append(bg)
                self._last_bg_masks[idx] = bg.detach()
        else:
            for idx, mask in enumerate(token_masks):
                background_masks.append(torch.zeros_like(mask, dtype=torch.bool, device=mask.device))
        background_s = time_module.perf_counter() - background_start

        prefix_build_start = time_module.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks, image_spans = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            image_embs=image_embs,
            return_image_spans=True,
        )
        prefix_build_s = time_module.perf_counter() - prefix_build_start

        important_masks: list[torch.Tensor] = []
        region_scores: list[torch.Tensor | None] = []
        token_scores: list[torch.Tensor] = []

        keep_masks: list[torch.Tensor] = []
        clipped_masks: list[torch.Tensor] = []
        effective_important_masks: list[torch.Tensor] = []

        mask_time = 0.0
        prune_time = 0.0

        if token_sel_enabled and not do_region_eval:
            mask_start = time_module.perf_counter()
            for idx, img_emb in enumerate(image_embs):
                imp = self._last_important_masks.get(idx)
                if imp is None:
                    imp = torch.zeros_like(token_masks[idx], dtype=torch.bool)
                imp = imp.to(device=img_emb.device)
                important_masks.append(imp)
                last_scores = self._last_token_scores.get(idx)
                if last_scores is None:
                    last_scores = img_emb.detach().float().norm(dim=-1)
                token_scores.append(last_scores.to(device=img_emb.device))
                last_region = self._last_region_scores.get(idx)
                if last_region is None:
                    last_region = self._aggregate_region_scores(
                        token_scores[-1], token_masks[idx], grid_infos[idx]
                    )
                region_scores.append(last_region.to(device=img_emb.device))

                keep_pre = (token_masks[idx] & (~background_masks[idx])) | imp
                keep_mask, clipped = self._apply_keep_constraints(
                    keep_pre,
                    token_scores[-1],
                    token_masks[idx],
                    grid_infos[idx][0],
                )
                eff_imp = imp & keep_mask
                keep_masks.append(keep_mask)
                clipped_masks.append(clipped)
                effective_important_masks.append(eff_imp)
                self._last_keep_masks[idx] = keep_mask.detach()
                self._last_clipped_masks[idx] = clipped.detach()
                self._last_effective_important_masks[idx] = eff_imp.detach()
            mask_time = time_module.perf_counter() - mask_start

            if token_prune_enabled:
                prune_start = time_module.perf_counter()
                prefix_embs, prefix_pad_masks, prefix_att_masks = self._apply_keep_mask_to_prefix(
                    prefix_embs, prefix_pad_masks, prefix_att_masks, image_spans, keep_masks
                )
                prune_time = time_module.perf_counter() - prune_start

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        llm_start = time_module.perf_counter()
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        llm_s = time_module.perf_counter() - llm_start

        if token_prune_enabled and token_sel_enabled and not do_region_eval:
            self._timing["llm_pruned_s"] += llm_s
            self._timing["llm_pruned_calls"] += 1
        else:
            self._timing["llm_unpruned_s"] += llm_s
            self._timing["llm_unpruned_calls"] += 1

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        denoise_start = time_module.perf_counter()
        grad_steps = min(num_steps, max(1, int(self.config.grad_denoise_steps)))
        grad_outputs: list[torch.Tensor] = []

        for step in range(num_steps):
            step_time = 1.0 + step * dt
            time_tensor = torch.tensor(step_time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
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
                    time=step_time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            if token_sel_enabled and do_region_eval and method == "grad" and step >= num_steps - grad_steps:
                grad_outputs.append(v_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=step_time, x_t=x_t, v_t=v_t)

        diffusion_s = time_module.perf_counter() - denoise_start

        region_eval_s = 0.0
        if token_sel_enabled and do_region_eval:
            if method != "grad":
                raise ValueError(f"Unsupported token_importance_method: {method}")
            region_start = time_module.perf_counter()
            if not grad_outputs:
                grad_outputs = [v_t]
            objective = self._compute_grad_objective(grad_outputs, x_t)
            if not objective.requires_grad:
                grads = [None for _ in image_embs]
            else:
                grads = torch.autograd.grad(objective, image_embs, allow_unused=True, retain_graph=False)

            for idx, (img_emb, grad) in enumerate(zip(image_embs, grads, strict=False)):
                if grad is None:
                    score = torch.zeros(img_emb.shape[:2], dtype=torch.float32, device=img_emb.device)
                else:
                    grad_norm = grad.float().pow(2).sum(dim=-1).sqrt()
                    emb_norm = img_emb.detach().float().pow(2).sum(dim=-1).sqrt()
                    score = grad_norm * emb_norm

                token_scores.append(score)
                region = self._aggregate_region_scores(score, token_masks[idx], grid_infos[idx])
                ema = float(self.config.grad_region_ema)
                if ema > 0.0:
                    prev = self._last_region_scores.get(idx)
                    if prev is not None and prev.shape == region.shape:
                        region = prev.to(device=region.device) * ema + region * (1.0 - ema)
                denom = region.sum(dim=1, keepdim=True).clamp(min=1e-6)
                region = region / denom

                keep_regions = self._select_regions(region)
                imp_mask = self._regions_to_token_mask(
                    keep_regions, grid_infos[idx], token_masks[idx].shape[1]
                )
                imp_mask = imp_mask & token_masks[idx]
                if self.config.grad_keep_prev:
                    prev_imp = self._last_important_masks.get(idx)
                    if prev_imp is not None and prev_imp.shape == imp_mask.shape:
                        imp_mask = imp_mask | prev_imp.to(device=imp_mask.device)

                important_masks.append(imp_mask)
                region_scores.append(region)

                self._last_token_scores[idx] = score.detach()
                self._last_region_scores[idx] = region.detach()
                self._last_important_masks[idx] = imp_mask.detach()

            for idx, imp in enumerate(important_masks):
                keep_pre = (token_masks[idx] & (~background_masks[idx])) | imp
                keep_mask, clipped = self._apply_keep_constraints(
                    keep_pre,
                    token_scores[idx],
                    token_masks[idx],
                    grid_infos[idx][0],
                )
                eff_imp = imp & keep_mask
                keep_masks.append(keep_mask)
                clipped_masks.append(clipped)
                effective_important_masks.append(eff_imp)
                self._last_keep_masks[idx] = keep_mask.detach()
                self._last_clipped_masks[idx] = clipped.detach()
                self._last_effective_important_masks[idx] = eff_imp.detach()
            region_eval_s = time_module.perf_counter() - region_start

        total_s = time_module.perf_counter() - total_start
        self._timing["num_calls"] += 1
        if do_region_eval:
            self._timing["num_eval_calls"] += 1
        else:
            self._timing["num_noeval_calls"] += 1

        self._accumulate_timing("total_s", total_s)
        self._accumulate_timing("vision_encode_s", vision_s)
        self._accumulate_timing("prefix_build_s", prefix_build_s)
        self._accumulate_timing("diffusion_total_s", diffusion_s)
        self._accumulate_timing("diffusion_denoise_s", diffusion_s)
        self._accumulate_timing("background_s", background_s)
        self._accumulate_timing("token_selection_s", background_s + region_eval_s)
        self._accumulate_timing("prune_s", mask_time + prune_time)
        if do_region_eval:
            self._accumulate_timing("total_s", total_s, bucket="sum_eval_s")
            self._accumulate_timing("region_eval_s", region_eval_s, bucket="sum_eval_s")
        else:
            self._accumulate_timing("total_s", total_s, bucket="sum_noeval_s")
            if token_prune_enabled and token_sel_enabled:
                self._accumulate_timing("prune_mask_s", mask_time, bucket="sum_noeval_s")
                self._accumulate_timing("prune_pack_s", prune_time, bucket="sum_noeval_s")

        if token_sel_enabled and token_masks:
            total_counts = 0.0
            bg_counts = 0.0
            imp_counts = 0.0
            for idx, mask in enumerate(token_masks):
                has_cls, _, _ = grid_infos[idx]
                count_mask = mask.clone()
                if has_cls and count_mask.shape[1] > 0:
                    count_mask[:, 0] = False
                total_counts += count_mask.sum(dim=1).float().mean().item()

                if idx < len(background_masks) and idx < len(effective_important_masks) and idx < len(important_masks):
                    bg_mask = background_masks[idx] & (~important_masks[idx])
                    if has_cls and bg_mask.shape[1] > 0:
                        bg_mask = bg_mask.clone()
                        bg_mask[:, 0] = False
                    bg_counts += bg_mask.sum(dim=1).float().mean().item()

                    imp_mask = effective_important_masks[idx]
                    if has_cls and imp_mask.shape[1] > 0:
                        imp_mask = imp_mask.clone()
                        imp_mask[:, 0] = False
                    imp_counts += imp_mask.sum(dim=1).float().mean().item()

            self._timing["token_counts"]["total"] += total_counts
            self._timing["token_counts"]["prunable_bg"] += bg_counts
            self._timing["token_counts"]["important"] += imp_counts
            self._timing["token_count_calls"] += 1

        if token_prune_enabled and token_sel_enabled and not do_region_eval and keep_masks:
            pruned_counts = 0.0
            pruned_ratios = 0.0
            total_tokens = None
            kept_tokens = None
            for idx, keep in enumerate(keep_masks):
                has_cls, _, _ = grid_infos[idx]
                count_mask = token_masks[idx].clone()
                if has_cls and count_mask.shape[1] > 0:
                    count_mask[:, 0] = False
                cur_total = count_mask.sum(dim=1).float()
                cur_kept = (keep & count_mask).sum(dim=1).float()
                if total_tokens is None:
                    total_tokens = cur_total
                    kept_tokens = cur_kept
                else:
                    total_tokens = total_tokens + cur_total
                    kept_tokens = kept_tokens + cur_kept
            if total_tokens is not None and kept_tokens is not None:
                pruned = (total_tokens - kept_tokens).clamp(min=0.0)
                ratio = pruned / total_tokens.clamp(min=1.0)
                pruned_counts += pruned.mean().item()
                pruned_ratios += ratio.mean().item()
            self._timing["pruned_tokens"] += pruned_counts
            self._timing["pruned_ratio"] += pruned_ratios
            self._timing["pruned_calls"] += 1

        if token_sel_enabled:
            for idx in range(len(image_embs)):
                overlay = {
                    "grid_h": grid_infos[idx][1],
                    "grid_w": grid_infos[idx][2],
                    "has_cls": grid_infos[idx][0],
                    "token_mask": token_masks[idx].detach(),
                    "background_mask": background_masks[idx].detach(),
                    "important_mask": effective_important_masks[idx].detach() if idx < len(effective_important_masks) else None,
                    "keep_mask": keep_masks[idx].detach() if idx < len(keep_masks) else None,
                    "clipped_mask": clipped_masks[idx].detach() if idx < len(clipped_masks) else None,
                    "region_scores": region_scores[idx].detach() if idx < len(region_scores) and region_scores[idx] is not None else None,
                    "region_patch_size": int(self.config.region_patch_size),
                }
                self._last_token_overlay[idx] = overlay

            if self.config.token_selection_log_frames:
                for idx in range(len(image_embs)):
                    total_tokens = int(token_masks[idx].sum().item())
                    bg_tokens = int(background_masks[idx].sum().item())
                    imp_tokens = (
                        int(effective_important_masks[idx].sum().item())
                        if idx < len(effective_important_masks)
                        else 0
                    )
                    region_count = (
                        int(region_scores[idx].shape[1]) if idx < len(region_scores) and region_scores[idx] is not None else 0
                    )
                    logging.info(
                        "Token selection frame=%d image=image%d total=%d bg=%d (%.3f) important=%d (%.3f) regions=%d eval=%d",
                        self._frame_counter - 1,
                        idx,
                        total_tokens,
                        bg_tokens,
                        bg_tokens / max(total_tokens, 1),
                        imp_tokens,
                        imp_tokens / max(total_tokens, 1),
                        region_count,
                        1 if do_region_eval else 0,
                    )
        else:
            self._last_token_overlay = {}

        if token_sel_enabled and do_region_eval and method == "grad":
            image_embs = [emb.detach() for emb in image_embs]

        return x_t.detach() if token_sel_enabled and do_region_eval and method == "grad" else x_t

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
