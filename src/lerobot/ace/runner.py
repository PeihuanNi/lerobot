#!/usr/bin/env python

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


RawImageInput = np.ndarray | torch.Tensor | Image.Image | str | Path
ImageInputs = dict[str, RawImageInput] | list[RawImageInput] | tuple[RawImageInput, ...]


def _format_cli_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return str(value)


def _overrides_to_cli_args(overrides: dict[str, Any] | None) -> list[str]:
    if not overrides:
        return []
    return [f"--{key}={_format_cli_override_value(value)}" for key, value in overrides.items()]


class AceInferenceRunner:
    """Standalone inference helper for PI0 / PI05 + ACE/token-pruning configs."""

    def __init__(
        self,
        policy_path: str | Path,
        *,
        device: str | None = None,
        use_amp: bool | None = None,
        policy_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.policy_path = Path(policy_path).expanduser().resolve()
        self.config = self._load_config(
            self.policy_path,
            device=device,
            use_amp=use_amp,
            policy_overrides=policy_overrides,
        )

        policy_cls = get_policy_class(self.config.type)
        self.policy = policy_cls.from_pretrained(
            pretrained_name_or_path=self.policy_path,
            config=self.config,
        )
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(self.config.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=str(self.policy_path),
            preprocessor_overrides=preprocessor_overrides,
        )

        self.device = torch.device(str(self.config.device))
        self.use_amp = bool(self.config.use_amp if use_amp is None else use_amp)
        self.image_keys = [key for key in self.config.image_features if "empty_camera" not in key]
        self.state_key = OBS_STATE if OBS_STATE in self.config.input_features else None
        self.state_dim = (
            int(self.config.input_features[self.state_key].shape[0]) if self.state_key is not None else 0
        )
        self.action_dim = int(self.config.output_features[ACTION].shape[0])
        self.reset()

    @staticmethod
    def _load_config(
        policy_path: Path,
        *,
        device: str | None,
        use_amp: bool | None,
        policy_overrides: dict[str, Any] | None,
    ) -> PreTrainedConfig:
        cli_overrides = _overrides_to_cli_args(policy_overrides)
        config = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        config.pretrained_path = policy_path
        # ACE/token-selection inference relies on dynamic control flow, timing and
        # gradient-based scoring that is fragile under torch.compile. Keep runner
        # inference on eager mode unless the caller explicitly overrides this.
        if not policy_overrides or "compile_model" not in policy_overrides:
            config.compile_model = False
        if device is not None:
            config.device = device
        if use_amp is not None:
            config.use_amp = use_amp
        config.__post_init__()
        return config

    def reset(self) -> None:
        self.policy.reset()

    def predict_action(
        self,
        images: ImageInputs,
        instruction: str,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None = None,
        *,
        use_policy_queue: bool = False,
    ) -> np.ndarray:
        batch = self.preprocessor(self._build_batch(images=images, instruction=instruction, state=state))
        with self._inference_context():
            if use_policy_queue:
                action = self.policy.select_action(batch)
                action = self.postprocessor(action)
                return action.squeeze(0).detach().cpu().numpy()

            action_chunk = self.policy.predict_action_chunk(batch)
            action_chunk = self.postprocessor(action_chunk)
            return action_chunk[0, 0].detach().cpu().numpy()

    def predict_action_chunk(
        self,
        images: ImageInputs,
        instruction: str,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ) -> np.ndarray:
        batch = self.preprocessor(self._build_batch(images=images, instruction=instruction, state=state))
        with self._inference_context():
            action_chunk = self.policy.predict_action_chunk(batch)
            action_chunk = self.postprocessor(action_chunk)
        return action_chunk[0].detach().cpu().numpy()

    def get_token_selection_state(self) -> dict[str, torch.Tensor] | None:
        if not hasattr(self.policy, "get_token_selection_state"):
            return None
        return self.policy.get_token_selection_state()

    @contextmanager
    def _inference_context(self):
        with ExitStack() as stack:
            # ACE scoring uses torch.enable_grad() inside the policy to compute
            # relevance scores. inference_mode() cannot be locally overridden,
            # so keep the outer context at no_grad().
            stack.enter_context(torch.no_grad())
            if self.device.type == "cuda" and self.use_amp:
                stack.enter_context(torch.autocast(device_type=self.device.type))
            else:
                stack.enter_context(nullcontext())
            yield

    def _build_batch(
        self,
        *,
        images: ImageInputs,
        instruction: str,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None,
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "task": instruction,
        }
        batch.update(self._prepare_images(images))
        if self.state_key is not None:
            batch[self.state_key] = self._prepare_state(state)
        return batch

    def _prepare_images(self, images: ImageInputs) -> dict[str, torch.Tensor]:
        if isinstance(images, dict):
            image_map = {str(key): value for key, value in images.items()}
        else:
            image_values = list(images)
            if len(image_values) > len(self.image_keys):
                raise ValueError(
                    f"Received {len(image_values)} images, but model expects at most {len(self.image_keys)} "
                    f"named image inputs: {self.image_keys}"
                )
            image_map = {key: value for key, value in zip(self.image_keys, image_values, strict=False)}

        if not image_map:
            raise ValueError("At least one image is required for ACE inference.")

        unknown_keys = set(image_map) - set(self.config.image_features)
        if unknown_keys:
            raise ValueError(f"Unknown image keys: {sorted(unknown_keys)}")

        return {key: self._to_image_tensor(value) for key, value in image_map.items()}

    def _prepare_state(
        self,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None,
    ) -> torch.Tensor:
        if self.state_dim <= 0:
            raise ValueError("This model does not define a state input.")

        if state is None:
            tensor = torch.zeros(self.state_dim, dtype=torch.float32)
        elif isinstance(state, torch.Tensor):
            tensor = state.detach().cpu().to(torch.float32).flatten()
        else:
            tensor = torch.as_tensor(state, dtype=torch.float32).flatten()

        if tensor.numel() < self.state_dim:
            tensor = torch.nn.functional.pad(tensor, (0, self.state_dim - tensor.numel()))
        elif tensor.numel() > self.state_dim:
            tensor = tensor[: self.state_dim]
        return tensor

    @staticmethod
    def _to_image_tensor(image: RawImageInput) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            array = np.asarray(pil_image)
        elif isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(f"Expected a 3D image tensor, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] == 3:
                image_tensor = tensor.to(torch.float32)
            elif tensor.shape[-1] == 3:
                image_tensor = tensor.permute(2, 0, 1).contiguous().to(torch.float32)
            else:
                raise ValueError(f"Unsupported image tensor shape: {tuple(tensor.shape)}")
            if image_tensor.min().item() >= -1.0 and image_tensor.max().item() <= 1.0:
                if image_tensor.min().item() < 0.0:
                    image_tensor = (image_tensor + 1.0) / 2.0
            else:
                image_tensor = image_tensor / 255.0
            return image_tensor.clamp(0.0, 1.0)
        else:
            array = np.asarray(image)

        if array.ndim != 3:
            raise ValueError(f"Expected a 3D image array, got shape {array.shape}")
        if array.shape[-1] != 3 and array.shape[0] != 3:
            raise ValueError(f"Unsupported image array shape: {array.shape}")

        image_tensor = torch.from_numpy(array)
        if image_tensor.shape[-1] == 3:
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()
        image_tensor = image_tensor.to(torch.float32)
        if image_tensor.min().item() >= -1.0 and image_tensor.max().item() <= 1.0:
            if image_tensor.min().item() < 0.0:
                image_tensor = (image_tensor + 1.0) / 2.0
        else:
            image_tensor = image_tensor / 255.0
        return image_tensor.clamp(0.0, 1.0)
