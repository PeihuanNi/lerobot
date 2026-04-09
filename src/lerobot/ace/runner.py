from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from .presets import DEFAULT_SPVLA_ENV_TASK, normalize_policy_mode, resolve_policy_mode

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is optional at import time.
    Image = None

ImageInput = str | Path | np.ndarray | torch.Tensor | Any
ImageInputs = ImageInput | Sequence[ImageInput]


class AceInferenceRunner:
    def __init__(
        self,
        policy_path: str | Path,
        *,
        mode: str | None = None,
        env_task: str = DEFAULT_SPVLA_ENV_TASK,
        policy_overrides: dict[str, Any] | None = None,
        device: str | None = None,
        use_amp: bool | None = None,
        strict: bool = False,
    ) -> None:
        self.policy_path = str(policy_path)
        self.mode = normalize_policy_mode(mode) if mode is not None else None
        self.env_task = env_task
        self._policy_overrides = resolve_policy_mode(self.mode, env_task=env_task)
        self._policy_overrides.update(policy_overrides or {})
        self.config = self._load_config(device=device, use_amp=use_amp)
        self.device = torch.device(self.config.device)
        self.use_amp = bool(self.config.use_amp)

        policy_class = get_policy_class(self.config.type)
        self.policy = policy_class.from_pretrained(
            self.policy_path,
            config=self.config,
            strict=strict,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=self.policy_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self._last_inference_info: dict[str, Any] | None = None

    def reset(self) -> None:
        self.policy.reset()
        self._last_inference_info = None

    def get_effective_policy_overrides(self) -> dict[str, Any]:
        return dict(self._policy_overrides)

    def get_last_inference_info(self) -> dict[str, Any] | None:
        if self._last_inference_info is None:
            return None
        return dict(self._last_inference_info)

    def get_token_selection_debug_state(self) -> dict[str, Any] | None:
        raw_state = self._get_raw_token_selection_state()
        if raw_state is None:
            return None

        eval_interval = self.config.region_eval_interval if self.config.region_eval_interval > 0 else 1
        next_frame_mode = "eval" if (self.config.score_debug_heatmap or raw_state.frame_idx % eval_interval == 0) else "prune"
        last_stats = dict(raw_state.last_stats or {})
        last_frame_mode = None
        if last_stats.get("eval") == "Y":
            last_frame_mode = "eval"
        elif last_stats.get("eval") == "N":
            last_frame_mode = "prune"

        return {
            "frame_idx": int(raw_state.frame_idx),
            "eval_interval": int(eval_interval),
            "next_frame_mode": next_frame_mode,
            "last_frame_mode": last_frame_mode,
            "eval_frame_count": int(raw_state.eval_frame_count),
            "token_selection_enabled": bool(getattr(self.config, "token_selection_enabled", False)),
            "token_prune_enabled": bool(getattr(self.config, "token_prune_enabled", False)),
            "model_scheduling_enabled": bool(getattr(self.config, "model_scheduling_enabled", False)),
            "last_stats": last_stats,
        }

    def predict_action(
        self,
        images: ImageInputs,
        instruction: str,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None = None,
        *,
        use_policy_queue: bool = False,
    ) -> np.ndarray:
        batch = self.preprocessor(self._build_batch(images=images, instruction=instruction, state=state))
        before = self.get_token_selection_debug_state()
        with self._inference_context():
            if use_policy_queue:
                action = self.policy.select_action(batch)
                action = self.postprocessor(action)
                action_np = action.squeeze(0).detach().cpu().numpy()
                self._record_inference_info(before=before, after=self.get_token_selection_debug_state(), via="queue_action")
                return action_np

            action_chunk = self.policy.predict_action_chunk(batch)
            action_chunk = self.postprocessor(action_chunk)
            action_np = action_chunk[0, 0].detach().cpu().numpy()
            self._record_inference_info(before=before, after=self.get_token_selection_debug_state(), via="action")
            return action_np

    def predict_action_chunk(
        self,
        images: ImageInputs,
        instruction: str,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ) -> np.ndarray:
        batch = self.preprocessor(self._build_batch(images=images, instruction=instruction, state=state))
        before = self.get_token_selection_debug_state()
        with self._inference_context():
            action_chunk = self.policy.predict_action_chunk(batch)
            action_chunk = self.postprocessor(action_chunk)
        action_chunk_np = action_chunk[0].detach().cpu().numpy()
        self._record_inference_info(before=before, after=self.get_token_selection_debug_state(), via="chunk")
        return action_chunk_np

    def get_token_selection_state(self) -> dict[str, torch.Tensor] | None:
        if not hasattr(self.policy, "get_token_selection_state"):
            return None
        return self.policy.get_token_selection_state()

    def predict_action_chunk_sequence(
        self,
        frames: Sequence[ImageInputs],
        instruction: str,
        states: Sequence[np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None] | None = None,
    ) -> list[dict[str, Any]]:
        if states is not None and len(states) != len(frames):
            raise ValueError("states must match frames in length.")

        outputs: list[dict[str, Any]] = []
        for index, frame_images in enumerate(frames):
            state = None if states is None else states[index]
            action_chunk = self.predict_action_chunk(
                images=frame_images,
                instruction=instruction,
                state=state,
            )
            outputs.append(
                {
                    "frame_index": index,
                    "action_chunk": action_chunk,
                    "inference_info": self.get_last_inference_info(),
                }
            )
        return outputs

    @contextmanager
    def _inference_context(self):
        with ExitStack() as stack:
            # ACE scoring uses torch.enable_grad() inside the policy.
            stack.enter_context(torch.no_grad())
            if self.device.type == "cuda" and self.use_amp:
                stack.enter_context(torch.autocast(device_type=self.device.type))
            else:
                stack.enter_context(nullcontext())
            yield

    def _load_config(self, *, device: str | None, use_amp: bool | None) -> PreTrainedConfig:
        cli_overrides = [
            f"--{key}={self._stringify_override_value(value)}"
            for key, value in self._policy_overrides.items()
        ]
        config = PreTrainedConfig.from_pretrained(
            self.policy_path,
            cli_overrides=cli_overrides,
        )
        config.pretrained_path = Path(self.policy_path)
        if hasattr(config, "compile_model") and "compile_model" not in self._policy_overrides:
            config.compile_model = False
        if device is not None:
            config.device = device
        if use_amp is not None:
            config.use_amp = use_amp
        config.__post_init__()
        return config

    def _build_batch(
        self,
        *,
        images: ImageInputs,
        instruction: str,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...] | None,
    ) -> dict[str, Any]:
        image_feature_items = list(self.config.image_features.items())
        image_inputs = list(self._normalize_image_inputs(images))
        batch: dict[str, Any] = {"task": instruction}

        non_empty_features = [item for item in image_feature_items if "empty_camera" not in item[0]]
        if len(image_inputs) != len(non_empty_features):
            raise ValueError(
                f"Expected {len(non_empty_features)} image(s) for {self.config.type}, got {len(image_inputs)}."
            )

        input_iter = iter(image_inputs)
        for key, feature in image_feature_items:
            if "empty_camera" in key:
                batch[key] = torch.zeros(feature.shape, dtype=torch.float32)
            else:
                batch[key] = next(input_iter)

        state_feature = self.config.robot_state_feature
        if state_feature is not None:
            if state is None:
                batch["observation.state"] = torch.zeros(state_feature.shape[0], dtype=torch.float32)
            else:
                batch["observation.state"] = self._normalize_state(state)

        return batch

    def _normalize_image_inputs(self, images: ImageInputs) -> Iterable[torch.Tensor]:
        if isinstance(images, (str, Path, np.ndarray, torch.Tensor)) or (
            Image is not None and isinstance(images, Image.Image)
        ):
            raw_images = [images]
        else:
            raw_images = list(images)

        for image in raw_images:
            yield self._normalize_image(image)

    def _normalize_image(self, image: ImageInput) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            if Image is None:
                raise ImportError("Pillow is required to load image paths.")
            image_np = np.asarray(Image.open(image).convert("RGB"))
        elif Image is not None and isinstance(image, Image.Image):
            image_np = np.asarray(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.asarray(image)

        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=-1)

        if image_np.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, got shape {image_np.shape}.")

        if image_np.shape[-1] in (1, 3):
            image_chw = np.transpose(image_np, (2, 0, 1))
        elif image_np.shape[0] in (1, 3):
            image_chw = image_np
        else:
            raise ValueError(f"Cannot infer channel dimension from image shape {image_np.shape}.")

        image_chw = np.ascontiguousarray(image_chw).astype(np.float32)
        if image_chw.min() >= -1.0 and image_chw.max() <= 1.0:
            if image_chw.min() < 0.0:
                image_chw = (image_chw + 1.0) / 2.0
        else:
            image_chw = image_chw / 255.0

        image_chw = np.clip(image_chw, 0.0, 1.0)
        return torch.from_numpy(image_chw)

    def _normalize_state(
        self,
        state: np.ndarray | torch.Tensor | list[float] | tuple[float, ...],
    ) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            state_tensor = state.detach().cpu().to(dtype=torch.float32).flatten()
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).flatten()
        return state_tensor

    def _get_raw_token_selection_state(self):
        model = getattr(self.policy, "model", None)
        return getattr(model, "_token_selection_state", None)

    def _record_inference_info(
        self,
        *,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        via: str,
    ) -> None:
        if before is None and after is None:
            self._last_inference_info = {
                "via": via,
                "fresh_policy_inference": None,
            }
            return

        fresh_policy_inference = None
        processed_frame_idx = None
        processed_frame_mode = None
        if before is not None and after is not None:
            fresh_policy_inference = before["frame_idx"] != after["frame_idx"]
            if fresh_policy_inference:
                processed_frame_idx = before["frame_idx"]
                processed_frame_mode = before["next_frame_mode"]

        self._last_inference_info = {
            "via": via,
            "fresh_policy_inference": fresh_policy_inference,
            "processed_frame_idx": processed_frame_idx,
            "processed_frame_mode": processed_frame_mode,
            "before": before,
            "after": after,
        }

    @staticmethod
    def _stringify_override_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value)
