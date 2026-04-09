from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lerobot.ace import AceInferenceRunner


def make_random_image(height: int = 256, width: int = 256) -> np.ndarray:
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def round_array(values: np.ndarray, digits: int = 5) -> list:
    return np.round(values.astype(np.float32), digits).tolist()


POLICY_PATH = "/home/nipeihuan/models/pi05_libero_finetuned"
ENV_TASK = "libero_object"
PRESET_NAME = "ace"
INSTRUCTION = "open the drawer"

# Real-robot-like setting:
# one policy request -> one fresh image observation -> one action chunk
# execute only the first n_action_steps actions, then request a new chunk again.
NUM_POLICY_REQUESTS = 5

POLICY_OVERRIDES = {
    "rollout_dir": "",
    "local_log_dir": "",
    "region_eval_interval": 2,
    "n_action_steps": 10,
}


runner = AceInferenceRunner(
    POLICY_PATH,
    mode=PRESET_NAME,
    env_task=ENV_TASK,
    policy_overrides=POLICY_OVERRIDES,
)

n_action_steps = int(runner.config.n_action_steps)
continuous_action_stream: list[list[float]] = []
policy_request_results: list[dict] = []

# Important: do not reset between policy requests.
# Reset only once when a new real-world episode starts.
runner.reset()

for request_idx in range(NUM_POLICY_REQUESTS):
    # Simulate grabbing fresh camera images right before a new policy request.
    images = [make_random_image(), make_random_image()]
    action_chunk = runner.predict_action_chunk(
        images=images,
        instruction=INSTRUCTION,
    )
    executed_actions = action_chunk[:n_action_steps]
    continuous_action_stream.extend(round_array(executed_actions))

    info = runner.get_last_inference_info() or {}
    before = info.get("before") or {}
    after = info.get("after") or {}
    policy_request_results.append(
        {
            "policy_request_index": request_idx,
            "processed_frame_idx": info.get("processed_frame_idx"),
            "policy_request_mode": info.get("processed_frame_mode"),
            "next_policy_request_mode": after.get("next_frame_mode"),
            "chunk_shape": list(action_chunk.shape),
            "executed_action_count": int(executed_actions.shape[0]),
            # "executed_action_preview": round_array(executed_actions[:3]),
            "camera_shapes": [list(image.shape) for image in images],
            "frame_state_before_request": before,
            "frame_state_after_request": after,
        }
    )

print("preset:", PRESET_NAME)
print("instruction:", INSTRUCTION)
print("env_task:", ENV_TASK)
print("n_action_steps:", n_action_steps)
print("region_eval_interval:", runner.get_effective_policy_overrides().get("region_eval_interval"))
print("max_kept_tokens_per_camera:", runner.get_effective_policy_overrides().get("max_kept_tokens"))
print("min_kept_tokens_per_camera:", runner.get_effective_policy_overrides().get("min_kept_tokens"))
print("dynamic_prune_mode:", runner.get_effective_policy_overrides().get("dynamic_prune_mode"))
print("num_policy_requests:", NUM_POLICY_REQUESTS)
print("policy request results:")
print(json.dumps(policy_request_results, indent=2, sort_keys=True))
print("continuous action stream length:", len(continuous_action_stream))
print("continuous action stream:")
# print(json.dumps(continuous_action_stream, indent=2))
