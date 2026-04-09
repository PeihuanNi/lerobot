from __future__ import annotations

import argparse
import json

import numpy as np

from lerobot.ace.presets import describe_policy_modes, get_all_policy_presets, list_policy_modes


def _parse_override(override: str) -> tuple[str, object]:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected KEY=VALUE.")
    key, raw_value = override.split("=", 1)
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    if lowered == "null":
        return key, None
    try:
        return key, json.loads(raw_value)
    except json.JSONDecodeError:
        return key, raw_value


def _parse_state(raw_state: str | None) -> list[float] | None:
    if raw_state is None or raw_state.strip() == "":
        return None
    return [float(item) for item in raw_state.split(",")]


def main() -> None:
    supported_modes = ", ".join(list_policy_modes())
    mode_descriptions = describe_policy_modes()
    parser = argparse.ArgumentParser(description="Standalone ACE/SP-VLA inference entrypoint.")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "Preset policy mode. "
            "Use 'ace' for ACE scoring/pruning or 'spvla' for the default SP-VLA paper_acc preset. "
            f"Supported canonical modes: {supported_modes}."
        ),
    )
    parser.add_argument(
        "--env-task",
        default="libero_object",
        help="Env-task preset used to fill SP-VLA schedule defaults such as step_skip and z_* thresholds.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print all available presets and their resolved policy parameters, then exit.",
    )
    parser.add_argument(
        "--print-resolved-overrides",
        action="store_true",
        help="Print the final resolved policy overrides after applying preset + manual overrides.",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--image", dest="images", action="append", default=None)
    parser.add_argument("--state", default=None, help="Comma-separated state vector.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--use-policy-queue", action="store_true")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Run the same observation through the runner multiple times without resetting state.",
    )
    parser.add_argument("--print-chunk", action="store_true")
    parser.add_argument("--policy-override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    if args.list_presets:
        payload = {
            "env_task": args.env_task,
            "preset_descriptions": mode_descriptions,
            "resolved_presets": get_all_policy_presets(env_task=args.env_task),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.policy_path is None:
        parser.error("--policy-path is required unless --list-presets is used.")
    if args.instruction is None:
        parser.error("--instruction is required unless --list-presets is used.")
    if not args.images:
        parser.error("At least one --image is required unless --list-presets is used.")
    if args.num_frames <= 0:
        parser.error("--num-frames must be >= 1.")

    policy_overrides = dict(_parse_override(item) for item in args.policy_override)
    from lerobot.ace import AceInferenceRunner

    runner = AceInferenceRunner(
        args.policy_path,
        mode=args.mode,
        env_task=args.env_task,
        policy_overrides=policy_overrides,
        device=args.device,
        use_amp=args.use_amp if args.use_amp else None,
    )
    if args.print_resolved_overrides:
        print(json.dumps(runner.get_effective_policy_overrides(), indent=2, sort_keys=True))

    state = _parse_state(args.state)
    for frame_idx in range(args.num_frames):
        print(f"frame {frame_idx}:")
        if args.print_chunk:
            chunk = runner.predict_action_chunk(
                images=args.images,
                instruction=args.instruction,
                state=state,
            )
            print("chunk shape:", chunk.shape)
            print(chunk)
        else:
            action = runner.predict_action(
                images=args.images,
                instruction=args.instruction,
                state=state,
                use_policy_queue=args.use_policy_queue,
            )
            print("action shape:", np.asarray(action).shape)
            print(action)
        print("inference info:")
        print(json.dumps(runner.get_last_inference_info(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
