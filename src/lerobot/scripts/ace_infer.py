#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lerobot.ace import AceInferenceRunner


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_state(raw: str | None) -> Any:
    if raw is None:
        return None
    if "," in raw and not raw.strip().startswith("["):
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    return _parse_value(raw)


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected KEY=VALUE.")
        key, raw_value = pair.split("=", 1)
        overrides[key] = _parse_value(raw_value)
    return overrides


def _parse_named_images(pairs: list[str]) -> dict[str, str]:
    images: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid image mapping '{pair}'. Expected KEY=PATH.")
        key, value = pair.split("=", 1)
        images[key] = value
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone ACE inference for PI0 / PI05.")
    parser.add_argument("--policy-path", required=True, help="Path to pretrained PI0 / PI05 model directory.")
    parser.add_argument("--instruction", required=True, help="Language instruction.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path. Repeat to provide multiple cameras in config order.",
    )
    parser.add_argument(
        "--image-key",
        action="append",
        default=[],
        help="Explicit image mapping in KEY=PATH format. Repeat as needed.",
    )
    parser.add_argument("--state", default=None, help="Optional state vector as JSON list or comma-separated values.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu / cuda / cuda:0.")
    parser.add_argument(
        "--use-amp",
        default=None,
        choices=["true", "false"],
        help="Override automatic mixed precision.",
    )
    parser.add_argument(
        "--policy-override",
        action="append",
        default=[],
        help="Policy config override in KEY=VALUE format. Repeat as needed.",
    )
    parser.add_argument(
        "--use-policy-queue",
        action="store_true",
        help="Use policy.select_action() queue semantics instead of a fresh chunk forward pass.",
    )
    parser.add_argument(
        "--return-chunk",
        action="store_true",
        help="Return the full action chunk in addition to the first action.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.image and args.image_key:
        raise ValueError("Use either --image or --image-key, not both.")
    if args.return_chunk and args.use_policy_queue:
        raise ValueError("--return-chunk cannot be used together with --use-policy-queue.")

    policy_overrides = _parse_overrides(args.policy_override)
    use_amp = None if args.use_amp is None else args.use_amp == "true"
    runner = AceInferenceRunner(
        policy_path=args.policy_path,
        device=args.device,
        use_amp=use_amp,
        policy_overrides=policy_overrides,
    )

    if args.image_key:
        images: Any = _parse_named_images(args.image_key)
    else:
        images = [Path(path) for path in args.image]

    state = _parse_state(args.state)
    action_chunk = None
    if args.return_chunk:
        action_chunk = runner.predict_action_chunk(
            images=images,
            instruction=args.instruction,
            state=state,
        )
        action = action_chunk[0]
    else:
        action = runner.predict_action(
            images=images,
            instruction=args.instruction,
            state=state,
            use_policy_queue=args.use_policy_queue,
        )

    result: dict[str, Any] = {
        "policy_type": runner.config.type,
        "device": str(runner.device),
        "action": action.tolist(),
    }
    if args.return_chunk:
        result["action_chunk"] = action_chunk.tolist()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
