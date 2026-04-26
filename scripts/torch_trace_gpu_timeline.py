#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}


def _load_events(trace_path: Path) -> list[dict]:
    with trace_path.open() as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def _collect_steps(events: list[dict]) -> list[dict]:
    steps = []
    for event in events:
        if event.get("cat") != "user_annotation" or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        if not name.startswith("ProfilerStep#"):
            continue
        ts = float(event["ts"])
        dur = float(event.get("dur", 0.0))
        steps.append(
            {
                "name": name,
                "ts": ts,
                "end": ts + dur,
                "dur": dur,
            }
        )
    steps.sort(key=lambda x: x["ts"])
    return steps


def _assign_step(event_ts: float, steps: list[dict], step_starts: list[float]) -> str:
    idx = bisect_right(step_starts, event_ts) - 1
    if idx < 0:
        return "no_step"
    step = steps[idx]
    if event_ts <= step["end"]:
        return step["name"]
    return "no_step"


def _format_duration_us(dur_us: float) -> str:
    if dur_us >= 1000.0:
        return f"{dur_us / 1000.0:.3f} ms"
    return f"{dur_us:.3f} us"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print GPU events from a PyTorch profiler trace in time order.")
    parser.add_argument("trace", type=Path, help="Path to *.pt.trace.json")
    parser.add_argument("--step", type=str, default=None, help="Only show one profiler step, e.g. ProfilerStep#4")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of GPU events to print")
    parser.add_argument(
        "--show-categories",
        type=str,
        default="kernel,gpu_memcpy,gpu_memset,gpu_user_annotation",
        help="Comma-separated GPU categories to include",
    )
    args = parser.parse_args()

    events = _load_events(args.trace)
    steps = _collect_steps(events)
    step_starts = [step["ts"] for step in steps]
    allowed_categories = {item.strip() for item in args.show_categories.split(",") if item.strip()}

    gpu_events = []
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = event.get("cat")
        if cat not in GPU_CATEGORIES or cat not in allowed_categories:
            continue
        ts = float(event["ts"])
        dur = float(event.get("dur", 0.0))
        step_name = _assign_step(ts, steps, step_starts)
        if args.step is not None and step_name != args.step:
            continue
        gpu_events.append(
            {
                "step": step_name,
                "ts": ts,
                "dur": dur,
                "cat": cat,
                "stream": event.get("args", {}).get("stream", "-"),
                "name": str(event.get("name", "")),
            }
        )

    gpu_events.sort(key=lambda x: x["ts"])
    if not gpu_events:
        print("No matching GPU events found.")
        return

    first_ts = gpu_events[0]["ts"]
    current_step = None
    for idx, event in enumerate(gpu_events[: args.limit], start=1):
        if event["step"] != current_step:
            current_step = event["step"]
            print(f"\n[{current_step}]")
        rel_ms = (event["ts"] - first_ts) / 1000.0
        print(
            f"{idx:04d}  +{rel_ms:10.3f} ms  {_format_duration_us(event['dur']):>10}  "
            f"{event['cat']:<19}  stream={event['stream']:<3}  {event['name']}"
        )

    print(f"\nPrinted {min(len(gpu_events), args.limit)} / {len(gpu_events)} GPU events.")


if __name__ == "__main__":
    main()
