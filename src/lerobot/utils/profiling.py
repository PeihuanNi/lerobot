from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.cuda.nvtx as nvtx

_EMIT_NVTX = False
_EMIT_RECORD_FUNCTION = False


def configure_runtime_profiling(*, emit_nvtx: bool = False, emit_record_function: bool = False) -> None:
    global _EMIT_NVTX, _EMIT_RECORD_FUNCTION
    _EMIT_NVTX = bool(emit_nvtx)
    _EMIT_RECORD_FUNCTION = bool(emit_record_function)


@contextmanager
def nvtx_range(name: str):
    enabled = _EMIT_NVTX and torch.cuda.is_available()
    if enabled:
        nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            nvtx.range_pop()


@contextmanager
def profile_range(name: str):
    # Keep the PyTorch trace labels when requested, but make the NVTX path
    # explicitly mirror the traditional push/pop style used in nsys_test.py.
    record_cm = torch.profiler.record_function(name) if _EMIT_RECORD_FUNCTION else nullcontext()
    with record_cm:
        with nvtx_range(name):
            yield


def create_torch_profiler(eval_cfg: Any, output_dir: Path):
    backend = getattr(eval_cfg, "profile_backend", "none")
    if backend != "pytorch":
        return nullcontext(None)

    trace_dir_cfg = getattr(eval_cfg, "profile_dir", None)
    if trace_dir_cfg:
        trace_dir = Path(trace_dir_cfg)
        if not trace_dir.is_absolute():
            trace_dir = output_dir / trace_dir
    else:
        trace_dir = output_dir / "torch_profile"
    trace_dir.mkdir(parents=True, exist_ok=True)

    wait = int(getattr(eval_cfg, "profile_wait_steps", 1))
    warmup = int(getattr(eval_cfg, "profile_warmup_steps", 1))
    active = int(getattr(eval_cfg, "profile_active_steps", 3))
    repeat = int(getattr(eval_cfg, "profile_repeat", 1))
    if active <= 0:
        raise ValueError("eval.profile_active_steps must be > 0 when eval.profile_backend='pytorch'")
    if repeat <= 0:
        raise ValueError("eval.profile_repeat must be > 0 when eval.profile_backend='pytorch'")
    record_shapes = bool(getattr(eval_cfg, "profile_record_shapes", True))
    with_stack = bool(getattr(eval_cfg, "profile_with_stack", False))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    logging.info("PyTorch profiler enabled. Writing traces to %s", trace_dir)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir), worker_name="lerobot_eval"),
        record_shapes=record_shapes,
        with_stack=with_stack,
    )
