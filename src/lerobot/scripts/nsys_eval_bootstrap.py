#!/usr/bin/env python

"""Bootstrap LeRobot eval under Nsight Systems.

This module emits a tiny CUDA+NVTX smoke range before importing and executing
``lerobot.scripts.lerobot_eval`` in the same Python process. The goal is to
force a known-good CUDA/NVTX event sequence early in process lifetime for
setups where ``nsys`` fails to capture later CUDA activity from the full eval
stack.
"""

from __future__ import annotations

import runpy
import sys

import torch
import torch.cuda.nvtx as nvtx


def _emit_cuda_nvtx_smoke() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    # Keep the smoke tiny so it does not meaningfully perturb the actual eval.
    x = torch.randn(256, 256, device=device)
    nvtx.range_push("nsys.bootstrap.cuda_smoke")
    _ = x @ x
    torch.cuda.synchronize(device)
    nvtx.range_pop()


def main() -> None:
    passthrough_args = sys.argv[1:]
    _emit_cuda_nvtx_smoke()
    sys.argv = ["lerobot.scripts.lerobot_eval", *passthrough_args]
    runpy.run_module("lerobot.scripts.lerobot_eval", run_name="__main__")


if __name__ == "__main__":
    main()
