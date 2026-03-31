#!/usr/bin/env python

from collections import deque

import torch

from lerobot.scripts.lerobot_eval import CUDAPolicyLatencyTracker, _policy_will_run_inference


class _DummyPolicy:
    pass


def test_policy_will_run_inference_without_action_queue():
    policy = _DummyPolicy()

    assert _policy_will_run_inference(policy) is True


def test_policy_will_run_inference_with_action_queue_state():
    policy = _DummyPolicy()
    policy._action_queue = deque()  # noqa: SLF001
    assert _policy_will_run_inference(policy) is True

    policy._action_queue.append(torch.tensor(1.0))  # noqa: SLF001
    assert _policy_will_run_inference(policy) is False


def test_cuda_policy_latency_tracker_skips_warmup_and_summarizes():
    tracker = CUDAPolicyLatencyTracker(torch.device("cuda:0"), warmup_steps=2)
    snapshot = tracker.snapshot()

    tracker.record_inference(10.0)  # warmup
    tracker.record_queue_pop()
    tracker.record_inference(20.0)  # warmup
    tracker.record_inference(30.0)  # measured
    tracker.record_inference(50.0)  # measured

    stats, samples = tracker.stats_since(snapshot)

    assert samples == [30.0, 50.0]
    assert stats["device"] == "cuda:0"
    assert stats["select_action_calls"] == 5
    assert stats["model_inference_calls"] == 4
    assert stats["action_queue_pop_calls"] == 1
    assert stats["warmup_skipped_inference_calls"] == 2
    assert stats["measured_inference_calls"] == 2
    assert stats["mean_ms"] == 40.0
    assert stats["p50_ms"] == 40.0
    assert stats["p95_ms"] == 49.0
    assert stats["min_ms"] == 30.0
    assert stats["max_ms"] == 50.0


def test_cuda_policy_latency_tracker_snapshot_is_incremental():
    tracker = CUDAPolicyLatencyTracker(torch.device("cuda:0"), warmup_steps=0)

    tracker.record_inference(12.0)
    first_snapshot = tracker.snapshot()
    tracker.record_queue_pop()
    tracker.record_inference(18.0)

    stats, samples = tracker.stats_since(first_snapshot)

    assert samples == [18.0]
    assert stats["select_action_calls"] == 2
    assert stats["model_inference_calls"] == 1
    assert stats["action_queue_pop_calls"] == 1
    assert stats["warmup_skipped_inference_calls"] == 0
    assert stats["measured_inference_calls"] == 1
    assert stats["mean_ms"] == 18.0
