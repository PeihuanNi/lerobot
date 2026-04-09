from __future__ import annotations

from .presets import describe_policy_modes, get_all_policy_presets, get_policy_preset, list_policy_modes, resolve_policy_mode

__all__ = [
    "AceInferenceRunner",
    "describe_policy_modes",
    "get_all_policy_presets",
    "get_policy_preset",
    "list_policy_modes",
    "resolve_policy_mode",
]


def __getattr__(name: str):
    if name == "AceInferenceRunner":
        from .runner import AceInferenceRunner

        return AceInferenceRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
