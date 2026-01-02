# src/dynlib/runtime/fastpath/__init__.py
"""
Lightweight execution backends for observer/diagnostic workloads.

The fastpath executor bypasses the full Sim wrapper when the requested run fits
strict constraints (fixed recording plan, apply-only events, no dynamic logs).
"""
from importlib import import_module

from .plans import (
    FixedStridePlan,
    TailWindowPlan,
    RecordingPlan,
    TracePlan,
    FixedTracePlan,
    TailTracePlan,
    HitTracePlan,
)

__all__ = [
    "FixedStridePlan",
    "TailWindowPlan",
    "RecordingPlan",
    "TracePlan",
    "FixedTracePlan",
    "TailTracePlan",
    "HitTracePlan",
    "run_single_fastpath",
    "run_batch_fastpath",
    "fastpath_for_sim",
    "fastpath_batch_for_sim",
    "FastpathSupport",
    "assess_capability",
]


def __getattr__(name):
    if name in {
        "run_single_fastpath",
        "run_batch_fastpath",
        "fastpath_for_sim",
        "fastpath_batch_for_sim",
    }:
        module = import_module("dynlib.runtime.fastpath.executor")
        return getattr(module, name)
    if name in {"FastpathSupport", "assess_capability"}:
        module = import_module("dynlib.runtime.fastpath.capability")
        return getattr(module, name)
    raise AttributeError(name)
