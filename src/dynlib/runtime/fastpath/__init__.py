"""
Lightweight execution backends for analysis workloads.

The fastpath runner bypasses the full Sim wrapper when the requested run fits
strict constraints (fixed recording plan, apply-only events, no dynamic logs).
"""
from .plans import FixedStridePlan, TailWindowPlan, RecordingPlan
from .runner import (
    fastpath_for_sim,
    fastpath_batch_for_sim,
    run_batch_fastpath,
    run_single_fastpath,
)
from .capability import FastpathSupport, assess_capability

__all__ = [
    "FixedStridePlan",
    "TailWindowPlan",
    "RecordingPlan",
    "run_single_fastpath",
    "run_batch_fastpath",
    "fastpath_for_sim",
    "fastpath_batch_for_sim",
    "FastpathSupport",
    "assess_capability",
]
