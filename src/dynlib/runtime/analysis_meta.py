# src/dynlib/runtime/analysis_meta.py
"""Helper utilities for attaching observer metadata to results."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from dynlib.runtime.observers import ObserverModule

__all__ = ["build_observer_metadata"]


def _trace_metadata(mod: ObserverModule, trace_capacity: int | None) -> Mapping[str, object] | None:
    if mod.trace is None:
        return None
    plan = mod.trace.plan
    return {
        "width": int(mod.trace.width),
        "stride": int(mod.trace.record_interval()),
        "capacity": trace_capacity,
        "plan": plan.__class__.__name__ if plan is not None else None,
        "plan_id": id(plan) if plan is not None else None,
    }


def build_observer_metadata(
    modules: Sequence[ObserverModule] | None,
    *,
    analysis_kind: int,
    trace_stride: int | None,
    trace_capacity: int | None,
    overflow: bool = False,
) -> Mapping[str, object] | None:
    """Collect lightweight metadata for runtime observers."""
    if not modules:
        return None
    meta_modules: list[Mapping[str, object]] = []
    mutates_state = False
    for mod in modules:
        req_summary = asdict(mod.requirements) if hasattr(mod, "requirements") else {}
        mutates_state = mutates_state or bool(req_summary.get("mutates_state", False))
        meta_modules.append(
            {
                "key": mod.key,
                "name": mod.name,
                "analysis_kind": int(mod.analysis_kind),
                "workspace_size": int(mod.workspace_size),
                "output_size": int(mod.output_size),
                "output_names": mod.output_names,
                "trace_names": mod.trace_names,
                "trace": _trace_metadata(mod, trace_capacity),
                "requirements": req_summary,
            }
        )
    return {
        "analysis_kind": int(analysis_kind),
        "mutates_state": mutates_state,
        "trace_stride": trace_stride,
        "trace_capacity": trace_capacity,
        "trace_overflow": bool(overflow),
        "modules": meta_modules,
    }
