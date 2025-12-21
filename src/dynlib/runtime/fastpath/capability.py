# src/dynlib/runtime/fastpath/capability.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from typing import TYPE_CHECKING

from dynlib.runtime.fastpath.plans import RecordingPlan

if TYPE_CHECKING:
    from dynlib.analysis.runtime import AnalysisModule
    from dynlib.runtime.sim import Sim

__all__ = ["FastpathSupport", "assess_capability"]


@dataclass(frozen=True)
class FastpathSupport:
    allowed: bool
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.allowed)


def _has_event_logs(spec) -> bool:
    return any(getattr(ev, "log", None) for ev in spec.events)


def assess_capability(
    sim: "Sim",
    *,
    plan: RecordingPlan,
    record_vars: Sequence[str] | None,
    dt: Optional[float],
    transient: float,
    adaptive: bool,
    analysis: "AnalysisModule" | None = None,
) -> FastpathSupport:
    """
    Static gate to decide if the fastpath runner can be used.

    Constraints:
      - No event logging (apply-only actions are fine)
      - Fixed-step time control (adaptive steppers fall back)
      - Record interval must be positive and fixed
      - No resume / stitching / snapshots
      - Dtype and stepper config are known
      - Optional analysis modules must provide JIT hooks and fixed trace plan
    """
    spec = sim.model.spec
    if _has_event_logs(spec):
        return FastpathSupport(False, "event logging requested")

    if adaptive:
        return FastpathSupport(False, "adaptive steppers are not supported on fast path")

    if dt is None:
        return FastpathSupport(False, "dt must be explicit for fast path")
    if dt <= 0.0:
        return FastpathSupport(False, "dt must be positive")

    if plan.record_interval() <= 0:
        return FastpathSupport(False, "record interval must be positive")

    if transient < 0.0:
        return FastpathSupport(False, "transient must be non-negative")

    # Require at least one recorded variable (states or aux)
    n_state = len(spec.states)
    n_aux = len(spec.aux)
    if record_vars is None and n_state == 0:
        return FastpathSupport(False, "no states to record")

    # Disallow lagged systems for now to avoid ring-buffer management drift.
    if getattr(spec, "uses_lag", False):
        return FastpathSupport(False, "lagged systems are not fast-path ready yet")

    if analysis is not None:
        has_jacobian = getattr(sim.model, "jacobian", None) is not None
        ok, reason = analysis.supports_fastpath(
            adaptive=adaptive,
            has_event_logs=_has_event_logs(spec),
            has_jacobian=has_jacobian,
        )
        if not ok:
            return FastpathSupport(False, reason)
        if analysis.needs_trace and analysis.trace is None:
            return FastpathSupport(False, "analysis trace requires a TracePlan")

    return FastpathSupport(True, None)
