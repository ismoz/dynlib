# src/dynlib/runtime/fastpath/capability.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from typing import TYPE_CHECKING

from dynlib.runtime.fastpath.plans import RecordingPlan

if TYPE_CHECKING:
    from dynlib.runtime.observers import ObserverModule
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


def _is_jitted_runner(fn) -> bool:
    """Detect numba-compiled runners by the presence of signatures."""
    return bool(getattr(fn, "signatures", None))


def assess_capability(
    sim: "Sim",
    *,
    plan: RecordingPlan,
    record_vars: Sequence[str] | None,
    dt: Optional[float],
    transient: float,
    adaptive: bool,
    observers: "ObserverModule" | None = None,
) -> FastpathSupport:
    """
    Static gate to decide if the fastpath runner can be used.

    Constraints:
      - No event logging (apply-only actions are fine)
      - Observers may not mutate state on fast path
      - Fixed-step time control (adaptive steppers fall back)
      - Record interval must be positive and fixed
      - No resume / stitching / snapshots
      - Dtype and stepper config are known
      - Optional observers must provide fixed trace plan
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

    if observers is not None:
        has_jvp = getattr(sim.model, "jvp", None) is not None
        has_jacobian = getattr(sim.model, "jacobian", None) is not None
        if observers.requirements.requires_event_log:
            return FastpathSupport(False, "observer requires event logging")
        if observers.requirements.mutates_state:
            return FastpathSupport(False, "observer mutates state")
        ok, reason = observers.supports_fastpath(
            adaptive=adaptive,
            has_event_logs=_has_event_logs(spec),
            has_jvp=has_jvp,
            has_dense_jacobian=has_jacobian,
        )
        if not ok:
            return FastpathSupport(False, reason)
        if observers.needs_trace and (observers.trace is None or observers.trace.plan is None):
            return FastpathSupport(False, "observer trace requires a TracePlan")
        if observers.trace is not None and observers.trace.record_interval() <= 0:
            return FastpathSupport(False, "observer trace stride must be positive")
        if _is_jitted_runner(sim.model.runner):
            try:
                observers.resolve_hooks(jit=True, dtype=sim.model.dtype)
            except Exception as exc:
                return FastpathSupport(False, f"observer hooks failed to jit-compile: {exc}")

    return FastpathSupport(True, None)
