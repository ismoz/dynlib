"""Runtime observer APIs and factory protocol."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING, runtime_checkable

from .core import (
    CombinedObserver,
    ObserverHooks,
    ObserverModule,
    ObserverRequirements,
    TraceSpec,
    observer_noop_hook,
    observer_noop_variational_step,
)
from .lyapunov import lyapunov_mle_observer, lyapunov_spectrum_observer

if TYPE_CHECKING:  # pragma: no cover
    from dynlib.runtime.sim import Sim


@runtime_checkable
class ObserverFactory(Protocol):
    """Callable factory that constructs an observer module for Sim.run."""

    __observer_factory__: bool

    def __call__(
        self, model: object, sim: "Sim", record_interval: int | None
    ) -> ObserverModule: ...


def mark_observer_factory(factory):
    """Mark a callable as an ObserverFactory for runtime validation."""
    setattr(factory, "__observer_factory__", True)
    return factory


__all__ = [
    "ObserverFactory",
    "ObserverHooks",
    "ObserverModule",
    "ObserverRequirements",
    "CombinedObserver",
    "TraceSpec",
    "observer_noop_hook",
    "observer_noop_variational_step",
    "lyapunov_mle_observer",
    "lyapunov_spectrum_observer",
    "mark_observer_factory",
]
