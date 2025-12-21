# src/dynlib/runtime/fastpath/plans.py
from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = [
    "RecordingPlan",
    "FixedStridePlan",
    "TailWindowPlan",
    "TracePlan",
    "FixedTracePlan",
    "TailTracePlan",
    "HitTracePlan",
]


@dataclass(frozen=True)
class RecordingPlan:
    """Abstract recording plan with fixed, predeclared capacity."""

    stride: int

    def record_interval(self) -> int:
        return int(self.stride)

    def capacity(self, *, total_steps: int) -> int:
        raise NotImplementedError

    def finalize_index(self, filled: int) -> slice | None:
        """Optional slice to apply after execution (for tail windows)."""
        return None


@dataclass(frozen=True)
class TracePlan:
    """
    Fixed-capacity trace plan used by analysis modules.

    Mirrors RecordingPlan but intentionally kept separate to decouple analysis
    traces from trajectory recording. All subclasses must produce a fixed
    capacity so fast-path execution can remain allocation free.
    """

    stride: int

    def record_interval(self) -> int:
        return int(self.stride)

    def capacity(self, *, total_steps: int | None) -> int:
        raise NotImplementedError

    def finalize_index(self, filled: int) -> slice | None:
        return None

    def hit_limit(self) -> int | None:
        """Optional hard cap on trace rows (stride-independent)."""
        return None


@dataclass(frozen=True)
class FixedStridePlan(RecordingPlan):
    """
    Record every ``stride`` steps. Capacity is derived from the step budget.
    """

    def capacity(self, *, total_steps: int) -> int:
        # Record initial point + every stride-aligned step.
        if total_steps <= 0:
            return 1
        return 1 + math.floor(total_steps / max(1, self.stride))


@dataclass(frozen=True)
class TailWindowPlan(RecordingPlan):
    """
    Keep only the last ``window`` samples (after applying stride thinning).

    Note: the execution still runs with a fixed buffer; ``finalize_index``
    trims to the last window to expose a bounded view without reallocation.
    """

    window: int

    def capacity(self, *, total_steps: int) -> int:
        # Allocate enough slots for the full run to avoid growth requests.
        # Final exposure is trimmed to the last ``window`` samples.
        if total_steps <= 0:
            return min(1, self.window)
        return max(self.window, 1 + math.floor(total_steps / max(1, self.stride)))

    def finalize_index(self, filled: int) -> slice | None:
        if filled <= self.window:
            return None
        start = filled - self.window
        return slice(start, filled)


# ------------------------------ Trace plans ----------------------------------

@dataclass(frozen=True)
class FixedTracePlan(TracePlan):
    """Record every ``stride`` steps into a fixed-capacity trace buffer."""

    def capacity(self, *, total_steps: int | None) -> int:
        if total_steps is None:
            return 0
        if total_steps <= 0:
            return 1
        return 1 + math.floor(total_steps / max(1, self.stride))


@dataclass(frozen=True)
class TailTracePlan(TracePlan):
    """Keep the last ``window`` trace samples after stride thinning."""

    window: int

    def capacity(self, *, total_steps: int | None) -> int:
        if total_steps is None:
            return max(1, self.window)
        if total_steps <= 0:
            return min(1, self.window)
        return max(self.window, 1 + math.floor(total_steps / max(1, self.stride)))

    def finalize_index(self, filled: int) -> slice | None:
        if filled <= self.window:
            return None
        start = filled - self.window
        return slice(start, filled)


@dataclass(frozen=True)
class HitTracePlan(TracePlan):
    """
    Fixed-capacity trace that is indexed directly by analysis hooks.

    Used for analyses that emit traces on irregular "hits" rather than at a
    fixed stride. The capacity is a hard cap; callers must gate or stop before
    exceeding it on fast-path backends.
    """

    max_hits: int

    def capacity(self, *, total_steps: int | None) -> int:
        return int(self.max_hits)

    def hit_limit(self) -> int | None:
        return int(self.max_hits)
