# src/dynlib/runtime/fastpath/plans.py
from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = [
    "RecordingPlan",
    "FixedStridePlan",
    "TailWindowPlan",
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
