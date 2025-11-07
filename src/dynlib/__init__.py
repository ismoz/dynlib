# src/dynlib/__init__.py
from __future__ import annotations

# Re-export frozen constants/types for stable imports
from dynlib.runtime.runner_api import (
    Status, OK, STEPFAIL, NAN_DETECTED, DONE, GROW_REC, GROW_EVT, USER_BREAK,
)
from .runtime.types import Kind, TimeCtrl, Scheme

from .steppers.base import (
    StepperMeta, StepperInfo, StructSpec, StepperSpec,
)
from .steppers.registry import register, get_stepper, registry

from .compiler.build import build
from .runtime.sim import Sim


__all__ = [
    # Core entry points
    "build", "Sim", "plot",
    # Status codes (for advanced use)
    "Status", "OK", "STEPFAIL", "NAN_DETECTED", "DONE", "GROW_REC", "GROW_EVT", "USER_BREAK",
    # Results
    "Results",
    # Stepper registry
    "list_steppers", "get_stepper_info",
]
