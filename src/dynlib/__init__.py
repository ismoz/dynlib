# src/dynlib/__init__.py
from __future__ import annotations

# Re-export frozen constants/types for stable imports
from .runtime.runner_api import (
    Status, OK, REJECT, STEPFAIL, NAN_DETECTED, DONE, GROW_REC, GROW_EVT, USER_BREAK,
)
from .runtime.types import Kind, TimeCtrl, Scheme

from .steppers.base import (
    StepperMeta, StepperInfo, StructSpec, StepperSpec,
)
from .steppers.registry import register, get_stepper, registry

from .utils.arrays import (
    require_c_contig, require_dtype, require_len1, carve_view,
)

__all__ = [
    # Status / constants
    "Status", "OK", "REJECT", "STEPFAIL", "NAN_DETECTED", "DONE", "GROW_REC", "GROW_EVT", "USER_BREAK",
    # type literals
    "Kind", "TimeCtrl", "Scheme",
    # steppers
    "StepperMeta", "StepperInfo", "StructSpec", "StepperSpec", "register", "get_stepper", "registry",
    # utils
    "require_c_contig", "require_dtype", "require_len1", "carve_view",
]
