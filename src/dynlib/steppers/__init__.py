# src/dynlib/steppers/__init__.py
from .base import StepperMeta, StepperInfo, StructSpec, StepperSpec
from .registry import register, get_stepper, registry

# Import concrete steppers to trigger auto-registration
from . import euler

__all__ = [
    "StepperMeta", "StepperInfo", "StructSpec", "StepperSpec",
    "register", "get_stepper", "registry",
]
