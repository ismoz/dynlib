# src/dynlib/steppers/__init__.py
from .base import StepperCaps, StepperMeta, StepperInfo, StepperSpec
from .registry import register, get_stepper, registry

# Import concrete steppers to trigger auto-registration
from .discrete import map
from .ode import euler, rk4, rk45
from .ode import ab2, ab3
from .ode import bdf2_jit, bdf2

__all__ = [
    "StepperCaps", "StepperMeta", "StepperInfo", "StepperSpec",
    "register", "get_stepper", "registry",
]
