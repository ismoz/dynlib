# src/dynlib/steppers/__init__.py
from .base import StepperCaps, StepperMeta, StepperInfo, StepperSpec
from .registry import register, get_stepper, registry

# Import concrete steppers to trigger auto-registration
from .discrete import map
from .ode import euler, rk4, rk45, rk2_midpoint
from .ode import ab2, ab3
from .ode import bdf2, bdf2a, tr_bdf2a
from .ode import sdirk2

__all__ = [
    "StepperCaps", "StepperMeta", "StepperInfo", "StepperSpec",
    "register", "get_stepper", "registry",
]
