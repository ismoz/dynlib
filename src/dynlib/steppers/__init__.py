# src/dynlib/steppers/__init__.py
from .base import StepperMeta, StepperInfo, StructSpec, StepperSpec
from .registry import register, get_stepper, registry

# Import concrete steppers to trigger auto-registration
from .ode import euler
from .ode import rk4
from .ode import rk45
from .ode import ab2
from .discrete import map

__all__ = [
    "StepperMeta", "StepperInfo", "StructSpec", "StepperSpec",
    "register", "get_stepper", "registry",
]
