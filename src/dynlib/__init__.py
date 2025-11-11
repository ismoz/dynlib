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
    "build", "setup", "Sim", "plot",
    # Status codes (for advanced use)
    "Status", "OK", "STEPFAIL", "NAN_DETECTED", "DONE", "GROW_REC", "GROW_EVT", "USER_BREAK",
    # Results
    "Results",
    # Stepper registry
    "list_steppers", "get_stepper_info",
]


def setup(
    model, *,
    stepper=None,
    mods=None,
    jit=True,
    dtype="float64",
    disk_cache=True,
    config=None,
    validate_stepper=True,
) -> Sim:
    """Compile and setup a simulation in one call.

    This is a convenience function that combines the `build()` and `Sim()`
    calls into a single step. It compiles the provided model and initializes
    a simulation instance with the specified stepper and configuration.
    
    For most users, this is the recommended entry point. Advanced users who
    need to inspect or modify the compiled model before simulation should use
    `build()` and `Sim()` separately.
    
    The compiled model remains accessible via the returned `Sim` instance's
    `model` attribute for advanced use cases.

    Parameters:
        model: The model to be compiled (ModelSpec or URI string).
        stepper: The stepper to use for the simulation (e.g., "euler", "rk4", "rk45").
            If None, uses the model's sim.stepper default.
        mods: Optional list of mod URIs to apply during compilation.
        jit: Whether to use JIT compilation (default True).
        dtype: The data type for computations (default "float64").
        disk_cache: Whether to use disk caching for compiled models (default True).
        config: Optional PathConfig for URI resolution.
        validate_stepper: Whether to validate the stepper against the model (default True).

    Returns:
        An instance of `Sim` initialized with the compiled model.
    
    Example:
        Simple usage with inline model::
        
            from dynlib import setup
            
            sim = setup(model_uri, stepper="euler", jit=True)
            sim.run(t_end=10.0)
            results = sim.results()
        
        Equivalent manual approach for advanced use::
        
            from dynlib import build, Sim
            
            model = build(model_uri, stepper="euler", jit=True)
            # Inspect or modify model here if needed
            sim = Sim(model)
            sim.run(t_end=10.0)
    
    See Also:
        build : Compile a model without creating a Sim instance.
        Sim : Simulation facade for compiled models.
    """
    compiled_model = build(
        model,
        stepper=stepper,
        mods=mods,
        jit=jit,
        dtype=dtype,
        disk_cache=disk_cache,
        config=config,
        validate_stepper=validate_stepper,
    )
    sim = Sim(compiled_model)
    return sim
