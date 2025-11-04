# src/dynlib/runtime/model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from dynlib.dsl.spec import ModelSpec
from dynlib.steppers.base import StructSpec

__all__ = ["Model"]

@dataclass(frozen=True)
class Model:
    """
    Compiled model ready for execution (Slice 4+).
    
    Attributes:
        spec: Original ModelSpec
        stepper_name: Name of the stepper used
        struct: StructSpec from the stepper
        rhs: Compiled RHS callable
        events_pre: Compiled pre-events callable
        events_post: Compiled post-events callable
        stepper: Compiled stepper callable
        runner: Compiled runner callable
        spec_hash: Content hash of the spec
        model_dtype: NumPy dtype for the model
    """
    spec: ModelSpec
    stepper_name: str
    struct: StructSpec
    rhs: Callable
    events_pre: Callable
    events_post: Callable
    stepper: Callable
    runner: Callable
    spec_hash: str
    model_dtype: np.dtype

