# src/dynlib/utils/arrays.py
from __future__ import annotations
from typing import Any
import numpy as np

#TODO: Add below functionality after Slice 6+
"""
When you want to harden the API (beyond Slice 4), add validation in Sim.run():

from dynlib.utils.arrays import require_c_contig, require_dtype

def run(self, ..., y0=None, params=None, ...):
    # ... (your current defaults code) ...
    
    # Add validation when user provides arrays
    if y0 is not None:
        y0 = require_c_contig(y0, "y0")
        y0 = require_dtype(y0, self.model.model_dtype, "y0")
        if y0.shape != (n_state,):
            raise ValueError(f"y0 shape must be ({n_state},), got {y0.shape}")
    
    # Similar for params...
"""


__all__ = [
    "require_c_contig", "require_dtype", "require_len1", "carve_view",
]

def require_c_contig(a: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Ensure 'a' is C-contiguous. Raise ValueError if not.
    (Guard only; not for hot loops.)
    """
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return a

def require_dtype(a: np.ndarray, dtype: np.dtype, name: str = "array") -> np.ndarray:
    """
    Ensure 'a' dtype matches 'dtype' exactly. Raise TypeError if not.
    """
    if a.dtype != np.dtype(dtype):
        raise TypeError(f"{name} dtype must be {np.dtype(dtype).name}; got {a.dtype.name}")
    return a

def require_len1(a: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Ensure 'a' is 1D with length 1. Raise ValueError if not.
    """
    if a.ndim != 1 or a.size != 1:
        raise ValueError(f"{name} must be a 1D array of length 1; got shape {a.shape}")
    return a

def carve_view(a: np.ndarray, start: int, length: int) -> np.ndarray:
    """
    Return a C-contiguous view 'a[start:start+length]'.
    Guard that the slice is in bounds; raises ValueError if not.
    """
    if start < 0 or length < 0 or start + length > a.size:
        raise ValueError(f"slice out of bounds: start={start}, length={length}, size={a.size}")
    # The slice is a view; caller may enforce contiguity if needed.
    return a[start : start + length]
