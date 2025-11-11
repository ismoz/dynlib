from __future__ import annotations

import math
from typing import Callable
import numpy as np

from dynlib.compiler.jit.compile import njit

__all__ = ["allfinite1d", "configure_allfinite_guard"]


def _allfinite1d_impl(x: np.ndarray) -> bool:
    for i in range(x.size):
        if not math.isfinite(x[i]):
            return False
    return True



_allfinite1d_py = _allfinite1d_impl
_allfinite1d_jit = njit(cache=True)(_allfinite1d_impl) if njit is not None else _allfinite1d_impl

# Mutable binding exported to callers; defaults to numba version when available.
allfinite1d: Callable[[np.ndarray], bool] = _allfinite1d_py


def configure_allfinite_guard(jit_enabled: bool) -> None:
    """Select Python or numba implementation based on jit flag."""

    global allfinite1d
    if jit_enabled and njit is not None:
        allfinite1d = _allfinite1d_jit
    else:
        allfinite1d = _allfinite1d_py
