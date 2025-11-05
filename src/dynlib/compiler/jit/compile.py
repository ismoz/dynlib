# src/dynlib/compiler/jit/compile.py
from __future__ import annotations
from typing import Callable, Dict, Tuple
import warnings

# JIT toggle applied *only here*.
# If numba missing or jit=False, we return original Python callables.

__all__ = ["maybe_jit_triplet", "jit_compile"]

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False
    njit = None  # type: ignore

def jit_compile(fn: Callable, *, jit: bool = True) -> Callable:
    """
    Centralized JIT compilation with consistent error handling.
    
    Behavior:
        - If jit=False: returns original Python function
        - If jit=True and numba not installed: warns and returns original function
        - If jit=True and numba installed but compilation fails: raises RuntimeError with details
    
    Args:
        fn: Function to compile
        jit: Whether to apply JIT compilation (default True)
    
    Returns:
        JIT-compiled function if successful, or original function with warning
    
    Raises:
        RuntimeError: If numba is installed but compilation fails
    """
    if not jit:
        return fn
    
    if not _NUMBA_OK:
        # Numba not installed: graceful fallback with warning
        warnings.warn(
            "Numba not found; falling back to pure Python (slower). "
            "Install numba for 10-100x speedup: pip install numba",
            RuntimeWarning,
            stacklevel=3  # Point to caller's caller
        )
        return fn
    
    # Numba is installed; attempt compilation
    try:
        return njit(cache=False)(fn)
    except Exception as e:
        # Numba installed but compilation failed: hard error
        raise RuntimeError(
            f"JIT compilation with numba failed: {type(e).__name__}: {e}"
        ) from e

def maybe_jit_triplet(rhs: Callable, events_pre: Callable, events_post: Callable, *, jit: bool):
    # Apply JIT here (and only here).
    return (
        jit_compile(rhs, jit=jit),
        jit_compile(events_pre, jit=jit),
        jit_compile(events_post, jit=jit),
    )
