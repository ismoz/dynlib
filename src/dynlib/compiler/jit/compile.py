# src/dynlib/compiler/jit/compile.py
from __future__ import annotations
from typing import Callable, Dict, Tuple

# JIT toggle applied *only here*.
# If numba missing or jit=False, we return original Python callables.

__all__ = ["maybe_jit_triplet"]

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False
    njit = None  # type: ignore

def _jit_if_enabled(fn: Callable, jit: bool) -> Callable:
    if jit and _NUMBA_OK:
        # nopython; error model left default; caching left to user’s global NUMBA cache settings
        return njit(cache=False)(fn)  # cache False to keep behavior simple at this slice
    return fn

def maybe_jit_triplet(rhs: Callable, events_pre: Callable, events_post: Callable, *, jit: bool):
    # Apply JIT here (and only here).
    return (
        _jit_if_enabled(rhs, jit),
        _jit_if_enabled(events_pre, jit),
        _jit_if_enabled(events_post, jit),
    )
