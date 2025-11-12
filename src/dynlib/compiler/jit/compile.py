# src/dynlib/compiler/jit/compile.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import warnings

# JIT toggle applied *only here*.
# If numba missing or jit=False, we return original Python callables.

__all__ = ["maybe_jit_triplet", "jit_compile"]


@dataclass(frozen=True)
class JittedCallable:
    fn: Callable
    cache_digest: Optional[str]
    cache_hit: bool
    component: Optional[str] = None

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False
    njit = None  # type: ignore

def jit_compile(fn: Callable, *, jit: bool = True, cache: bool = False) -> JittedCallable:
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
        return JittedCallable(fn=fn, cache_digest=None, cache_hit=False, component=None)
    
    if not _NUMBA_OK:
        # Numba not installed: graceful fallback with warning
        warnings.warn(
            "Numba not found; falling back to pure Python (slower). "
            "Install numba for 10-100x speedup: pip install numba",
            RuntimeWarning,
            stacklevel=3  # Point to caller's caller
        )
        return JittedCallable(fn=fn, cache_digest=None, cache_hit=False, component=None)

    if cache:
        artifact = _jit_compile_with_disk_cache(fn)
        if artifact is not None:
            return artifact
    
    # Numba is installed; attempt compilation
    try:
        compiled = njit(cache=False)(fn)
        return JittedCallable(
            fn=compiled,
            cache_digest=None,
            cache_hit=False,
            component=None,
        )
    except Exception as e:
        # Numba installed but compilation failed: hard error
        raise RuntimeError(
            f"JIT compilation with numba failed: {type(e).__name__}: {e}"
        ) from e

def maybe_jit_triplet(
    rhs: Callable,
    events_pre: Callable,
    events_post: Callable,
    *,
    jit: bool,
    cache: bool = False,
    cache_setup: Optional[Callable[[str], None]] = None,
) -> Tuple[JittedCallable, JittedCallable, JittedCallable]:
    """Apply JIT to RHS/events, optionally wiring up disk cache."""
    components = (
        ("rhs", rhs),
        ("events_pre", events_pre),
        ("events_post", events_post),
    )
    results = []
    for name, fn in components:
        if cache and cache_setup is not None:
            cache_setup(name)
        results.append(jit_compile(fn, jit=jit, cache=cache))
    return results[0], results[1], results[2]


def _jit_compile_with_disk_cache(fn: Callable) -> Optional[JittedCallable]:
    from dynlib.compiler.codegen import runner as runner_codegen

    request = runner_codegen.consume_callable_disk_cache_request()
    if request is None:
        raise RuntimeError(
            "jit_compile(cache=True) called without configure_triplet_disk_cache() or configure_stepper_disk_cache()"
        )

    cache_instance: Optional[object]
    if request.family == "triplet":
        cache_instance = runner_codegen.JitTripletCache(request)
    elif request.family == "stepper":
        cache_instance = runner_codegen._StepperDiskCache(request)
    else:
        raise RuntimeError(f"Unknown cache family: {request.family}")

    try:
        compiled, digest, hit = cache_instance.get_or_build()  # type: ignore[attr-defined]
        return JittedCallable(
            fn=compiled,
            cache_digest=digest,
            cache_hit=hit,
            component=request.component,
        )
    except runner_codegen.DiskCacheUnavailable as exc:
        runner_codegen._warn_disk_cache_disabled(str(exc))
        return None
