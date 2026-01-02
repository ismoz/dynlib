# src/dynlib/compiler/jit/compile.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Tuple

from dynlib.runtime.softdeps import softdeps
from dynlib.errors import JITUnavailableError

# JIT toggle applied *only here*.
# If numba is missing and jit=True, we raise to avoid silent fallbacks.

__all__ = ["maybe_jit_triplet", "jit_compile"]


@dataclass(frozen=True)
class JittedCallable:
    fn: Callable
    cache_digest: Optional[str]
    cache_hit: bool
    component: Optional[str] = None


@lru_cache(maxsize=1)
def _get_njit():
    from numba import njit  # type: ignore

    return njit

def jit_compile(fn: Callable, *, jit: bool = True, cache: bool = False) -> JittedCallable:
    """
    Centralized JIT compilation with consistent error handling.
    
    Behavior:
        - If jit=False: returns original Python function
        - If jit=True and numba not installed: raises RuntimeError
        - If jit=True and numba installed but compilation fails: raises RuntimeError with details
    
    Args:
        fn: Function to compile
        jit: Whether to apply JIT compilation (default True)
    
    Returns:
        JIT-compiled function if successful
    
    Raises:
        RuntimeError: If numba is missing or compilation fails
    """
    if not jit:
        return JittedCallable(fn=fn, cache_digest=None, cache_hit=False, component=None)
    
    deps = softdeps()
    if not deps.numba:
        raise JITUnavailableError(
            "jit=True requires numba, but numba is not installed. "
            "Install numba or pass jit=False."
        )

    if cache:
        artifact = _jit_compile_with_disk_cache(fn)
        if artifact is not None:
            return artifact
    
    # Numba is installed; attempt compilation
    try:
        njit = _get_njit()
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
    update_aux: Callable,
    *,
    jit: bool,
    cache: bool = False,
    cache_setup: Optional[Callable[[str], None]] = None,
) -> Tuple[JittedCallable, JittedCallable, JittedCallable, JittedCallable]:
    """Apply JIT to RHS/events/update_aux, optionally wiring up disk cache."""
    components = (
        ("rhs", rhs),
        ("events_pre", events_pre),
        ("events_post", events_post),
        ("update_aux", update_aux),
    )
    results = []
    for name, fn in components:
        if cache and cache_setup is not None:
            cache_setup(name)
        results.append(jit_compile(fn, jit=jit, cache=cache))
    return results[0], results[1], results[2], results[3]


def _jit_compile_with_disk_cache(fn: Callable) -> Optional[JittedCallable]:
    from dynlib.compiler.codegen import runner_cache as runner_codegen

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
