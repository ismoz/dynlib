# src/dynlib/compiler/guards.py
"""
NaN/Inf detection utilities compatible with numba JIT compilation.

These guards are conditionally compiled based on the jit flag passed to build().
When jit=True, they are decorated with @njit and integrated with disk cache.
When jit=False, they run as ordinary Python functions.

Design rationale:
- Follows the same pattern as rhs, events, stepper, and runner functions
- Uses inline=True for better performance in hot loops
- Provides both 1D array and scalar checks
- Compatible with numba's type inference and compilation
"""
from __future__ import annotations
import math
from typing import Callable, Optional
from pathlib import Path
import textwrap
import inspect

from dynlib.runtime.softdeps import softdeps
from dynlib.errors import JITUnavailableError

__all__ = [
    "allfinite1d",
    "allfinite_scalar",
    "get_guards",
    "configure_guards_disk_cache",
    "register_guards_consumer",
    "_render_guards_inline_source",
]


def allfinite1d(arr):
    """
    Check if all elements in a 1D array are finite (not NaN or Inf).
    
    Args:
        arr: 1D numpy array (any numeric dtype)
    
    Returns:
        True if all elements are finite, False otherwise
    
    Note:
        This function is designed to be jit-compiled with inline='always'
        for zero overhead in hot loops.
    """
    for i in range(len(arr)):
        val = arr[i]
        if math.isnan(val) or math.isinf(val):
            return False
    return True


def allfinite_scalar(val):
    """
    Check if a scalar value is finite (not NaN or Inf).
    
    Args:
        val: numeric scalar
    
    Returns:
        True if value is finite, False otherwise
    
    Note:
        This function is designed to be jit-compiled with inline='always'
        for zero overhead in hot loops.
    """
    return not (math.isnan(val) or math.isinf(val))


_current_allfinite1d = allfinite1d
_current_allfinite_scalar = allfinite_scalar
_guard_consumers: list[tuple[dict[str, object], dict[str, str]]] = []


def register_guards_consumer(
    namespace: dict[str, object],
    *,
    mapping: Optional[dict[str, str]] = None,
) -> None:
    """
    Register a module namespace so guard updates propagate to existing steppers.

    Args:
        namespace: Module globals where guard names live.
        mapping: Optional mapping from guard identifiers to namespace keys.
                 Defaults to {"allfinite1d": "allfinite1d",
                              "allfinite_scalar": "allfinite_scalar"}.
    """
    if mapping is None:
        mapping = {
            "allfinite1d": "allfinite1d",
            "allfinite_scalar": "allfinite_scalar",
        }
    _guard_consumers.append((namespace, mapping))
    _install_guard_consumers(_current_allfinite1d, _current_allfinite_scalar)


def _install_guard_consumers(allfinite1d_fn, allfinite_scalar_fn) -> None:
    """Update registered namespaces with the active guard functions."""
    global _current_allfinite1d, _current_allfinite_scalar
    _current_allfinite1d = allfinite1d_fn
    _current_allfinite_scalar = allfinite_scalar_fn

    for namespace, mapping in _guard_consumers:
        target_1d = mapping.get("allfinite1d")
        if target_1d and target_1d in namespace:
            namespace[target_1d] = allfinite1d_fn

        target_scalar = mapping.get("allfinite_scalar")
        if target_scalar and target_scalar in namespace:
            namespace[target_scalar] = allfinite_scalar_fn


# ============================================================================
# Disk cache integration (follows same pattern as triplet/stepper/runner)
# ============================================================================

_pending_guards_cache_request: Optional["_GuardsCacheRequest"] = None


class _GuardsCacheRequest:
    """Request object for guards disk cache configuration."""
    def __init__(
        self,
        *,
        spec_hash: str,
        stepper_name: str,
        dtype: str,
        cache_root: Path,
    ):
        self.spec_hash = spec_hash
        self.stepper_name = stepper_name
        self.dtype = dtype
        self.cache_root = Path(cache_root).expanduser().resolve()


def configure_guards_disk_cache(
    *,
    spec_hash: str,
    stepper_name: str,
    dtype: str,
    cache_root: Path,
) -> None:
    """
    Store disk cache context for the next guards build.
    
    This follows the same pattern as configure_triplet_disk_cache and
    configure_stepper_disk_cache. Called by build() before get_guards().
    """
    global _pending_guards_cache_request
    _pending_guards_cache_request = _GuardsCacheRequest(
        spec_hash=spec_hash,
        stepper_name=stepper_name,
        dtype=dtype,
        cache_root=cache_root,
    )


def _consume_guards_cache_request() -> Optional[_GuardsCacheRequest]:
    """Consume pending guards cache request."""
    global _pending_guards_cache_request
    req = _pending_guards_cache_request
    _pending_guards_cache_request = None
    return req


# ============================================================================
# Guards factory with JIT toggle and disk cache
# ============================================================================

_SOFTDEPS = softdeps()
_NUMBA_AVAILABLE = _SOFTDEPS.numba

if _NUMBA_AVAILABLE:
    from numba import njit  # type: ignore
else:
    njit = None  # type: ignore


# In-process cache for guards (avoids recompilation within same process)
_inproc_guards_cache = {}


def get_guards(*, jit: bool = True, disk_cache: bool = True) -> tuple[Callable, Callable]:
    """
    Get NaN/Inf guard functions, optionally JIT-compiled with disk caching.
    
    Args:
        jit: If True, compile with numba @njit(inline=True). If False, return pure Python.
        disk_cache: If True and jit=True, attempt to use disk-backed cache.
    
    Returns:
        Tuple of (allfinite1d, allfinite_scalar) functions
    
    Behavior:
        - If jit=False: returns pure Python functions
        - If jit=True and numba not installed: raises RuntimeError
        - If jit=True and numba installed: returns JIT-compiled inline functions
        - Disk cache follows same pattern as runner/stepper/triplet caching
    
    Design notes:
        - Uses inline='always' for better performance (no function call overhead)
        - Integrates with dynlib's existing disk cache infrastructure
        - Guards are cached per (spec_hash, stepper_name, dtype) tuple
    """
    if not jit:
        result = (allfinite1d, allfinite_scalar)
        _install_guard_consumers(*result)
        return result
    
    if not _NUMBA_AVAILABLE:
        raise JITUnavailableError(
            "jit=True requires numba, but numba is not installed. "
            "Install numba or pass jit=False."
        )
    
    # Check in-process cache first
    cache_key = f"guards:jit={jit}:disk={disk_cache}"
    if cache_key in _inproc_guards_cache:
        result = _inproc_guards_cache[cache_key]
        _install_guard_consumers(*result)
        return result
    
    # Disk cache not implemented yet (guards are simple enough that
    # in-process cache + inline compilation is sufficient)
    # If needed in future, follow the JitTripletCache pattern from runner.py
    
    # JIT compile with inline='always' for zero overhead
    allfinite1d_jit = njit(inline='always')(allfinite1d)
    allfinite_scalar_jit = njit(inline='always')(allfinite_scalar)
    
    result = (allfinite1d_jit, allfinite_scalar_jit)
    _inproc_guards_cache[cache_key] = result
    _install_guard_consumers(*result)
    return result


# ============================================================================
# Source code rendering for disk cache (if needed in future)
# ============================================================================

def _render_guards_module_source() -> str:
    """
    Render guards module source for disk cache.
    
    This is here for future extension if disk caching becomes necessary.
    Currently guards use inline='always' so they get compiled into the calling
    function and don't need separate disk cache entries.
    """
    allfinite1d_src = textwrap.dedent(inspect.getsource(allfinite1d)).lstrip()
    allfinite_scalar_src = textwrap.dedent(inspect.getsource(allfinite_scalar)).lstrip()
    
    header = inspect.cleandoc(
        """
        # Auto-generated by dynlib.compiler.guards
        from __future__ import annotations
        import math
        from numba import njit
        
        __all__ = ["allfinite1d", "allfinite_scalar"]
        """
    )
    
    # Decorate with inline='always'
    allfinite1d_decorated = allfinite1d_src.replace(
        "def allfinite1d(", "@njit(inline='always')\ndef allfinite1d(", 1
    )
    allfinite_scalar_decorated = allfinite_scalar_src.replace(
        "def allfinite_scalar(", "@njit(inline='always')\ndef allfinite_scalar(", 1
    )
    
    return f"{header}\n\n{allfinite1d_decorated}\n\n{allfinite_scalar_decorated}\n"


def _render_guards_inline_source() -> str:
    """
    Render guards source for inline inclusion in runner/stepper modules.
    
    This version emits the guards with @njit(inline='always') decoration
    so they can be inlined directly into the runner module source.
    """
    allfinite1d_src = textwrap.dedent(inspect.getsource(allfinite1d)).lstrip()
    allfinite_scalar_src = textwrap.dedent(inspect.getsource(allfinite_scalar)).lstrip()
    
    # Decorate with inline='always' and remove docstrings for cleaner output
    allfinite1d_decorated = allfinite1d_src.replace(
        "def allfinite1d(", "@njit(inline='always')\ndef allfinite1d(", 1
    )
    allfinite_scalar_decorated = allfinite_scalar_src.replace(
        "def allfinite_scalar(", "@njit(inline='always')\ndef allfinite_scalar(", 1
    )
    
    header = inspect.cleandoc(
        """
        # ============================================================================
        # NaN/Inf Guards (auto-generated by dynlib.compiler.guards)
        # ============================================================================
        """
    )
    
    return f"{header}\n\n{allfinite1d_decorated}\n\n{allfinite_scalar_decorated}"
