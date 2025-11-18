# src/dynlib/runtime/softdeps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "SoftDepState",
    "softdeps",
    "has_numba",
    "has_scipy",
    "reload_softdeps",
]


@dataclass(frozen=True)
class SoftDepState:
    """
    Snapshot of optional dependency availability.

    Fields capture both basic availability (bool) and any metadata that callers
    might need (e.g., version strings for cache invalidation).
    """

    numba: bool
    scipy: bool
    numba_version: Optional[str] = None
    llvmlite_version: Optional[str] = None


_STATE: SoftDepState | None = None


def _probe_softdeps() -> SoftDepState:
    """Import soft dependencies once and capture availability metadata."""
    numba_available = False
    numba_version: Optional[str] = None

    try:
        import numba  # type: ignore

        numba_available = True
        numba_version = getattr(numba, "__version__", "unknown")
    except Exception:
        numba = None  # type: ignore

    try:
        import llvmlite  # type: ignore

        llvmlite_version: Optional[str] = getattr(llvmlite, "__version__", "unknown")
    except Exception:
        llvmlite = None  # type: ignore
        llvmlite_version = None

    try:
        import scipy  # type: ignore  # noqa: F401

        scipy_available = True
    except Exception:
        scipy_available = False

    return SoftDepState(
        numba=numba_available,
        scipy=scipy_available,
        numba_version=numba_version,
        llvmlite_version=llvmlite_version,
    )


def softdeps() -> SoftDepState:
    """Return cached snapshot of soft dependency availability."""
    global _STATE
    if _STATE is None:
        _STATE = _probe_softdeps()
    return _STATE


def reload_softdeps() -> SoftDepState:
    """
    Force a fresh probe of dependency availability.

    Useful in tests that monkeypatch imports.
    """
    global _STATE
    _STATE = _probe_softdeps()
    return _STATE


def has_numba() -> bool:
    """Convenience helper for the most common dependency check."""
    return softdeps().numba


def has_scipy() -> bool:
    """Convenience helper for SciPy availability checks."""
    return softdeps().scipy
