# src/dynlib/dsl/constants.py
"""
Shared DSL constants and reserved identifiers.

These values are inlined during code generation (no runtime lookups).
"""
from __future__ import annotations

from typing import Dict

__all__ = [
    "BUILTIN_CONSTS",
    "RUNTIME_RESERVED_NAMES",
    "RESERVED_IDENTIFIERS",
    "cast_constants",
]

# Built-in numeric constants that can appear directly in DSL expressions
BUILTIN_CONSTS: Dict[str, float] = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}

# Names reserved for runtime symbols (not user-definable)
RUNTIME_RESERVED_NAMES = frozenset({"t"})

# Union of runtime-reserved names and builtin constants; identifiers must not shadow these
RESERVED_IDENTIFIERS = frozenset(set(RUNTIME_RESERVED_NAMES) | set(BUILTIN_CONSTS.keys()))


def cast_constants(dtype: str, extra: Dict[str, float | int] | None = None) -> Dict[str, float | int]:
    """
    Optionally cast built-in constants to a target dtype.

    This keeps codegen literals aligned with the model dtype; callers that don't
    care about dtype can keep using Python floats directly.
    """
    all_consts: Dict[str, float | int] = dict(BUILTIN_CONSTS)
    if extra:
        all_consts.update(extra)

    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy should be available, but stay defensive
        return all_consts

    np_dtype = np.dtype(dtype)
    return {name: np_dtype.type(val).item() for name, val in all_consts.items()}
