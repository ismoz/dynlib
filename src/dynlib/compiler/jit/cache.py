# src/dynlib/compiler/jit/cache.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# Tiny in-process cache for compiled callables (rhs/events).
# Keyed by (spec_hash, stepper_name, workspace_signature, dtype, version_pins)

__all__ = ["CacheKey", "JITCache"]

@dataclass(frozen=True)
class CacheKey:
    spec_hash: str
    stepper: str
    structsig: Tuple[int, ...]   # derived from workspace structure (immutable)
    dtype: str
    version_pins: Tuple[str, ...]  # e.g. ("dynlib=2", "numba=0.60")

class JITCache:
    def __init__(self):
        self._store: Dict[CacheKey, Dict[str, Any]] = {}

    def get(self, key: CacheKey):
        return self._store.get(key)

    def put(self, key: CacheKey, payload: Dict[str, Any]):
        self._store[key] = payload
