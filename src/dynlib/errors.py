# src/dynlib/errors.py
from __future__ import annotations

__all__ = [
    "DynlibError",
    "ModelLoadError",
]

class DynlibError(Exception):
    """Base error for the dynlib package."""


class ModelLoadError(DynlibError):
    """Raised when a model (TOML + mods) fails validation or parsing."""
    def __init__(self, message: str):
        super().__init__(message)