# src/dynlib/errors.py
from __future__ import annotations
from typing import List

__all__ = [
    "DynlibError",
    "ModelLoadError",
    "ModelNotFoundError",
    "ConfigError",
    "PathTraversalError",
    "AmbiguousModelError",
]

class DynlibError(Exception):
    """Base error for the dynlib package."""


class ModelLoadError(DynlibError):
    """Raised when a model (TOML + mods) fails validation or parsing."""
    def __init__(self, message: str):
        super().__init__(message)


class ModelNotFoundError(DynlibError):
    """Raised when a model URI cannot be resolved to an existing file."""
    def __init__(self, uri: str, candidates: List[str]):
        self.uri = uri
        self.candidates = candidates
        msg = f"Model not found: {uri}\n"
        if candidates:
            msg += "Searched locations:\n"
            for c in candidates:
                msg += f"  - {c}\n"
        super().__init__(msg)


class ConfigError(DynlibError):
    """Raised when configuration file is malformed or invalid."""
    def __init__(self, message: str):
        super().__init__(message)


class PathTraversalError(DynlibError):
    """Raised when a URI attempts to traverse outside allowed roots."""
    def __init__(self, uri: str, attempted_path: str, root: str):
        self.uri = uri
        self.attempted_path = attempted_path
        self.root = root
        msg = f"Path traversal detected: {uri}\n"
        msg += f"Attempted: {attempted_path}\n"
        msg += f"Root: {root}"
        super().__init__(msg)


class AmbiguousModelError(DynlibError):
    """Raised when a model URI matches multiple files (e.g., extensionless)."""
    def __init__(self, uri: str, matches: List[str]):
        self.uri = uri
        self.matches = matches
        msg = f"Ambiguous model reference: {uri}\n"
        msg += "Multiple matches found:\n"
        for m in matches:
            msg += f"  - {m}\n"
        msg += "Please use an explicit filename."
        super().__init__(msg)