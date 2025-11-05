# src/dynlib/steppers/registry.py
from __future__ import annotations
from typing import Dict

from .base import StepperSpec, StepperMeta

__all__ = ["register", "get_stepper", "registry"]

# name -> spec instance
_registry: Dict[str, StepperSpec] = {}

def register(spec: StepperSpec) -> None:
    """
    Register a stepper spec by its meta.name and meta.aliases.
    Enforces uniqueness of the canonical name; aliases may overlap only
    if they point to the same spec instance.
    """
    name = spec.meta.name
    if name in _registry and _registry[name] is not spec:
        raise ValueError(f"Stepper '{name}' already registered with a different spec.")
    _registry[name] = spec

    for alias in spec.meta.aliases:
        if alias in _registry and _registry[alias] is not spec:
            raise ValueError(f"Alias '{alias}' already registered for a different spec.")
        _registry[alias] = spec

def get_stepper(name: str) -> StepperSpec:
    """
    Return the registered spec for 'name' or raise KeyError.
    """
    return _registry[name]

def registry() -> Dict[str, StepperSpec]:
    """
    Read-only-ish view (do not mutate externally).
    """
    return dict(_registry)
