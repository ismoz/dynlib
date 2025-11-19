# src/dynlib/steppers/registry.py
from __future__ import annotations
from typing import Dict, Iterable
from dataclasses import fields

from .base import StepperSpec, StepperInfo, StepperMeta, StepperCaps

__all__ = ["register", "get_stepper", "registry"]

# name -> spec instance (canonical name + aliases)
_registry: Dict[str, StepperSpec] = {}

# Introspect dataclass fields once
_META_FIELDS = {f.name for f in fields(StepperMeta)}
_CAPS_FIELDS = {f.name for f in fields(StepperCaps)}


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

# ---------------------- Registry Helpers ----------------------

def _iter_unique_specs() -> Iterable[StepperSpec]:
    """
    Iterate over unique specs (one per canonical stepper).
    Aliases are collapsed by meta.name.
    """
    seen: set[str] = set()
    for spec in _registry.values():
        name = spec.meta.name
        if name in seen:
            continue
        seen.add(name)
        yield spec

def iter_specs() -> Iterable[StepperSpec]:
    """
    Iterate over unique specs (no alias duplicates).
    """
    return _iter_unique_specs()


def iter_infos() -> Iterable[StepperInfo]:
    """
    Iterate over StepperInfo (StepperMeta) for each canonical stepper.
    """
    for spec in _iter_unique_specs():
        yield spec.meta


# ---------------------- Public Helpers ----------------------

from fnmatch import fnmatch
from typing import Any, Callable

def select_steppers(
    *,
    name_pattern: str | None = None,
    predicate: Callable[[StepperInfo], bool] | None = None,
    **filters: Any,
) -> list[StepperInfo]:
    """
    Select steppers by StepperMeta / StepperCaps fields.

    Usage:
        select_steppers(kind="ode", scheme="implicit")
        select_steppers(stiff=True, jit_capable=True)
        select_steppers(family="bdf", requires_scipy=True)

    Args:
        name_pattern: Optional glob on meta.name (e.g. "bdf2*").
        predicate: Optional extra filter on StepperInfo.
        **filters: Keys must match StepperMeta or StepperCaps field names.

    Returns:
        List of StepperInfo (alias-free, one per canonical stepper).
    """
    # Split filters into meta vs caps using the dataclass field sets
    meta_filters: dict[str, Any] = {}
    caps_filters: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in filters.items():
        if key in _META_FIELDS:
            meta_filters[key] = value
        elif key in _CAPS_FIELDS:
            caps_filters[key] = value
        else:
            unknown.append(key)

    if unknown:
        # You can also choose to ignore unknowns instead of raising
        raise TypeError(
            f"Unknown stepper filter fields: {', '.join(sorted(unknown))}. "
            f"Valid StepperMeta fields: {sorted(_META_FIELDS)}; "
            f"StepperCaps fields: {sorted(_CAPS_FIELDS)}"
        )

    selected: list[StepperInfo] = []

    for spec in _iter_unique_specs():
        meta = spec.meta
        caps = meta.caps

        # name_pattern first since it's cheap and useful
        if name_pattern is not None and not fnmatch(meta.name, name_pattern):
            continue

        # Apply StepperMeta filters
        ok = True
        for key, expected in meta_filters.items():
            if getattr(meta, key) != expected:
                ok = False
                break
        if not ok:
            continue

        # Apply StepperCaps filters
        for key, expected in caps_filters.items():
            if getattr(caps, key) != expected:
                ok = False
                break
        if not ok:
            continue

        # Custom predicate on meta (StepperInfo)
        if predicate is not None and not predicate(meta):
            continue

        selected.append(meta)

    return selected


def list_steppers(**filters: Any) -> list[str]:
    """
    Convenience: like select_steppers() but returns canonical stepper names.

    Accepts the same filters as select_steppers (kind, scheme, jit_capable, etc.).
    """
    infos = select_steppers(**filters)
    return sorted(info.name for info in infos)
