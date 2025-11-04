# src/dynlib/dsl/schema.py
from __future__ import annotations
from typing import Dict, Any, Iterable

from dynlib.errors import ModelLoadError

__all__ = [
    "validate_model_header",
    "validate_tables",
    "validate_name_collisions",
]


def _require_table(doc: Dict[str, Any], key: str) -> None:
    if key not in doc or not isinstance(doc[key], dict):
        raise ModelLoadError(f"Missing required table [{key}]")


def _is_float_dtype(dtype: str) -> bool:
    # Accept common aliases; exact canon happens later in spec._canon_dtype
    alias = {
        "float": "float64",
        "double": "float64",
        "single": "float32",
    }
    norm = alias.get(dtype, dtype)
    return norm in {"float32", "float64", "float16", "bfloat16"}


def validate_model_header(doc: Dict[str, Any]) -> None:
    """Structural checks for [model].

    Rules (frozen):
      - [model] exists and has type = "ode" | "map".
      - dtype defaults to "float64". ODE requires a floating dtype.
    """
    _require_table(doc, "model")
    model = doc["model"]
    if not isinstance(model, dict):
        raise ModelLoadError("[model] must be a table")

    mtype = model.get("type")
    if mtype not in {"ode", "map"}:
        raise ModelLoadError("[model].type must be 'ode' or 'map'")

    dtype = model.get("dtype", "float64")
    if not isinstance(dtype, str):
        raise ModelLoadError("[model].dtype must be a string if present")

    if mtype == "ode" and not _is_float_dtype(dtype):
        raise ModelLoadError("ODE models require a floating dtype (float32/float64/float16/bfloat16)")


def validate_tables(doc: Dict[str, Any]) -> None:
    """Presence and shape checks for top-level tables.

    Required: [model], [states], [params]
    Optional: [equations], [equations.rhs], [aux], [functions], [events.*], [sim]
    """
    validate_model_header(doc)

    _require_table(doc, "states")
    if not isinstance(doc["states"], dict):
        raise ModelLoadError("[states] must be a table of name = value")

    _require_table(doc, "params")
    if not isinstance(doc["params"], dict):
        raise ModelLoadError("[params] must be a table (may be empty)")

    eq = doc.get("equations")
    if eq is not None and not isinstance(eq, dict):
        raise ModelLoadError("[equations] must be a table if present")

    if isinstance(eq, dict):
        rhs = eq.get("rhs")
        if rhs is not None and not isinstance(rhs, dict):
            raise ModelLoadError("[equations.rhs] must be a table of name = expr")
        expr = eq.get("expr")
        if expr is not None and not isinstance(expr, str):
            raise ModelLoadError("[equations].expr must be a string block if present")

    funcs = doc.get("functions")
    if funcs is not None and not isinstance(funcs, dict):
        raise ModelLoadError("[functions] must be a table of subtables")

    events = doc.get("events")
    if events is not None and not isinstance(events, dict):
        raise ModelLoadError("[events] must be a table of named event subtables")

    sim = doc.get("sim")
    if sim is not None and not isinstance(sim, dict):
        raise ModelLoadError("[sim] must be a table if present")


def _targets_from_block(expr: str) -> Iterable[str]:
    """Extract naive LHS targets from a multi-line block like:
        dx = -a*x
        du = v
    This is *syntactic* and conservative; parser will re-validate later.
    """
    targets: list[str] = []
    for line in expr.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            lhs = line.split("=", 1)[0].strip()
            # allow patterns like 'dx' or 'x'
            if lhs:
                targets.append(lhs)
    return targets


def validate_name_collisions(doc: Dict[str, Any]) -> None:
    """Forbid duplicate equation targets across [equations.rhs] and block expr.

    - If both are present, the union of targets must not duplicate the same state.
    - Targets must exist in [states].
    """
    states = doc.get("states", {})
    eq = doc.get("equations", {}) if isinstance(doc.get("equations"), dict) else {}

    rhs = eq.get("rhs") if isinstance(eq.get("rhs"), dict) else None
    expr = eq.get("expr") if isinstance(eq.get("expr"), str) else None

    rhs_targets = set(rhs.keys()) if rhs else set()
    block_targets = set(_targets_from_block(expr)) if expr else set()

    dup = rhs_targets.intersection(block_targets)
    if dup:
        raise ModelLoadError(f"Duplicate equation targets across rhs and block: {sorted(dup)}")

    # Ensure targets address declared states only (best-effort here; parser rechecks)
    unknown = (rhs_targets | block_targets) - set(states.keys())
    if unknown:
        raise ModelLoadError(f"Equation targets must be declared in [states], unknown: {sorted(unknown)}")