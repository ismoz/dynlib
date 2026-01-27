# src/dynlib/dsl/schema.py
from __future__ import annotations
from typing import Dict, Any, Iterable
import re
import difflib

from dynlib.errors import ModelLoadError

__all__ = [
    "validate_model_header",
    "validate_tables",
    "validate_name_collisions",
]

# Regex patterns for derivative notation (ODE only)
_DFUNC_PAREN = re.compile(r'^d\(\s*([A-Za-z_]\w*)\s*\)$')
_DFUNC_FLAT = re.compile(r'^d([A-Za-z_]\w*)$')


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

    if "label" in model:
        raise ModelLoadError("[model].label is no longer supported; use [model].name")

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

    Required: [model], [states]
    Optional: [params], [constants], [equations], [equations.rhs], [equations.inverse], [equations.jacobian], [aux], [functions], [events.*], [sim]
    """
    validate_model_header(doc)

    _require_table(doc, "states")
    if not isinstance(doc["states"], dict):
        raise ModelLoadError("[states] must be a table of name = value")

    # params/constants are optional; if present, must be a table
    params = doc.get("params")
    if params is not None and not isinstance(params, dict):
        raise ModelLoadError("[params] must be a table if present")

    constants = doc.get("constants")
    if constants is not None and not isinstance(constants, dict):
        raise ModelLoadError("[constants] must be a table if present")

    eq = doc.get("equations")
    if eq is not None and not isinstance(eq, dict):
        raise ModelLoadError("[equations] must be a table if present")

    if isinstance(eq, dict):
        # Fail loudly on unknown [equations] keys.
        # This prevents silent no-op dynamics when users typo 'expr' as 'exprs', etc.
        allowed_eq_keys = {"rhs", "expr", "jacobian", "inverse"}
        unknown_eq_keys = sorted(k for k in eq.keys() if k not in allowed_eq_keys)
        if unknown_eq_keys:
            hints = []
            for bad in unknown_eq_keys:
                close = difflib.get_close_matches(bad, list(allowed_eq_keys), n=1, cutoff=0.6)
                if close:
                    hints.append(f"'{bad}' (did you mean '{close[0]}'?)")
                else:
                    hints.append(f"'{bad}'")
            raise ModelLoadError(
                f"Unknown key(s) in [equations]: {', '.join(hints)}. "
                "Valid keys: rhs, expr, jacobian"
            )

        rhs = eq.get("rhs")
        if rhs is not None and not isinstance(rhs, dict):
            raise ModelLoadError("[equations.rhs] must be a table of name = expr")
        expr = eq.get("expr")
        if expr is not None and not isinstance(expr, str):
            raise ModelLoadError("[equations].expr must be a string block if present")
        inv = eq.get("inverse")
        if inv is not None:
            if doc.get("model", {}).get("type") != "map":
                raise ModelLoadError("[equations.inverse] is only supported for map models")
            if not isinstance(inv, dict):
                raise ModelLoadError("[equations.inverse] must be a table if present")
            allowed_inv_keys = {"rhs", "expr"}
            unknown_inv_keys = sorted(k for k in inv.keys() if k not in allowed_inv_keys)
            if unknown_inv_keys:
                hints = []
                for bad in unknown_inv_keys:
                    close = difflib.get_close_matches(bad, list(allowed_inv_keys), n=1, cutoff=0.6)
                    if close:
                        hints.append(f"'{bad}' (did you mean '{close[0]}'?)")
                    else:
                        hints.append(f"'{bad}'")
                raise ModelLoadError(
                    f"Unknown key(s) in [equations.inverse]: {', '.join(hints)}. "
                    "Valid keys: rhs, expr"
                )
            inv_rhs = inv.get("rhs")
            if inv_rhs is not None and not isinstance(inv_rhs, dict):
                raise ModelLoadError("[equations.inverse.rhs] must be a table of name = expr")
            inv_expr = inv.get("expr")
            if inv_expr is not None and not isinstance(inv_expr, str):
                raise ModelLoadError("[equations.inverse].expr must be a string block if present")
        jac = eq.get("jacobian")
        if jac is not None and not isinstance(jac, dict):
            raise ModelLoadError("[equations.jacobian] must be a table if present")

        if isinstance(jac, dict):
            allowed_jac_keys = {"expr"}
            unknown_jac_keys = sorted(k for k in jac.keys() if k not in allowed_jac_keys)
            if unknown_jac_keys:
                hints = []
                for bad in unknown_jac_keys:
                    close = difflib.get_close_matches(bad, list(allowed_jac_keys), n=1, cutoff=0.6)
                    if close:
                        hints.append(f"'{bad}' (did you mean '{close[0]}'?)")
                    else:
                        hints.append(f"'{bad}'")
                raise ModelLoadError(
                    f"Unknown key(s) in [equations.jacobian]: {', '.join(hints)}. "
                    "Valid keys: expr"
                )

    funcs = doc.get("functions")
    if funcs is not None and not isinstance(funcs, dict):
        raise ModelLoadError("[functions] must be a table of subtables")

    events = doc.get("events")
    if events is not None and not isinstance(events, dict):
        raise ModelLoadError("[events] must be a table of named event subtables")

    presets = doc.get("presets")
    if presets is not None and not isinstance(presets, dict):
        raise ModelLoadError("[presets] must be a table of named preset subtables")

    sim = doc.get("sim")
    if sim is not None and not isinstance(sim, dict):
        raise ModelLoadError("[sim] must be a table if present")


def _targets_from_block(expr: str, model_type: str = "ode", states: Iterable[str] = ()) -> Iterable[str]:
    """Extract state names from a multi-line block like:
        dx = -a*x  (ODE: derivative notation)
        x = v      (ODE/map: direct assignment)
    
    For ODE models: Parses both 'd(x)' / 'dx' -> 'x' and direct 'x = expr'.
    For map models: Parses direct assignment 'x = expr'.
    
    This is used at schema validation time for duplicate detection. The stricter
    derivative notation validation happens later in emitter.py where we have full context.
    
    If states are provided and model_type is ODE, validates that derivative targets
    refer to declared states (to prevent 'delta' from being parsed as 'd(elta)').
    """
    states_set = set(states)
    targets: list[str] = []
    
    for line in expr.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        
        lhs = line.split("=", 1)[0].strip()
        
        # Try to parse as derivative notation (for both ODE and map, but validate for ODE)
        m = _DFUNC_PAREN.match(lhs) or _DFUNC_FLAT.match(lhs)
        if m:
            name = m.group(1)
            # If we have states info and this is ODE, verify it's a real state
            # This prevents "delta" from matching as "d(elta)" 
            if states_set and name in states_set:
                targets.append(name)
            elif not states_set:
                # No states info yet - accept tentatively
                targets.append(name)
            else:
                # Matched derivative pattern but name not in states
                # For map models, this is an error (derivative notation not allowed)
                # For ODE models, might be a typo or might be direct assignment
                # Fall through to treat as direct assignment
                targets.append(lhs)
        else:
            # Direct assignment (no derivative notation)
            targets.append(lhs)
    
    return targets


def validate_name_collisions(doc: Dict[str, Any]) -> None:
    """Forbid duplicate equation targets across [equations.rhs] and block expr.

    - If both are present, the union of targets must not duplicate the same state.
    - Targets must exist in [states].
    - Derivative notation (dx/d(x)) only allowed for ODE models and only for declared states.
    """
    states = doc.get("states", {})
    model_type = doc.get("model", {}).get("type", "ode")
    eq = doc.get("equations", {}) if isinstance(doc.get("equations"), dict) else {}

    # Common typo guard: docs specify [equations].expr (singular).
    # If a model provides [equations].exprs as a string, it will otherwise be
    # ignored downstream and can silently yield a "frozen" trajectory.
    if isinstance(eq.get("exprs"), str) and not isinstance(eq.get("expr"), str):
        raise ModelLoadError(
            "Invalid equations block key: use [equations].expr (singular), not 'exprs'. "
            "Example: [equations]\nexpr = \"\"\"dx = ...\n\"\"\""
        )

    rhs = eq.get("rhs") if isinstance(eq.get("rhs"), dict) else None
    expr = eq.get("expr") if isinstance(eq.get("expr"), str) else None

    rhs_targets = set(rhs.keys()) if rhs else set()
    block_targets = set(_targets_from_block(expr, model_type=model_type, states=states.keys())) if expr else set()

    dup = rhs_targets.intersection(block_targets)
    if dup:
        raise ModelLoadError(f"Duplicate equation targets across rhs and block: {sorted(dup)}")

    # Ensure targets address declared states only (best-effort here; parser rechecks)
    unknown = (rhs_targets | block_targets) - set(states.keys())
    if unknown:
        raise ModelLoadError(f"Equation targets must be declared in [states], unknown: {sorted(unknown)}")

    inv_tbl = eq.get("inverse") if isinstance(eq.get("inverse"), dict) else None
    if inv_tbl:
        inv_rhs = inv_tbl.get("rhs") if isinstance(inv_tbl.get("rhs"), dict) else None
        inv_expr = inv_tbl.get("expr") if isinstance(inv_tbl.get("expr"), str) else None
        inv_rhs_targets = set(inv_rhs.keys()) if inv_rhs else set()
        inv_block_targets = set(
            _targets_from_block(inv_expr, model_type=model_type, states=states.keys())
        ) if inv_expr else set()

        inv_dup = inv_rhs_targets.intersection(inv_block_targets)
        if inv_dup:
            raise ModelLoadError(
                f"Duplicate inverse equation targets across rhs and block: {sorted(inv_dup)}"
            )

        inv_unknown = (inv_rhs_targets | inv_block_targets) - set(states.keys())
        if inv_unknown:
            raise ModelLoadError(
                f"Inverse equation targets must be declared in [states], unknown: {sorted(inv_unknown)}"
            )
