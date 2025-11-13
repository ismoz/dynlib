# src/dynlib/dsl/parser.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List

from dynlib.errors import ModelLoadError
from .schema import validate_tables, validate_name_collisions

__all__ = [
    "parse_model_v2",
]


def _ordered_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    # Python 3.7+ preserves insertion order; we keep it explicit here.
    return list(d.items())


def _read_functions(funcs_tbl: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, body in funcs_tbl.items():
        if not isinstance(body, dict):
            raise ModelLoadError(f"[functions.{name}] must be a table")
        args = body.get("args", [])
        expr = body.get("expr")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ModelLoadError(f"[functions.{name}].args must be a list of strings")
        if not isinstance(expr, str):
            raise ModelLoadError(f"[functions.{name}].expr must be a string")
        out[name] = {"args": list(args), "expr": expr}
    return out


def _read_events(ev_tbl: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for name, body in ev_tbl.items():
        if not isinstance(body, dict):
            raise ModelLoadError(f"[events.{name}] must be a table")
        phase = body.get("phase")
        if phase not in {"pre", "post", "both"}:
            raise ModelLoadError(f"[events.{name}].phase must be 'pre'|'post'|'both'")
        cond = body.get("cond")
        if not isinstance(cond, str):
            raise ModelLoadError(f"[events.{name}].cond must be a string expression")
        # action: either keyed assignments or a block string
        action_keyed = None
        action_block = None
        if "action" in body:
            if isinstance(body["action"], str):
                action_block = body["action"]
            elif isinstance(body["action"], dict):
                # TOML dotted keys like action.x create a nested dict
                action_ns = body["action"]
                for tgt, expr in action_ns.items():
                    if not isinstance(expr, str):
                        raise ModelLoadError(f"[events.{name}].action.{tgt} must be a string expression")
                action_keyed = action_ns
        else:
            # Alternative: keyed form via 'action.*' flat keys (fallback)
            action_ns = {k[7:]: v for k, v in body.items() if k.startswith("action.")}
            if action_ns:
                for tgt, expr in action_ns.items():
                    if not isinstance(expr, str):
                        raise ModelLoadError(f"[events.{name}].action.{tgt} must be a string expression")
                action_keyed = action_ns
        
        # Reject deprecated 'record' key
        if "record" in body:
            raise ModelLoadError(
                f"[events.{name}].record is no longer supported. "
                f"Use log=['t'] to record event occurrence times, or log=['t', 'x', ...] to log time and values."
            )
        
        log = body.get("log", [])
        if log is None:
            log = []
        if not isinstance(log, list) or not all(isinstance(s, str) for s in log):
            raise ModelLoadError(f"[events.{name}].log must be a list of strings if present")
        
        # Tags
        tags = body.get("tags", [])
        if tags is None:
            tags = []
        if not isinstance(tags, list) or not all(isinstance(s, str) for s in tags):
            raise ModelLoadError(f"[events.{name}].tags must be a list of strings if present")
        
        events.append({
            "name": name,
            "phase": phase,
            "cond": cond,
            "action_keyed": action_keyed,
            "action_block": action_block,
            "log": list(log),
            "tags": list(tags),
        })
    return events


def _read_presets(presets_tbl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse [presets.<name>] blocks from TOML.
    
    Returns a list of dicts with keys: name, params, states (optional).
    """
    presets: List[Dict[str, Any]] = []
    for name, body in presets_tbl.items():
        if not isinstance(body, dict):
            raise ModelLoadError(f"[presets.{name}] must be a table")
        
        # Read params (optional, may be empty)
        params = body.get("params")
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise ModelLoadError(f"[presets.{name}].params must be a table")
        
        # Validate param values are numeric
        for key, val in params.items():
            if not isinstance(val, (int, float)):
                raise ModelLoadError(
                    f"[presets.{name}].params.{key} must be a number, got {type(val).__name__}"
                )
        
        # Read states (optional, may be empty)
        states = body.get("states")
        if states is not None:
            if not isinstance(states, dict):
                raise ModelLoadError(f"[presets.{name}].states must be a table if present")
            
            # Validate state values are numeric
            for key, val in states.items():
                if not isinstance(val, (int, float)):
                    raise ModelLoadError(
                        f"[presets.{name}].states.{key} must be a number, got {type(val).__name__}"
                    )
        else:
            states = {}

        if len(params) == 0 and len(states) == 0:
            raise ModelLoadError(
                f"[presets.{name}] must define at least one param or state"
            )
        
        presets.append({
            "name": name,
            "params": dict(params),
            "states": dict(states) if states else None,
        })
    
    return presets


def parse_model_v2(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a v2 DSL TOML dict into a normalized model dict (no codegen).

    Returns keys:
      model, states, params, equations:{rhs|expr}, aux, functions, events(list), sim
    """
    validate_tables(doc)
    validate_name_collisions(doc)

    model = doc["model"].copy()
    if "dtype" not in model:
        model["dtype"] = "float64"

    # Preserve order of declaration
    states_in = doc["states"]
    params_in = doc.get("params", {})  # params is optional, default to empty
    states = {k: states_in[k] for k in states_in.keys()}
    params = {k: params_in[k] for k in params_in.keys()}

    # Equations
    eq_tbl = doc.get("equations") or {}
    rhs_tbl = eq_tbl.get("rhs") or None
    block_expr = eq_tbl.get("expr") or None
    if rhs_tbl is not None and not isinstance(rhs_tbl, dict):
        raise ModelLoadError("[equations.rhs] must be a table")
    if block_expr is not None and not isinstance(block_expr, str):
        raise ModelLoadError("[equations].expr must be a string")

    # Aux
    aux_tbl = doc.get("aux") or {}
    if not isinstance(aux_tbl, dict):
        raise ModelLoadError("[aux] must be a table of name = expr")
    for k, v in aux_tbl.items():
        if not isinstance(v, str):
            raise ModelLoadError(f"[aux].{k} must be a string expression")

    # Functions
    funcs_tbl = doc.get("functions") or {}
    functions = _read_functions(funcs_tbl) if funcs_tbl else {}

    # Events
    ev_tbl = doc.get("events") or {}
    events = _read_events(ev_tbl) if ev_tbl else []

    # Presets
    presets_tbl = doc.get("presets") or {}
    presets = _read_presets(presets_tbl) if presets_tbl else []

    # Sim (defaults finalized in spec.build_spec)
    sim_tbl = doc.get("sim") or {}
    if not isinstance(sim_tbl, dict):
        raise ModelLoadError("[sim] must be a table if present")

    return {
        "model": {"type": model["type"], "label": model.get("label"), "dtype": model["dtype"]},
        "states": states,
        "params": params,
        "equations": {"rhs": rhs_tbl, "expr": block_expr},
        "aux": aux_tbl,
        "functions": functions,
        "events": events,
        "presets": presets,
        "sim": sim_tbl,
    }
