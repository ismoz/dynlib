# src/dynlib/dsl/parser.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import ast

from dynlib.errors import ModelLoadError
from .schema import validate_tables, validate_name_collisions
from .constants import BUILTIN_CONSTS

__all__ = [
    "parse_model_v2",
]

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_UNARY_OPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}


def _sanitize_literal_expr(expr: str) -> str:
    """Normalize caret power `^` to Python's `**` for literal math."""
    return expr.replace("^", "**")


def _allowed_numeric_names(extra: Dict[str, float | int] | None = None) -> Dict[str, float | int]:
    allowed = dict(BUILTIN_CONSTS)
    if extra:
        allowed.update(extra)
    return allowed


def _evaluate_numeric_expr(expr: str, context: str, allowed_names: Dict[str, float | int] | None = None) -> float | int:
    txt = expr.strip()
    if not txt:
        raise ModelLoadError(f"{context} must be a numeric literal or expression, got empty string")
    allowed = _allowed_numeric_names(allowed_names)
    try:
        node = ast.parse(_sanitize_literal_expr(txt), mode="eval")
    except SyntaxError as err:
        raise ModelLoadError(
            f"{context} must be a numeric literal or arithmetic expression (e.g. '8/3')"
        ) from err
    return _eval_node(node.body, context, allowed)


def _eval_node(node: ast.AST, context: str, allowed_names: Dict[str, float | int]) -> float | int:
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ModelLoadError(
                f"{context} must evaluate to a number, got {type(val).__name__}"
            )
        return val
    if isinstance(node, ast.Name):
        if node.id in allowed_names:
            return allowed_names[node.id]
        raise ModelLoadError(
            f"{context} contains unknown identifier '{node.id}'. "
            f"Allowed constants: {', '.join(sorted(allowed_names))}."
        )
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ModelLoadError(f"{context} contains an unsupported unary operator")
        return op(_eval_node(node.operand, context, allowed_names))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ModelLoadError(f"{context} contains an unsupported operator")
        left = _eval_node(node.left, context, allowed_names)
        right = _eval_node(node.right, context, allowed_names)
        try:
            return op(left, right)
        except ZeroDivisionError as err:
            raise ModelLoadError(f"{context} has a division by zero") from err
    raise ModelLoadError(
        f"{context} must be composed of numbers and arithmetic operators (e.g. '8/3')"
    )


def _coerce_numeric_table(tbl: Dict[str, Any], section: str, allowed_names: Dict[str, float | int] | None = None) -> Dict[str, float | int]:
    """Return an ordered dict where values are numeric scalars."""
    allowed = _allowed_numeric_names(allowed_names)
    out: Dict[str, float | int] = {}
    for key in tbl.keys():
        val = tbl[key]
        context = f"[{section}].{key}"
        if isinstance(val, (int, float)):
            out[key] = val
        elif isinstance(val, str):
            out[key] = _evaluate_numeric_expr(val, context, allowed)
        else:
            raise ModelLoadError(
                f"{context} must be a number or numeric expression string, got {type(val).__name__}"
            )
    return out


def _coerce_constants_table(tbl: Dict[str, Any]) -> Dict[str, float | int]:
    """Return constants as numeric scalars, allowing earlier constants by name."""
    constants: Dict[str, float | int] = {}
    for key in tbl.keys():
        val = tbl[key]
        context = f"[constants].{key}"
        if isinstance(val, (int, float)):
            constants[key] = val
        elif isinstance(val, str):
            constants[key] = _evaluate_numeric_expr(val, context, allowed_names=constants)
        else:
            raise ModelLoadError(
                f"{context} must be a number or numeric expression string, got {type(val).__name__}"
            )
    return constants


def _ordered_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    # Python 3.7+ preserves insertion order; we keep it explicit here.
    return list(d.items())

def _coerce_jacobian_exprs(jac_tbl: Dict[str, Any]) -> Dict[str, list[list[str]]]:
    """Normalize [equations.jacobian] table to a list-of-lists of strings."""
    exprs = jac_tbl.get("expr")
    if exprs is None:
        if "exprs" in jac_tbl:
            raise ModelLoadError(
                "Invalid Jacobian key: use [equations.jacobian].expr (singular), not '.exprs'."
            )
        raise ModelLoadError("[equations.jacobian].expr is required and must be a square list")
    if not isinstance(exprs, list):
        raise ModelLoadError("[equations.jacobian].expr must be a list of rows")

    rows: list[list[str]] = []
    for i, row in enumerate(exprs):
        if not isinstance(row, list):
            raise ModelLoadError(f"[equations.jacobian].expr[{i}] must be a list")
        row_out: list[str] = []
        for j, entry in enumerate(row):
            if isinstance(entry, (int, float)):
                row_out.append(str(entry))
            elif isinstance(entry, str):
                row_out.append(entry)
            else:
                raise ModelLoadError(
                    f"[equations.jacobian].expr[{i}][{j}] must be a string or number"
                )
        rows.append(row_out)
    return {"expr": rows}


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
        phase = body.get("phase", "post")
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
      model, states, params, constants, equations:{rhs|expr}, aux, functions, events(list), sim
    """
    validate_tables(doc)
    validate_name_collisions(doc)

    model = doc["model"].copy()
    if "dtype" not in model:
        model["dtype"] = "float64"

    # Constants (optional)
    constants_in = doc.get("constants") or {}
    constants = _coerce_constants_table(constants_in) if constants_in else {}

    # Preserve order of declaration
    states_in = doc["states"]
    params_in = doc.get("params", {})  # params is optional, default to empty
    states = _coerce_numeric_table(states_in, "states", allowed_names=constants)
    params = _coerce_numeric_table(params_in, "params", allowed_names=constants) if params_in else {}

    # Equations
    eq_tbl = doc.get("equations") or {}
    rhs_tbl = eq_tbl.get("rhs") or None
    block_expr = eq_tbl.get("expr") or None
    jac_tbl = eq_tbl.get("jacobian") or None
    inv_tbl = eq_tbl.get("inverse") or None
    if rhs_tbl is not None and not isinstance(rhs_tbl, dict):
        raise ModelLoadError("[equations.rhs] must be a table")
    if block_expr is not None and not isinstance(block_expr, str):
        raise ModelLoadError("[equations].expr must be a string")
    if jac_tbl is not None and not isinstance(jac_tbl, dict):
        raise ModelLoadError("[equations.jacobian] must be a table if present")
    if inv_tbl is not None and not isinstance(inv_tbl, dict):
        raise ModelLoadError("[equations.inverse] must be a table if present")
    jacobian = _coerce_jacobian_exprs(jac_tbl) if jac_tbl else None
    inv_rhs_tbl = None
    inv_block_expr = None
    if isinstance(inv_tbl, dict):
        inv_rhs_tbl = inv_tbl.get("rhs") or None
        inv_block_expr = inv_tbl.get("expr") or None
        if inv_rhs_tbl is not None and not isinstance(inv_rhs_tbl, dict):
            raise ModelLoadError("[equations.inverse.rhs] must be a table")
        if inv_block_expr is not None and not isinstance(inv_block_expr, str):
            raise ModelLoadError("[equations.inverse].expr must be a string")

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
        "model": {"type": model["type"], "name": model.get("name"), "dtype": model["dtype"]},
        "states": states,
        "params": params,
        "constants": constants,
        "equations": {
            "rhs": rhs_tbl,
            "expr": block_expr,
            "jacobian": jacobian,
            "inverse": {"rhs": inv_rhs_tbl, "expr": inv_block_expr} if inv_tbl is not None else None,
        },
        "aux": aux_tbl,
        "functions": functions,
        "events": events,
        "presets": presets,
        "sim": sim_tbl,
    }
