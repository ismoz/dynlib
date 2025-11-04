# src/dynlib/dsl/astcheck.py
from __future__ import annotations
from typing import Dict, Any, Set, List
import re

from dynlib.errors import ModelLoadError

__all__ = [
    "collect_names",
    "validate_expr_acyclic",
    "validate_equation_targets",
    "validate_event_legality",
    "validate_dtype_rules",
    "validate_functions_signature",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def collect_names(normal: Dict[str, Any]) -> Dict[str, Set[str]]:
    states = set(normal["states"].keys())
    params = set(normal["params"].keys())
    aux = set((normal.get("aux") or {}).keys())
    functions = set((normal.get("functions") or {}).keys())
    events = set(ev["name"] for ev in (normal.get("events") or []))
    return {
        "states": states,
        "params": params,
        "aux": aux,
        "functions": functions,
        "events": events,
    }


def _find_idents(expr: str) -> Set[str]:
    return set(m.group(0) for m in _IDENT.finditer(expr))


def _edges_for_aux_and_functions(normal: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build a conservative dependency map among aux and functions.

    Node set: aux names and function names.
    Edge A->B means A depends on B.
    """
    names = collect_names(normal)
    aux_names = names["aux"]
    fn_names = names["functions"]

    edges: Dict[str, Set[str]] = {n: set() for n in aux_names | fn_names}

    # aux dependencies (on states/params/aux/functions)
    for a, expr in (normal.get("aux") or {}).items():
        used = _find_idents(expr)
        deps = (used & aux_names) | (used & fn_names)
        if deps:
            edges[a].update(deps)

    # function dependencies (may call other functions)
    for f, fdef in (normal.get("functions") or {}).items():
        expr = fdef["expr"]
        used = _find_idents(expr)
        deps = used & fn_names
        if deps:
            edges[f].update(deps)
    return edges


def _dfs_cycle_check(graph: Dict[str, Set[str]]) -> None:
    temp: Set[str] = set()
    perm: Set[str] = set()

    def visit(n: str) -> None:
        if n in perm:
            return
        if n in temp:
            raise ModelLoadError(f"Cyclic dependency detected involving '{n}'")
        temp.add(n)
        for m in graph.get(n, ()):  # iterate deps
            visit(m)
        temp.remove(n)
        perm.add(n)

    for node in graph.keys():
        if node not in perm:
            visit(node)


def validate_expr_acyclic(normal: Dict[str, Any]) -> None:
    graph = _edges_for_aux_and_functions(normal)
    _dfs_cycle_check(graph)


def validate_equation_targets(normal: Dict[str, Any]) -> None:
    states = set(normal["states"].keys())
    rhs = normal["equations"].get("rhs") or {}
    expr = normal["equations"].get("expr")

    # Collect block targets
    block_targets: Set[str] = set()
    if isinstance(expr, str):
        for line in expr.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                lhs = line.split("=", 1)[0].strip()
                if lhs:
                    block_targets.add(lhs)

    rhs_targets = set(rhs.keys())
    all_targets = rhs_targets | block_targets

    unknown = all_targets - states
    if unknown:
        raise ModelLoadError(f"Equation targets must be declared in [states], unknown: {sorted(unknown)}")

    dup = rhs_targets & block_targets
    if dup:
        raise ModelLoadError(f"Duplicate equation targets across rhs and block: {sorted(dup)}")


def validate_event_legality(normal: Dict[str, Any]) -> None:
    states = set(normal["states"].keys())
    params = set(normal["params"].keys())

    for ev in (normal.get("events") or []):
        name = ev["name"]
        ak = ev.get("action_keyed")
        if ak:
            illegal = [t for t in ak.keys() if t not in states and t not in params]
            if illegal:
                raise ModelLoadError(
                    f"[events.{name}] may mutate only states/params; illegal: {illegal}"
                )
        # action_block legality is deferred to codegen parsing; enforced here by absence of aux/buffer names is not trivial.


def validate_dtype_rules(normal: Dict[str, Any]) -> None:
    model = normal["model"]
    mtype = model["type"]
    dtype = model.get("dtype", "float64")
    if mtype == "ode":
        if dtype not in {"float32", "float64", "float16", "bfloat16"}:
            raise ModelLoadError("ODE models require a floating dtype (float32/float64/float16/bfloat16)")


def validate_functions_signature(normal: Dict[str, Any]) -> None:
    for name, fdef in (normal.get("functions") or {}).items():
        args = fdef.get("args") or []
        expr = fdef.get("expr")
        if not isinstance(expr, str):
            raise ModelLoadError(f"[functions.{name}].expr must be a string")
        if not isinstance(args, list) or not all(isinstance(a, str) and _IDENT.fullmatch(a) for a in args):
            raise ModelLoadError(f"[functions.{name}].args must be a list of identifiers")
        if len(set(args)) != len(args):
            raise ModelLoadError(f"[functions.{name}].args must be unique")