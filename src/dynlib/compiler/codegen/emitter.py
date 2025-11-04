# src/dynlib/compiler/codegen/emitter.py
from __future__ import annotations
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass

from dynlib.dsl.spec import ModelSpec, EventSpec
from .rewrite import NameMaps, compile_scalar_expr, sanitize_expr, lower_expr_node

__all__ = ["emit_rhs_and_events", "CompiledCallables"]

@dataclass(frozen=True)
class CompiledCallables:
    rhs: Callable
    events_pre: Callable
    events_post: Callable

def _state_param_maps(spec: ModelSpec) -> Tuple[Dict[str, int], Dict[str, int]]:
    s2i = {name: i for i, name in enumerate(spec.states)}
    p2i = {name: i for i, name in enumerate(spec.params)}
    return s2i, p2i

def _functions_table(spec: ModelSpec) -> Dict[str, Tuple[Tuple[str, ...], str]]:
    """Return functions as {name: (argnames, expr_str)}."""
    return {fname: (tuple(args), expr) for fname, (args, expr) in (spec.functions or {}).items()}

def _build_name_maps(spec: ModelSpec) -> NameMaps:
    s2i, p2i = _state_param_maps(spec)
    aux_names = tuple((spec.aux or {}).keys())
    funcs = _functions_table(spec)
    return NameMaps(s2i, p2i, aux_names, functions=funcs)

def _aux_defs(spec: ModelSpec) -> Dict[str, str]:
    return dict(spec.aux or {})

def _rhs_items(spec: ModelSpec) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    if spec.equations_rhs:
        for sname, expr in spec.equations_rhs.items():
            # index known; schema already validated names
            items.append((sname, expr))
    # Block form can be added in Slice 3+; for now we honor rhs only in tests.
    return [(list(_build_name_maps(spec).state_to_ix.keys()).index(nm), ex) for nm, ex in items]

def _compile_rhs(spec: ModelSpec, nmap: NameMaps):
    """
    Emit a single function:
        def rhs(t, y_vec, dy_out, params): dy_out[i] = <lowered_expr>; ...
    """
    import ast
    body: List[ast.stmt] = []
    if spec.equations_rhs:
        for sname, expr in spec.equations_rhs.items():
            idx = nmap.state_to_ix[sname]
            node = lower_expr_node(expr, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
            assign = ast.Assign(
                targets=[ast.Subscript(value=ast.Name(id="dy_out", ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Store())],
                value=node,
            )
            body.append(assign)
    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name="rhs",
                args=ast.arguments(posonlyargs=[], args=[
                    ast.arg(arg="t"),
                    ast.arg(arg="y_vec"),
                    ast.arg(arg="dy_out"),
                    ast.arg(arg="params"),
                ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=body if body else [ast.Pass()],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, "<dsl-rhs>", "exec"), ns, ns)
    return ns["rhs"]

def _legal_lhs(name: str, spec: ModelSpec) -> Tuple[str, int, str]:
    if name in spec.states:
        return ("state", spec.states.index(name), name)
    if name in spec.params:
        return ("param", spec.params.index(name), name)
    raise ValueError(f"Illegal assignment target in event action: {name!r} (only states/params are assignable)")

def _compile_action_block_ast(block_lines: List[Tuple[str, str]], spec: ModelSpec, nmap: NameMaps):
    """Return a list of AST statements that mutate y_vec/params in place."""
    import ast
    stmts: List[ast.stmt] = []
    for lhs, rhs_expr in block_lines:
        kind, ix, _ = _legal_lhs(lhs, spec)
        rhs_node = lower_expr_node(rhs_expr, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
        target = ast.Subscript(value=ast.Name(id="y_vec" if kind == "state" else "params", ctx=ast.Load()),
                               slice=ast.Constant(value=ix), ctx=ast.Store())
        stmts.append(ast.Assign(targets=[target], value=rhs_node))
    return stmts

def _emit_events_function(spec: ModelSpec, phase: str, nmap: NameMaps):
    """
    Emit a single function:
        def events_phase(t, y_vec, params):
            if <cond>: <mutations> ; (in declared order)
    """
    import ast
    body: List[ast.stmt] = []
    for ev in spec.events:
        if ev.phase not in ("both", phase):
            continue
        cond_node = lower_expr_node(ev.cond or "1", nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
        # collect actions (keyed or block)
        actions: List[Tuple[str, str]] = []
        if ev.action_keyed:
            actions = list(ev.action_keyed.items())
        elif ev.action_block:
            for line in sanitize_expr(ev.action_block).splitlines():
                line = line.strip()
                if not line:
                    continue
                lhs, rhs = [p.strip() for p in line.split("=", 1)]
                actions.append((lhs, rhs))
        act_stmts = _compile_action_block_ast(actions, spec, nmap)
        body.append(ast.If(test=cond_node, body=act_stmts or [ast.Pass()], orelse=[]))

    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name=f"events_{phase}",
                args=ast.arguments(posonlyargs=[], args=[
                    ast.arg(arg="t"), ast.arg(arg="y_vec"), ast.arg(arg="params")
                ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=body if body else [ast.Pass()],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, f"<dsl-events-{phase}>", "exec"), ns, ns)
    return ns[f"events_{phase}"]

def emit_rhs_and_events(spec: ModelSpec) -> CompiledCallables:
    nmap = _build_name_maps(spec)
    rhs = _compile_rhs(spec, nmap)
    events_pre = _emit_events_function(spec, "pre", nmap)
    events_post = _emit_events_function(spec, "post", nmap)

    return CompiledCallables(rhs=rhs, events_pre=events_pre, events_post=events_post)
