# src/dynlib/compiler/codegen/emitter.py
from __future__ import annotations
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass
import re

from dynlib.dsl.spec import ModelSpec, EventSpec
from .rewrite import NameMaps, compile_scalar_expr, sanitize_expr, lower_expr_node

__all__ = ["emit_rhs_and_events", "CompiledCallables"]

# Regex patterns for derivative notation (ODE only)
_DFUNC_PAREN = re.compile(r'^d\(\s*([A-Za-z_]\w*)\s*\)$')
_DFUNC_FLAT = re.compile(r'^d([A-Za-z_]\w*)$')

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
    
    # Case 1: Per-state RHS form
    if spec.equations_rhs:
        for sname, expr in spec.equations_rhs.items():
            idx = nmap.state_to_ix[sname]
            node = lower_expr_node(expr, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
            assign = ast.Assign(
                targets=[ast.Subscript(value=ast.Name(id="dy_out", ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Store())],
                value=node,
            )
            body.append(assign)
    
    # Case 2: Block form
    if spec.equations_block:
        for line in sanitize_expr(spec.equations_block).splitlines():
            line = line.strip()
            if not line:
                continue
            
            # Parse "dx = expr" or "d(x) = expr" or "x = expr" (all valid for ODE)
            # For map: only "x = expr" is valid
            if "=" not in line:
                from dynlib.errors import ModelLoadError
                raise ModelLoadError(f"Block equation line must contain '=': {line!r}")
            
            lhs, rhs = [p.strip() for p in line.split("=", 1)]
            
            # Try to match derivative notation
            m = _DFUNC_PAREN.match(lhs) or _DFUNC_FLAT.match(lhs)
            if m:
                # Looks like derivative notation (d(x) or dx)
                name = m.group(1)
                
                # Only treat as derivative if the name is actually a declared state
                # This prevents 'delta' from being parsed as 'd(elta)'
                if name in nmap.state_to_ix:
                    # It's a real derivative
                    if spec.kind == "map":
                        from dynlib.errors import ModelLoadError
                        raise ModelLoadError(
                            f"Map models do not support derivative notation (d(x) or dx). "
                            f"Use direct assignment (x = expr). Got: {lhs!r} in line: {line!r}"
                        )
                    sname = name
                else:
                    # Pattern matched but not a state - treat as direct assignment
                    # This handles 'delta' matching as 'd' + 'elta' where 'elta' is not a state
                    sname = lhs
                    if sname not in nmap.state_to_ix:
                        from dynlib.errors import ModelLoadError
                        raise ModelLoadError(
                            f"Unknown state in block equation: {sname!r}"
                        )
            else:
                # Direct assignment: x = expr (valid for both ODE and map)
                sname = lhs
                if sname not in nmap.state_to_ix:
                    from dynlib.errors import ModelLoadError
                    raise ModelLoadError(
                        f"Unknown state in block equation: {sname!r}"
                    )
            
            idx = nmap.state_to_ix[sname]
            node = lower_expr_node(rhs, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
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
            Returns event_code (int) if an event fired and has logging enabled, else -1
    """
    import ast
    body: List[ast.stmt] = []
    
    # Assign unique event codes to events in this phase that have logging enabled
    event_code_counter = 0
    for ev_idx, ev in enumerate(spec.events):
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
        
        # If this event has record=True and log items, return its code after mutations
        if ev.record and ev.log:
            # Add return statement with event code after mutations
            act_stmts.append(ast.Return(value=ast.Constant(value=event_code_counter)))
            event_code_counter += 1
        
        body.append(ast.If(test=cond_node, body=act_stmts or [ast.Pass()], orelse=[]))
    
    # Default return -1 (no event fired with logging)
    body.append(ast.Return(value=ast.Constant(value=-1)))

    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name=f"events_{phase}",
                args=ast.arguments(posonlyargs=[], args=[
                    ast.arg(arg="t"), ast.arg(arg="y_vec"), ast.arg(arg="params")
                ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=body if body else [ast.Return(value=ast.Constant(value=-1))],
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
