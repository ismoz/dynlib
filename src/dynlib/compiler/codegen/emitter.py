# src/dynlib/compiler/codegen/emitter.py
from __future__ import annotations
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass
import re

from dynlib.dsl.spec import ModelSpec, EventSpec
from dynlib.dsl.constants import cast_constants
from .rewrite import NameMaps, compile_scalar_expr, sanitize_expr, lower_expr_node, lower_expr_with_preamble

__all__ = ["emit_rhs_and_events", "emit_jacobian", "CompiledCallables", "CompiledJacobian"]

# Regex patterns for derivative notation (ODE only)
_DFUNC_PAREN = re.compile(r'^d\(\s*([A-Za-z_]\w*)\s*\)$')
_DFUNC_FLAT = re.compile(r'^d([A-Za-z_]\w*)$')

@dataclass(frozen=True)
class CompiledCallables:
    rhs: Callable
    events_pre: Callable
    events_post: Callable
    update_aux: Callable
    rhs_source: str
    events_pre_source: str
    events_post_source: str
    update_aux_source: str
    inv_rhs: Callable | None = None
    inv_rhs_source: str | None = None


@dataclass(frozen=True)
class CompiledJacobian:
    jvp: Callable
    jvp_source: str
    jacobian: Callable | None = None
    jacobian_source: str | None = None

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
    consts = cast_constants(spec.dtype, extra=dict(zip(spec.constants, spec.constant_vals)))
    return NameMaps(
        s2i,
        p2i,
        aux_names,
        functions=funcs,
        constants=consts,
        lag_map=spec.lag_map,
    )

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

def _compile_equations(
    spec: ModelSpec,
    nmap: NameMaps,
    rhs_tbl: Dict[str, str] | None,
    block_expr: str | None,
    func_name: str,
):
    """
    Emit a single function:
        def <func_name>(t, y_vec, dy_out, params, runtime_ws): dy_out[i] = <lowered_expr>; ...
    """
    import ast
    body: List[ast.stmt] = []
    
    # Case 1: Per-state RHS form
    if rhs_tbl:
        for sname, expr in rhs_tbl.items():
            idx = nmap.state_to_ix[sname]
            preamble, node = lower_expr_with_preamble(expr, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
            assign = ast.Assign(
                targets=[ast.Subscript(value=ast.Name(id="dy_out", ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Store())],
                value=node,
            )
            body.extend(preamble)
            body.append(assign)
    
    # Case 2: Block form
    if block_expr:
        for line in sanitize_expr(block_expr).splitlines():
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
            preamble, node = lower_expr_with_preamble(rhs, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
            assign = ast.Assign(
                targets=[ast.Subscript(value=ast.Name(id="dy_out", ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Store())],
                value=node,
            )
            body.extend(preamble)
            body.append(assign)
    
    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name=func_name,
                args=ast.arguments(posonlyargs=[], args=[
                    ast.arg(arg="t"),
                    ast.arg(arg="y_vec"),
                    ast.arg(arg="dy_out"),
                    ast.arg(arg="params"),
                    ast.arg(arg="runtime_ws"),
                ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=body if body else [ast.Pass()],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    module_source = ast.unparse(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, "<dsl-rhs>", "exec"), ns, ns)
    return ns[func_name], module_source


def _compile_rhs(spec: ModelSpec, nmap: NameMaps):
    return _compile_equations(
        spec,
        nmap,
        spec.equations_rhs,
        spec.equations_block,
        "rhs",
    )


def _compile_inv_rhs(spec: ModelSpec, nmap: NameMaps):
    if not spec.inverse_rhs and not spec.inverse_block:
        return None, None
    return _compile_equations(
        spec,
        nmap,
        spec.inverse_rhs,
        spec.inverse_block,
        "inv_rhs",
    )


def _compile_jvp(spec: ModelSpec, nmap: NameMaps):
    """Emit a JVP operator from a dense Jacobian declaration."""
    import ast

    aux_defs = _aux_defs(spec)
    fn_defs = nmap.functions
    body: List[ast.stmt] = []
    # Hoist aux once per call to avoid re-evaluating inside each entry
    for aux_name, expr in aux_defs.items():
        # Compute aux once with full inlining to avoid order-dependency issues,
        # then reuse the hoisted locals in the Jacobian body.
        preamble, node = lower_expr_with_preamble(
            expr,
            nmap,
            aux_defs=aux_defs,
            fn_defs=fn_defs,
            hoist_aux=False,
        )
        body.extend(preamble)
        body.append(
            ast.Assign(
                targets=[ast.Name(id=aux_name, ctx=ast.Store())],
                value=node,
            )
        )

    for i, row in enumerate(spec.jacobian_exprs or ()):
        acc_name = f"_acc_{i}"
        body.append(
            ast.Assign(targets=[ast.Name(id=acc_name, ctx=ast.Store())], value=ast.Constant(value=0.0))
        )
        for j, expr in enumerate(row):
            preamble, node = lower_expr_with_preamble(
                expr,
                nmap,
                aux_defs=aux_defs,
                fn_defs=fn_defs,
                hoist_aux=True,
            )
            body.extend(preamble)
            term = ast.BinOp(
                left=node,
                op=ast.Mult(),
                right=ast.Subscript(
                    value=ast.Name(id="v_in", ctx=ast.Load()),
                    slice=ast.Constant(value=j),
                    ctx=ast.Load(),
                ),
            )
            body.append(
                ast.AugAssign(
                    target=ast.Name(id=acc_name, ctx=ast.Store()),
                    op=ast.Add(),
                    value=term,
                )
            )
        body.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id="v_out", ctx=ast.Load()),
                        slice=ast.Constant(value=i),
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Name(id=acc_name, ctx=ast.Load()),
            )
        )

    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name="jvp",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg="t"),
                        ast.arg(arg="y_vec"),
                        ast.arg(arg="params"),
                        ast.arg(arg="v_in"),
                        ast.arg(arg="v_out"),
                        ast.arg(arg="runtime_ws"),
                    ],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=body if body else [ast.Pass()],
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    module_source = ast.unparse(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, "<dsl-jvp>", "exec"), ns, ns)
    return ns["jvp"], module_source


def _compile_jacobian_fill(spec: ModelSpec, nmap: NameMaps):
    """Emit a dense Jacobian filler: J_out[i, j] = expr."""
    import ast

    aux_defs = _aux_defs(spec)
    fn_defs = nmap.functions
    body: List[ast.stmt] = []

    # Hoist aux computations once per call and reuse in entries
    for aux_name, expr in aux_defs.items():
        # Compute aux once with inlining (order-independent), then reuse locals.
        preamble, node = lower_expr_with_preamble(
            expr,
            nmap,
            aux_defs=aux_defs,
            fn_defs=fn_defs,
            hoist_aux=False,
        )
        body.extend(preamble)
        body.append(
            ast.Assign(
                targets=[ast.Name(id=aux_name, ctx=ast.Store())],
                value=node,
            )
        )

    for i, row in enumerate(spec.jacobian_exprs or ()):
        for j, expr in enumerate(row):
            preamble, node = lower_expr_with_preamble(
                expr,
                nmap,
                aux_defs=aux_defs,
                fn_defs=fn_defs,
                hoist_aux=True,
            )
            body.extend(preamble)
            body.append(
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id="J_out", ctx=ast.Load()),
                            slice=ast.Tuple(
                                elts=[ast.Constant(value=i), ast.Constant(value=j)],
                                ctx=ast.Load(),
                            ),
                            ctx=ast.Store(),
                        )
                    ],
                    value=node,
                )
            )

    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name="jacobian",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg="t"),
                        ast.arg(arg="y_vec"),
                        ast.arg(arg="params"),
                        ast.arg(arg="J_out"),
                        ast.arg(arg="runtime_ws"),
                    ],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=body if body else [ast.Pass()],
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    module_source = ast.unparse(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, "<dsl-jacobian>", "exec"), ns, ns)
    return ns["jacobian"], module_source


def emit_jacobian(spec: ModelSpec) -> CompiledJacobian | None:
    """Emit JVP (and dense fill) from a DSL-declared Jacobian."""
    if not spec.jacobian_exprs:
        return None
    nmap = _build_name_maps(spec)
    jvp_fn, jvp_src = _compile_jvp(spec, nmap)
    jac_fn, jac_src = _compile_jacobian_fill(spec, nmap)
    return CompiledJacobian(
        jvp=jvp_fn,
        jvp_source=jvp_src,
        jacobian=jac_fn,
        jacobian_source=jac_src,
    )

def _legal_lhs(name: str, spec: ModelSpec) -> Tuple[str, int, str]:
    if name in spec.states:
        return ("state", spec.states.index(name), name)
    if name in spec.params:
        return ("param", spec.params.index(name), name)
    raise ValueError(f"Illegal assignment target in event action: {name!r} (only states/params are assignable)")

def _parse_log_signal(signal: str, nmap: NameMaps, spec: ModelSpec) -> str:
    """
    Parse a log signal specification and return the expression to evaluate.
    
    Formats:
      - "x"         → state x
      - "param:a"   → parameter a
      - "aux:E"     → auxiliary variable E
      - "t"         → time (special case)
    
    Returns an expression string that can be lowered.
    """
    signal = signal.strip()
    
    # Special case: time
    if signal == "t":
        return "t"
    
    # Check for prefix notation
    if ":" in signal:
        prefix, name = signal.split(":", 1)
        prefix = prefix.strip()
        name = name.strip()
        
        if prefix == "param":
            if name not in spec.params:
                raise ValueError(f"Unknown parameter in log signal: {name!r}")
            return name  # Will be resolved as parameter by lower_expr_node
        elif prefix == "aux":
            if name not in (spec.aux or {}):
                raise ValueError(f"Unknown auxiliary variable in log signal: {name!r}")
            return name  # Will be resolved as aux by lower_expr_node
        elif prefix == "state":
            if name not in spec.states:
                raise ValueError(f"Unknown state in log signal: {name!r}")
            return name
        else:
            raise ValueError(f"Unknown log signal prefix: {prefix!r} (use 'state:', 'param:', or 'aux:')")
    
    # No prefix: assume it's a state, param, or aux (in that priority order)
    if signal in spec.states:
        return signal
    elif signal in spec.params:
        return signal
    elif signal in (spec.aux or {}):
        return signal
    else:
        raise ValueError(f"Unknown signal in log specification: {signal!r}")

def _compile_action_block_ast(block_lines: List[Tuple[str, str]], spec: ModelSpec, nmap: NameMaps):
    """Return a list of AST statements that mutate y_vec/params in place."""
    import ast
    stmts: List[ast.stmt] = []
    for lhs, rhs_expr in block_lines:
        kind, ix, _ = _legal_lhs(lhs, spec)
        preamble, rhs_node = lower_expr_with_preamble(rhs_expr, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
        target = ast.Subscript(value=ast.Name(id="y_vec" if kind == "state" else "params", ctx=ast.Load()),
                               slice=ast.Constant(value=ix), ctx=ast.Store())
        stmts.extend(preamble)
        stmts.append(ast.Assign(targets=[target], value=rhs_node))
    return stmts

def _emit_events_function(spec: ModelSpec, phase: str, nmap: NameMaps):
    """
    Emit a single function:
        def events_phase(t, y_vec, params, evt_log_scratch, runtime_ws):
            if <cond>: 
                <mutations>
                [fill evt_log_scratch with log values if log is non-empty]
                return (event_code, log_width)
            ...
            return (-1, 0)  # no event fired
    
    Where:
      - event_code: unique int identifying which event fired (0, 1, 2...)
      - log_width: number of values written to evt_log_scratch (len(ev.log))
    
    All log values (including "t" if present) are written to EVT_LOG_DATA.
    The "t" signal is treated like any other signal - no special EVT_TIME buffer.
    """
    import ast
    body: List[ast.stmt] = []
    
    # Assign unique event codes to events in this phase
    event_code_counter = 0
    for ev_idx, ev in enumerate(spec.events):
        if ev.phase not in ("both", phase):
            continue
        
        cond_preamble, cond_node = lower_expr_with_preamble(ev.cond or "1", nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
        
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
        
        # Generate log assignments if this event has log items
        log_width = len(ev.log) if ev.log else 0
        if log_width > 0:
            for log_idx, log_signal in enumerate(ev.log):
                # Parse log signal: "x", "param:a", "aux:E", etc.
                log_expr = _parse_log_signal(log_signal, nmap, spec)
                log_preamble, log_node = lower_expr_with_preamble(log_expr, nmap, aux_defs=_aux_defs(spec), fn_defs=nmap.functions)
                
                # evt_log_scratch[log_idx] = <value>
                assign = ast.Assign(
                    targets=[ast.Subscript(
                        value=ast.Name(id="evt_log_scratch", ctx=ast.Load()),
                        slice=ast.Constant(value=log_idx),
                        ctx=ast.Store()
                    )],
                    value=log_node,
                )
                act_stmts.extend(log_preamble)
                act_stmts.append(assign)
        
        # Return (event_code, log_width)
        act_stmts.append(ast.Return(value=ast.Tuple(elts=[
            ast.Constant(value=event_code_counter),
            ast.Constant(value=log_width),
        ], ctx=ast.Load())))
        
        event_code_counter += 1
        
        body.extend(cond_preamble)
        body.append(ast.If(test=cond_node, body=act_stmts or [ast.Pass()], orelse=[]))
    
    # Default return (-1, 0) - no event fired
    body.append(ast.Return(value=ast.Tuple(elts=[
        ast.Constant(value=-1),
        ast.Constant(value=0),
    ], ctx=ast.Load())))

    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name=f"events_{phase}",
                args=ast.arguments(posonlyargs=[], args=[
                    ast.arg(arg="t"), 
                    ast.arg(arg="y_vec"), 
                    ast.arg(arg="params"),
                    ast.arg(arg="evt_log_scratch"),
                    ast.arg(arg="runtime_ws"),
                ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=body if body else [ast.Return(value=ast.Tuple(elts=[
                    ast.Constant(value=-1),
                    ast.Constant(value=0),
                ], ctx=ast.Load()))],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    module_source = ast.unparse(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, f"<dsl-events-{phase}>", "exec"), ns, ns)
    return ns[f"events_{phase}"], module_source

def _compile_update_aux(spec: ModelSpec, nmap: NameMaps):
    """
    Emit a function that computes all aux values from current state:
        def update_aux(t, y_vec, params, aux_out, runtime_ws):
            aux_out[0] = <lowered_expr_0>
            aux_out[1] = <lowered_expr_1>
            ...
    
    If no aux variables exist, returns a no-op function.
    """
    import ast
    aux_defs = _aux_defs(spec)
    stop_spec = getattr(getattr(spec, "sim", None), "stop", None)
    stop_expr = getattr(stop_spec, "cond", None) if stop_spec is not None else None

    if not aux_defs and not stop_expr:
        # No aux variables and no stop condition - return a no-op function
        mod = ast.Module(
            body=[
                ast.Import(names=[ast.alias(name="math", asname=None)]),
                ast.FunctionDef(
                    name="update_aux",
                    args=ast.arguments(posonlyargs=[], args=[
                        ast.arg(arg="t"),
                        ast.arg(arg="y_vec"),
                        ast.arg(arg="params"),
                        ast.arg(arg="aux_out"),
                        ast.arg(arg="runtime_ws"),
                    ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                    body=[ast.Pass()],
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(mod)
        module_source = ast.unparse(mod)
        ns: Dict[str, object] = {}
        exec(compile(mod, "<dsl-update-aux>", "exec"), ns, ns)
        return ns["update_aux"], module_source
    
    # Build assignment statements for each aux variable
    body: List[ast.stmt] = []
    for idx, (name, expr) in enumerate(aux_defs.items()):
        # Lower the aux expression (may reference states, params, other aux)
        preamble, node = lower_expr_with_preamble(
            expr, nmap, aux_defs=aux_defs, fn_defs=nmap.functions
        )
        assign = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id="aux_out", ctx=ast.Load()),
                slice=ast.Constant(value=idx),
                ctx=ast.Store()
            )],
            value=node,
        )
        body.extend(preamble)
        body.append(assign)

    # Optional stop flag update (numba-friendly): runtime_ws.stop_flag[0] = 1/0
    if stop_expr:
        stop_preamble, stop_node = lower_expr_with_preamble(
            stop_expr, nmap, aux_defs=aux_defs, fn_defs=nmap.functions
        )
        stop_assign = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Attribute(value=ast.Name(id="runtime_ws", ctx=ast.Load()), attr="stop_flag", ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=ast.Store(),
            )],
            value=ast.IfExp(
                test=stop_node,
                body=ast.Constant(value=1),
                orelse=ast.Constant(value=0),
            ),
        )
        body.extend(stop_preamble)
        body.append(
            ast.If(
                test=ast.Compare(
                    left=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="runtime_ws", ctx=ast.Load()),
                            attr="stop_flag",
                            ctx=ast.Load(),
                        ),
                        attr="size",
                        ctx=ast.Load(),
                    ),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[stop_assign],
                orelse=[],
            )
        )
    
    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name="update_aux",
                args=ast.arguments(posonlyargs=[], args=[
                    ast.arg(arg="t"),
                    ast.arg(arg="y_vec"),
                    ast.arg(arg="params"),
                    ast.arg(arg="aux_out"),
                    ast.arg(arg="runtime_ws"),
                ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=body,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    module_source = ast.unparse(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, "<dsl-update-aux>", "exec"), ns, ns)
    return ns["update_aux"], module_source

def emit_rhs_and_events(spec: ModelSpec) -> CompiledCallables:
    nmap = _build_name_maps(spec)
    rhs_fn, rhs_src = _compile_rhs(spec, nmap)
    inv_rhs_fn, inv_rhs_src = _compile_inv_rhs(spec, nmap)
    events_pre_fn, events_pre_src = _emit_events_function(spec, "pre", nmap)
    events_post_fn, events_post_src = _emit_events_function(spec, "post", nmap)
    update_aux_fn, update_aux_src = _compile_update_aux(spec, nmap)

    return CompiledCallables(
        rhs=rhs_fn,
        inv_rhs=inv_rhs_fn,
        events_pre=events_pre_fn,
        events_post=events_post_fn,
        update_aux=update_aux_fn,
        rhs_source=rhs_src,
        inv_rhs_source=inv_rhs_src,
        events_pre_source=events_pre_src,
        events_post_source=events_post_src,
        update_aux_source=update_aux_src,
    )
