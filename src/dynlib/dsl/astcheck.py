# src/dynlib/dsl/astcheck.py
from __future__ import annotations
from typing import Dict, Any, Set, List
import re

from dynlib.errors import ModelLoadError
from .constants import BUILTIN_CONSTS, RESERVED_IDENTIFIERS, RUNTIME_RESERVED_NAMES

__all__ = [
    "collect_names",
    "collect_lag_requests",
    "detect_equation_lag_usage",
    "validate_expr_acyclic",
    "validate_event_legality",
    "validate_event_tags",
    "validate_functions_signature",
    "validate_no_duplicate_equation_targets",
    "validate_presets",
    "validate_aux_names",
    "validate_identifier_uniqueness",
    "validate_reserved_identifiers",
    "validate_constants",
    "validate_identifiers_resolved",
    "validate_jacobian_matrix",
]

#NOTE: emitter.py and schema.py also perform the same regex matching;
# this module only uses the patterns to help validate equations.
# emitter.py converts validated equations to AST.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Pattern for valid tag identifiers/slugs: alphanumeric + underscore + hyphen
# Must start with letter or underscore
_TAG_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

# Regex patterns for derivative notation (ODE only)
_DFUNC_PAREN = re.compile(r'^d\(\s*([A-Za-z_]\w*)\s*\)$')
_DFUNC_FLAT = re.compile(r'^d([A-Za-z_]\w*)$')

# Regex pattern for lag notation (lag_<name>() or lag_<name>(k))
_LAG_CALL = re.compile(r'lag_([A-Za-z_]\w*)\s*\(\s*(\d*)\s*\)')
_MACRO_LAG_CALL = re.compile(
    r'\b(cross_up|cross_down|cross_either|changed|enters_interval|leaves_interval|increasing|decreasing)\s*\(\s*([A-Za-z_]\w*)'
)

# Aux identifiers that are reserved for runtime/time symbols and must not be shadowed
_AUX_RESERVED_NAMES = RUNTIME_RESERVED_NAMES


def collect_names(normal: Dict[str, Any]) -> Dict[str, Set[str]]:
    states = set(normal["states"].keys())
    params = set(normal["params"].keys())
    aux = set((normal.get("aux") or {}).keys())
    functions = set((normal.get("functions") or {}).keys())
    events = set(ev["name"] for ev in (normal.get("events") or []))
    constants = set((normal.get("constants") or {}).keys())
    return {
        "states": states,
        "params": params,
        "aux": aux,
        "functions": functions,
        "events": events,
        "constants": constants,
    }


def _collect_free_identifiers(expr: str) -> Set[str]:
    """
    Return identifiers that are *used* in an expression (Load context) but not
    bound by a comprehension target or lambda argument within that expression.
    """
    import ast

    # Keep expression parsing consistent with downstream lowering (support "^" as power)
    expr = expr.strip().replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ModelLoadError(f"Invalid expression syntax: {exc.msg}") from exc

    class _Collector(ast.NodeVisitor):
        def __init__(self):
            self.scope_stack: List[Set[str]] = [set()]
            self.free: Set[str] = set()

        def _is_bound(self, name: str) -> bool:
            return any(name in scope for scope in self.scope_stack)

        def _push(self, names: Set[str] | None = None) -> None:
            self.scope_stack.append(set(names or ()))

        def _pop(self) -> None:
            self.scope_stack.pop()

        def _bind_target(self, target: ast.AST) -> None:
            if isinstance(target, ast.Name):
                self.scope_stack[-1].add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    self._bind_target(elt)
            elif isinstance(target, ast.Starred):
                self._bind_target(target.value)

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load) and not self._is_bound(node.id):
                self.free.add(node.id)

        def visit_Lambda(self, node: ast.Lambda):
            arg_names = {arg.arg for arg in node.args.args}
            self._push(arg_names)
            self.visit(node.body)
            self._pop()

        def _visit_comprehension(self, generators: List[ast.comprehension], visit_elt) -> None:
            def walk(idx: int) -> None:
                if idx == len(generators):
                    visit_elt()
                    return
                gen = generators[idx]
                self.visit(gen.iter)
                self._push()
                self._bind_target(gen.target)
                for cond in gen.ifs:
                    self.visit(cond)
                walk(idx + 1)
                self._pop()

            walk(0)

        def visit_ListComp(self, node: ast.ListComp):
            self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

        def visit_SetComp(self, node: ast.SetComp):
            self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

        def visit_GeneratorExp(self, node: ast.GeneratorExp):
            self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

        def visit_DictComp(self, node: ast.DictComp):
            self._visit_comprehension(node.generators, lambda: (self.visit(node.key), self.visit(node.value)))

        def visit_NamedExpr(self, node: ast.NamedExpr):
            # Walrus assigns within the expression; treat target as bound for the value
            self.visit(node.value)
            self.scope_stack[-1].add(node.target.id)

    collector = _Collector()
    collector.visit(tree)
    return collector.free


def _find_idents(expr: str) -> Set[str]:
    return set(m.group(0) for m in _IDENT.finditer(expr))


def _find_lag_requests(expr: str) -> Dict[str, int]:
    """
    Scan expression for lag_<name>(k) patterns.
    Returns {state_name: max_lag_depth}.
    
    Example: "lag_x() + lag_x(5)" -> {"x": 5}
    """
    lag_depths: Dict[str, int] = {}
    
    # Find all lag_<name>(k) calls
    for match in _LAG_CALL.finditer(expr):
        name = match.group(1)
        depth_str = match.group(2)
        depth = int(depth_str) if depth_str else 1
        if depth < 1:
            raise ModelLoadError(f"Lag depth must be positive, got lag_{name}({depth})")
        if depth > 1000:
            raise ModelLoadError(f"Lag depth {depth} exceeds sanity limit (1000) for lag_{name}")
        lag_depths[name] = max(lag_depths.get(name, 0), depth)
    for match in _MACRO_LAG_CALL.finditer(expr):
        name = match.group(2)
        lag_depths[name] = max(lag_depths.get(name, 0), 1)
    return lag_depths


def collect_lag_requests(normal: Dict[str, Any]) -> Dict[str, int]:
    """
    Scan all expressions in the model for lag notation.
    Returns {state_name: max_lag_depth} for all lagged states.
    
    Validates:
    - Lagged names must be declared states (not params or aux)
    - Lag depths are positive integers within sanity limits
    """
    states = set(normal["states"].keys())
    lag_requests: Dict[str, int] = {}
    
    def merge_requests(expr: str, location: str) -> None:
        if not expr:
            return
        found = _find_lag_requests(expr)
        for name, depth in found.items():
            # Validate that lagged variable is a state
            if name not in states:
                raise ModelLoadError(
                    f"lag_{name}() used in {location}, "
                    f"but '{name}' is not a declared state. "
                    f"Lag notation only applies to state variables."
                )
            lag_requests[name] = max(lag_requests.get(name, 0), depth)
    
    # Scan equations
    eq = normal.get("equations", {})
    if eq.get("rhs"):
        for name, expr in eq["rhs"].items():
            merge_requests(expr, f"[equations.rhs.{name}]")
    if eq.get("expr"):
        merge_requests(eq["expr"], "[equations].expr")
    inv = eq.get("inverse") or {}
    if isinstance(inv, dict):
        if inv.get("rhs"):
            for name, expr in inv["rhs"].items():
                merge_requests(expr, f"[equations.inverse.rhs.{name}]")
        if inv.get("expr"):
            merge_requests(inv["expr"], "[equations.inverse].expr")
    
    # Scan aux
    for name, expr in (normal.get("aux") or {}).items():
        merge_requests(expr, f"[aux.{name}]")
    
    # Scan functions
    for name, fdef in (normal.get("functions") or {}).items():
        merge_requests(fdef.get("expr", ""), f"[functions.{name}].expr")
    
    # Scan events
    for ev in (normal.get("events") or []):
        ev_name = ev["name"]
        merge_requests(ev.get("cond", ""), f"[events.{ev_name}].cond")
        
        if ev.get("action_keyed"):
            for tgt, expr in ev["action_keyed"].items():
                merge_requests(expr, f"[events.{ev_name}].action.{tgt}")
        
        if ev.get("action_block"):
            merge_requests(ev["action_block"], f"[events.{ev_name}].action (block)")

    # Scan sim.stop
    sim = normal.get("sim") or {}
    stop = sim.get("stop")
    if isinstance(stop, dict):
        cond = stop.get("cond")
        if isinstance(cond, str):
            merge_requests(cond, "[sim.stop].cond")
    elif isinstance(stop, str):
        merge_requests(stop, "[sim].stop")
    
    return lag_requests


def detect_equation_lag_usage(normal: Dict[str, Any]) -> bool:
    """
    Determine whether any equation (rhs entry or block expression)
    depends on lag() either directly or through aux/functions that use lag.
    """
    aux_map = normal.get("aux") or {}
    fn_map = normal.get("functions") or {}

    edges = _edges_for_aux_and_functions(normal)
    direct_lag: Dict[str, bool] = {}

    for name, expr in aux_map.items():
        direct_lag[name] = bool(_find_lag_requests(expr))
    for name, fdef in fn_map.items():
        direct_lag[name] = bool(_find_lag_requests(fdef.get("expr", "")))

    memo: Dict[str, bool] = {}

    def _uses_lag(node: str) -> bool:
        if node in memo:
            return memo[node]
        val = direct_lag.get(node, False)
        if not val:
            for dep in edges.get(node, ()):
                if _uses_lag(dep):
                    val = True
                    break
        memo[node] = val
        return val

    aux_with_lag = {name for name in aux_map.keys() if _uses_lag(name)}
    fn_with_lag = {name for name in fn_map.keys() if _uses_lag(name)}

    def _expr_uses_lag(expr: str | None) -> bool:
        if not expr:
            return False
        if _find_lag_requests(expr):
            return True
        used = _find_idents(expr)
        if used & aux_with_lag:
            return True
        if used & fn_with_lag:
            return True
        return False

    eq = normal.get("equations") or {}
    rhs_map = eq.get("rhs") or {}
    for expr in rhs_map.values():
        if _expr_uses_lag(expr):
            return True

    block_expr = eq.get("expr")
    if isinstance(block_expr, str) and _expr_uses_lag(block_expr):
        return True

    return False


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

    # function dependencies (may call other functions or reference aux)
    for f, fdef in (normal.get("functions") or {}).items():
        expr = fdef["expr"]
        used = _find_idents(expr)
        deps = (used & fn_names) | (used & aux_names)
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


def validate_event_tags(normal: Dict[str, Any]) -> None:
    """Validate event tags are well-formed identifiers/slugs.
    
    Tags must:
    - Match pattern: [A-Za-z_][A-Za-z0-9_-]*
    - Not be empty
    
    Note: Duplicates are handled by normalization (deduplication) in build_spec,
    not treated as an error.
    """
    for ev in (normal.get("events") or []):
        name = ev["name"]
        tags = ev.get("tags", [])
        
        if not tags:
            continue
        
        # Validate each tag format
        for tag in tags:
            if not isinstance(tag, str):
                raise ModelLoadError(f"[events.{name}] tag must be a string, got {type(tag).__name__}")
            
            if not tag:
                raise ModelLoadError(f"[events.{name}] tag cannot be empty")
            
            if not _TAG_PATTERN.match(tag):
                raise ModelLoadError(
                    f"[events.{name}] tag '{tag}' is invalid. "
                    f"Tags must start with a letter or underscore and contain only "
                    f"letters, digits, underscores, and hyphens."
                )


def _validate_reserved_names(pool: Set[str], section: str) -> None:
    """
    Raise if any identifiers in `pool` shadow reserved runtime symbols or builtin constants.
    """
    bad = pool & RESERVED_IDENTIFIERS
    if bad:
        bad_list = ", ".join(sorted(bad))
        reserved_list = ", ".join(sorted(RESERVED_IDENTIFIERS))
        raise ModelLoadError(
            f"{section} name(s) {bad_list} are reserved (reserved: {reserved_list})"
        )

def validate_aux_names(normal: Dict[str, Any]) -> None:
    """Reject aux identifiers that shadow reserved runtime symbols or builtin constants."""
    aux_names = set((normal.get("aux") or {}).keys())
    _validate_reserved_names(aux_names, "[aux]")


def validate_constants(normal: Dict[str, Any]) -> None:
    """Validate constant identifiers: reserved words and collisions."""
    const_names = set((normal.get("constants") or {}).keys())
    _validate_reserved_names(const_names, "[constants]")

    collisions = (
        ("states", "state"),
        ("params", "parameter"),
        ("aux", "auxiliary variable"),
    )
    for section, friendly in collisions:
        dup = const_names & set((normal.get(section) or {}).keys())
        if dup:
            name = sorted(dup)[0]
            article = "an" if friendly[0].lower() in {"a", "e", "i", "o", "u"} else "a"
            raise ModelLoadError(
                f"{name} is defined as a constant and cannot be reused as {article} {friendly}."
            )


def validate_reserved_identifiers(normal: Dict[str, Any]) -> None:
    """Reject reserved names across states/params/aux/functions."""
    names = collect_names(normal)
    _validate_reserved_names(names["states"], "[states]")
    _validate_reserved_names(names["params"], "[params]")
    _validate_reserved_names(names["aux"], "[aux]")
    _validate_reserved_names(names["functions"], "[functions]")
    _validate_reserved_names(names["constants"], "[constants]")


def validate_identifier_uniqueness(normal: Dict[str, Any]) -> None:
    """Ensure identifiers are not reused across states/params/aux/functions/constants."""
    names = collect_names(normal)
    section_order = ("states", "params", "aux", "functions", "constants")

    owners: Dict[str, Set[str]] = {}
    for section in section_order:
        for name in names[section]:
            owners.setdefault(name, set()).add(section)

    conflicts = {
        name: [sec for sec in section_order if sec in seen]
        for name, seen in owners.items()
        if len(seen) > 1
    }
    if conflicts:
        detail = "; ".join(f"{name} ({', '.join(sections)})" for name, sections in sorted(conflicts.items()))
        raise ModelLoadError(
            "Identifiers must be unique across states/params/aux/functions/constants; "
            f"conflicts: {detail}"
        )


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


def validate_no_duplicate_equation_targets(normal: Dict[str, Any]) -> None:
    """Ensure states aren't defined in both [equations.rhs] and [equations].expr forms.
    
    Also enforces:
    - Map models must not use derivative notation (d(x) or dx)
    - Derivative notation targets must refer to declared states (prevents 'delta' -> 'd(elta)')
    """
    equations = normal.get("equations", {})
    model_type = normal.get("model", {}).get("type", "ode")
    states = set(normal.get("states", {}).keys())

    # Common typo guard: [equations].expr is the block-form key (singular).
    # If 'exprs' is provided as a string, it would otherwise be ignored and can
    # silently result in missing dynamics.
    if isinstance(equations.get("exprs"), str) and not equations.get("expr"):
        raise ModelLoadError("Invalid equations key: use [equations].expr (singular), not 'exprs'.")
    rhs_dict = equations.get("rhs")
    rhs_targets = set(rhs_dict.keys()) if rhs_dict else set()

    def _block_targets(expr: str | None) -> Set[str]:
        if not expr:
            return set()
        block_targets: Set[str] = set()
        for line in expr.splitlines():
            line = line.strip()
            if not line:
                continue

            if "=" not in line:
                raise ModelLoadError(f"Block equation line must contain '=': {line!r}")

            lhs, rhs = [p.strip() for p in line.split("=", 1)]

            m = _DFUNC_PAREN.match(lhs) or _DFUNC_FLAT.match(lhs)
            if m:
                name = m.group(1)
                if name in states:
                    if model_type == "map":
                        raise ModelLoadError(
                            f"Map models do not support derivative notation (d(x) or dx). "
                            f"Use direct assignment (x = expr). Got: {lhs!r} in line: {line!r}"
                        )
                    block_targets.add(name)
                else:
                    block_targets.add(lhs)
            else:
                block_targets.add(lhs)
        return block_targets

    block_targets = _block_targets(equations.get("expr"))
    overlap = rhs_targets & block_targets
    if overlap:
        raise ModelLoadError(
            f"States defined in both [equations.rhs] and [equations].expr: {sorted(overlap)}"
        )

    inv_tbl = equations.get("inverse") if isinstance(equations.get("inverse"), dict) else None
    if inv_tbl:
        inv_rhs = inv_tbl.get("rhs")
        inv_rhs_targets = set(inv_rhs.keys()) if inv_rhs else set()
        inv_block_targets = _block_targets(inv_tbl.get("expr"))
        inv_overlap = inv_rhs_targets & inv_block_targets
        if inv_overlap:
            raise ModelLoadError(
                "States defined in both [equations.inverse.rhs] and "
                f"[equations.inverse].expr: {sorted(inv_overlap)}"
            )


def validate_presets(normal: Dict[str, Any]) -> None:
    """Validate preset definitions at spec-build time.
    
    Checks:
    - All param keys in preset exist in model params
    - State keys (if present) exist in model states
    - Each preset defines at least one param or state
    
    This catches typos early during model loading instead of waiting until runtime.
    """
    presets = normal.get("presets") or []
    if not presets:
        return
    
    param_names = set(normal["params"].keys())
    state_names = set(normal["states"].keys())
    
    for preset in presets:
        name = preset["name"]
        
        # Validate param keys
        preset_params = set(preset["params"].keys())
        unknown_params = preset_params - param_names
        if unknown_params:
            raise ModelLoadError(
                f"[presets.{name}].params contains unknown parameter(s): {sorted(unknown_params)}. "
                f"Valid params: {sorted(param_names)}"
            )
        
        # Validate state keys (if present)
        preset_states_dict = preset.get("states") or {}
        preset_states = set(preset_states_dict.keys())
        unknown_states = preset_states - state_names
        if unknown_states:
            raise ModelLoadError(
                f"[presets.{name}].states contains unknown state(s): {sorted(unknown_states)}. "
                f"Valid states: {sorted(state_names)}"
            )

        if not preset_params and not preset_states:
            raise ModelLoadError(
                f"[presets.{name}] must define at least one param or state"
            )


def validate_identifiers_resolved(normal: Dict[str, Any]) -> None:
    """
    Ensure every identifier used in an expression can be resolved to a declared
    state/param/aux/function/constant (or a supported builtin/macro).
    """
    names = collect_names(normal)
    states = names["states"]
    params = names["params"]
    aux = names["aux"]
    functions = names["functions"]
    constants = names["constants"]

    allowed_builtins = {
        # math-style functions (rewritten downstream)
        "abs", "min", "max", "round", "exp", "expm1", "log", "log10", "log2", "log1p",
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "floor", "ceil", "trunc", "hypot", "copysign", "erf", "erfc",
        # DSL macros
        "sign", "heaviside", "step", "relu", "clip", "approx",
        "cross_up", "cross_down", "cross_either", "changed",
        "in_interval", "enters_interval", "leaves_interval",
        "increasing", "decreasing",
        # Builtins used in comprehensions/aggregations
        "sum", "prod", "range", "int", "float", "bool", "len",
        # Module prefixes allowed in user expressions
        "math",
    }

    allowed_lag = {f"lag_{s}" for s in states}
    base_allowed = (
        states | params | aux | functions | constants |
        set(RUNTIME_RESERVED_NAMES) | set(BUILTIN_CONSTS.keys())
    )
    base_allowed = base_allowed | allowed_builtins | allowed_lag

    def _check_expr(expr: str, location: str, extra_allowed: Set[str] | None = None) -> None:
        if not expr:
            return
        allowed = base_allowed | (extra_allowed or set())
        free_idents = _collect_free_identifiers(expr)
        unknown = [name for name in free_idents if name not in allowed]
        if unknown:
            missing = ", ".join(sorted(set(unknown)))
            raise ModelLoadError(
                f"Unknown identifier(s) in {location}: {missing}. "
                "Declare them as states/params/aux/functions or constants."
            )

    # functions.<name>.expr
    for fname, fdef in (normal.get("functions") or {}).items():
        args = set(fdef.get("args") or [])
        _check_expr(fdef.get("expr", ""), f"[functions.{fname}].expr", extra_allowed=args)

    # aux.<name>
    for aname, expr in (normal.get("aux") or {}).items():
        _check_expr(expr, f"[aux.{aname}]")

    # equations.rhs entries
    equations = normal.get("equations", {}) or {}
    for sname, expr in (equations.get("rhs") or {}).items():
        _check_expr(expr, f"[equations.rhs.{sname}]")

    # equations.expr block form
    block_expr = equations.get("expr")
    if isinstance(block_expr, str):
        for line in block_expr.splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" not in line:
                continue  # other validators handle this
            _, rhs = [p.strip() for p in line.split("=", 1)]
            _check_expr(rhs, "[equations].expr")

    # equations.inverse
    inv_tbl = equations.get("inverse") or {}
    if isinstance(inv_tbl, dict):
        for sname, expr in (inv_tbl.get("rhs") or {}).items():
            _check_expr(expr, f"[equations.inverse.rhs.{sname}]")
        inv_block_expr = inv_tbl.get("expr")
        if isinstance(inv_block_expr, str):
            for line in inv_block_expr.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "=" not in line:
                    continue
                _, rhs = [p.strip() for p in line.split("=", 1)]
                _check_expr(rhs, "[equations.inverse].expr")

    # events cond + actions
    for ev in (normal.get("events") or []):
        ev_name = ev["name"]
        _check_expr(ev.get("cond", ""), f"[events.{ev_name}].cond")
        if ev.get("action_keyed"):
            for tgt, expr in ev["action_keyed"].items():
                _check_expr(expr, f"[events.{ev_name}].action.{tgt}")
        if ev.get("action_block"):
            for line in ev["action_block"].splitlines():
                line = line.strip()
                if not line:
                    continue
                if "=" not in line:
                    continue
                _, rhs = [p.strip() for p in line.split("=", 1)]
                _check_expr(rhs, f"[events.{ev_name}].action")

    # sim.stop
    sim = normal.get("sim") or {}
    stop = sim.get("stop")
    if isinstance(stop, dict):
        _check_expr(str(stop.get("cond") or ""), "[sim.stop].cond")
    elif isinstance(stop, str):
        _check_expr(stop, "[sim].stop")

    # equations.jacobian entries
    jac_tbl = (normal.get("equations") or {}).get("jacobian") or {}
    for i, row in enumerate(jac_tbl.get("expr", []) or []):
        for j, expr in enumerate(row):
            _check_expr(expr, f"[equations.jacobian].expr[{i}][{j}]")


def validate_jacobian_matrix(normal: Dict[str, Any]) -> None:
    """Validate shape and types of [equations.jacobian]."""
    jac_tbl = (normal.get("equations") or {}).get("jacobian")
    if not jac_tbl:
        return

    exprs = jac_tbl.get("expr") or []
    n_state = len(normal["states"])
    if len(exprs) != n_state:
        raise ModelLoadError(
            f"[equations.jacobian].expr must have {n_state} rows to match [states]; got {len(exprs)}"
        )
    for i, row in enumerate(exprs):
        if len(row) != n_state:
            raise ModelLoadError(
                f"[equations.jacobian].expr[{i}] must have {n_state} columns to match [states]; got {len(row)}"
            )
        for j, expr in enumerate(row):
            if not isinstance(expr, str):
                raise ModelLoadError(
                    f"[equations.jacobian].expr[{i}][{j}] must be a string expression"
                )
