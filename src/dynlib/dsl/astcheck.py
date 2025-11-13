# src/dynlib/dsl/astcheck.py
from __future__ import annotations
from typing import Dict, Any, Set, List
import re

from dynlib.errors import ModelLoadError

__all__ = [
    "collect_names",
    "validate_expr_acyclic",
    "validate_event_legality",
    "validate_event_tags",
    "validate_functions_signature",
    "validate_no_duplicate_equation_targets",
    "validate_presets",
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
    rhs_dict = equations.get("rhs")
    rhs_targets = set(rhs_dict.keys()) if rhs_dict else set()
    
    block_expr = equations.get("expr")
    if not block_expr:
        return  # No block form, nothing to check
    
    block_targets: Set[str] = set()
    # Parse block form to extract state names
    for line in block_expr.splitlines():
        line = line.strip()
        if not line:
            continue
        
        # Split on '=' to get LHS and RHS
        if "=" not in line:
            raise ModelLoadError(f"Block equation line must contain '=': {line!r}")
        
        lhs, rhs = [p.strip() for p in line.split("=", 1)]
        
        # Try to match derivative notation
        m = _DFUNC_PAREN.match(lhs) or _DFUNC_FLAT.match(lhs)
        if m:
            # Looks like derivative notation, but verify the extracted name is a state
            name = m.group(1)
            
            # Only treat as derivative if the name is actually a declared state
            if name in states:
                # For map models, derivative notation is not allowed
                if model_type == "map":
                    raise ModelLoadError(
                        f"Map models do not support derivative notation (d(x) or dx). "
                        f"Use direct assignment (x = expr). Got: {lhs!r} in line: {line!r}"
                    )
                block_targets.add(name)
            else:
                # Pattern matched but name not a state - treat as direct assignment
                # This handles cases like 'delta' which matches 'd' + 'elta' but 'elta' is not a state
                block_targets.add(lhs)
        else:
            # Direct assignment (x = expr)
            # For both ODE and map models, this is valid
            block_targets.add(lhs)
    
    # Check for overlap
    overlap = rhs_targets & block_targets
    if overlap:
        raise ModelLoadError(
            f"States defined in both [equations.rhs] and [equations].expr: {sorted(overlap)}"
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
