# src/dynlib/compiler/mods.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from dynlib.errors import ModelLoadError

__all__ = [
    "ModSpec",
    "apply_mods_v2",
]


@dataclass(frozen=True)
class ModSpec:
    name: str
    group: str | None = None
    exclusive: bool = False
    # verbs payloads are normalized dicts mirroring parser output shapes
    remove: Dict[str, Any] | None = None
    replace: Dict[str, Any] | None = None
    add: Dict[str, Any] | None = None
    set: Dict[str, Any] | None = None


# ---- helpers ----------------------------------------------------------------

def _events_index(events: List[Dict[str, Any]]) -> Dict[str, int]:
    return {ev["name"]: i for i, ev in enumerate(events)}


def _apply_remove(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not payload:
        return
    
    # Validate supported targets
    allowed_targets = {"events", "params", "aux", "functions"}
    for target in payload.keys():
        if target not in allowed_targets:
            raise ModelLoadError(
                f"remove.{target}: unsupported target. "
                f"Supported targets: {', '.join(sorted(allowed_targets))}"
            )
    
    # remove.events
    names = payload.get("events", {}).get("names", [])
    if names:
        idx = _events_index(normal["events"]) if normal.get("events") else {}
        for name in names:
            if name not in idx:
                raise ModelLoadError(f"remove.events: event '{name}' does not exist")
        keep: List[Dict[str, Any]] = []
        for ev in normal.get("events", []):
            if ev["name"] not in names:
                keep.append(ev)
        normal["events"] = keep
    
    # remove.params
    remove_params = payload.get("params", {}).get("names", [])
    if remove_params:
        existing_params = set(normal.get("params", {}).keys())
        for k in remove_params:
            if k not in existing_params:
                raise ModelLoadError(f"remove.params.{k}: param does not exist")
            del normal["params"][k]
    
    # remove.aux
    remove_aux = payload.get("aux", {}).get("names", [])
    if remove_aux:
        existing_aux = set(normal.get("aux", {}).keys())
        for k in remove_aux:
            if k not in existing_aux:
                raise ModelLoadError(f"remove.aux.{k}: aux does not exist")
            del normal["aux"][k]
    
    # remove.functions
    remove_funcs = payload.get("functions", {}).get("names", [])
    if remove_funcs:
        existing_funcs = set(normal.get("functions", {}).keys())
        for k in remove_funcs:
            if k not in existing_funcs:
                raise ModelLoadError(f"remove.functions.{k}: function does not exist")
            del normal["functions"][k]


def _normalize_event(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal normalization identical to parser output
    phase = body["phase"]
    cond = body["cond"]
    action_keyed = None
    action_block = None
    if "action" in body:
        if isinstance(body.get("action"), str):
            action_block = body.get("action")
        elif isinstance(body.get("action"), dict):
            # nested TOML dict: action = { x = "...", y = "..." }
            ak_dict = body.get("action") or {}
            action_keyed = {k: v for k, v in ak_dict.items()}
        else:
            action_keyed = None
    # dotted fallback: action.x = "..."
    if action_keyed is None:
        ak = {k[7:]: v for k, v in body.items() if k.startswith("action.")}
        action_keyed = ak or None
    
    # Reject deprecated 'record' key
    if "record" in body:
        raise ModelLoadError(
            f"events.{name}.record is no longer supported. "
            f"Use log=['t'] to record event occurrence times."
        )
    
    log = list(body.get("log", []) or [])
    return {
        "name": name,
        "phase": phase,
        "cond": cond,
        "action_keyed": action_keyed,
        "action_block": action_block,
        "log": log,
    }


def _normalize_function(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert TOML function def to internal format."""
    args = body.get("args", [])
    expr = body.get("expr")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ModelLoadError(f"functions.{name}.args must be list of strings")
    if not isinstance(expr, str):
        raise ModelLoadError(f"functions.{name}.expr must be a string")
    return {"args": list(args), "expr": expr}


def _apply_replace(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not payload:
        return
    
    # Validate supported targets
    allowed_targets = {"events", "aux", "functions"}
    for target in payload.keys():
        if target not in allowed_targets:
            raise ModelLoadError(
                f"replace.{target}: unsupported target. "
                f"Supported targets: {', '.join(sorted(allowed_targets))}"
            )
    
    # replace.events
    repl = payload.get("events", {})
    if repl:
        idx = _events_index(normal.get("events", []))
        for name, body in repl.items():
            if name not in idx:
                raise ModelLoadError(f"replace.events.{name}: event does not exist")
            normal["events"][idx[name]] = _normalize_event(name, body)
    
    # replace.aux
    repl_aux = payload.get("aux", {})
    if repl_aux:
        for k, v in repl_aux.items():
            if k not in normal.get("aux", {}):
                raise ModelLoadError(f"replace.aux.{k}: aux does not exist")
            if not isinstance(v, str):
                raise ModelLoadError(f"replace.aux.{k}: value must be a string expression")
            normal["aux"][k] = v
    
    # replace.functions
    repl_funcs = payload.get("functions", {})
    if repl_funcs:
        for fname, fbody in repl_funcs.items():
            if fname not in normal.get("functions", {}):
                raise ModelLoadError(f"replace.functions.{fname}: function does not exist")
            normal["functions"][fname] = _normalize_function(fname, fbody)


def _apply_add(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not payload:
        return
    
    # Validate supported targets
    allowed_targets = {"events", "params", "aux", "functions"}
    for target in payload.keys():
        if target not in allowed_targets:
            raise ModelLoadError(
                f"add.{target}: unsupported target. "
                f"Supported targets: {', '.join(sorted(allowed_targets))}"
            )
    
    # add.events
    add = payload.get("events", {})
    if add:
        existing = set(ev["name"] for ev in normal.get("events", []))
        for name, body in add.items():
            if name in existing:
                raise ModelLoadError(f"add.events.{name}: event already exists")
            normal.setdefault("events", []).append(_normalize_event(name, body))
    
    # add.params
    add_params = payload.get("params", {})
    if add_params:
        existing_params = set(normal.get("params", {}).keys())
        for k, v in add_params.items():
            if k in existing_params:
                raise ModelLoadError(f"add.params.{k}: param already exists")
            normal.setdefault("params", {})[k] = v
    
    # add.aux
    add_aux = payload.get("aux", {})
    if add_aux:
        existing_aux = set(normal.get("aux", {}).keys())
        for k, v in add_aux.items():
            if k in existing_aux:
                raise ModelLoadError(f"add.aux.{k}: aux already exists")
            if not isinstance(v, str):
                raise ModelLoadError(f"add.aux.{k}: value must be a string expression")
            normal.setdefault("aux", {})[k] = v
    
    # add.functions
    add_funcs = payload.get("functions", {})
    if add_funcs:
        existing_funcs = set(normal.get("functions", {}).keys())
        for fname, fbody in add_funcs.items():
            if fname in existing_funcs:
                raise ModelLoadError(f"add.functions.{fname}: function already exists")
            normal.setdefault("functions", {})[fname] = _normalize_function(fname, fbody)


def _apply_set(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not payload:
        return
    allowed_targets = {"states", "params", "aux", "functions", "sim"}
    for target in payload.keys():
        if target not in allowed_targets:
            raise ModelLoadError(
                f"set.{target}: unsupported target. "
                f"Supported targets: {', '.join(sorted(allowed_targets))}"
            )
    # set.states
    s = payload.get("states")
    if isinstance(s, dict):
        for k, v in s.items():
            if k not in normal["states"]:
                raise ModelLoadError(f"set.states.{k}: unknown state")
            normal["states"][k] = v
    # set.params
    p = payload.get("params")
    if isinstance(p, dict):
        for k, v in p.items():
            if k not in normal["params"]:
                raise ModelLoadError(f"set.params.{k}: unknown param")
            normal["params"][k] = v
    # set.aux (upsert semantics - can create or update)
    a = payload.get("aux")
    if isinstance(a, dict):
        for k, v in a.items():
            if not isinstance(v, str):
                raise ModelLoadError(f"set.aux.{k}: value must be a string expression")
            normal.setdefault("aux", {})[k] = v
    # set.functions (upsert semantics - can create or update)
    f = payload.get("functions")
    if isinstance(f, dict):
        for fname, fbody in f.items():
            normal.setdefault("functions", {})[fname] = _normalize_function(fname, fbody)

    # set.sim (upsert semantics - can create or update)
    sim = payload.get("sim")
    if isinstance(sim, dict):
        normal.setdefault("sim", {}).update(sim)


# ---- main -------------------------------------------------------------------

def apply_mods_v2(normal: Dict[str, Any], selected: List[ModSpec]) -> Dict[str, Any]:
    """Apply mods deterministically: remove → replace → add → set.

    Group/exclusive rules:
      - If multiple mods from the same exclusive group are provided, raise an error.
      - Otherwise, mods apply in the *original* provided order.
    """
    # Filter by group exclusivity
    by_group: Dict[str, List[ModSpec]] = {}
    passthrough: List[ModSpec] = []
    for m in selected:
        if m.group:
            by_group.setdefault(m.group, []).append(m)
        else:
            passthrough.append(m)

    # Enforce exclusivity per group
    exclusive_groups: Dict[str, List[ModSpec]] = {}
    for g, mods in by_group.items():
        has_exclusive = any(m.exclusive for m in mods)
        if has_exclusive and len(mods) > 1:
            names = ", ".join(sorted(m.name for m in mods))
            raise ModelLoadError(
                f"group '{g}' allows only one active mod but received: {names}"
            )
        if has_exclusive:
            exclusive_groups[g] = mods

    # Final order strictly follows caller order of 'selected',
    # dropping groups blocked by exclusivity violations (already raised).
    final_order: List[ModSpec] = []
    for m in selected:
        if m.group is None:
            final_order.append(m)
        else:
            if m.group in exclusive_groups:
                # exclusive group validated above; only the sole member is appended
                if exclusive_groups[m.group][0] is m:
                    final_order.append(m)
            else:
                # non-exclusive group: keep original caller order
                final_order.append(m)

    # Work on a shallow copy of normal
    out = {
        "model": dict(normal["model"]),
        "states": dict(normal["states"]),
        "params": dict(normal.get("params", {})),  # params is optional
        "equations": dict(normal["equations"]),
        "aux": dict(normal.get("aux", {})),
        "functions": dict(normal.get("functions", {})),
        "events": list(normal.get("events", [])),
        "sim": dict(normal.get("sim", {})),
        "constants": dict(normal.get("constants", {})),
    }

    # Apply verbs per mod in fixed verb order
    for mod in final_order:
        _apply_remove(out, mod.remove or {})
        _apply_replace(out, mod.replace or {})
        _apply_add(out, mod.add or {})
        _apply_set(out, mod.set or {})

    return out
