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
    priority: int = 0
    # verbs payloads are normalized dicts mirroring parser output shapes
    remove: Dict[str, Any] | None = None
    replace: Dict[str, Any] | None = None
    add: Dict[str, Any] | None = None
    set: Dict[str, Any] | None = None


# ---- helpers ----------------------------------------------------------------

def _events_index(events: List[Dict[str, Any]]) -> Dict[str, int]:
    return {ev["name"]: i for i, ev in enumerate(events)}


def _apply_remove(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    names = payload.get("events", {}).get("names", []) if payload else []
    if names:
        idx = _events_index(normal["events"]) if normal.get("events") else {}
        keep: List[Dict[str, Any]] = []
        for ev in normal.get("events", []):
            if ev["name"] not in names:
                keep.append(ev)
        normal["events"] = keep


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
    record = bool(body.get("record", False))
    log = list(body.get("log", []) or [])
    return {
        "name": name,
        "phase": phase,
        "cond": cond,
        "action_keyed": action_keyed,
        "action_block": action_block,
        "record": record,
        "log": log,
    }


def _apply_replace(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    repl = payload.get("events", {}) if payload else {}
    if repl:
        idx = _events_index(normal.get("events", []))
        for name, body in repl.items():
            if name not in idx:
                raise ModelLoadError(f"replace.events.{name}: event does not exist")
            normal["events"][idx[name]] = _normalize_event(name, body)


def _apply_add(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    add = payload.get("events", {}) if payload else {}
    if add:
        existing = set(ev["name"] for ev in normal.get("events", []))
        for name, body in add.items():
            if name in existing:
                raise ModelLoadError(f"add.events.{name}: event already exists")
            normal.setdefault("events", []).append(_normalize_event(name, body))


def _apply_set(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not payload:
        return
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


# ---- main -------------------------------------------------------------------

def apply_mods_v2(normal: Dict[str, Any], selected: List[ModSpec]) -> Dict[str, Any]:
    """Apply mods deterministically: remove → replace → add → set.

    Group/exclusive rules:
      - If multiple mods from the same exclusive group are provided, the one with
        the lowest priority (then name) wins; others from that group are ignored.
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

    # Resolve exclusives per group (choose lowest (priority, name)); non-exclusive keep all.
    winners: Dict[str, ModSpec] = {}
    for g, mods in by_group.items():
        exclusives = [m for m in mods if m.exclusive]
        if exclusives:
            winners[g] = sorted(exclusives, key=lambda x: (x.priority, x.name))[0]

    # Final order strictly follows caller order of 'selected',
    # dropping losers from exclusive groups.
    final_order: List[ModSpec] = []
    for m in selected:
        if m.group is None:
            final_order.append(m)
        else:
            if m.group in winners:
                if winners[m.group] is m:
                    final_order.append(m)
            else:
                # non-exclusive group: keep original caller order
                final_order.append(m)

    # Work on a shallow copy of normal
    out = {
        "model": dict(normal["model"]),
        "states": dict(normal["states"]),
        "params": dict(normal["params"]),
        "equations": dict(normal["equations"]),
        "aux": dict(normal.get("aux", {})),
        "functions": dict(normal.get("functions", {})),
        "events": list(normal.get("events", [])),
        "sim": dict(normal.get("sim", {})),
    }

    # Apply verbs per mod in fixed verb order
    for mod in final_order:
        _apply_remove(out, mod.remove or {})
        _apply_replace(out, mod.replace or {})
        _apply_add(out, mod.add or {})
        _apply_set(out, mod.set or {})

    return out