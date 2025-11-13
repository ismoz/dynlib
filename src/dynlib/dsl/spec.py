# src/dynlib/dsl/spec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json
import hashlib

__all__ = [
    "SimDefaults",
    "EventSpec",
    "PresetSpec",
    "ModelSpec",
    "build_spec",
    "compute_spec_hash",
]


@dataclass(frozen=True)
class PresetSpec:
    """A preset defined in the model DSL."""
    name: str
    params: Dict[str, float | int]
    states: Dict[str, float | int] | None


@dataclass(frozen=True)
class SimDefaults:
    t0: float = 0.0
    t_end: float = 1.0
    dt: float = 1e-2
    stepper: str = "rk4"
    record: bool = True
    # Adaptive stepper tolerances (used by RK45, etc.)
    atol: float = 1e-8
    rtol: float = 1e-5


@dataclass(frozen=True)
class EventSpec:
    name: str
    phase: str  # "pre" | "post" | "both"
    cond: str
    action_keyed: Dict[str, str] | None
    action_block: str | None
    log: Tuple[str, ...]
    tags: Tuple[str, ...]  # compile-time only, order-stable, canonical


@dataclass(frozen=True)
class ModelSpec:
    kind: str                 # "ode" | "map"
    label: str | None
    dtype: str               # canonical dtype string
    states: Tuple[str, ...]   # ordered
    state_ic: Tuple[float | int, ...]
    params: Tuple[str, ...]
    param_vals: Tuple[float | int, ...]
    equations_rhs: Dict[str, str] | None
    equations_block: str | None
    aux: Dict[str, str]
    functions: Dict[str, tuple[list[str], str]]
    events: Tuple[EventSpec, ...]
    sim: SimDefaults
    tag_index: Dict[str, Tuple[str, ...]]  # tag -> event names, compile-time only
    presets: Tuple[PresetSpec, ...]  # inline presets from DSL
    lag_map: Dict[str, Tuple[int, int, int]]  # state_name -> (max_depth, ss_offset, iw0_index)


# ---- builders ----------------------------------------------------------------

def _canon_dtype(dtype: str) -> str:
    # Keep as-is but normalize common aliases
    alias = {
        "float": "float64",
        "double": "float64",
        "single": "float32",
    }
    return alias.get(dtype, dtype)


def _build_tag_index(events: Tuple[EventSpec, ...]) -> Dict[str, Tuple[str, ...]]:
    """Build a reverse index: tag -> tuple of event names.
    
    Returns a dict where each tag maps to an ordered tuple of event names
    that have that tag. Event names maintain their declaration order.
    """
    index: Dict[str, list[str]] = {}
    for event in events:
        for tag in event.tags:
            if tag not in index:
                index[tag] = []
            index[tag].append(event.name)
    # Convert lists to tuples for immutability
    return {tag: tuple(names) for tag, names in index.items()}


def build_spec(normal: Dict[str, Any]) -> ModelSpec:
    # VALIDATE BEFORE PARSING
    from .astcheck import (
        validate_expr_acyclic,
        validate_event_legality,
        validate_event_tags,
        validate_functions_signature,
        validate_no_duplicate_equation_targets,
        validate_presets,
        collect_lag_requests,
    )
    
    validate_expr_acyclic(normal)
    validate_event_legality(normal)
    validate_event_tags(normal)
    validate_functions_signature(normal)
    validate_no_duplicate_equation_targets(normal)
    validate_presets(normal)
    
    # Collect lag requests from all expressions
    lag_requests = collect_lag_requests(normal)
    
    model = normal["model"]
    kind = model["type"]
    dtype = _canon_dtype(model.get("dtype", "float64"))
    label = model.get("label")

    states = tuple(normal["states"].keys())
    state_ic = tuple(normal["states"].values())

    params = tuple(normal["params"].keys())
    param_vals = tuple(normal["params"].values())

    eq = normal["equations"]
    eq_rhs = dict(eq["rhs"]) if eq.get("rhs") else None
    eq_block = eq.get("expr") if isinstance(eq.get("expr"), str) else None

    aux = dict(normal.get("aux", {}))
    functions = {k: (v["args"], v["expr"]) for k, v in (normal.get("functions") or {}).items()}

    events = tuple(
        EventSpec(
            name=e["name"],
            phase=e["phase"],
            cond=e["cond"],
            action_keyed=dict(e["action_keyed"]) if e.get("action_keyed") else None,
            action_block=e.get("action_block"),
            log=tuple(e.get("log", []) or []),
            tags=tuple(sorted(set(e.get("tags", []) or []))),  # normalize: dedupe & sort
        )
        for e in (normal.get("events") or [])
    )

    sim_in = normal.get("sim", {})
    sim = SimDefaults(
        t0=float(sim_in.get("t0", SimDefaults.t0)),
        t_end=float(sim_in.get("t_end", SimDefaults.t_end)),
        dt=float(sim_in.get("dt", SimDefaults.dt)),
        stepper=str(sim_in.get("stepper", SimDefaults.stepper)),
        record=bool(sim_in.get("record", SimDefaults.record)),
        atol=float(sim_in.get("atol", SimDefaults.atol)),
        rtol=float(sim_in.get("rtol", SimDefaults.rtol)),
    )

    presets = tuple(
        PresetSpec(
            name=p["name"],
            params=dict(p["params"]),
            states=dict(p["states"]) if p.get("states") else None,
        )
        for p in (normal.get("presets") or [])
    )
    
    # Build lag_map: state_name -> (buffer_len, ss_offset, iw0_index)
    # buffer_len = max_requested_lag + 1 (extra slot preserves current head)
    # ss_offset is the starting lane in ss for this state's circular buffer
    # iw0_index is the slot in iw0 for this state's head pointer
    lag_map: Dict[str, Tuple[int, int, int]] = {}
    ss_offset = 0
    iw0_index = 0

    # Process lagged states in the order they appear in states tuple (deterministic)
    for state_name in states:
        if state_name in lag_requests:
            requested_depth = lag_requests[state_name]
            buffer_len = requested_depth + 1  # extra slot ensures lag_depth indexing works
            lag_map[state_name] = (buffer_len, ss_offset, iw0_index)
            ss_offset += buffer_len  # each lagged state gets buffer_len lanes
            iw0_index += 1           # each lagged state gets one iw0 slot

    return ModelSpec(
        kind=kind,
        label=label,
        dtype=dtype,
        states=states,
        state_ic=state_ic,
        params=params,
        param_vals=param_vals,
        equations_rhs=eq_rhs,
        equations_block=eq_block,
        aux=aux,
        functions=functions,
        events=events,
        sim=sim,
        tag_index=_build_tag_index(events),
        presets=presets,
        lag_map=lag_map,
    )


# ---- hashing -----------------------------------------------------------------

def _json_canon(obj: Any) -> str:
    # Convert dataclasses/tuples to plain structures deterministically
    def encode(o: Any) -> Any:
        if isinstance(o, ModelSpec):
            return {
                "kind": o.kind,
                "label": o.label,
                "dtype": o.dtype,
                "states": list(o.states),
                "state_ic": list(o.state_ic),
                "params": list(o.params),
                "param_vals": list(o.param_vals),
                "equations_rhs": o.equations_rhs,
                "equations_block": o.equations_block,
                "aux": o.aux,
                "functions": o.functions,
                "events": [encode(e) for e in o.events],
                "sim": encode(o.sim),
                "tag_index": {k: list(v) for k, v in o.tag_index.items()},
                "presets": [encode(p) for p in o.presets],
                "lag_map": {k: list(v) for k, v in o.lag_map.items()},
            }
        if isinstance(o, PresetSpec):
            return {
                "name": o.name,
                "params": o.params,
                "states": o.states,
            }
        if isinstance(o, EventSpec):
            return {
                "name": o.name,
                "phase": o.phase,
                "cond": o.cond,
                "action_keyed": o.action_keyed,
                "action_block": o.action_block,
                "log": list(o.log),
                "tags": list(o.tags),
            }
        if isinstance(o, SimDefaults):
            return {
                "t0": o.t0,
                "t_end": o.t_end,
                "dt": o.dt,
                "stepper": o.stepper,
                "record": o.record,
                "atol": o.atol,
                "rtol": o.rtol,
            }
        if isinstance(o, tuple):
            return list(o)
        return o

    return json.dumps(encode(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_spec_hash(spec: ModelSpec) -> str:
    canon = _json_canon(spec)
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return h
