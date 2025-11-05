# src/dynlib/dsl/spec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json
import hashlib

__all__ = [
    "SimDefaults",
    "EventSpec",
    "ModelSpec",
    "build_spec",
    "compute_spec_hash",
]


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
    record: bool
    log: Tuple[str, ...]


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


# ---- builders ----------------------------------------------------------------

def _canon_dtype(dtype: str) -> str:
    # Keep as-is but normalize common aliases
    alias = {
        "float": "float64",
        "double": "float64",
        "single": "float32",
    }
    return alias.get(dtype, dtype)


def build_spec(normal: Dict[str, Any]) -> ModelSpec:
    # VALIDATE BEFORE PARSING
    from .astcheck import (
        validate_expr_acyclic,
        validate_event_legality,
        validate_functions_signature,
    )
    
    validate_expr_acyclic(normal)
    validate_event_legality(normal)
    validate_functions_signature(normal)
    
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
            record=bool(e.get("record", False)),
            log=tuple(e.get("log", []) or []),
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
            }
        if isinstance(o, EventSpec):
            return {
                "name": o.name,
                "phase": o.phase,
                "cond": o.cond,
                "action_keyed": o.action_keyed,
                "action_block": o.action_block,
                "record": o.record,
                "log": list(o.log),
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
