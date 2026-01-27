# src/dynlib/dsl/spec.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Mapping
import json
import hashlib
from types import MappingProxyType
from dynlib.errors import ModelLoadError

__all__ = [
    "SimDefaults",
    "StopSpec",
    "EventSpec",
    "PresetSpec",
    "ModelSpec",
    "build_spec",
    "compute_spec_hash",
    "choose_default_stepper",
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
    max_steps: int = 1_000_000
    stop: "StopSpec | None" = None
    _stepper_defaults: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_stepper_defaults",
            MappingProxyType(dict(self._stepper_defaults)),
        )

    def __getattr__(self, name: str) -> Any:
        step_defaults = object.__getattribute__(self, "_stepper_defaults")
        if name in step_defaults:
            return step_defaults[name]
        raise AttributeError(name)

    def stepper_defaults(self) -> Mapping[str, Any]:
        """Return a read-only view of extra stepper configuration defaults."""
        return object.__getattribute__(self, "_stepper_defaults")


@dataclass(frozen=True)
class StopSpec:
    """Early-exit stop condition.

    The stop condition is evaluated inside the runner loop on the committed
    state/time (phase semantics controlled by runner placement).
    """

    cond: str
    phase: str = "post"  # only "post" is supported; pre/both reserved for future use


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
    name: str | None
    dtype: str               # canonical dtype string
    constants: Tuple[str, ...]
    constant_vals: Tuple[float | int, ...]
    states: Tuple[str, ...]   # ordered
    state_ic: Tuple[float | int, ...]
    params: Tuple[str, ...]
    param_vals: Tuple[float | int, ...]
    equations_rhs: Dict[str, str] | None
    equations_block: str | None
    inverse_rhs: Dict[str, str] | None
    inverse_block: str | None
    jacobian_exprs: Tuple[Tuple[str, ...], ...] | None
    aux: Dict[str, str]
    functions: Dict[str, tuple[list[str], str]]
    events: Tuple[EventSpec, ...]
    sim: SimDefaults
    tag_index: Dict[str, Tuple[str, ...]]  # tag -> event names, compile-time only
    presets: Tuple[PresetSpec, ...]  # inline presets from DSL
    lag_map: Dict[str, Tuple[int, int, int]]  # state_name -> (buffer_len, ring_offset, head_index)
    uses_lag: bool
    equations_use_lag: bool


# ---- builders ----------------------------------------------------------------

def choose_default_stepper(kind: str) -> str:
    """Return the default stepper name for the given model kind."""
    if kind == "ode":
        return "rk4"
    if kind == "map":
        return "map"
    raise ModelLoadError(f"Unknown model kind for default stepper: {kind!r}")


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


_SIM_KNOWN_KEYS = frozenset(
    {"t0", "t_end", "dt", "stepper", "record", "atol", "rtol", "max_steps", "stop"}
)


def build_spec(normal: Dict[str, Any]) -> ModelSpec:
    # VALIDATE BEFORE PARSING
    from .astcheck import (
        validate_expr_acyclic,
        validate_event_legality,
        validate_event_tags,
        validate_functions_signature,
        validate_no_duplicate_equation_targets,
        validate_presets,
        validate_aux_names,
        validate_identifier_uniqueness,
        validate_reserved_identifiers,
        validate_constants,
        validate_identifiers_resolved,
        validate_jacobian_matrix,
        collect_lag_requests,
        detect_equation_lag_usage,
    )
    
    validate_reserved_identifiers(normal)
    validate_constants(normal)
    validate_identifier_uniqueness(normal)
    validate_expr_acyclic(normal)
    validate_event_legality(normal)
    validate_event_tags(normal)
    validate_functions_signature(normal)
    validate_no_duplicate_equation_targets(normal)
    validate_presets(normal)
    validate_aux_names(normal)
    validate_identifiers_resolved(normal)
    validate_jacobian_matrix(normal)
    
    # Collect lag requests from all expressions
    lag_requests = collect_lag_requests(normal)
    
    model = normal["model"]
    kind = model["type"]
    dtype = _canon_dtype(model.get("dtype", "float64"))
    name = model.get("name")

    constants = tuple((normal.get("constants") or {}).keys())
    constant_vals = tuple((normal.get("constants") or {}).values())

    states = tuple(normal["states"].keys())
    state_ic = tuple(normal["states"].values())

    params = tuple(normal["params"].keys())
    param_vals = tuple(normal["params"].values())

    eq = normal["equations"]
    eq_rhs = dict(eq["rhs"]) if eq.get("rhs") else None
    eq_block = eq.get("expr") if isinstance(eq.get("expr"), str) else None
    inv_tbl = eq.get("inverse") if isinstance(eq.get("inverse"), dict) else None
    inv_rhs = dict(inv_tbl.get("rhs")) if inv_tbl and inv_tbl.get("rhs") else None
    inv_block = inv_tbl.get("expr") if inv_tbl and isinstance(inv_tbl.get("expr"), str) else None
    jacobian_exprs = None
    if isinstance(eq.get("jacobian"), dict):
        expr_rows = eq["jacobian"].get("expr") or []
        jacobian_exprs = tuple(tuple(str(item) for item in row) for row in expr_rows)

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
    stop_in = sim_in.get("stop")
    stop_spec: StopSpec | None = None
    if stop_in is not None:
        if isinstance(stop_in, str):
            stop_spec = StopSpec(cond=str(stop_in), phase="post")
        elif isinstance(stop_in, dict):
            cond = stop_in.get("cond")
            if not isinstance(cond, str) or not cond.strip():
                raise ModelLoadError("[sim.stop].cond must be a non-empty string")
            phase = stop_in.get("phase", "post")
            if phase != "post":
                raise ModelLoadError("[sim.stop].phase must be 'post'")
            stop_spec = StopSpec(cond=str(cond), phase=str(phase))
        else:
            raise ModelLoadError("[sim].stop must be a table ([sim.stop]) or a string expression")

    sim_extras = {
        key: value for key, value in sim_in.items() if key not in _SIM_KNOWN_KEYS
    }
    stepper_value = sim_in.get("stepper")
    if stepper_value is None:
        stepper_value = choose_default_stepper(kind)
    sim = SimDefaults(
        t0=float(sim_in.get("t0", SimDefaults.t0)),
        t_end=float(sim_in.get("t_end", SimDefaults.t_end)),
        dt=float(sim_in.get("dt", SimDefaults.dt)),
        stepper=str(stepper_value),
        record=bool(sim_in.get("record", SimDefaults.record)),
        atol=float(sim_in.get("atol", SimDefaults.atol)),
        rtol=float(sim_in.get("rtol", SimDefaults.rtol)),
        max_steps=int(sim_in.get("max_steps", SimDefaults.max_steps)),
        stop=stop_spec,
        _stepper_defaults=sim_extras,
    )

    presets = tuple(
        PresetSpec(
            name=p["name"],
            params=dict(p["params"]),
            states=dict(p["states"]) if p.get("states") else None,
        )
        for p in (normal.get("presets") or [])
    )
    
    # Build lag_map: state_name -> (buffer_len, ring_offset, head_index)
    # buffer_len = max_requested_lag + 1 (extra slot preserves current head)
    # ring_offset is the starting index in runtime_ws.lag_ring for this state's circular buffer
    # head_index is the slot in runtime_ws.lag_head for this state's head pointer
    lag_map: Dict[str, Tuple[int, int, int]] = {}
    ring_offset = 0
    head_index = 0

    # Process lagged states in the order they appear in states tuple (deterministic)
    for state_name in states:
        if state_name in lag_requests:
            requested_depth = lag_requests[state_name]
            buffer_len = requested_depth + 1  # extra slot ensures lag_depth indexing works
            lag_map[state_name] = (buffer_len, ring_offset, head_index)
            ring_offset += buffer_len  # each lagged state gets buffer_len slots
            head_index += 1           # each lagged state gets one head slot

    equations_use_lag = detect_equation_lag_usage(normal)
    uses_lag = bool(lag_map)

    return ModelSpec(
        kind=kind,
        name=name,
        dtype=dtype,
        constants=constants,
        constant_vals=constant_vals,
        states=states,
        state_ic=state_ic,
        params=params,
        param_vals=param_vals,
        equations_rhs=eq_rhs,
        equations_block=eq_block,
        inverse_rhs=inv_rhs,
        inverse_block=inv_block,
        jacobian_exprs=jacobian_exprs,
        aux=aux,
        functions=functions,
        events=events,
        sim=sim,
        tag_index=_build_tag_index(events),
        presets=presets,
        lag_map=lag_map,
        uses_lag=uses_lag,
        equations_use_lag=equations_use_lag,
    )


# ---- hashing -----------------------------------------------------------------

def _json_canon(obj: Any) -> str:
    # Convert dataclasses/tuples to plain structures deterministically
    def encode(o: Any) -> Any:
        if isinstance(o, ModelSpec):
            return {
                "kind": o.kind,
                "name": o.name,
                "dtype": o.dtype,
                "constants": list(o.constants),
                "constant_vals": list(o.constant_vals),
                "states": list(o.states),
                "state_ic": list(o.state_ic),
                "params": list(o.params),
                "param_vals": list(o.param_vals),
                "equations_rhs": o.equations_rhs,
                "equations_block": o.equations_block,
                "inverse_rhs": o.inverse_rhs,
                "inverse_block": o.inverse_block,
                "jacobian_exprs": [list(row) for row in o.jacobian_exprs] if o.jacobian_exprs else None,
                "aux": o.aux,
                "functions": o.functions,
                "events": [encode(e) for e in o.events],
                "sim": encode(o.sim),
                "tag_index": {k: list(v) for k, v in o.tag_index.items()},
                "presets": [encode(p) for p in o.presets],
                "lag_map": {k: list(v) for k, v in o.lag_map.items()},
                "uses_lag": o.uses_lag,
                "equations_use_lag": o.equations_use_lag,
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
            base = {
                "t0": o.t0,
                "t_end": o.t_end,
                "dt": o.dt,
                "stepper": o.stepper,
                "record": o.record,
                "atol": o.atol,
                "rtol": o.rtol,
                "max_steps": o.max_steps,
            }
            if o.stop is not None:
                base["stop"] = {"cond": o.stop.cond, "phase": o.stop.phase}
            stepper_defaults = dict(o.stepper_defaults())
            if stepper_defaults:
                base["stepper_defaults"] = stepper_defaults
            return base
        if isinstance(o, tuple):
            return list(o)
        return o

    return json.dumps(encode(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_spec_hash(spec: ModelSpec) -> str:
    canon = _json_canon(spec)
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return h
