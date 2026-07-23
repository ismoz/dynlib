# src/dynlib/analysis/basin.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
import warnings
import numpy as np

from dynlib.analysis.basin_codes import BLOWUP, OUTSIDE, UNRESOLVED
from dynlib.errors import JITUnavailableError
from dynlib.runtime.fastpath.plans import FixedStridePlan
from dynlib.runtime.fastpath.capability import assess_capability
from dynlib.runtime.results_api import ResultsView
from dynlib.runtime.sim import Sim
from dynlib.runtime.softdeps import softdeps

_SOFTDEPS = softdeps()
_NUMBA_AVAILABLE = _SOFTDEPS.numba

if _NUMBA_AVAILABLE:  # pragma: no cover - optional dependency
    from numba import njit, prange  # type: ignore
    from numba import types as nb_types  # type: ignore
    from numba.typed import List as NumbaList  # type: ignore
else:  # pragma: no cover - fallback when numba missing
    njit = None
    prange = range
    nb_types = None
    NumbaList = None

__all__ = [
    "BLOWUP",
    "OUTSIDE",
    "UNRESOLVED",
    "Attractor",
    "BasinResult",
    "BasinAxis",
    "BasinValues",
    "BasinPoints",
    "FixedPoint",
    "ReferenceRun",
    "KnownAttractorLibrary",
    "basin_axis",
    "basin_values",
    "basin_points",
    "build_known_attractors_psc",
]


@dataclass
class Attractor:
    id: int
    fingerprint: set[int]  # merge-key (on merge grid)
    cells: set[int]        # accumulated discovered set (on detection grid)


@dataclass
class BasinResult:
    labels: np.ndarray
    registry: list[Attractor]
    meta: dict[str, object]


@dataclass(frozen=True)
class BasinAxis:
    min: float
    max: float
    n: int
    sample: Literal["edge", "center"] = "edge"


@dataclass(frozen=True)
class BasinValues:
    values: Sequence[float]


@dataclass(frozen=True)
class BasinPoints:
    points: Sequence[Sequence[float]] | np.ndarray
    vars: Sequence[str | int]


@dataclass(frozen=True)
class FixedPoint:
    name: str
    loc: Sequence[float]
    radius: Sequence[float] | float = 0.0


@dataclass(frozen=True)
class ReferenceRun:
    name: str
    ic: Sequence[float] | np.ndarray
    params: Sequence[float] | np.ndarray | None = None
    transient_samples: int | None = None
    signature_samples: int | None = None


@dataclass(frozen=True)
class KnownAttractorLibrary:
    """Library of known attractors for basin classification.
    
    Simple trajectory-based matching: stores reference trajectories and uses
    distance-based classification rather than complex probabilistic scoring.
    """
    obs_idx: np.ndarray  # indices of observed state variables
    names: tuple[str, ...]  # attractor names
    trajectories: list[np.ndarray]  # list of reference trajectories (n_samples x n_dims)
    obs_min: np.ndarray  # attractor observation bounds (for matching threshold)
    obs_max: np.ndarray
    escape_min: np.ndarray  # escape bounds (for blowup/outside detection)
    escape_max: np.ndarray
    attractor_radii: list[np.ndarray | None]  # per-attractor radii (for fixed points)
    meta: dict[str, object] | None = None
    
    @property
    def n_attr(self) -> int:
        return len(self.trajectories)


def _require_numba(who: str) -> None:
    if not _NUMBA_AVAILABLE or njit is None:
        raise JITUnavailableError(f"{who} requires numba for nopython execution")


def basin_axis(
    min: float,
    max: float,
    *,
    n: int,
    sample: Literal["edge", "center"] = "edge",
) -> BasinAxis:
    """Describe a uniformly sampled basin initial-condition axis."""
    n_int = int(n)
    if n_int <= 0:
        raise ValueError("n must be positive")
    if sample not in ("edge", "center"):
        raise ValueError("sample must be 'edge' or 'center'")
    min_f = float(min)
    max_f = float(max)
    if max_f <= min_f:
        raise ValueError("max must be greater than min")
    return BasinAxis(min=min_f, max=max_f, n=n_int, sample=sample)


def basin_values(values: Sequence[float]) -> BasinValues:
    """Describe an explicit basin initial-condition axis."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1D sequence")
    return BasinValues(values=tuple(float(x) for x in arr.tolist()))


def basin_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    vars: Sequence[str | int],
) -> BasinPoints:
    """Describe an advanced point cloud with named columns."""
    if not vars:
        raise ValueError("vars must be non-empty")
    return BasinPoints(points=points, vars=tuple(vars))


def _state_index_map(sim: Sim) -> tuple[list[str], dict[str, int]]:
    state_names = list(sim.model.spec.states)
    return state_names, {name: idx for idx, name in enumerate(state_names)}


def _resolve_state_name(
    item: str | int,
    *,
    state_names: Sequence[str],
) -> str:
    if isinstance(item, (int, np.integer)):
        idx = int(item)
        if idx < 0 or idx >= len(state_names):
            raise ValueError(f"state index {idx} out of range")
        return state_names[idx]
    name = str(item)
    if name not in state_names:
        raise ValueError(f"Unknown state variable '{name}'")
    return name


def _axis_values(spec: BasinAxis | BasinValues, *, n_override: int | None = None) -> np.ndarray:
    if isinstance(spec, BasinAxis):
        n = int(spec.n if n_override is None else n_override)
        if n <= 0:
            raise ValueError("axis resolution must be positive")
        if spec.sample == "center":
            step = (float(spec.max) - float(spec.min)) / float(n)
            return float(spec.min) + (np.arange(n, dtype=np.float64) + 0.5) * step
        return np.linspace(float(spec.min), float(spec.max), n, dtype=np.float64)
    arr = np.asarray(spec.values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("basin_values axes must be non-empty 1D sequences")
    return arr


def _default_ic_vector(sim: Sim, dtype: np.dtype) -> np.ndarray:
    return np.asarray(sim.model.spec.state_ic, dtype=dtype)


def _resolve_basin_points(
    sim: Sim,
    spec: BasinPoints,
    dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, object]]:
    state_names, state_to_idx = _state_index_map(sim)
    var_names = tuple(_resolve_state_name(item, state_names=state_names) for item in spec.vars)
    if len(set(var_names)) != len(var_names):
        raise ValueError("basin_points vars must be unique")

    points = np.asarray(spec.points, dtype=dtype)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("basin_points points must be a non-empty 2D array")
    if points.shape[1] != len(var_names):
        raise ValueError(
            f"basin_points expected {len(var_names)} columns for vars={list(var_names)}, "
            f"got {points.shape[1]}"
        )

    base = _default_ic_vector(sim, dtype)
    ic_arr = np.repeat(base[None, :], points.shape[0], axis=0)
    for col, name in enumerate(var_names):
        ic_arr[:, state_to_idx[name]] = points[:, col]

    meta = {
        "ic_kind": "points",
        "ic_vars": var_names,
        "ic_fixed": {
            name: float(base[idx])
            for idx, name in enumerate(state_names)
            if name not in var_names
        },
    }
    return np.ascontiguousarray(ic_arr, dtype=dtype), meta


def _resolve_basin_ic(
    sim: Sim,
    ic: Mapping[str | int, object] | BasinPoints,
    dtype: np.dtype,
    *,
    grid_shape: Sequence[int] | None = None,
    flat_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resolve named basin IC specs into full state arrays plus plotting metadata."""
    if isinstance(ic, BasinPoints):
        if grid_shape is not None or flat_indices is not None:
            raise ValueError("basin_points cannot be used with grid refinement")
        return _resolve_basin_points(sim, ic, dtype)

    if isinstance(ic, np.ndarray):
        raise TypeError("raw IC arrays are no longer accepted; use basin_points(array, vars=[...])")
    if not isinstance(ic, Mapping) or not ic:
        raise TypeError("ic must be a non-empty mapping of state names to basin specs")

    state_names, state_to_idx = _state_index_map(sim)
    base = _default_ic_vector(sim, dtype)
    fixed = base.astype(np.float64, copy=True)
    fixed_overrides: dict[str, float] = {}
    axis_names: list[str] = []
    axis_specs: list[BasinAxis | BasinValues] = []

    seen: set[str] = set()
    for key, value in ic.items():
        name = _resolve_state_name(key, state_names=state_names)
        if name in seen:
            raise ValueError(f"Duplicate state variable '{name}' in ic")
        seen.add(name)
        if isinstance(value, (BasinAxis, BasinValues)):
            axis_names.append(name)
            axis_specs.append(value)
            continue
        if isinstance(value, (str, bytes)):
            raise TypeError(f"ic value for '{name}' must be a scalar, basin_axis, or basin_values")
        try:
            scalar = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TypeError(f"ic value for '{name}' must be a scalar, basin_axis, or basin_values") from exc
        fixed[state_to_idx[name]] = scalar
        fixed_overrides[name] = scalar

    if not axis_names:
        if flat_indices is not None or grid_shape is not None:
            raise ValueError("refinement requires at least one swept basin_axis")
        return fixed.astype(dtype, copy=False)[None, :], {
            "ic_kind": "single",
            "ic_vars": (),
            "ic_fixed": {name: float(fixed[idx]) for idx, name in enumerate(state_names)},
        }

    if grid_shape is not None:
        if len(grid_shape) != len(axis_specs):
            raise ValueError("grid_shape length must match the number of swept IC axes")
        if any(not isinstance(spec, BasinAxis) for spec in axis_specs):
            raise ValueError("refinement only supports basin_axis IC axes")

    axes: list[np.ndarray] = []
    for idx, spec in enumerate(axis_specs):
        n_override = None if grid_shape is None else int(grid_shape[idx])
        axes.append(_axis_values(spec, n_override=n_override))

    shape = tuple(int(axis.size) for axis in axes)
    if any(n <= 0 for n in shape):
        raise ValueError("IC axes must be non-empty")

    if flat_indices is None:
        mesh = np.meshgrid(*axes, indexing="ij")
        point_count = int(np.prod(shape))
        ic_arr = np.repeat(fixed.astype(dtype, copy=False)[None, :], point_count, axis=0)
        for name, grid in zip(axis_names, mesh):
            ic_arr[:, state_to_idx[name]] = grid.ravel(order="C").astype(dtype, copy=False)
    else:
        flat = np.asarray(flat_indices, dtype=np.int64)
        if flat.ndim != 1:
            raise ValueError("flat_indices must be a 1D array")
        if flat.size and (np.any(flat < 0) or np.any(flat >= int(np.prod(shape)))):
            raise ValueError("flat_indices contains values outside the IC grid")
        coords = np.unravel_index(flat, shape)
        ic_arr = np.repeat(fixed.astype(dtype, copy=False)[None, :], flat.size, axis=0)
        for dim, name in enumerate(axis_names):
            ic_arr[:, state_to_idx[name]] = axes[dim][coords[dim]].astype(dtype, copy=False)

    bounds: list[tuple[float, float]] = []
    axis_values_meta: list[tuple[float, ...]] = []
    axis_samples: list[str] = []
    refinable = True
    for axis, spec in zip(axes, axis_specs):
        bounds.append((float(np.min(axis)), float(np.max(axis))))
        axis_values_meta.append(tuple(float(x) for x in axis.tolist()))
        if isinstance(spec, BasinAxis):
            axis_samples.append(spec.sample)
        else:
            axis_samples.append("values")
            refinable = False

    meta = {
        "ic_kind": "grid",
        "ic_grid": shape,
        "ic_bounds": tuple(bounds),
        "ic_vars": tuple(axis_names),
        "ic_axis_values": tuple(axis_values_meta),
        "ic_axis_sample": tuple(axis_samples),
        "ic_fixed": {
            name: float(fixed[idx])
            for idx, name in enumerate(state_names)
            if name not in axis_names or name in fixed_overrides
        },
        "ic_refinable": refinable,
    }
    return np.ascontiguousarray(ic_arr, dtype=dtype), meta


def _resolve_mode(
    *,
    mode: Literal["map", "ode", "auto"],
    sim: Sim,
) -> Literal["map", "ode"]:
    if mode in ("map", "ode"):
        return mode
    if mode != "auto":
        raise ValueError("mode must be 'map', 'ode', or 'auto'")
    kind = getattr(sim.model.spec, "kind", None)
    if kind == "map":
        return "map"
    if kind == "ode":
        return "ode"
    raise ValueError("mode='auto' requires sim.model.spec.kind in {'map','ode'}")


def _normalize_dims(name: str, values: Sequence[float] | float, d: int) -> np.ndarray:
    if isinstance(values, (float, int, np.floating, np.integer)):
        return np.full((d,), float(values), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (d,):
        raise ValueError(f"{name} must have length {d}")
    return arr


def _normalize_grid(grid_res: Sequence[int] | int, d: int) -> np.ndarray:
    if isinstance(grid_res, (int, np.integer)):
        arr = np.full((d,), int(grid_res), dtype=np.int64)
    else:
        arr = np.asarray(grid_res, dtype=np.int64)
        if arr.shape != (d,):
            raise ValueError(f"grid_res must have length {d}")
    if np.any(arr <= 0):
        raise ValueError("grid_res values must be positive")
    return arr


def _seq_len(value: object) -> int | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _prepare_record_vars(
    sim: Sim,
    observe_vars: Sequence[str | int] | None,
    blowup_vars: Sequence[str | int] | None,
    d: int,
) -> tuple[list[str], np.ndarray]:
    state_names = list(sim.model.spec.states)

    if observe_vars is None:
        if len(state_names) < d:
            raise ValueError("Not enough state variables to infer observe_vars")
        observe_list = state_names[:d]
    else:
        observe_list = []
        for item in observe_vars:
            if isinstance(item, (int, np.integer)):
                if item < 0 or item >= len(state_names):
                    raise ValueError(f"observe_vars index {item} out of range")
                observe_list.append(state_names[int(item)])
            else:
                if item not in state_names:
                    raise ValueError(f"Unknown observe variable '{item}'")
                observe_list.append(str(item))

    if len(observe_list) != d:
        raise ValueError(f"observe_vars must have length {d}")

    blowup_list: list[str] = []
    if blowup_vars is not None:
        for item in blowup_vars:
            if isinstance(item, (int, np.integer)):
                if item < 0 or item >= len(state_names):
                    raise ValueError(f"blowup_vars index {item} out of range")
                name = state_names[int(item)]
            else:
                if item not in state_names:
                    raise ValueError(f"Unknown blowup variable '{item}'")
                name = str(item)
            blowup_list.append(name)

    record_vars: list[str] = []
    seen = set()
    for name in list(observe_list) + blowup_list:
        if name in seen:
            continue
        record_vars.append(name)
        seen.add(name)

    blowup_idx = np.array([record_vars.index(name) for name in blowup_list], dtype=np.int64)
    return record_vars, blowup_idx


def _coerce_batch(
    *,
    ic: np.ndarray,
    params: np.ndarray,
    n_state: int,
    n_params: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    ic_arr = np.asarray(ic, dtype=dtype)
    if ic_arr.ndim == 1:
        ic_arr = ic_arr[None, :]
    if ic_arr.shape[1] != n_state:
        raise ValueError(f"ic shape mismatch: expected (*, {n_state}), got {ic_arr.shape}")

    params_arr = np.asarray(params, dtype=dtype)
    if params_arr.ndim == 1:
        params_arr = params_arr[None, :]
    if params_arr.shape[1] != n_params:
        raise ValueError(f"params shape mismatch: expected (*, {n_params}), got {params_arr.shape}")

    if ic_arr.shape[0] == 1 and params_arr.shape[0] > 1:
        ic_arr = np.repeat(ic_arr, params_arr.shape[0], axis=0)
    if params_arr.shape[0] == 1 and ic_arr.shape[0] > 1:
        params_arr = np.repeat(params_arr, ic_arr.shape[0], axis=0)
    if ic_arr.shape[0] != params_arr.shape[0]:
        raise ValueError(f"batch size mismatch: ic has {ic_arr.shape[0]}, params has {params_arr.shape[0]}")
    return np.ascontiguousarray(ic_arr), np.ascontiguousarray(params_arr)


def build_known_attractors_psc(
    sim: Sim,
    attractor_specs: Sequence[FixedPoint | ReferenceRun],
    *,
    observe_vars: Sequence[str | int] | None = None,
    escape_bounds: Sequence[tuple[float, float]] | None = None,
    mode: Literal["map", "ode", "auto"] = "auto",
    dt_obs: float | None = None,
    transient_samples: int = 100,
    signature_samples: int = 500,
) -> KnownAttractorLibrary:
    """
    Build a Known-Attractor library from reference trajectories.
    
    Simplified API - just runs trajectories and stores them for matching.
    No complex grid parameters or probabilistic scoring needed.
    
    escape_bounds: Optional bounds for escape/blowup detection. Sequence of (min, max) 
                   tuples per dimension. If None, computed as attractor_bounds * 1.5 margin.
    
    Note: signature_samples can be 0 if all attractors are FixedPoints (no
    trajectory capture needed). For ReferenceRun attractors, signature_samples
    must be positive to capture the attractor signature. Individual ReferenceRun
    objects may override transient_samples and signature_samples.
    """
    if not attractor_specs:
        raise ValueError("attractor_specs must be non-empty")
    if transient_samples < 0:
        raise ValueError("transient_samples must be non-negative")
    
    mode_use = _resolve_mode(mode=mode, sim=sim)
    adaptive = getattr(sim._stepper_spec.meta, "time_control", "fixed") == "adaptive"
    if mode_use == "ode" and adaptive:
        raise ValueError("build_known_attractors_psc requires a fixed-step stepper for ODE mode")
    
    # Determine observation variables
    if observe_vars is None:
        # Use all state variables
        observe_vars = list(sim.model.spec.states)
    
    record_vars, _ = _prepare_record_vars(sim, observe_vars, None, len(observe_vars))
    state_names = list(sim.model.spec.states)
    state_to_idx = {name: idx for idx, name in enumerate(state_names)}
    obs_names = record_vars[:len(observe_vars)]
    obs_idx = np.array([state_to_idx[name] for name in obs_names], dtype=np.int64)
    d = len(obs_idx)
    
    # Determine escape bounds (for blowup/outside detection)
    if escape_bounds is None:
        # Will be computed from trajectories with margin
        escape_min_arr = None
        escape_max_arr = None
    else:
        # Convert sequence of (min, max) tuples to separate min/max arrays
        escape_min_list = []
        escape_max_list = []
        for min_val, max_val in escape_bounds:
            escape_min_list.append(float(min_val))
            escape_max_list.append(float(max_val))
        escape_min_arr = np.array(escape_min_list, dtype=np.float64)
        escape_max_arr = np.array(escape_max_list, dtype=np.float64)
        if np.any(escape_max_arr <= escape_min_arr):
            raise ValueError("escape_max must be greater than escape_min for all dimensions")
    
    # Determine timestep
    dt_use = float(dt_obs) if mode_use == "ode" else float(
        dt_obs if dt_obs is not None else sim.model.spec.sim.dt
    )
    if mode_use == "ode" and dt_obs is None:
        raise ValueError("dt_obs required for ODE mode")
    
    # Record every step for signature capture
    record_stride = 1
    plan = FixedStridePlan(stride=record_stride)
    
    support = assess_capability(
        sim,
        plan=plan,
        record_vars=obs_names,
        dt=dt_use,
        transient=0.0,
        adaptive=adaptive,
        observers=None,
    )
    use_fastpath = support.ok
    
    (
        state_rec_indices,
        aux_rec_indices,
        state_rec_names,
        aux_names,
    ) = sim._resolve_recording_selection(obs_names)
    stepper_config = sim.stepper_config()
    n_state = len(sim.model.spec.states)
    n_params = len(sim.model.spec.params)
    dtype = sim.model.dtype
    
    trajectories: list[np.ndarray] = []
    names: list[str] = []
    all_points: list[np.ndarray] = []
    attractor_radii: list[np.ndarray | None] = []
    reference_timing: list[dict[str, int] | None] = []
    
    for idx, spec in enumerate(attractor_specs):
        name = getattr(spec, "name", f"attr_{idx}")
        names.append(str(name))
        
        if isinstance(spec, FixedPoint):
            # For fixed points, store the point and its radius
            loc = np.asarray(spec.loc, dtype=dtype).reshape(1, d)
            trajectories.append(loc)
            all_points.append(loc)
            
            # Extract radius (can be scalar or per-dimension)
            if isinstance(spec.radius, (list, tuple, np.ndarray)):
                radius_arr = np.asarray(spec.radius, dtype=dtype)
                if radius_arr.size != d:
                    raise ValueError(f"FixedPoint '{name}' radius must have {d} elements")
                attractor_radii.append(radius_arr)
            else:
                radius_arr = np.full(d, float(spec.radius), dtype=dtype)
                attractor_radii.append(radius_arr)
            reference_timing.append(None)
            continue
        elif isinstance(spec, ReferenceRun):
            ref_transient_samples = (
                int(spec.transient_samples)
                if spec.transient_samples is not None
                else int(transient_samples)
            )
            ref_signature_samples = (
                int(spec.signature_samples)
                if spec.signature_samples is not None
                else int(signature_samples)
            )
            if ref_transient_samples < 0:
                raise ValueError(f"ReferenceRun '{name}' transient_samples must be non-negative")
            if ref_signature_samples <= 0:
                raise ValueError(f"ReferenceRun '{name}' signature_samples must be positive")

            t0 = float(sim.model.spec.sim.t0)
            max_steps = int(ref_transient_samples + ref_signature_samples + 1)
            if mode_use == "ode":
                T = t0 + float(max_steps) * dt_use
                N = None
            else:
                T = None
                N = int(max_steps)
            reference_timing.append(
                {
                    "transient_samples": int(ref_transient_samples),
                    "signature_samples": int(ref_signature_samples),
                    "max_steps": int(max_steps),
                }
            )

            # Run the trajectory and record it
            ic_arr, params_arr = _coerce_batch(
                ic=np.asarray(spec.ic, dtype=dtype),
                params=np.asarray(spec.params, dtype=dtype) if spec.params is not None else sim.param_vector(
                    source="session",
                    copy=True,
                ),
                n_state=n_state,
                n_params=n_params,
                dtype=dtype,
            )
            if ic_arr.shape[0] != 1:
                raise ValueError(f"ReferenceRun '{name}' ic must define a single state vector")
            
            views: list[ResultsView] = []
            if use_fastpath:
                from dynlib.runtime.fastpath import fastpath_batch_for_sim

                views = fastpath_batch_for_sim(
                    sim,
                    plan=plan,
                    t0=t0,
                    T=T,
                    N=N,
                    dt=dt_use,
                    record_vars=obs_names,
                    transient=0.0,
                    record_interval=record_stride,
                    max_steps=max_steps,
                    ic=ic_arr,
                    params=params_arr,
                    parallel_mode="none",
                    max_workers=None,
                    observers=None,
                )
                if views is None:
                    use_fastpath = False
                    views = []

            if not use_fastpath:
                seed = sim._select_seed(
                    resume=False,
                    t0=t0,
                    dt=dt_use,
                    ic=ic_arr[0],
                    params=params_arr[0],
                )
                result = sim._execute_run(
                    seed=seed,
                    t_end=float(T if T is not None else seed.t + float(max_steps) * dt_use),
                    target_steps=int(N) if N is not None else None,
                    max_steps=int(max_steps),
                    record=True,
                    record_interval=record_stride,
                    cap_rec=max_steps + 10,
                    cap_evt=1,
                    stepper_config=stepper_config,
                    adaptive=adaptive,
                    wrms_cfg=None,
                    state_rec_indices=state_rec_indices,
                    aux_rec_indices=aux_rec_indices,
                    state_names=state_rec_names,
                    aux_names=aux_names,
                    observers=None,
                )
                views = [ResultsView(result, sim.model.spec)]

            if not views:
                warnings.warn(f"Failed to capture trajectory for '{name}'", RuntimeWarning)
                trajectories.append(np.zeros((0, d), dtype=dtype))
                continue
                
            view = views[0]
            # Extract all observed state variables
            try:
                traj_full = view[obs_names]  # This returns (n_steps, n_dims) array
            except Exception as e:
                warnings.warn(f"Failed to extract trajectory for '{name}': {e}", RuntimeWarning)
                trajectories.append(np.zeros((0, d), dtype=dtype))
                continue
            
            if traj_full.size == 0:
                warnings.warn(f"No trajectory data captured for '{name}'", RuntimeWarning)
                trajectories.append(np.zeros((0, d), dtype=dtype))
                continue
            
            # Extract trajectory after transient
            traj = traj_full[ref_transient_samples:, :]  # Skip transient
            if traj.shape[0] == 0:
                warnings.warn(f"Transient too long for '{name}'", RuntimeWarning)
                trajectories.append(np.zeros((0, d), dtype=dtype))
                attractor_radii.append(None)
                continue
                
            trajectories.append(np.asarray(traj, dtype=dtype))
            all_points.append(traj)
            attractor_radii.append(None)  # ReferenceRun attractors have no fixed radius
        else:
            raise TypeError(f"Unsupported attractor spec type: {type(spec)!r}")
    
    # Compute observation bounds from attractor data (for matching threshold)
    if not all_points:
        raise ValueError("No valid trajectories captured")
    all_data = np.vstack(all_points)
    obs_min_arr = np.min(all_data, axis=0)
    obs_max_arr = np.max(all_data, axis=0)
    
    # Compute escape bounds (for blowup/outside detection)
    # If user provided explicit bounds, use those; otherwise use attractor bounds with margin
    if escape_min_arr is None or escape_max_arr is None:
        # Add 50% margin around attractor bounds for escape detection
        margin = 0.5 * np.ptp(all_data, axis=0)
        escape_min_arr = obs_min_arr - margin
        escape_max_arr = obs_max_arr + margin
    
    meta = {
        "mode": mode_use,
        "observe_vars": tuple(obs_names),
        "dt_obs": float(dt_use),
        "transient_samples": int(transient_samples),
        "signature_samples": int(signature_samples),
        "reference_timing": tuple(reference_timing),
    }

    return KnownAttractorLibrary(
        obs_idx=np.ascontiguousarray(obs_idx, dtype=np.int64),
        names=tuple(names),
        trajectories=trajectories,
        obs_min=np.ascontiguousarray(obs_min_arr, dtype=np.float64),
        obs_max=np.ascontiguousarray(obs_max_arr, dtype=np.float64),
        escape_min=np.ascontiguousarray(escape_min_arr, dtype=np.float64),
        escape_max=np.ascontiguousarray(escape_max_arr, dtype=np.float64),
        attractor_radii=attractor_radii,
        meta=meta,
    )
