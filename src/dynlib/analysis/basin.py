# src/dynlib/analysis/basin.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
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
    "FixedPoint",
    "ReferenceRun",
    "KnownAttractorLibrary",
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
class FixedPoint:
    name: str
    loc: Sequence[float]
    radius: Sequence[float] | float = 0.0


@dataclass(frozen=True)
class ReferenceRun:
    name: str
    ic: Sequence[float] | np.ndarray
    params: Sequence[float] | np.ndarray | None = None


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
    must be positive to capture the attractor signature.
    """
    if not attractor_specs:
        raise ValueError("attractor_specs must be non-empty")
    if transient_samples < 0:
        raise ValueError("transient_samples must be non-negative")
    
    # Check if we have any ReferenceRun attractors that need signature capture
    has_reference_runs = any(isinstance(spec, ReferenceRun) for spec in attractor_specs)
    if has_reference_runs and signature_samples <= 0:
        raise ValueError("signature_samples must be positive when using ReferenceRun attractors")
    # For fixed-point-only basins, signature_samples=0 is valid

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
    
    t0 = float(sim.model.spec.sim.t0)
    max_steps = int(transient_samples + signature_samples + 1)
    if mode_use == "ode":
        T = t0 + float(max_steps) * dt_use
        N = None
    else:
        T = None
        N = int(max_steps)
    
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
            continue
        elif isinstance(spec, ReferenceRun):
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
            traj = traj_full[transient_samples:, :]  # Skip transient  
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


