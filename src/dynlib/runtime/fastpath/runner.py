from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence
import math
import numpy as np

from dynlib.runtime.fastpath.plans import RecordingPlan
from dynlib.runtime.fastpath.capability import assess_capability, FastpathSupport
from dynlib.runtime.results import Results
from dynlib.runtime.workspace import make_runtime_workspace, snapshot_workspace
from dynlib.runtime.sim import Segment, Sim
from dynlib.runtime.results_api import ResultsView
from dynlib.runtime.runner_api import DONE, GROW_REC, GROW_EVT

__all__ = [
    "run_single_fastpath",
    "run_batch_fastpath",
    "fastpath_for_sim",
    "fastpath_batch_for_sim",
]


def _max_event_log_width(events) -> int:
    width = 0
    for event in events:
        width = max(width, len(getattr(event, "log", ()) or ()))
    return width


def _resolve_recording_selection(
    *,
    spec,
    record_vars: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if record_vars is None:
        state_indices = np.arange(len(spec.states), dtype=np.int32)
        aux_indices = np.array([], dtype=np.int32)
        return state_indices, aux_indices, list(spec.states), []

    state_names = list(spec.states)
    aux_names_all = list(spec.aux.keys()) if spec.aux else []

    state_idx: list[int] = []
    aux_idx: list[int] = []
    sel_states: list[str] = []
    sel_aux: list[str] = []

    for name in record_vars:
        if name.startswith("aux."):
            key = name[4:]
            if key not in aux_names_all:
                raise ValueError(f"Unknown aux variable: {key}")
            aux_idx.append(aux_names_all.index(key))
            sel_aux.append(key)
            continue
        if name in state_names:
            state_idx.append(state_names.index(name))
            sel_states.append(name)
            continue
        if name in aux_names_all:
            aux_idx.append(aux_names_all.index(name))
            sel_aux.append(name)
            continue
        raise ValueError(
            f"Unknown variable '{name}'. "
            f"States: {state_names}. Aux: {aux_names_all}."
        )

    return (
        np.array(state_idx, dtype=np.int32),
        np.array(aux_idx, dtype=np.int32),
        sel_states,
        sel_aux,
    )


def _is_jitted_runner(fn) -> bool:
    """
    Best-effort detection of a numba-compiled runner. Dispatchers expose
    ``signatures`` when compiled; pure Python runners do not.
    """
    return bool(getattr(fn, "signatures", None))


@dataclass(frozen=True)
class _RunContext:
    t0: float
    t_end: float
    target_steps: Optional[int]
    dt: float
    max_steps: int
    transient: float
    record_interval: int


def _call_runner(
    *,
    model,
    ctx: _RunContext,
    state_rec_indices: np.ndarray,
    aux_rec_indices: np.ndarray,
    state_names: list[str],
    aux_names: list[str],
    plan: RecordingPlan,
    stepper_config: np.ndarray,
    params: np.ndarray,
    ic: np.ndarray,
    runtime_ws,
    stepper_ws,
):
    dtype = model.dtype
    n_state = len(model.spec.states)
    n_aux = len(model.spec.aux)
    n_rec_states = len(state_rec_indices)
    n_rec_aux = len(aux_rec_indices)

    rec_every = int(ctx.record_interval)
    total_steps = ctx.target_steps if ctx.target_steps is not None else math.ceil(max(0.0, ctx.t_end - ctx.t0) / ctx.dt)
    cap_rec = 1 if rec_every <= 0 else max(1, int(plan.capacity(total_steps=total_steps)))
    max_log_width = _max_event_log_width(model.spec.events)

    # Buffers
    T = np.zeros((cap_rec,), dtype=np.float64)
    Y = np.zeros((n_rec_states, cap_rec), dtype=dtype) if n_rec_states > 0 else np.zeros((0, cap_rec), dtype=dtype)
    AUX = np.zeros((n_rec_aux, cap_rec), dtype=dtype) if n_rec_aux > 0 else np.zeros((0, cap_rec), dtype=dtype)
    STEP = np.zeros((cap_rec,), dtype=np.int64)
    FLAGS = np.zeros((cap_rec,), dtype=np.int32)

    cap_evt = 1
    EVT_CODE = np.zeros((cap_evt,), dtype=np.int32)
    EVT_INDEX = np.zeros((cap_evt,), dtype=np.int32)
    EVT_LOG_DATA = np.zeros((cap_evt, max(1, max_log_width)), dtype=dtype)

    # Work arrays
    y_curr = np.array(ic, dtype=dtype, copy=True)
    y_prev = np.array(ic, dtype=dtype, copy=True)
    y_prop = np.zeros((n_state,), dtype=dtype)
    t_prop = np.zeros((1,), dtype=np.float64)
    dt_next = np.zeros((1,), dtype=np.float64)
    err_est = np.zeros((1,), dtype=np.float64)
    evt_log_scratch = np.zeros((max(1, max_log_width),), dtype=dtype)

    user_break_flag = np.zeros((1,), dtype=np.int32)
    status_out = np.zeros((1,), dtype=np.int32)
    hint_out = np.zeros((1,), dtype=np.int32)
    i_out = np.zeros((1,), dtype=np.int64)
    step_out = np.zeros((1,), dtype=np.int64)
    t_out = np.zeros((1,), dtype=np.float64)

    status = model.runner(
        float(ctx.t0),
        float(ctx.target_steps if ctx.target_steps is not None else ctx.t_end),
        float(ctx.dt),
        int(ctx.max_steps),
        int(n_state),
        int(rec_every),
        y_curr,
        y_prev,
        params,
        runtime_ws,
        stepper_ws,
        stepper_config,
        y_prop,
        t_prop,
        dt_next,
        err_est,
        T,
        Y,
        AUX,
        STEP,
        FLAGS,
        EVT_CODE,
        EVT_INDEX,
        EVT_LOG_DATA,
        evt_log_scratch,
        np.int64(0),
        np.int64(0),
        int(cap_rec),
        int(cap_evt),
        user_break_flag,
        status_out,
        hint_out,
        i_out,
        step_out,
        t_out,
        model.stepper,
        model.rhs,
        model.events_pre,
        model.events_post,
        model.update_aux,
        state_rec_indices,
        aux_rec_indices,
        n_rec_states,
        n_rec_aux,
    )

    filled = int(i_out[0])
    evt_filled = max(0, int(hint_out[0]))
    if status == GROW_REC:
        raise RuntimeError("Fastpath runner ran out of record capacity; adjust plan.")
    if status == GROW_EVT:
        raise RuntimeError("Fastpath runner received unexpected event growth request.")
    if status != DONE:
        raise RuntimeError(f"Fastpath runner exited with status {int(status)}")

    # Optional tail trimming
    trim = plan.finalize_index(filled)
    if trim is not None:
        T = T[trim]
        Y = Y[:, trim]
        AUX = AUX[:, trim] if AUX.shape[0] > 0 else AUX
        STEP = STEP[trim]
        FLAGS = FLAGS[trim]
        filled = STEP.shape[0]
    else:
        T = T[:filled]
        Y = Y[:, :filled]
        AUX = AUX[:, :filled] if AUX.shape[0] > 0 else AUX
        STEP = STEP[:filled]
        FLAGS = FLAGS[:filled]

    final_ws = {
        "stepper": snapshot_workspace(stepper_ws),
        "runtime": snapshot_workspace(runtime_ws),
    }

    return Results(
        T=T,
        Y=Y,
        AUX=(AUX if AUX.shape[0] > 0 else None),
        STEP=STEP,
        FLAGS=FLAGS,
        EVT_CODE=EVT_CODE,
        EVT_INDEX=EVT_INDEX,
        EVT_LOG_DATA=EVT_LOG_DATA,
        n=filled,
        m=evt_filled,
        status=int(status),
        final_state=np.array(y_curr, copy=True),
        final_params=np.array(params, copy=True),
        t_final=float(t_out[0]),
        final_dt=float(dt_next[0]) if filled > 0 else float(ctx.dt),
        step_count_final=int(step_out[0]),
        final_workspace=final_ws,
        state_names=state_names,
        aux_names=aux_names,
    )


def _normalize_batch_inputs(
    *,
    ic: np.ndarray,
    params: np.ndarray,
    n_state: int,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    ic_arr = np.atleast_2d(np.asarray(ic))
    params_arr = np.atleast_2d(np.asarray(params))

    if ic_arr.shape[1] != n_state:
        raise ValueError(f"ic shape mismatch: expected (*, {n_state}), got {ic_arr.shape}")
    if params_arr.shape[1] != n_params:
        raise ValueError(f"params shape mismatch: expected (*, {n_params}), got {params_arr.shape}")

    if ic_arr.shape[0] == 1 and params_arr.shape[0] > 1:
        ic_arr = np.repeat(ic_arr, params_arr.shape[0], axis=0)
    if params_arr.shape[0] == 1 and ic_arr.shape[0] > 1:
        params_arr = np.repeat(params_arr, ic_arr.shape[0], axis=0)

    if ic_arr.shape[0] != params_arr.shape[0]:
        raise ValueError(f"Batch size mismatch: ic has {ic_arr.shape[0]}, params has {params_arr.shape[0]}")

    batch = ic_arr.shape[0]
    return ic_arr, params_arr, batch


def run_single_fastpath(
    *,
    model,
    plan: RecordingPlan,
    t0: float,
    t_end: float | None,
    target_steps: int | None,
    dt: float,
    max_steps: int,
    transient: float,
    state_rec_indices: np.ndarray,
    aux_rec_indices: np.ndarray,
    state_names: list[str],
    aux_names: list[str],
    params: np.ndarray,
    ic: np.ndarray,
    stepper_config: np.ndarray | None = None,
) -> Results:
    """Core fastpath execution using the compiled runner."""
    if stepper_config is None:
        stepper_config = np.array([], dtype=np.float64)
    dtype = model.dtype
    n_aux = len(model.spec.aux)
    runtime_ws = make_runtime_workspace(
        lag_state_info=model.lag_state_info,
        dtype=dtype,
        n_aux=n_aux,
    )
    stepper_ws = model.make_stepper_workspace() if model.make_stepper_workspace else None

    # Optional transient warm-up (no recording)
    if transient > 0.0:
        warm_ctx = _RunContext(
            t0=float(t0),
            t_end=float(t0 + transient) if t_end is not None else float(t0 + transient),
            target_steps=int(transient) if target_steps is not None else None,
            dt=float(dt),
            max_steps=max_steps,
            transient=0.0,
            record_interval=0,
        )
        warm_result = _call_runner(
            model=model,
            ctx=warm_ctx,
            state_rec_indices=np.array([], dtype=np.int32),
            aux_rec_indices=np.array([], dtype=np.int32),
            state_names=[],
            aux_names=[],
            plan=plan,
            stepper_config=stepper_config,
            params=params,
            ic=ic,
            runtime_ws=runtime_ws,
            stepper_ws=stepper_ws,
        )
        t0 = warm_ctx.t_end
        if target_steps is not None:
            target_steps = max(0, target_steps - int(transient))
        # Warm-up updates runtime_ws/stepper_ws in place; reuse them.
        ic = warm_result.final_state
        params = warm_result.final_params
        dt = float(warm_result.final_dt) if warm_result.final_dt != 0.0 else dt

    run_ctx = _RunContext(
        t0=float(t0),
        t_end=float(t_end if t_end is not None else t0),
        target_steps=target_steps,
        dt=float(dt),
        max_steps=max_steps,
        transient=0.0,
        record_interval=plan.record_interval(),
    )
    return _call_runner(
        model=model,
        ctx=run_ctx,
        state_rec_indices=state_rec_indices,
        aux_rec_indices=aux_rec_indices,
        state_names=state_names,
        aux_names=aux_names,
        plan=plan,
        stepper_config=stepper_config,
        params=params,
        ic=ic,
        runtime_ws=runtime_ws,
        stepper_ws=stepper_ws,
    )


def run_batch_fastpath(
    *,
    model,
    plan: RecordingPlan,
    t0: float,
    t_end: float | None,
    target_steps: int | None,
    dt: float,
    max_steps: int,
    transient: float,
    state_rec_indices: np.ndarray,
    aux_rec_indices: np.ndarray,
    state_names: list[str],
    aux_names: list[str],
    params: np.ndarray,
    ic: np.ndarray,
    stepper_config: np.ndarray | None = None,
    parallel_mode: Literal["auto", "threads", "process", "none"] = "auto",
    max_workers: Optional[int] = None,
) -> list[Results]:
    """
    Batch fastpath execution across multiple IC/parameter sets.

    For JIT builds, threads will leverage the numba-compiled runner (GIL-free).
    For pure Python builds, a thread pool is used unless ``parallel_mode="none"``.
    """
    n_state = len(model.spec.states)
    n_params = len(model.spec.params)
    ic_batch, params_batch, batch = _normalize_batch_inputs(
        ic=ic, params=params, n_state=n_state, n_params=n_params
    )
    if batch == 0:
        return []
    if batch == 1:
        return [
            run_single_fastpath(
                model=model,
                plan=plan,
                t0=t0,
                t_end=t_end,
                target_steps=target_steps,
                dt=dt,
                max_steps=max_steps,
                transient=transient,
                state_rec_indices=state_rec_indices,
                aux_rec_indices=aux_rec_indices,
                state_names=state_names,
                aux_names=aux_names,
                params=params_batch[0],
                ic=ic_batch[0],
                stepper_config=stepper_config,
            )
        ]

    backend = parallel_mode
    is_jit = _is_jitted_runner(model.runner)
    if backend == "auto":
        backend = "threads" if is_jit else "threads"

    def _run(idx: int) -> Results:
        return run_single_fastpath(
            model=model,
            plan=plan,
            t0=t0,
            t_end=t_end,
            target_steps=target_steps,
            dt=dt,
            max_steps=max_steps,
            transient=transient,
            state_rec_indices=state_rec_indices,
            aux_rec_indices=aux_rec_indices,
            state_names=state_names,
            aux_names=aux_names,
            params=params_batch[idx],
            ic=ic_batch[idx],
            stepper_config=stepper_config,
        )

    if backend == "none" or max_workers == 1:
        return [_run(i) for i in range(batch)]

    if backend == "process":
        # ProcessPool can be brittle for compiled runners; fall back to threads.
        backend = "threads"

    if backend != "threads":
        raise ValueError(f"Unknown parallel_mode {parallel_mode!r}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_run, range(batch)))


def fastpath_for_sim(
    sim: Sim,
    *,
    plan: RecordingPlan,
    t0: float | None,
    T: float | None,
    N: int | None,
    dt: float | None,
    record_vars: Sequence[str] | None,
    transient: float | None,
    record_interval: int | None,
    max_steps: int | None,
    ic: np.ndarray,
    params: np.ndarray,
    support: FastpathSupport | None = None,
) -> ResultsView | None:
    """
    Fastpath convenience entry point for :class:`Sim`.

    Returns a ResultsView on success, or None when the capability gate fails.
    """
    sim_defaults = sim.model.spec.sim
    dt_use = float(dt if dt is not None else sim._nominal_dt if sim._nominal_dt else sim_defaults.dt)
    transient_use = float(transient) if transient is not None else 0.0
    max_steps_use = int(max_steps if max_steps is not None else sim_defaults.max_steps)

    stepper_spec = sim._stepper_spec
    adaptive = getattr(stepper_spec.meta, "time_control", "fixed") == "adaptive"

    support = support or assess_capability(
        sim,
        plan=plan,
        record_vars=record_vars,
        dt=dt_use,
        transient=transient_use,
        adaptive=adaptive,
    )
    if not support.ok:
        return None

    state_rec_indices, aux_rec_indices, state_names, aux_names = _resolve_recording_selection(
        spec=sim.model.spec,
        record_vars=record_vars,
    )

    stepper_cfg = sim._default_stepper_cfg

    is_discrete = sim.model.spec.kind == "map"
    if is_discrete:
        if N is None:
            if T is None:
                raise ValueError("Provide N or T for discrete systems on fast path.")
            target_steps = int(round((float(T) - (t0 or sim_defaults.t0)) / dt_use))
        else:
            target_steps = int(N)
        horizon = None
    else:
        horizon = float(T if T is not None else sim_defaults.t_end)
        target_steps = None

    result = run_single_fastpath(
        model=sim.model,
        plan=plan,
        t0=float(t0 if t0 is not None else sim_defaults.t0),
        t_end=horizon,
        target_steps=target_steps,
        dt=dt_use,
        max_steps=max_steps_use,
        transient=transient_use,
        state_rec_indices=state_rec_indices,
        aux_rec_indices=aux_rec_indices,
        state_names=state_names,
        aux_names=aux_names,
        params=params,
        ic=ic,
        stepper_config=stepper_cfg,
    )

    seg = Segment(
        id=0,
        name=None,
        rec_start=0,
        rec_len=int(result.n),
        evt_start=0,
        evt_len=int(result.m),
        t_start=float(result.T[0]) if result.n > 0 else float(t0 or 0.0),
        t_end=float(result.t_final),
        step_start=0,
        step_end=int(result.step_count_final),
        resume=False,
        cfg_hash="fastpath",
    )
    return ResultsView(result, sim.model.spec, segments=[seg])


def fastpath_batch_for_sim(
    sim: Sim,
    *,
    plan: RecordingPlan,
    t0: float | None,
    T: float | None,
    N: int | None,
    dt: float | None,
    record_vars: Sequence[str] | None,
    transient: float | None,
    record_interval: int | None,
    max_steps: int | None,
    ic: np.ndarray,
    params: np.ndarray,
    support: FastpathSupport | None = None,
    parallel_mode: Literal["auto", "threads", "process", "none"] = "auto",
    max_workers: Optional[int] = None,
) -> list[ResultsView] | None:
    """
    Batch fastpath entry point for :class:`Sim`.

    Accepts stacked ``ic``/``params`` (shape (B, n_state)/(B, n_params)) and
    returns one ResultsView per run. Returns None when capability gate fails.
    """
    sim_defaults = sim.model.spec.sim
    dt_use = float(dt if dt is not None else sim._nominal_dt if sim._nominal_dt else sim_defaults.dt)
    transient_use = float(transient) if transient is not None else 0.0
    max_steps_use = int(max_steps if max_steps is not None else sim_defaults.max_steps)

    stepper_spec = sim._stepper_spec
    adaptive = getattr(stepper_spec.meta, "time_control", "fixed") == "adaptive"

    support = support or assess_capability(
        sim,
        plan=plan,
        record_vars=record_vars,
        dt=dt_use,
        transient=transient_use,
        adaptive=adaptive,
    )
    if not support.ok:
        return None

    state_rec_indices, aux_rec_indices, state_names, aux_names = _resolve_recording_selection(
        spec=sim.model.spec,
        record_vars=record_vars,
    )

    stepper_cfg = sim._default_stepper_cfg

    is_discrete = sim.model.spec.kind == "map"
    if is_discrete:
        if N is None:
            if T is None:
                raise ValueError("Provide N or T for discrete systems on fast path.")
            target_steps = int(round((float(T) - (t0 or sim_defaults.t0)) / dt_use))
        else:
            target_steps = int(N)
        horizon = None
    else:
        horizon = float(T if T is not None else sim_defaults.t_end)
        target_steps = None

    batch_results = run_batch_fastpath(
        model=sim.model,
        plan=plan,
        t0=float(t0 if t0 is not None else sim_defaults.t0),
        t_end=horizon,
        target_steps=target_steps,
        dt=dt_use,
        max_steps=max_steps_use,
        transient=transient_use,
        state_rec_indices=state_rec_indices,
        aux_rec_indices=aux_rec_indices,
        state_names=state_names,
        aux_names=aux_names,
        params=params,
        ic=ic,
        stepper_config=stepper_cfg,
        parallel_mode=parallel_mode,
        max_workers=max_workers,
    )

    views: list[ResultsView] = []
    for idx, result in enumerate(batch_results):
        seg = Segment(
            id=idx,
            name=None,
            rec_start=0,
            rec_len=int(result.n),
            evt_start=0,
            evt_len=int(result.m),
            t_start=float(result.T[0]) if result.n > 0 else float(t0 or 0.0),
            t_end=float(result.t_final),
            step_start=0,
            step_end=int(result.step_count_final),
            resume=False,
            cfg_hash="fastpath",
        )
        views.append(ResultsView(result, sim.model.spec, segments=[seg]))
    return views
