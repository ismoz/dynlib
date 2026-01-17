# src/dynlib/runtime/fastpath/executor.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Literal, NamedTuple, Optional, Sequence
import math
import os
import threading
import numpy as np

from dynlib.runtime.observers import ObserverModule, observer_noop_variational_step
from dynlib.runtime.fastpath.plans import RecordingPlan
from dynlib.runtime.fastpath.capability import assess_capability, FastpathSupport
from dynlib.runtime.results import Results
from dynlib.runtime.analysis_meta import build_observer_metadata
from dynlib.runtime.workspace import (
    initialize_lag_runtime_workspace,
    make_runtime_workspace,
    snapshot_workspace,
)
from dynlib.runtime.sim import Segment, Sim
from dynlib.runtime.results_api import ResultsView
from dynlib.runtime.runner_api import DONE, EARLY_EXIT, NAN_DETECTED, TRACE_OVERFLOW
from dynlib.compiler.codegen.runner_variants import RunnerVariant, get_runner

__all__ = [
    "run_single_fastpath",
    "run_batch_fastpath",
    "run_batch_fastpath_optimized",
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
    state_to_idx = {name: idx for idx, name in enumerate(state_names)}
    aux_names_all = list(spec.aux.keys()) if spec.aux else []
    aux_to_idx = {name: idx for idx, name in enumerate(aux_names_all)}

    state_idx: list[int] = []
    aux_idx: list[int] = []
    sel_states: list[str] = []
    sel_aux: list[str] = []

    for name in record_vars:
        if name.startswith("aux."):
            key = name[4:]
            aux_pos = aux_to_idx.get(key)
            if aux_pos is None:
                raise ValueError(f"Unknown aux variable: {key}")
            aux_idx.append(aux_pos)
            sel_aux.append(key)
            continue
        state_pos = state_to_idx.get(name)
        if state_pos is not None:
            state_idx.append(state_pos)
            sel_states.append(name)
            continue
        aux_pos = aux_to_idx.get(name)
        if aux_pos is not None:
            aux_idx.append(aux_pos)
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


class FastpathAnalysisResult(NamedTuple):
    status: int
    analysis_status: float
    analysis_matched_id: float
    step_count: int


class _WorkspaceBundle:
    def __init__(
        self,
        *,
        model,
        plan: RecordingPlan,
        ctx: _RunContext,
        state_rec_indices: np.ndarray,
        aux_rec_indices: np.ndarray,
        state_names: list[str],
        aux_names: list[str],
        stepper_config: np.ndarray | None,
        analysis: ObserverModule | None,
    ) -> None:
        self.model = model
        self.plan = plan
        self.ctx = ctx
        self.state_rec_indices = state_rec_indices
        self.aux_rec_indices = aux_rec_indices
        self.state_names = state_names
        self.aux_names = aux_names
        self.analysis = analysis

        self.dtype = model.dtype
        self.n_state = len(model.spec.states)
        self.n_aux = len(model.spec.aux)
        self.n_rec_states = len(state_rec_indices)
        self.n_rec_aux = len(aux_rec_indices)
        self.is_jit = _is_jitted_runner(model.runner)
        self.is_discrete = model.spec.kind == "map"

        if stepper_config is None:
            self.stepper_config = np.array([], dtype=np.float64)
        else:
            self.stepper_config = np.asarray(stepper_config, dtype=np.float64)

        rec_every = int(ctx.record_interval)
        total_steps = (
            ctx.target_steps
            if ctx.target_steps is not None
            else math.ceil(max(0.0, ctx.t_end - ctx.t0) / ctx.dt)
        )
        self.rec_every = rec_every
        self.total_steps = total_steps
        self.cap_rec = 1 if rec_every <= 0 else max(1, int(plan.capacity(total_steps=total_steps) + 1))
        self.cap_evt = 1
        self.max_log_width = _max_event_log_width(model.spec.events)

        # Recording buffers
        self.T = np.zeros((self.cap_rec,), dtype=np.float64)
        self.Y = (
            np.zeros((self.n_rec_states, self.cap_rec), dtype=self.dtype)
            if self.n_rec_states > 0
            else np.zeros((0, self.cap_rec), dtype=self.dtype)
        )
        self.AUX = (
            np.zeros((self.n_rec_aux, self.cap_rec), dtype=self.dtype)
            if self.n_rec_aux > 0
            else np.zeros((0, self.cap_rec), dtype=self.dtype)
        )
        self.STEP = np.zeros((self.cap_rec,), dtype=np.int64)
        self.FLAGS = np.zeros((self.cap_rec,), dtype=np.int32)

        self.EVT_CODE = np.zeros((self.cap_evt,), dtype=np.int32)
        self.EVT_INDEX = np.zeros((self.cap_evt,), dtype=np.int32)
        self.EVT_LOG_DATA = np.zeros((self.cap_evt, max(1, self.max_log_width)), dtype=self.dtype)

        # Work arrays
        self.y_curr = np.zeros((self.n_state,), dtype=self.dtype)
        self.y_prev = np.zeros((self.n_state,), dtype=self.dtype)
        self.y_prop = np.zeros((self.n_state,), dtype=self.dtype)
        self.t_prop = np.zeros((1,), dtype=np.float64)
        self.dt_next = np.zeros((1,), dtype=np.float64)
        self.err_est = np.zeros((1,), dtype=np.float64)
        self.evt_log_scratch = np.zeros((max(1, self.max_log_width),), dtype=self.dtype)

        self.user_break_flag = np.zeros((1,), dtype=np.int32)
        self.status_out = np.zeros((1,), dtype=np.int32)
        self.hint_out = np.zeros((1,), dtype=np.int32)
        self.i_out = np.zeros((1,), dtype=np.int64)
        self.step_out = np.zeros((1,), dtype=np.int64)
        self.t_out = np.zeros((1,), dtype=np.float64)

        # Analysis buffers (present even for base runner for ABI compatibility)
        self.variational_step_enabled = 0
        self.variational_step_fn = observer_noop_variational_step()
        if analysis is not None:
            ws_size = int(analysis.workspace_size)
            out_size = int(analysis.output_size)
            self.analysis_ws = np.zeros((ws_size,), dtype=self.dtype) if ws_size > 0 else np.zeros((0,), dtype=self.dtype)
            self.analysis_out = np.zeros((out_size,), dtype=self.dtype) if out_size > 0 else np.zeros((0,), dtype=self.dtype)

            trace_width = analysis.trace.width if analysis.trace else 0
            if trace_width > 0:
                if analysis.trace is None:
                    raise RuntimeError("observer trace requested but TracePlan is missing")
                trace_cap = int(analysis.trace.capacity(total_steps=total_steps))
                if trace_cap <= 0:
                    raise RuntimeError("TracePlan must provide positive capacity for fastpath observers.")
                self.analysis_trace = np.zeros((trace_cap, trace_width), dtype=self.dtype)
                self.analysis_trace_cap = int(trace_cap)
                self.analysis_trace_stride = int(analysis.trace.record_interval())
            else:
                self.analysis_trace = np.zeros((0, 0), dtype=self.dtype)
                self.analysis_trace_cap = 0
                self.analysis_trace_stride = 0
            self.analysis_trace_count = np.zeros((1,), dtype=np.int64)

            runner_var = getattr(analysis, "runner_variational_step", None)
            if callable(runner_var):
                var_fn = runner_var(jit=self.is_jit)
                if var_fn is not None:
                    self.variational_step_enabled = 1
                    self.variational_step_fn = var_fn
        else:
            self.analysis_ws = np.zeros((0,), dtype=self.dtype)
            self.analysis_out = np.zeros((0,), dtype=self.dtype)
            self.analysis_trace = np.zeros((0, 0), dtype=self.dtype)
            self.analysis_trace_count = np.zeros((1,), dtype=np.int64)
            self.analysis_trace_cap = 0
            self.analysis_trace_stride = 0

        stop_phase_mask = 0
        stop_spec = getattr(model.spec.sim, "stop", None)
        if stop_spec is not None:
            phase = stop_spec.phase
            if phase in ("pre", "both"):
                stop_phase_mask |= 1
            if phase in ("post", "both"):
                stop_phase_mask |= 2
        if analysis is not None:
            stop_phase_mask |= int(getattr(analysis, "stop_phase_mask", 0))

        self.runtime_ws = make_runtime_workspace(
            lag_state_info=model.lag_state_info,
            dtype=self.dtype,
            n_aux=self.n_aux,
            stop_enabled=stop_phase_mask != 0,
            stop_phase_mask=stop_phase_mask,
        )
        self.stepper_ws = model.make_stepper_workspace() if model.make_stepper_workspace else None
        self.lag_state_info = model.lag_state_info

        variant = RunnerVariant.FASTPATH_ANALYSIS if analysis is not None else RunnerVariant.FASTPATH
        self.runner = get_runner(
            variant,
            model_hash=model.spec_hash,
            stepper_name=model.stepper_name,
            analysis=analysis,
            dtype=self.dtype,
            jit=self.is_jit,
            discrete=self.is_discrete,
        )

        self.i_start = np.int64(0)
        self.step_start = np.int64(0)

    def reset(self, ic: np.ndarray) -> None:
        np.copyto(self.y_curr, ic, casting="unsafe")
        np.copyto(self.y_prev, ic, casting="unsafe")
        if self.runtime_ws.stop_flag.size > 0:
            self.runtime_ws.stop_flag[0] = 0
        if self.runtime_ws.aux_values.size > 0:
            self.runtime_ws.aux_values.fill(0)
        if self.lag_state_info:
            initialize_lag_runtime_workspace(
                self.runtime_ws,
                lag_state_info=self.lag_state_info,
                y_curr=self.y_curr,
            )
        if self.analysis_trace_count.size > 0:
            self.analysis_trace_count[0] = 0
        if self.analysis_out.size > 0:
            self.analysis_out.fill(0.0)
        self.user_break_flag[0] = 0
        self.status_out[0] = 0
        self.hint_out[0] = 0
        self.i_out[0] = 0
        self.step_out[0] = 0
        self.t_out[0] = 0.0

    def run_analysis_only(self, *, ic: np.ndarray, params: np.ndarray) -> FastpathAnalysisResult:
        self.reset(ic)
        status = self.runner(
            float(self.ctx.t0),
            float(self.ctx.target_steps if self.ctx.target_steps is not None else self.ctx.t_end),
            float(self.ctx.dt),
            int(self.ctx.max_steps),
            int(self.n_state),
            int(self.rec_every),
            self.y_curr,
            self.y_prev,
            params,
            self.runtime_ws,
            self.stepper_ws,
            self.stepper_config,
            self.y_prop,
            self.t_prop,
            self.dt_next,
            self.err_est,
            self.T,
            self.Y,
            self.AUX,
            self.STEP,
            self.FLAGS,
            self.EVT_CODE,
            self.EVT_INDEX,
            self.EVT_LOG_DATA,
            self.evt_log_scratch,
            self.analysis_ws,
            self.analysis_out,
            self.analysis_trace,
            self.analysis_trace_count,
            int(self.analysis_trace_cap),
            int(self.analysis_trace_stride),
            int(self.variational_step_enabled),
            self.variational_step_fn,
            self.i_start,
            self.step_start,
            int(self.cap_rec),
            int(self.cap_evt),
            self.user_break_flag,
            self.status_out,
            self.hint_out,
            self.i_out,
            self.step_out,
            self.t_out,
            self.model.stepper,
            self.model.rhs,
            self.model.events_pre,
            self.model.events_post,
            self.model.update_aux,
            self.state_rec_indices,
            self.aux_rec_indices,
            self.n_rec_states,
            self.n_rec_aux,
        )

        status_value = int(status)
        if status_value not in (DONE, EARLY_EXIT, TRACE_OVERFLOW, NAN_DETECTED):
            raise RuntimeError(f"Fastpath runner exited with status {status_value}")

        analysis_status = 0.0
        analysis_matched_id = -1.0
        if self.analysis_out.size >= 2:
            analysis_status = float(self.analysis_out[0])
            analysis_matched_id = float(self.analysis_out[1])

        return FastpathAnalysisResult(
            status=status_value,
            analysis_status=analysis_status,
            analysis_matched_id=analysis_matched_id,
            step_count=int(self.step_out[0]),
        )

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
    analysis: ObserverModule | None = None,
):
    dtype = model.dtype
    n_state = len(model.spec.states)
    n_aux = len(model.spec.aux)
    n_rec_states = len(state_rec_indices)
    n_rec_aux = len(aux_rec_indices)
    is_jit = _is_jitted_runner(model.runner)
    is_discrete = model.spec.kind == "map"

    rec_every = int(ctx.record_interval)
    total_steps = ctx.target_steps if ctx.target_steps is not None else math.ceil(max(0.0, ctx.t_end - ctx.t0) / ctx.dt)
    # Pad by one to avoid edge cases where endpoint snapping records an extra sample.
    cap_rec = 1 if rec_every <= 0 else max(1, int(plan.capacity(total_steps=total_steps) + 1))
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

    # Analysis buffers (present even for base runner for ABI compatibility)
    variational_step_enabled = 0
    variational_step_fn = observer_noop_variational_step()

    if analysis is not None:
        analysis_kind = int(analysis.analysis_kind)
        ws_size = int(analysis.workspace_size)
        out_size = int(analysis.output_size)
        analysis_ws = np.zeros((ws_size,), dtype=dtype) if ws_size > 0 else np.zeros((0,), dtype=dtype)
        analysis_out = np.zeros((out_size,), dtype=dtype) if out_size > 0 else np.zeros((0,), dtype=dtype)

        trace_width = analysis.trace.width if analysis.trace else 0
        if trace_width > 0:
            if analysis.trace is None:
                raise RuntimeError("analysis trace requested but TracePlan is missing")
            trace_cap = int(analysis.trace.capacity(total_steps=total_steps))
            if trace_cap <= 0:
                raise RuntimeError("TracePlan must provide positive capacity for fastpath analysis.")
            analysis_trace = np.zeros((trace_cap, trace_width), dtype=dtype)
            analysis_trace_cap = int(trace_cap)
            analysis_trace_stride = int(analysis.trace.record_interval())
        else:
            analysis_trace = np.zeros((0, 0), dtype=dtype)
            analysis_trace_cap = 0
            analysis_trace_stride = 0
        analysis_trace_count = np.zeros((1,), dtype=np.int64)
        runner_var = getattr(analysis, "runner_variational_step", None)
        if callable(runner_var):
            var_fn = runner_var(jit=is_jit)
            if var_fn is not None:
                variational_step_enabled = 1
                variational_step_fn = var_fn
    else:
        analysis_kind = 0
        analysis_ws = np.zeros((0,), dtype=dtype)
        analysis_out = np.zeros((0,), dtype=dtype)
        analysis_trace = np.zeros((0, 0), dtype=dtype)
        analysis_trace_count = np.zeros((1,), dtype=np.int64)
        analysis_trace_cap = 0
        analysis_trace_stride = 0
        variational_step_enabled = 0
        variational_step_fn = observer_noop_variational_step()

    variant = RunnerVariant.FASTPATH_ANALYSIS if analysis is not None else RunnerVariant.FASTPATH
    runner = get_runner(
        variant,
        model_hash=model.spec_hash,
        stepper_name=model.stepper_name,
        analysis=analysis,
        dtype=dtype,
        jit=is_jit,
        discrete=is_discrete,
    )

    # Call the runner (note: no analysis_kind or hook dispatch arguments - hooks are baked in)
    status = runner(
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
        analysis_ws,
        analysis_out,
        analysis_trace,
        analysis_trace_count,
        int(analysis_trace_cap),
        int(analysis_trace_stride),
        int(variational_step_enabled),
        variational_step_fn,
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

    status_value = int(status)
    filled = int(i_out[0])
    evt_filled = max(0, int(hint_out[0]))
    overflowed = status_value == TRACE_OVERFLOW
    if status_value not in (DONE, EARLY_EXIT, TRACE_OVERFLOW, NAN_DETECTED):
        raise RuntimeError(f"Fastpath runner exited with status {status_value}")

    analysis_trace_view = None
    analysis_trace_filled = None
    analysis_trace_offset = None
    analysis_out_payload = None
    analysis_stride_payload = None
    analysis_modules = None
    analysis_meta = None
    if analysis is not None:
        analysis_modules = tuple(getattr(analysis, "modules", (analysis,)))
        analysis_out_payload = analysis_out if analysis_out.size > 0 else None
        filled_raw = int(analysis_trace_count[0])
        analysis_trace_filled = filled_raw
        analysis_stride_payload = int(analysis_trace_stride) if analysis_trace_stride else None
        cap_payload = int(analysis_trace_cap) if analysis_trace_cap else None
        if analysis_trace.shape[0] > 0:
            trace_view = analysis_trace[:filled_raw, :]
            analysis_trace_filled = trace_view.shape[0]
            trace_offset = 0
            if analysis.trace:
                sl = analysis.finalize_trace(filled_raw)
                if sl is not None:
                    trace_view = trace_view[sl]
                    analysis_trace_filled = trace_view.shape[0]
                    trace_offset = int(sl.start) if sl.start is not None else 0
            analysis_trace_view = trace_view
            analysis_trace_offset = trace_offset
        analysis_meta = build_observer_metadata(
            analysis_modules,
            analysis_kind=analysis_kind,
            trace_stride=analysis_stride_payload,
            trace_capacity=cap_payload,
            overflow=overflowed,
        )

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
        status=status_value,
        final_state=np.array(y_curr, copy=True),
        final_params=np.array(params, copy=True),
        t_final=float(t_out[0]),
        final_dt=float(dt_next[0]) if filled > 0 else float(ctx.dt),
        step_count_final=int(step_out[0]),
        final_workspace=final_ws,
        state_names=state_names,
        aux_names=aux_names,
        analysis_out=analysis_out_payload,
        analysis_trace=analysis_trace_view,
        analysis_trace_filled=analysis_trace_filled,
        analysis_trace_stride=analysis_stride_payload,
        analysis_trace_offset=analysis_trace_offset,
        analysis_modules=analysis_modules,
        analysis_meta=analysis_meta,
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
    return np.ascontiguousarray(ic_arr), np.ascontiguousarray(params_arr), batch


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
    observers: ObserverModule | None = None,
) -> Results:
    """Core fastpath execution using the compiled runner."""
    analysis = observers
    if stepper_config is None:
        stepper_config = np.array([], dtype=np.float64)
    dtype = model.dtype
    n_aux = len(model.spec.aux)
    is_discrete = model.spec.kind == "map"

    stop_phase_mask = 0
    stop_spec = getattr(model.spec.sim, "stop", None)
    if stop_spec is not None:
        phase = stop_spec.phase
        if phase in ("pre", "both"):
            stop_phase_mask |= 1
        if phase in ("post", "both"):
            stop_phase_mask |= 2
    if analysis is not None:
        stop_phase_mask |= int(getattr(analysis, "stop_phase_mask", 0))
    runtime_ws = make_runtime_workspace(
        lag_state_info=model.lag_state_info,
        dtype=dtype,
        n_aux=n_aux,
        stop_enabled=stop_phase_mask != 0,
        stop_phase_mask=stop_phase_mask,
    )
    stepper_ws = model.make_stepper_workspace() if model.make_stepper_workspace else None

    base_t0 = float(t0)
    time_shift = 0.0
    run_t_end = t_end

    # Optional transient warm-up (no recording)
    if transient > 0.0:
        if is_discrete:
            trans_steps = int(transient)
            warm_t_end = float(t0 + float(trans_steps) * float(dt))
            warm_target_steps = trans_steps
        else:
            trans_steps = None
            warm_t_end = float(t0 + float(transient))
            warm_target_steps = None
        warm_ctx = _RunContext(
            t0=float(t0),
            t_end=warm_t_end,
            target_steps=warm_target_steps if target_steps is not None else None,
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
            analysis=None,
        )
        t0 = warm_result.t_final
        # Note: target_steps is the number of steps to RECORD after transient.
        # Do NOT reduce it - the transient warm-up has already been performed.
        # Warm-up updates runtime_ws/stepper_ws in place; reuse them.
        ic = warm_result.final_state
        params = warm_result.final_params
        dt = float(warm_result.final_dt) if warm_result.final_dt != 0.0 else dt
        time_shift = float(warm_result.t_final) - base_t0
        if not is_discrete and run_t_end is not None:
            run_t_end = float(run_t_end + time_shift)

    run_ctx = _RunContext(
        t0=float(t0),
        t_end=float(run_t_end if run_t_end is not None else t0),
        target_steps=target_steps,
        dt=float(dt),
        max_steps=max_steps,
        transient=0.0,
        record_interval=plan.record_interval(),
    )
    result = _call_runner(
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
        analysis=analysis,
    )
    if time_shift != 0.0 and result.n > 0:
        # Match Sim.run transient semantics: recorded time starts at original t0.
        result.T[: result.n] = result.T[: result.n] - time_shift
    return result


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
    observers: ObserverModule | None = None,
) -> list[Results]:
    """
    Batch fastpath execution across multiple IC/parameter sets.

    For JIT builds, threads will leverage the numba-compiled runner (GIL-free).
    For pure Python builds, a thread pool is used unless ``parallel_mode="none"``.
    """
    analysis = observers
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
                observers=analysis,
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
            observers=analysis,
        )

    if backend == "none" or max_workers == 1:
        return [_run(i) for i in range(batch)]

    if backend == "process":
        # ProcessPool can be brittle for compiled runners; fall back to threads.
        backend = "threads"

    if backend != "threads":
        raise ValueError(f"Unknown parallel_mode {parallel_mode!r}")

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(32, cpu_count + 4)
    max_workers = max(1, int(max_workers))

    if batch <= max_workers:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_run, range(batch)))

    chunk_size = (batch + max_workers - 1) // max_workers
    chunks = [
        (start, min(start + chunk_size, batch))
        for start in range(0, batch, chunk_size)
    ]

    def _run_chunk(start: int, stop: int) -> list[Results]:
        return [_run(i) for i in range(start, stop)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_chunk, start, stop) for start, stop in chunks]
        results: list[Results] = []
        for fut in futures:
            results.extend(fut.result())
        return results


def run_batch_fastpath_optimized(
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
    observers: ObserverModule | None = None,
    analysis_only: bool = False,
) -> list[Results] | list[FastpathAnalysisResult]:
    """
    Optimized batch fastpath execution with reusable workspaces.

    When analysis_only is True, returns FastpathAnalysisResult entries and skips
    full Results construction.
    """
    analysis = observers
    if not analysis_only:
        return run_batch_fastpath(
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
            params=params,
            ic=ic,
            stepper_config=stepper_config,
            parallel_mode=parallel_mode,
            max_workers=max_workers,
            observers=analysis,
        )

    if analysis is None:
        raise ValueError("analysis_only=True requires an analysis module.")
    if transient:
        raise ValueError("analysis_only fastpath does not support transient warm-up.")

    n_state = len(model.spec.states)
    n_params = len(model.spec.params)
    ic_batch, params_batch, batch = _normalize_batch_inputs(
        ic=ic, params=params, n_state=n_state, n_params=n_params
    )
    if batch == 0:
        return []

    ctx = _RunContext(
        t0=float(t0),
        t_end=float(t_end if t_end is not None else t0),
        target_steps=target_steps,
        dt=float(dt),
        max_steps=max_steps,
        transient=0.0,
        record_interval=plan.record_interval(),
    )

    backend = parallel_mode
    is_jit = _is_jitted_runner(model.runner)
    if backend == "auto":
        backend = "threads" if is_jit else "threads"

    if backend == "process":
        backend = "threads"
    if backend not in {"threads", "none"}:
        raise ValueError(f"Unknown parallel_mode {parallel_mode!r}")

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(32, cpu_count + 4)
    max_workers = max(1, int(max_workers))

    def _make_bundle() -> _WorkspaceBundle:
        return _WorkspaceBundle(
            model=model,
            plan=plan,
            ctx=ctx,
            state_rec_indices=state_rec_indices,
            aux_rec_indices=aux_rec_indices,
            state_names=state_names,
            aux_names=aux_names,
            stepper_config=stepper_config,
            analysis=analysis,
        )

    if backend == "none" or max_workers == 1 or batch == 1:
        bundle = _make_bundle()
        return [
            bundle.run_analysis_only(ic=ic_batch[i], params=params_batch[i])
            for i in range(batch)
        ]

    call_token = object()
    thread_local = threading.local()

    def _get_bundle() -> _WorkspaceBundle:
        bundle = getattr(thread_local, "bundle", None)
        token = getattr(thread_local, "bundle_token", None)
        if bundle is None or token is not call_token:
            bundle = _make_bundle()
            thread_local.bundle = bundle
            thread_local.bundle_token = call_token
        return bundle

    def _run(idx: int) -> FastpathAnalysisResult:
        bundle = _get_bundle()
        return bundle.run_analysis_only(ic=ic_batch[idx], params=params_batch[idx])

    if batch <= max_workers:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_run, range(batch)))

    chunk_size = (batch + max_workers - 1) // max_workers
    chunks = [
        (start, min(start + chunk_size, batch))
        for start in range(0, batch, chunk_size)
    ]

    def _run_chunk(start: int, stop: int) -> list[FastpathAnalysisResult]:
        return [_run(i) for i in range(start, stop)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_chunk, start, stop) for start, stop in chunks]
        results: list[FastpathAnalysisResult] = []
        for fut in futures:
            results.extend(fut.result())
        return results


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
    observers: ObserverModule | None = None,
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

    analysis = observers
    if analysis is not None:
        analysis.validate_stepper(stepper_spec)

    support = support or assess_capability(
        sim,
        plan=plan,
        record_vars=record_vars,
        dt=dt_use,
        transient=transient_use,
        adaptive=adaptive,
        observers=analysis,
    )
    if not support.ok:
        return None

    if record_interval is not None and record_interval != plan.record_interval():
        raise ValueError(
            f"record_interval ({record_interval}) must match plan stride ({plan.record_interval()}) for fast path"
        )

    state_rec_indices, aux_rec_indices, state_names, aux_names = _resolve_recording_selection(
        spec=sim.model.spec,
        record_vars=record_vars,
    )

    stepper_cfg = sim._default_stepper_cfg
    t0_use = float(t0) if t0 is not None else float(sim_defaults.t0)

    is_discrete = sim.model.spec.kind == "map"
    if is_discrete:
        if N is None:
            if T is None:
                raise ValueError("Provide N or T for discrete systems on fast path.")
            target_steps = int(round((float(T) - t0_use) / dt_use))
        else:
            target_steps = int(N)
        horizon = None
    else:
        horizon = float(T if T is not None else sim_defaults.t_end)
        target_steps = None

    result = run_single_fastpath(
        model=sim.model,
        plan=plan,
        t0=t0_use,
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
        observers=analysis,
    )

    seg = Segment(
        id=0,
        name=None,
        rec_start=0,
        rec_len=int(result.n),
        evt_start=0,
        evt_len=int(result.m),
        t_start=float(result.T[0]) if result.n > 0 else t0_use,
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
    observers: ObserverModule | None = None,
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

    analysis = observers
    if analysis is not None:
        analysis.validate_stepper(stepper_spec)

    support = support or assess_capability(
        sim,
        plan=plan,
        record_vars=record_vars,
        dt=dt_use,
        transient=transient_use,
        adaptive=adaptive,
        observers=analysis,
    )
    if not support.ok:
        return None

    if record_interval is not None and record_interval != plan.record_interval():
        raise ValueError(
            f"record_interval ({record_interval}) must match plan stride ({plan.record_interval()}) for fast path"
        )

    state_rec_indices, aux_rec_indices, state_names, aux_names = _resolve_recording_selection(
        spec=sim.model.spec,
        record_vars=record_vars,
    )

    stepper_cfg = sim._default_stepper_cfg
    t0_use = float(t0) if t0 is not None else float(sim_defaults.t0)

    is_discrete = sim.model.spec.kind == "map"
    if is_discrete:
        if N is None:
            if T is None:
                raise ValueError("Provide N or T for discrete systems on fast path.")
            target_steps = int(round((float(T) - t0_use) / dt_use))
        else:
            target_steps = int(N)
        horizon = None
    else:
        horizon = float(T if T is not None else sim_defaults.t_end)
        target_steps = None

    batch_results = run_batch_fastpath(
        model=sim.model,
        plan=plan,
        t0=t0_use,
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
        observers=analysis,
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
            t_start=float(result.T[0]) if result.n > 0 else t0_use,
            t_end=float(result.t_final),
            step_start=0,
            step_end=int(result.step_count_final),
            resume=False,
            cfg_hash="fastpath",
        )
        views.append(ResultsView(result, sim.model.spec, segments=[seg]))
    return views
