from __future__ import annotations
from typing import Callable, Mapping, Dict, Optional, Tuple
import warnings
import math
import numpy as np

from dynlib.runtime.runner_api import (
    OK, STEPFAIL, NAN_DETECTED, EARLY_EXIT, DONE, GROW_REC, GROW_EVT, USER_BREAK, TRACE_OVERFLOW, Status,
)
from dynlib.runtime.buffers import (
    allocate_pools, grow_rec_arrays, grow_evt_arrays,
)
from dynlib.runtime.results import Results
from dynlib.runtime.analysis_meta import build_observer_metadata
from dynlib.runtime.workspace import (
    make_runtime_workspace,
    initialize_lag_runtime_workspace,
    snapshot_workspace,
    restore_workspace,
)
from dynlib.runtime.initial_step import WRMSConfig, choose_initial_dt_wrms
from dynlib.runtime.observers import ObserverModule, observer_noop_variational_step
from dynlib.compiler.codegen.runner_variants import RunnerVariant, get_runner

__all__ = ["run_with_wrapper"]

# Shared helper: detect numba-compiled callables
def _is_jitted(obj) -> bool:
    """Best-effort detection of a numba-compiled callable (has signatures)."""
    return bool(getattr(obj, "signatures", None))

# ------------------------------ Wrapper --------------------------------------

def run_with_wrapper(
    *,
    runner: Callable[..., np.int32],
    stepper: Callable[..., np.int32],
    rhs: Callable[..., None],
    events_pre: Callable[..., None],
    events_post: Callable[..., None],
    update_aux: Callable[..., None],
    dtype: np.dtype,
    n_state: int,
    n_aux: int,
    stop_phase_mask: int = 0,
    # sim config
    t0: float,
    t_end: float,
    dt_init: float | None,
    max_steps: int,
    record: bool,
    record_interval: int,
    # NEW: selective recording
    state_record_indices: np.ndarray,  # indices of states to record
    aux_record_indices: np.ndarray,    # indices of aux to record
    state_names: list[str],            # names of states being recorded
    aux_names: list[str],              # names of aux being recorded
    # initial state/params
    ic: np.ndarray,
    params: np.ndarray,
    # capacities (can be small to force growth)
    cap_rec: int = 1024,
    cap_evt: int = 1,
    max_log_width: int = 0,  # maximum log width across all events
    # NEW: stepper configuration
    stepper_config: np.ndarray = None,
    workspace_seed: Mapping[str, object] | None = None,
    discrete: bool = False,
    target_steps: Optional[int] = None,
    lag_state_info: Tuple[Tuple[int, int, int, int], ...] | None = None,
    make_stepper_workspace: Callable[[], object] | None = None,
    wrms_cfg: WRMSConfig | None = None,
    observers: ObserverModule | None = None,
    adaptive: bool = False,
    model_hash: str | None = None,
    stepper_name: str | None = None,
    use_runner_variants: bool = True,
) -> Results:
    """
    Orchestrates runner calls while managing workspaces and growth/re-entry.

    Runner ABI (frozen) — summarized here (see runner_api.py for full doc):
        status = runner(
            t0, horizon, dt_init,
            max_steps, n_state, record_interval,
            y_curr, y_prev, params,
            runtime_ws, stepper_ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est,
            T, Y, AUX, STEP, FLAGS,
            EVT_CODE, EVT_INDEX, EVT_LOG_DATA,
            evt_log_scratch,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
            variational_step_enabled, variational_step_fn,
            i_start, step_start, cap_rec, cap_evt,
            user_break_flag, status_out, hint_out,
            i_out, step_out, t_out,
            stepper, rhs, events_pre, events_post, update_aux,
            state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux
        ) -> int32

    Notes:
      - Workspaces are allocated once per wrapper call and passed through to the runner.
      - hint_out[0] is used here as the current **event log cursor m** (by convention).
      - stepper_config is a read-only float64 array containing runtime configuration.
      - When ``discrete=True`` the runner horizon is interpreted as an iteration
        budget ``N`` (second argument). Otherwise it is ``t_end`` (continuous time).
      - state_rec_indices and aux_rec_indices control selective recording.
      - observer hooks are baked into runner variants; pre-step runs after pre-events
        and post-step runs after commit + aux update.
    """
    analysis = observers
    if discrete:
        if target_steps is None:
            raise ValueError("target_steps must be provided when discrete=True")
        steps_horizon = int(target_steps)
        if steps_horizon < 0:
            raise ValueError("target_steps must be non-negative")
    else:
        steps_horizon = None

    assert ic.shape == (n_state,)
    y_curr = np.array(ic, dtype=dtype, copy=True)
    y_prev = np.array(ic, dtype=dtype, copy=True)  # will be set by runner after first commit
    
    # Default stepper_config to empty array if not provided
    if stepper_config is None:
        stepper_config = np.array([], dtype=np.float64)
    else:
        # Ensure it's the right dtype
        stepper_config = np.asarray(stepper_config, dtype=np.float64)

    effective_stop_phase_mask = int(stop_phase_mask)
    if analysis is not None:
        effective_stop_phase_mask |= int(getattr(analysis, "stop_phase_mask", 0))

    runtime_ws = make_runtime_workspace(
        lag_state_info=lag_state_info,
        dtype=dtype,
        n_aux=n_aux,
        stop_enabled=effective_stop_phase_mask != 0,
        stop_phase_mask=effective_stop_phase_mask,
    )
    stepper_ws = make_stepper_workspace() if make_stepper_workspace else None

    # Allocate recording/event pools with selective recording
    n_rec_states = len(state_record_indices)
    n_rec_aux = len(aux_record_indices)
    
    # Allocate Y for selected states only
    T = np.zeros((cap_rec,), dtype=np.float64)
    Y = np.zeros((n_rec_states, cap_rec), dtype=dtype) if n_rec_states > 0 else np.zeros((0, cap_rec), dtype=dtype)
    AUX = np.zeros((n_rec_aux, cap_rec), dtype=dtype) if n_rec_aux > 0 else np.zeros((0, cap_rec), dtype=dtype)
    STEP = np.zeros((cap_rec,), dtype=np.int64)
    FLAGS = np.zeros((cap_rec,), dtype=np.int32)
    
    # Event log arrays
    EVT_CODE = np.zeros((cap_evt,), dtype=np.int32)
    EVT_INDEX = np.zeros((cap_evt,), dtype=np.int32)
    EVT_LOG_DATA = np.zeros((cap_evt, max(1, max_log_width)), dtype=dtype)

    # Apply workspace seed (for resume scenarios) before entering runner
    if workspace_seed:
        restore_workspace(stepper_ws, workspace_seed.get("stepper"))  # type: ignore[arg-type]
        restore_workspace(runtime_ws, workspace_seed.get("runtime"))  # type: ignore[arg-type]
    elif lag_state_info:
        initialize_lag_runtime_workspace(runtime_ws, lag_state_info=lag_state_info, y_curr=y_curr)

    # Proposals / outs (len-1 where applicable)
    y_prop  = np.zeros((n_state,), dtype=dtype)
    # Stepper control values must be float64 (not model dtype) to avoid truncation
    t_prop  = np.zeros((1,), dtype=np.float64)
    dt_next = np.zeros((1,), dtype=np.float64)
    err_est = np.zeros((1,), dtype=np.float64)
    
    # Event log scratch buffer
    evt_log_scratch = np.zeros((max(1, max_log_width),), dtype=dtype)

    # Control + cursors/outs
    user_break_flag = np.zeros((1,), dtype=np.int32)
    status_out      = np.zeros((1,), dtype=np.int32)
    hint_out        = np.zeros((1,), dtype=np.int32)    # convention: event cursor m
    i_out           = np.zeros((1,), dtype=np.int64)    # record cursor n
    step_out        = np.zeros((1,), dtype=np.int64)    # global step count
    t_out           = np.zeros((1,), dtype=np.float64)  # committed time

    # Start cursors
    i_start = np.int64(0)
    step_start = np.int64(0)

    # Recording at t0 is part of the **runner** discipline; wrapper just passes flags
    rec_every = int(record_interval) if record else 0  # runner may treat 0 as "no record except explicit"

    # Determine initial dt based on stepping mode.
    if discrete:
        if dt_init is None:
            raise ValueError("Discrete steppers require dt to be specified.")
        dt_curr = float(dt_init)
    elif not adaptive:
        if dt_init is None:
            raise ValueError("Fixed-step steppers require dt to be specified.")
        dt_curr = float(dt_init)
    else:
        if wrms_cfg is not None:
            dt_curr = choose_initial_dt_wrms(
                rhs=rhs,
                runtime_ws=runtime_ws,
                t0=float(t0),
                y0=y_curr,
                params=params,
                t_end=float(t_end),
                cfg=wrms_cfg,
            )
        elif dt_init is not None:
            dt_curr = float(dt_init)
        else:
            dt_curr = abs(float(t_end) - float(t0)) * 1e-3
            if dt_curr == 0.0:
                dt_curr = 1e-6

    is_jit_runner = all(
        _is_jitted(obj) for obj in (stepper, rhs, events_pre, events_post, update_aux)
    )
    if model_hash is None or stepper_name is None:
        raise ValueError("model_hash and stepper_name must be provided for runner selection")

    # Analysis buffers (persist across re-entry)
    if analysis is not None:
        analysis_kind = int(analysis.analysis_kind)
        ws_size = int(analysis.workspace_size)
        out_size = int(analysis.output_size)
        analysis_ws = np.zeros((ws_size,), dtype=dtype) if ws_size > 0 else np.zeros((0,), dtype=dtype)
        analysis_out = np.zeros((out_size,), dtype=dtype) if out_size > 0 else np.zeros((0,), dtype=dtype)

        trace_width = analysis.trace.width if analysis.trace else 0
        if trace_width > 0:
            if analysis.trace is None:
                raise ValueError("observer trace requested but TracePlan is missing")
            if adaptive:
                raise ValueError("observer traces require fixed-step execution")
            if discrete and steps_horizon is not None:
                total_steps = int(steps_horizon)
            else:
                total_steps = int(max_steps)
            trace_cap = int(analysis.trace.capacity(total_steps=total_steps))
            if trace_cap <= 0:
                raise ValueError("TracePlan must provide positive capacity for runtime observers")
            analysis_trace = np.zeros((trace_cap, trace_width), dtype=dtype)
            analysis_trace_cap = np.int64(trace_cap)
            analysis_trace_stride = np.int64(analysis.trace.record_interval())
        else:
            analysis_trace = np.zeros((0, 0), dtype=dtype)
            analysis_trace_cap = np.int64(0)
            analysis_trace_stride = np.int64(0)
        analysis_trace_count = np.zeros((1,), dtype=np.int64)
        variational_step_enabled = np.int32(0)
        variational_step_fn = observer_noop_variational_step()
        runner_var = getattr(analysis, "runner_variational_step", None)
        if callable(runner_var):
            var_fn = runner_var(jit=is_jit_runner)
            if var_fn is not None:
                variational_step_enabled = np.int32(1)
                variational_step_fn = var_fn
    else:
        analysis_kind = 0
        analysis_ws = np.zeros((0,), dtype=dtype)
        analysis_out = np.zeros((0,), dtype=dtype)
        analysis_trace = np.zeros((0, 0), dtype=dtype)
        analysis_trace_count = np.zeros((1,), dtype=np.int64)
        analysis_trace_cap = np.int64(0)
        analysis_trace_stride = np.int64(0)
        variational_step_enabled = np.int32(0)
        variational_step_fn = observer_noop_variational_step()
    analysis_modules = tuple(getattr(analysis, "modules", (analysis,))) if analysis is not None else None

    def _analysis_meta(overflow: bool = False):
        if analysis_modules is None:
            return None
        stride_payload = int(analysis_trace_stride) if int(analysis_trace_stride) > 0 else None
        cap_payload = int(analysis_trace_cap) if int(analysis_trace_cap) > 0 else None
        return build_observer_metadata(
            analysis_modules,
            analysis_kind=analysis_kind,
            trace_stride=stride_payload,
            trace_capacity=cap_payload,
            overflow=overflow,
        )

    def _finalize_analysis_trace_view() -> tuple[np.ndarray | None, int | None, int]:
        if analysis is None or analysis_trace.shape[0] == 0:
            return None, None, 0
        filled_raw = int(analysis_trace_count[0])
        trace_view = analysis_trace[:filled_raw, :]
        filled = trace_view.shape[0]
        offset = 0
        if analysis.trace:
            sl = analysis.finalize_trace(filled_raw)
            if sl is not None:
                trace_view = trace_view[sl]
                filled = trace_view.shape[0]
                offset = int(sl.start) if sl.start is not None else 0
        return trace_view, filled, offset

    # Track the committed (t, dt) so re-entries resume from the correct point.
    t_curr = float(t0)

    variant = RunnerVariant.ANALYSIS if analysis is not None else RunnerVariant.BASE
    runner = get_runner(
        variant,
        model_hash=model_hash,
        stepper_name=stepper_name,
        analysis=analysis,
        dtype=dtype,
        jit=is_jit_runner,
        discrete=discrete,
    )

    # Attempt/re-entry loop
    while True:
        horizon_arg = steps_horizon if discrete else float(t_end)
        status = runner(
            t_curr, horizon_arg, dt_curr,
            int(max_steps), int(n_state), int(rec_every),
            y_curr, y_prev, params,
            runtime_ws,
            stepper_ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est,
            T, Y, AUX, STEP, FLAGS,
            EVT_CODE, EVT_INDEX, EVT_LOG_DATA,
            evt_log_scratch,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, int(analysis_trace_cap), int(analysis_trace_stride),
            int(variational_step_enabled), variational_step_fn,
            i_start, step_start, int(cap_rec), int(cap_evt),
            user_break_flag, status_out, hint_out,
            i_out, step_out, t_out,
            stepper, rhs, events_pre, events_post, update_aux,
            state_record_indices, aux_record_indices, n_rec_states, n_rec_aux,
        )

        status_value = int(status)

        # Filled cursors reported by runner
        n_filled = int(i_out[0])            # records
        m_filled = max(0, int(hint_out[0])) # events (by convention)
        step_curr = int(step_out[0])

        if status_value in (DONE, EARLY_EXIT):
            final_state = np.array(y_curr, copy=True)
            final_params = np.array(params, copy=True)
            final_ws = {
                "stepper": snapshot_workspace(stepper_ws),
                "runtime": snapshot_workspace(runtime_ws),
            }
            final_dt = float(dt_next[0]) if step_curr > 0 else float(dt_curr)
            t_final = float(t_out[0])
            trace_view, trace_filled, trace_offset = _finalize_analysis_trace_view()
            analysis_out_payload = analysis_out if (analysis is not None and analysis_out.size > 0) else None
            analysis_stride_payload = int(analysis_trace_stride) if (analysis is not None and int(analysis_trace_stride) > 0) else None
            return Results(
                T=T, Y=Y, AUX=(AUX if n_rec_aux > 0 else None), 
                STEP=STEP, FLAGS=FLAGS,
                EVT_CODE=EVT_CODE, EVT_INDEX=EVT_INDEX,
                EVT_LOG_DATA=EVT_LOG_DATA,
                n=n_filled, m=m_filled,
                status=status_value,
                final_state=final_state,
                final_params=final_params,
                t_final=t_final,
                final_dt=final_dt,
                step_count_final=step_curr,
                final_workspace=final_ws,
                state_names=state_names,
                aux_names=aux_names,
                analysis_out=analysis_out_payload,
                analysis_trace=trace_view,
                analysis_trace_filled=trace_filled,
                analysis_trace_stride=analysis_stride_payload,
                analysis_trace_offset=trace_offset,
                analysis_modules=analysis_modules,
                analysis_meta=_analysis_meta(status_value == TRACE_OVERFLOW),
            )

        if status_value == GROW_REC:
            # Require at least one more slot beyond current n_filled
            new_cap = cap_rec
            while new_cap < n_filled + 1:
                new_cap *= 2
            
            T_new = np.zeros((new_cap,), dtype=np.float64)
            Y_new = np.zeros((n_rec_states, new_cap), dtype=dtype) if n_rec_states > 0 else np.zeros((0, new_cap), dtype=dtype)
            AUX_new = np.zeros((n_rec_aux, new_cap), dtype=dtype) if n_rec_aux > 0 else np.zeros((0, new_cap), dtype=dtype)
            STEP_new = np.zeros((new_cap,), dtype=np.int64)
            FLAGS_new = np.zeros((new_cap,), dtype=np.int32)
            
            if n_filled > 0:
                T_new[:n_filled] = T[:n_filled]
                if n_rec_states > 0:
                    Y_new[:, :n_filled] = Y[:, :n_filled]
                if n_rec_aux > 0:
                    AUX_new[:, :n_filled] = AUX[:, :n_filled]
                STEP_new[:n_filled] = STEP[:n_filled]
                FLAGS_new[:n_filled] = FLAGS[:n_filled]
            
            T = T_new
            Y = Y_new
            AUX = AUX_new
            STEP = STEP_new
            FLAGS = FLAGS_new
            cap_rec = new_cap
            
            # Re-enter: update cursors/caps
            i_start = np.int64(n_filled)
            step_start = np.int64(step_curr)
            t_curr = float(t_out[0])
            if step_curr > 0:
                dt_candidate = float(dt_next[0])
                if dt_candidate != 0.0:
                    dt_curr = dt_candidate
            continue

        if status_value == GROW_EVT:
            new_cap = cap_evt
            while new_cap < m_filled + 1:
                new_cap *= 2
            
            EVT_CODE_new = np.zeros((new_cap,), dtype=np.int32)
            EVT_INDEX_new = np.zeros((new_cap,), dtype=np.int32)
            EVT_LOG_DATA_new = np.zeros((new_cap, max(1, max_log_width)), dtype=dtype)
            
            if m_filled > 0:
                EVT_CODE_new[:m_filled] = EVT_CODE[:m_filled]
                EVT_INDEX_new[:m_filled] = EVT_INDEX[:m_filled]
                EVT_LOG_DATA_new[:m_filled, :] = EVT_LOG_DATA[:m_filled, :]
            
            EVT_CODE = EVT_CODE_new
            EVT_INDEX = EVT_INDEX_new
            EVT_LOG_DATA = EVT_LOG_DATA_new
            cap_evt = new_cap
            
            # Keep event cursor in hint_out for re-entry
            hint_out[0] = np.int32(m_filled)
            i_start = np.int64(n_filled)
            step_start = np.int64(step_curr)
            t_curr = float(t_out[0])
            if step_curr > 0:
                dt_candidate = float(dt_next[0])
                if dt_candidate != 0.0:
                    dt_curr = dt_candidate
            continue

        if status_value in (USER_BREAK, STEPFAIL, NAN_DETECTED, TRACE_OVERFLOW):
            status_name = Status(status_value).name
            t_final = float(t_out[0])
            warnings.warn(
                f"run_with_wrapper exited early at t = {t_final:.2f} with status {status_name} ({status_value})",
                RuntimeWarning,
                stacklevel=2,
            )
            # Early termination or error; return what we have (viewed via n/m)
            final_state = np.array(y_curr, copy=True)
            final_params = np.array(params, copy=True)
            final_ws = {
                "stepper": snapshot_workspace(stepper_ws),
                "runtime": snapshot_workspace(runtime_ws),
            }
            final_dt = float(dt_next[0]) if step_curr > 0 else float(dt_curr)
            trace_view, trace_filled, trace_offset = _finalize_analysis_trace_view()
            analysis_out_payload = analysis_out if (analysis is not None and analysis_out.size > 0) else None
            analysis_stride_payload = int(analysis_trace_stride) if (analysis is not None and int(analysis_trace_stride) > 0) else None
            return Results(
                T=T, Y=Y, AUX=(AUX if n_rec_aux > 0 else None), 
                STEP=STEP, FLAGS=FLAGS,
                EVT_CODE=EVT_CODE, EVT_INDEX=EVT_INDEX,
                EVT_LOG_DATA=EVT_LOG_DATA,
                n=n_filled, m=m_filled, status=status_value,
                final_state=final_state,
                final_params=final_params,
                t_final=t_final,
                final_dt=final_dt,
                step_count_final=step_curr,
                final_workspace=final_ws,
                state_names=state_names,
                aux_names=aux_names,
                analysis_out=analysis_out_payload,
                analysis_trace=trace_view,
                analysis_trace_filled=trace_filled,
                analysis_trace_stride=analysis_stride_payload,
                analysis_trace_offset=trace_offset,
                analysis_modules=analysis_modules,
                analysis_meta=_analysis_meta(status_value == TRACE_OVERFLOW),
            )

        # Any other code is unexpected in wrapper-level exit contract.
        raise RuntimeError(f"Runner returned unexpected status {status_value}")
