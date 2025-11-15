from __future__ import annotations
from typing import Callable, Mapping, Dict, Optional, Tuple
import warnings
import numpy as np

from dynlib.runtime.runner_api import (
    OK, STEPFAIL, NAN_DETECTED, DONE, GROW_REC, GROW_EVT, USER_BREAK, Status,
)
from dynlib.runtime.buffers import (
    allocate_pools, grow_rec_arrays, grow_evt_arrays,
)
from dynlib.runtime.results import Results
from dynlib.runtime.workspace import (
    make_runtime_workspace,
    initialize_lag_runtime_workspace,
    snapshot_workspace,
    restore_workspace,
)

__all__ = ["run_with_wrapper"]

# ------------------------------ Wrapper --------------------------------------

def run_with_wrapper(
    *,
    runner: Callable[..., np.int32],
    stepper: Callable[..., np.int32],
    rhs: Callable[..., None],
    events_pre: Callable[..., None],
    events_post: Callable[..., None],
    dtype: np.dtype,
    n_state: int,
    # sim config
    t0: float,
    t_end: float,
    dt_init: float,
    max_steps: int,
    record: bool,
    record_interval: int,
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
            T, Y, STEP, FLAGS,
            EVT_CODE, EVT_INDEX, EVT_LOG_DATA,
            evt_log_scratch,
            i_start, step_start, cap_rec, cap_evt,
            user_break_flag, status_out, hint_out,
            i_out, step_out, t_out,
            stepper, rhs, events_pre, events_post
        ) -> int32

    Notes:
      - Workspaces are allocated once per wrapper call and passed through to the runner.
      - hint_out[0] is used here as the current **event log cursor m** (by convention).
      - stepper_config is a read-only float64 array containing runtime configuration.
      - When ``discrete=True`` the runner horizon is interpreted as an iteration
        budget ``N`` (second argument). Otherwise it is ``t_end`` (continuous time).
    """
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

    runtime_ws = make_runtime_workspace(
        lag_state_info=lag_state_info,
        dtype=dtype,
    )
    stepper_ws = make_stepper_workspace() if make_stepper_workspace else None

    # Allocate recording/event pools
    rec, ev = allocate_pools(
        n_state=n_state,
        dtype=dtype,
        cap_rec=cap_rec,
        cap_evt=cap_evt,
        max_log_width=max_log_width,
    )

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

    # Track the committed (t, dt) so re-entries resume from the correct point.
    t_curr = float(t0)
    dt_curr = float(dt_init)

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
            rec.T, rec.Y, rec.STEP, rec.FLAGS,
            ev.EVT_CODE, ev.EVT_INDEX, ev.EVT_LOG_DATA,
            evt_log_scratch,
            i_start, step_start, int(rec.cap_rec), int(ev.cap_evt),
            user_break_flag, status_out, hint_out,
            i_out, step_out, t_out,
            stepper, rhs, events_pre, events_post,
        )

        status_value = int(status)

        # Filled cursors reported by runner
        n_filled = int(i_out[0])            # records
        m_filled = max(0, int(hint_out[0])) # events (by convention)
        step_curr = int(step_out[0])

        if status_value == DONE:
            final_state = np.array(y_curr, copy=True)
            final_params = np.array(params, copy=True)
            final_ws = {
                "stepper": snapshot_workspace(stepper_ws),
                "runtime": snapshot_workspace(runtime_ws),
            }
            final_dt = float(dt_next[0]) if step_curr > 0 else float(dt_curr)
            t_final = float(t_out[0])
            return Results(
                T=rec.T, Y=rec.Y, STEP=rec.STEP, FLAGS=rec.FLAGS,
                EVT_CODE=ev.EVT_CODE, EVT_INDEX=ev.EVT_INDEX,
                EVT_LOG_DATA=ev.EVT_LOG_DATA,
                n=n_filled, m=m_filled,
                status=status_value,
                final_state=final_state,
                final_params=final_params,
                t_final=t_final,
                final_dt=final_dt,
                step_count_final=step_curr,
                final_workspace=final_ws,
            )

        if status_value == GROW_REC:
            # Require at least one more slot beyond current n_filled
            rec = grow_rec_arrays(rec, filled=n_filled, min_needed=n_filled + 1)
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
            ev = grow_evt_arrays(ev, filled=m_filled, min_needed=m_filled + 1, dtype=dtype)
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

        if status_value in (USER_BREAK, STEPFAIL, NAN_DETECTED):
            status_name = Status(status_value).name
            warnings.warn(
                f"run_with_wrapper exited early with status {status_name} ({status_value})",
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
            t_final = float(t_out[0])
            return Results(
                T=rec.T, Y=rec.Y, STEP=rec.STEP, FLAGS=rec.FLAGS,
                EVT_CODE=ev.EVT_CODE, EVT_INDEX=ev.EVT_INDEX,
                EVT_LOG_DATA=ev.EVT_LOG_DATA,
                n=n_filled, m=m_filled, status=status_value,
                final_state=final_state,
                final_params=final_params,
                t_final=t_final,
                final_dt=final_dt,
                step_count_final=step_curr,
                final_workspace=final_ws,
            )

        # Any other code is unexpected in wrapper-level exit contract.
        raise RuntimeError(f"Runner returned unexpected status {status_value}")
