from __future__ import annotations
from typing import Callable
import warnings
import numpy as np

from dynlib.runtime.runner_api import (
    OK, STEPFAIL, NAN_DETECTED, DONE, GROW_REC, GROW_EVT, USER_BREAK, Status,
)
from dynlib.runtime.buffers import (
    allocate_pools, grow_rec_arrays, grow_evt_arrays,
)
from dynlib.runtime.results import Results
from dynlib.steppers.base import StructSpec

__all__ = ["run_with_wrapper"]

# ------------------------------ Wrapper --------------------------------------

def run_with_wrapper(
    *,
    runner: Callable[..., np.int32],
    stepper: Callable[..., np.int32],
    rhs: Callable[..., None],
    events_pre: Callable[..., None],
    events_post: Callable[..., None],
    struct: StructSpec,
    model_dtype: np.dtype,
    n_state: int,
    # sim config
    t0: float,
    t_end: float,
    dt_init: float,
    max_steps: int,
    record: bool,
    record_every_step: int,
    # initial state/params
    y0: np.ndarray,
    params: np.ndarray,
    # capacities (can be small to force growth)
    cap_rec: int = 1024,
    cap_evt: int = 1,
    max_log_width: int = 0,  # maximum log width across all events
    # NEW: stepper configuration
    stepper_config: np.ndarray = None,
) -> Results:
    """
    JIT-free orchestrator. Allocates banks/buffers, calls the compiled runner
    exactly once per attempt, handles growth and re-entry, and returns Results.

    Runner ABI (frozen) — summarized here (see runner_api.py for full doc):
        status = runner(
            t0, t_end, dt_init,
            max_steps, n_state, record_every_step,
            y_curr, y_prev, params,
            sp, ss, sw0, sw1, sw2, sw3, iw0, bw0,
            stepper_config,
            y_prop, t_prop, dt_next, err_est,
            T, Y, STEP, FLAGS,
            EVT_CODE, EVT_INDEX, EVT_LOG_DATA,
            i_start, step_start, cap_rec, cap_evt,
            user_break_flag, status_out, hint_out,
            i_out, step_out, t_out,
            stepper, rhs, events_pre, events_post
        ) -> int32

    Notes:
      - This wrapper never allocates inside the runner attempt; growth is handled
        outside by doubling the relevant buffers, copying only filled regions,
        and re-entering with updated caps/cursors.
      - hint_out[0] is used here as the current **event log cursor m** (by convention).
      - stepper_config is a read-only float64 array containing runtime configuration.
    """
    assert y0.shape == (n_state,)
    y_curr = np.array(y0, dtype=model_dtype, copy=True)
    y_prev = np.array(y0, dtype=model_dtype, copy=True)  # will be set by runner after first commit
    
    # Default stepper_config to empty array if not provided
    if stepper_config is None:
        stepper_config = np.array([], dtype=np.float64)
    else:
        # Ensure it's the right dtype
        stepper_config = np.asarray(stepper_config, dtype=np.float64)

    # Allocate banks and pools
    banks, rec, ev = allocate_pools(
        n_state=n_state, struct=struct, model_dtype=model_dtype,
        cap_rec=cap_rec, cap_evt=cap_evt, max_log_width=max_log_width,
    )

    # Proposals / outs (len-1 where applicable)
    y_prop  = np.zeros((n_state,), dtype=model_dtype)
    t_prop  = np.zeros((1,), dtype=model_dtype)
    dt_next = np.zeros((1,), dtype=model_dtype)
    err_est = np.zeros((1,), dtype=model_dtype)
    
    # Event log scratch buffer
    evt_log_scratch = np.zeros((max(1, max_log_width),), dtype=model_dtype)

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
    rec_every = int(record_every_step) if record else 0  # runner may treat 0 as "no record except explicit"

    # Track the committed (t, dt) so re-entries resume from the correct point.
    t_curr = float(t0)
    dt_curr = float(dt_init)

    # Attempt/re-entry loop
    while True:
        status = runner(
            t_curr, float(t_end), dt_curr,
            int(max_steps), int(n_state), int(rec_every),
            y_curr, y_prev, params,
            banks.sp, banks.ss, banks.sw0, banks.sw1, banks.sw2, banks.sw3,
            banks.iw0, banks.bw0,
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
            return Results(
                T=rec.T, Y=rec.Y, STEP=rec.STEP, FLAGS=rec.FLAGS,
                EVT_CODE=ev.EVT_CODE, EVT_INDEX=ev.EVT_INDEX,
                EVT_LOG_DATA=ev.EVT_LOG_DATA,
                n=n_filled, m=m_filled,
                status=status_value,
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
            ev = grow_evt_arrays(ev, filled=m_filled, min_needed=m_filled + 1, model_dtype=model_dtype)
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
            return Results(
                T=rec.T, Y=rec.Y, STEP=rec.STEP, FLAGS=rec.FLAGS,
                EVT_CODE=ev.EVT_CODE, EVT_INDEX=ev.EVT_INDEX,
                EVT_LOG_DATA=ev.EVT_LOG_DATA,
                n=n_filled, m=m_filled, status=status_value,
            )

        # Any other code is unexpected in wrapper-level exit contract.
        raise RuntimeError(f"Runner returned unexpected status {status_value}")
