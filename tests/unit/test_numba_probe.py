# tests/test_numba_probe.py
from __future__ import annotations
import os
import numpy as np
import pytest

from numba import njit, int32, int64, float64, uint8

from dynlib import DONE, OK  # re-exported; stable constants


# ---------------------------------------------------------------------------
# JIT dummy callables
# ---------------------------------------------------------------------------

@njit
def _rhs(t: float, y_vec: np.ndarray, dy_out: np.ndarray, params: np.ndarray) -> None:
    # Simple ODE: dy/dt = -y  (params unused; concrete dtype float64[:])
    for i in range(y_vec.size):
        dy_out[i] = -y_vec[i]


@njit
def _events_pre(t: float, y_vec: np.ndarray, params: np.ndarray) -> None:
    # No-op pre events
    return


@njit
def _events_post(t: float, y_vec: np.ndarray, params: np.ndarray) -> None:
    # No-op post events
    return


@njit
def _stepper(
    t: float, dt: float,
    y_curr: np.ndarray, rhs,
    params: np.ndarray,
    sp: np.ndarray, ss: np.ndarray,
    sw0: np.ndarray, sw1: np.ndarray, sw2: np.ndarray, sw3: np.ndarray,
    iw0: np.ndarray, bw0: np.ndarray,
    y_prop: np.ndarray, t_prop: np.ndarray, dt_next: np.ndarray, err_est: np.ndarray
) -> np.int32:
    """
    Minimal explicit Euler stepper:
      y_prop = y_curr + dt * rhs(t, y_curr)
      t_prop = t + dt
      dt_next = dt  (fixed)
      err_est = 0
    Returns OK (0). Never touches record/log buffers.
    """
    # Workspace for dy (reuse sw0 if available, else small loop variable)
    n = y_curr.size
    # Use sw0 as scratch if it has enough space; else fallback to local tmp
    if sw0.size >= n:
        dy = sw0
    else:
        # small local array is fine in nopython mode; but keep it fixed-sized use pattern
        dy = np.empty(n, dtype=y_curr.dtype)

    # Evaluate RHS into dy
    rhs(t, y_curr, dy[:n], params)

    # Propose
    for i in range(n):
        y_prop[i] = y_curr[i] + dt * dy[i]

    t_prop[0] = t + dt
    dt_next[0] = dt
    err_est[0] = 0.0
    return np.int32(OK)


# ---------------------------------------------------------------------------
# JIT runner with the FROZEN ABI (names/order/shapes/dtypes)
# ---------------------------------------------------------------------------

@njit
def _runner(
  # scalars
  t0, t_end, dt_init,
  max_steps, n_state, record_every_step,
  # state/params
  y_curr, y_prev, params,
  # struct banks (views)
  sp, ss,
  sw0, sw1, sw2, sw3,
  iw0, bw0,
  # proposals/outs (len-1 arrays where applicable)
  y_prop, t_prop, dt_next, err_est,
  # recording
  T, Y, STEP, FLAGS,
  # event log (present; cap may be 1 if disabled)
  EVT_TIME, EVT_CODE, EVT_INDEX,
  # cursors & caps
  i_start, step_start, cap_rec, cap_evt,
  # control/outs (len-1)
  user_break_flag, status_out, hint_out,
  i_out, step_out, t_out,
  # function symbols (jittable callables)
  stepper, rhs, events_pre, events_post
) -> np.int32:
    """
    Minimal runner that:
      - executes pre-events on committed state
      - calls stepper once
      - commits y_prop -> y_curr; advances t
      - records one row (if capacity permits)
      - sets outs and returns DONE
    This is only a probe for JIT-ability and ABI conformance.
    """
    t = t0
    i = i_start
    step = step_start

    # Pre events on committed state
    events_pre(t, y_curr, params)

    # Single attempt
    status = stepper(
        t, dt_init, y_curr, rhs, params,
        sp, ss, sw0, sw1, sw2, sw3, iw0, bw0,
        y_prop, t_prop, dt_next, err_est
    )

    if status != OK:
        status_out[0] = np.int32(status)
        return np.int32(status)

    # Commit
    for k in range(n_state):
        y_prev[k] = y_curr[k]
        y_curr[k] = y_prop[k]
    t = t_prop[0]

    # Post events on committed state
    events_post(t, y_curr, params)

    # Record (one row)
    if i < cap_rec:
        T[i] = t
        for k in range(n_state):
            Y[i, k] = y_curr[k]
        STEP[i] = step
        FLAGS[i] = np.int32(OK)
        i += 1
        step += 1

    # Set outs
    t_out[0] = t
    i_out[0] = i
    step_out[0] = step
    status_out[0] = np.int32(DONE)
    hint_out[0] = np.int32(0)

    # We don't touch event log in the probe
    return np.int32(DONE)


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------

# @pytest.mark.skipif(
#     os.environ.get("NUMBA_DISABLE_JIT", "") == "1",
#     reason="Probe skipped when NUMBA is disabled (CI-B)."
# )
def test_numba_probe_minimal_ok():
    # Model dims
    n_state = 2
    cap_rec = 8
    cap_evt = 4

    # Scalars
    t0 = 0.0
    t_end = 0.1
    dt_init = 0.1
    max_steps = np.int64(4)
    record_every_step = np.int64(1)
    i_start = np.int64(0)
    step_start = np.int64(0)

    # Primary dtype: float64 (ODE)
    y_curr = np.array([1.0, 0.0], dtype=np.float64)
    y_prev = np.zeros(n_state, dtype=np.float64)
    params = np.array([0.0], dtype=np.float64)  # concrete float64[:] for ODEs

    # Struct banks (allow zero-length; still concrete dtypes)
    sp  = np.zeros(0, dtype=np.float64)
    ss  = np.zeros(0, dtype=np.float64)
    sw0 = np.zeros(n_state, dtype=np.float64)   # stepper scratch (>= n_state)
    sw1 = np.zeros(0, dtype=np.float64)
    sw2 = np.zeros(0, dtype=np.float64)
    sw3 = np.zeros(0, dtype=np.float64)
    iw0 = np.zeros(0, dtype=np.int32)
    bw0 = np.zeros(0, dtype=np.uint8)

    # Proposals/outs (len-1 where applicable)
    y_prop  = np.zeros(n_state, dtype=np.float64)
    t_prop  = np.zeros(1, dtype=np.float64)
    dt_next = np.zeros(1, dtype=np.float64)
    err_est = np.zeros(1, dtype=np.float64)

    # Recording buffers
    T    = np.zeros(cap_rec, dtype=np.float64)
    Y    = np.zeros((cap_rec, n_state), dtype=np.float64)
    STEP = np.zeros(cap_rec, dtype=np.int64)
    FLAGS= np.zeros(cap_rec, dtype=np.int32)

    # Event log buffers (not used here)
    EVT_TIME  = np.zeros(cap_evt, dtype=np.float64)
    EVT_CODE  = np.zeros(cap_evt, dtype=np.int32)
    EVT_INDEX = np.zeros(cap_evt, dtype=np.int32)

    # Control/outs (len-1)
    user_break_flag = np.zeros(1, dtype=np.int32)
    status_out = np.zeros(1, dtype=np.int32)
    hint_out = np.zeros(1, dtype=np.int32)
    i_out = np.zeros(1, dtype=np.int64)
    step_out = np.zeros(1, dtype=np.int64)
    t_out = np.zeros(1, dtype=np.float64)

    # Call the runner (function-pointer args are all jitted)
    ret = _runner(
        t0, t_end, dt_init,
        max_steps, np.int64(n_state), record_every_step,
        y_curr, y_prev, params,
        sp, ss,
        sw0, sw1, sw2, sw3,
        iw0, bw0,
        y_prop, t_prop, dt_next, err_est,
        T, Y, STEP, FLAGS,
        EVT_TIME, EVT_CODE, EVT_INDEX,
        i_start, step_start, np.int64(cap_rec), np.int64(cap_evt),
        user_break_flag, status_out, hint_out,
        i_out, step_out, t_out,
        _stepper, _rhs, _events_pre, _events_post
    )

    # Assertions: runner exits DONE; y committed; time advanced; recorded one row
    assert int(ret) == DONE
    assert status_out[0] == DONE
    assert t_out[0] > t0
    # Euler with dt=0.1: y1 = y0 + dt*(-y0) = 0.9*y0
    assert np.allclose(y_prev, np.array([1.0, 0.0], dtype=np.float64))
    assert np.allclose(y_curr, np.array([0.9, 0.0], dtype=np.float64))
    assert i_out[0] == 1
    assert step_out[0] == 1
    assert T[0] == pytest.approx(t_out[0])
    assert np.allclose(Y[0], y_curr)
