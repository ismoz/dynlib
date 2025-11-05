# src/dynlib/compiler/codegen/runner.py
"""
Generic runner (Slice 4).

Defines a single runner function with the frozen ABI that:
  1. Pre-events on committed state
  2. Stepper loop (single attempt for fixed-step; adaptive may retry internally)
  3. Commit: y_prev, y_curr, t
  4. Post-events on committed state
  5. Record (with capacity checks)
  6. Loop until t >= t_end or max_steps
  7. Return status codes per runner_api.py

The runner accepts the stepper as a callable parameter, so steppers can be
regular Python functions instead of generated source code.

Per guardrails: same function body is used with/without JIT; decoration happens
only in compiler/jit/*.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from dynlib.steppers.base import StructSpec

# Import status codes from canonical source
from dynlib.runtime.runner_api import (
    OK, REJECT, STEPFAIL, NAN_DETECTED,
    DONE, GROW_REC, GROW_EVT, USER_BREAK
)

__all__ = ["runner", "get_runner"]


def runner(
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
):
    """
    Generic runner (Slice 4): fixed-step execution with events and recording.
    
    Frozen ABI signature - must match runner_api.py specification.
    
    Returns status code (int32).
    """
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)         # record cursor
    step = int(step_start)   # global step counter
    m = 0                    # event log cursor (not used in Slice 4 minimal; placeholder)
    
    # Recording at t0 (if record_every_step > 0)
    if record_every_step > 0 and step == 0:
        # Record initial condition
        if i >= cap_rec:
            # Need growth before recording
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = GROW_REC
            hint_out[0] = m
            return GROW_REC
        
        T[i] = t
        for k in range(n_state):
            Y[k, i] = y_curr[k]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1
    
    # Main integration loop
    while step < max_steps and t < t_end:
        # 1. Pre-events on committed state
        events_pre(t, y_curr, params)
        
        # 2. Clip dt to avoid overshooting t_end
        if t + dt > t_end:
            dt = t_end - t
        
        # 3. Stepper attempt (fixed-step: single call; adaptive may loop internally)
        step_status = stepper(
            t, dt, y_curr, rhs, params,
            sp, ss, sw0, sw1, sw2, sw3, iw0, bw0,
            y_prop, t_prop, dt_next, err_est
        )
        
        # Check for stepper failure
        if step_status != OK:
            # STEPFAIL, NAN_DETECTED, or other error
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status
        
        # 4. Commit: y_prev <- y_curr, y_curr <- y_prop, t <- t_prop
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        t = t_prop[0]
        dt = dt_next[0]
        step += 1
        
        # 5. Post-events on committed state
        events_post(t, y_curr, params)
        
        # 6. Record (if enabled and step matches record_every_step)
        if record_every_step > 0 and (step % record_every_step == 0):
            if i >= cap_rec:
                # Need growth
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_state):
                Y[k, i] = y_curr[k]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1
        
        # Check user break flag (if implemented)
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK
        
        # Check for completion
        if t >= t_end:
            break
    
    # Successful completion
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE


def get_runner(*, jit: bool = True) -> Callable:
    """
    Get the runner function, optionally JIT-compiled.
    
    Per guardrails: JIT decoration happens only in compiler/jit/*.
    This is a convenience wrapper for build.py to avoid direct JIT logic there.
    
    Args:
        jit: Whether to apply JIT compilation (default True)
    
    Returns:
        Runner function (JIT-compiled if requested and available)
    """
    if jit:
        try:
            from numba import njit
            return njit(cache=False)(runner)
        except ImportError:
            pass  # Fall back to non-JIT
    return runner
