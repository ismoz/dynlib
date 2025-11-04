# src/dynlib/compiler/codegen/runner.py
"""
Generic runner source generation (Slice 4).

Generates a single @njit runner with the frozen ABI that:
  1. Pre-events on committed state
  2. Stepper loop (single attempt for fixed-step; adaptive may retry internally)
  3. Commit: y_prev, y_curr, t
  4. Post-events on committed state
  5. Record (with capacity checks)
  6. Loop until t >= t_end or max_steps
  7. Return status codes per runner_api.py
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynlib.steppers.base import StructSpec

__all__ = ["generate_runner_source"]


def generate_runner_source(
    *,
    n_state: int,
    struct: StructSpec,
    stepper_src: str,
) -> str:
    """
    Generate the complete runner source code as a string.
    
    Args:
        n_state: Number of state variables
        struct: StructSpec from the chosen stepper (for maintenance hooks)
        stepper_src: Source code for the stepper function (emitted by StepperSpec.emit())
    
    Returns:
        Python source code defining the runner function with frozen ABI signature.
    """
    # The runner follows the exact signature from runner_api.py.
    # We'll generate it as a string template.
    
    # Note: We import status codes at module level in the generated source.
    # The stepper_src should return OK or other status codes.
    
    source = f"""
# Auto-generated runner (Slice 4)
import numpy as np

# Status codes (must match runtime.runner_api)
OK = 0
REJECT = 1
STEPFAIL = 2
NAN_DETECTED = 3
DONE = 9
GROW_REC = 10
GROW_EVT = 11
USER_BREAK = 12

# ---- Stepper function (injected) ----
{stepper_src}

# ---- Generic Runner (frozen ABI) ----
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
    \"\"\"
    Generic runner (Slice 4): fixed-step execution with events and recording.
    
    Returns status code (int32).
    \"\"\"
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
        
        # 2. Stepper attempt (fixed-step: single call; adaptive may loop internally)
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
        
        # 3. Commit: y_prev <- y_curr, y_curr <- y_prop, t <- t_prop
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        t = t_prop[0]
        dt = dt_next[0]
        step += 1
        
        # 4. Post-events on committed state
        events_post(t, y_curr, params)
        
        # 5. Record (if enabled and step matches record_every_step)
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
"""
    
    return source
