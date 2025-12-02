# src/dynlib/steppers/ode/rk45.py
"""
RK45 (Dormand-Prince, adaptive) stepper implementation.

Adaptive RK method with embedded error estimation.
Uses internal accept/reject loop until step is accepted or fails.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..base import StepperMeta
from ..config_base import ConfigMixin
from dynlib.runtime.runner_api import OK, STEPFAIL

# Import guards for NaN/Inf detection
# When jit=False, this makes allfinite1d available in the closure
# When jit=True, the stepper source is rendered with guards inlined
from dynlib.compiler.guards import allfinite1d, allfinite_scalar, register_guards_consumer

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["RK45Spec"]

register_guards_consumer(globals())


class RK45Spec(ConfigMixin):
    """
    Dormand-Prince RK45: 5th-order method with embedded 4th-order error estimate.
    
    Adaptive time-stepping with internal accept/reject loop.
    DOPRI5(4): 7 RHS evaluations per step. FSAL reuse optional (not used here).
    
    Adaptive, order 5, explicit scheme for ODEs.
    """
    
    @dataclass
    class Config:
        """Runtime configuration for RK45 stepper."""
        atol: float = 1e-6
        rtol: float = 1e-3
        safety: float = 0.9
        min_factor: float = 0.2
        max_factor: float = 5.0
        max_tries: int = 10
        min_step: float = 1e-12
        dt_max: float = np.inf
    
    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="rk45",
                kind="ode",
                time_control="adaptive",
                scheme="explicit",
                geometry=frozenset(),
                family="runge-kutta",
                order=5,
                embedded_order=4,
                stiff=False,
                aliases=("dopri5", "dormand_prince"),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        y_stage: np.ndarray
        k1: np.ndarray
        k2: np.ndarray
        k3: np.ndarray
        k4: np.ndarray
        k5: np.ndarray
        k6: np.ndarray
        k7: np.ndarray

    def workspace_type(self) -> type | None:
        return RK45Spec.Workspace

    def make_workspace(self, n_state: int, dtype: np.dtype, model_spec=None) -> Workspace:
        zeros = lambda: np.zeros((n_state,), dtype=dtype)
        return RK45Spec.Workspace(
            y_stage=zeros(),
            k1=zeros(),
            k2=zeros(),
            k3=zeros(),
            k4=zeros(),
            k5=zeros(),
            k6=zeros(),
            k7=zeros(),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable RK45 stepper that consumes the workspace tuple.

        Runtime configuration is passed via the float64 stepper_config array (7 floats):
            [0] atol, [1] rtol, [2] safety, [3] min_factor, [4] max_factor,
            [5] max_tries (as float), [6] min_step.
        Defaults from the ModelSpec are used when the config array is empty.
        """
        # Get defaults from model_spec (closure values for when config array is empty)
        default_cfg = self.default_config(model_spec)
        default_atol = default_cfg.atol
        default_rtol = default_cfg.rtol
        default_safety = default_cfg.safety
        default_min_factor = default_cfg.min_factor
        default_max_factor = default_cfg.max_factor
        default_max_tries = default_cfg.max_tries
        default_min_step = default_cfg.min_step
        default_dt_max = default_cfg.dt_max
        
        # Dormand-Prince coefficients
        # Butcher tableau for DOPRI5(4)
        # a_ij coefficients (lower triangular)
        a21 = 1.0/5.0
        
        a31 = 3.0/40.0
        a32 = 9.0/40.0
        
        a41 = 44.0/45.0
        a42 = -56.0/15.0
        a43 = 32.0/9.0
        
        a51 = 19372.0/6561.0
        a52 = -25360.0/2187.0
        a53 = 64448.0/6561.0
        a54 = -212.0/729.0
        
        a61 = 9017.0/3168.0
        a62 = -355.0/33.0
        a63 = 46732.0/5247.0
        a64 = 49.0/176.0
        a65 = -5103.0/18656.0
        
        # c_i coefficients (time offsets)
        c2 = 1.0/5.0
        c3 = 3.0/10.0
        c4 = 4.0/5.0
        c5 = 8.0/9.0
        c6 = 1.0
        
        # b_i coefficients (5th order solution)
        b1 = 35.0/384.0
        b2 = 0.0
        b3 = 500.0/1113.0
        b4 = 125.0/192.0
        b5 = -2187.0/6784.0
        b6 = 11.0/84.0
        
        # b_i* coefficients (4th order embedded solution for error estimate)
        bs1 = 5179.0/57600.0
        bs2 = 0.0
        bs3 = 7571.0/16695.0
        bs4 = 393.0/640.0
        bs5 = -92097.0/339200.0
        bs6 = 187.0/2100.0
        bs7 = 1.0/40.0
        
        def rk45_stepper(
            t,
            dt,
            y_curr,
            rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop,
            t_prop,
            dt_next,
            err_est,
        ):
            # RK45: Dormand-Prince adaptive method (DOPRI5(4))
            # 7 RHS evaluations per step (k1-k7)
            n = y_curr.size

            k1 = ws.k1
            k2 = ws.k2
            k3 = ws.k3
            k4 = ws.k4
            k5 = ws.k5
            k6 = ws.k6
            k7 = ws.k7
            y_stage = ws.y_stage
            
            # Read runtime config with fallback to defaults
            # Config array format: [atol, rtol, safety, min_factor, max_factor, max_tries, min_step, dt_max]
            if stepper_config.size >= 8:
                atol = stepper_config[0]
                rtol = stepper_config[1]
                safety = stepper_config[2]
                min_factor = stepper_config[3]
                max_factor = stepper_config[4]
                max_tries = int(stepper_config[5])
                min_step = stepper_config[6]
                dt_max = stepper_config[7]
            else:
                # Fallback to closure defaults
                atol = default_atol
                rtol = default_rtol
                safety = default_safety
                min_factor = default_min_factor
                max_factor = default_max_factor
                max_tries = default_max_tries
                min_step = default_min_step
                dt_max = default_dt_max
            
            if max_tries < 1:
                max_tries = 1

            # Stage 1: k1 = f(t, y) is independent of h.
            # Compute it once outside the retry loop.
            rhs(t, y_curr, k1, params, runtime_ws)
            if not allfinite1d(k1):
                # If f(t, y) itself is non-finite, reducing h won't help.
                # Treat this as a hard failure.
                err_est[0] = float("inf")
                return STEPFAIL

            # Adaptive loop: keep trying until accept or fail
            h = dt
            if h > dt_max:
                h = dt_max
            error = float("inf")
            
            for attempt in range(max_tries):
                # k1 already computed outside the loop for this (t, y_curr)

                # Stage 2: k2 = f(t + c2*h, y + h*(a21*k1))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * a21 * k1[i]
                rhs(t + c2 * h, y_stage, k2, params, runtime_ws)
                if not allfinite1d(k2):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Stage 3: k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (a31 * k1[i] + a32 * k2[i])
                rhs(t + c3 * h, y_stage, k3, params, runtime_ws)
                if not allfinite1d(k3):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Stage 4: k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
                rhs(t + c4 * h, y_stage, k4, params, runtime_ws)
                if not allfinite1d(k4):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Stage 5: k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (
                        a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]
                    )
                rhs(t + c5 * h, y_stage, k5, params, runtime_ws)
                if not allfinite1d(k5):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Stage 6: k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (
                        a61 * k1[i] + a62 * k2[i] + a63 * k3[i] +
                        a64 * k4[i] + a65 * k5[i]
                    )
                rhs(t + c6 * h, y_stage, k6, params, runtime_ws)
                if not allfinite1d(k6):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Compute 5th order solution (y_prop = y5)
                for i in range(n):
                    y_prop[i] = y_curr[i] + h * (
                        b1 * k1[i] + b3 * k3[i] + b4 * k4[i] +
                        b5 * k5[i] + b6 * k6[i]
                    )
                if not allfinite1d(y_prop):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Stage 7: k7 = f(t + h, y5) for embedded error estimate
                rhs(t + h, y_prop, k7, params, runtime_ws)
                if not allfinite1d(k7):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Error estimate: e_i = h * |(b1-bs1)*k1 + ... + (b6-bs6)*k6 - bs7*k7|
                err_acc = 0.0
                for i in range(n):
                    e_i = h * abs(
                        (b1 - bs1) * k1[i] +
                        (b3 - bs3) * k3[i] +
                        (b4 - bs4) * k4[i] +
                        (b5 - bs5) * k5[i] +
                        (b6 - bs6) * k6[i] -
                        bs7 * k7[i]
                    )
                    scale_i = atol + rtol * max(abs(y_curr[i]), abs(y_prop[i]))
                    err_acc += (e_i / scale_i) ** 2

                error = (err_acc / n) ** 0.5

                if not math.isfinite(error):
                    error = float("inf")
                    if h <= min_step:
                        err_est[0] = error
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Accept or reject
                if error <= 1.0 or h <= min_step:
                    t_prop[0] = t + h
                    err_est[0] = error

                    if error > 0.0:
                        factor = safety * (1.0 / error) ** 0.2  # 1/(embedded_order+1) = 1/5
                        factor = max(min_factor, min(factor, max_factor))
                        dt_next[0] = h * factor
                    else:
                        dt_next[0] = h * max_factor
                    
                    # Cap at dt_max
                    if dt_next[0] > dt_max:
                        dt_next[0] = dt_max

                    return OK

                # Reject step, reduce h
                factor = safety * (1.0 / error) ** 0.25  # 1/embedded_order = 1/4
                factor = max(min_factor, factor)
                h = h * factor

                if h < min_step:
                    err_est[0] = error
                    return STEPFAIL

            # Max tries exceeded
            err_est[0] = error
            return STEPFAIL
        
        return rk45_stepper


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = RK45Spec()
    register(spec)

_auto_register()
