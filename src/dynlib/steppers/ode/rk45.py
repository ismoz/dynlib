# src/dynlib/steppers/rk45.py
"""
RK45 (Dormand-Prince, adaptive) stepper implementation.

Adaptive RK method with embedded error estimation.
Uses internal accept/reject loop until step is accepted or fails.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING
import numpy as np

from ..base import StepperMeta, StructSpec
from dynlib.runtime.runner_api import OK, NAN_DETECTED, STEPFAIL

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["RK45Spec", "RK45Config"]


@dataclass
class RK45Config:
    """Runtime configuration for RK45 stepper."""
    atol: float = 1e-8
    rtol: float = 1e-5
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 10.0
    max_tries: int = 10
    min_step: float = 1e-12


class RK45Spec:
    """
    Dormand-Prince RK45: 5th-order method with embedded 4th-order error estimate.
    
    Adaptive time-stepping with internal accept/reject loop.
    DOPRI5(4): 7 RHS evaluations per step. FSAL reuse optional (not used here).
    
    Adaptive, order 5, explicit scheme for ODEs.
    """
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
                dense_output=False,  # Not implemented yet
                stiff_ok=False,
                aliases=("dopri5", "dormand_prince"),
            )
        self.meta = meta

    def struct_spec(self) -> StructSpec:
        """
        RK45 workspace (lane-based allocation):
          - sp:  1 lane for y_stage (intermediate state scratch)
          - sw0: 2 lanes for k1, k2
          - sw1: 2 lanes for k3, k4
          - sw2: 2 lanes for k5, k6
          - sw3: 1 lane for k7 (embedded error term)
          - ss:  0 lanes (no persistence; FSAL optional later)
        
        DOPRI5(4): 7 RHS evaluations per step.
        Lane-based sizes: sp/ss/sw* sizes are lane counts (multiples of n_state).
        """
        return StructSpec(
            sp_size=1,    # y_stage (scratch)
            ss_size=0,    # no persistence (FSAL optional later)
            sw0_size=2,   # k1, k2
            sw1_size=2,   # k3, k4
            sw2_size=2,   # k5, k6
            sw3_size=1,   # k7 (embedded term)
            iw0_size=0,   # unused
            bw0_size=0,   # unused
            use_history=False,
            use_f_history=False,
            dense_output=False,
            needs_jacobian=False,
            embedded_order=4,
            stiff_ok=False,
        )

    def config_spec(self) -> type:
        """Return RK45Config dataclass for runtime configuration."""
        return RK45Config
    
    def default_config(self, model_spec=None) -> RK45Config:
        """
        Create default RK45 config, optionally reading from model_spec.
        
        Args:
            model_spec: Optional ModelSpec to read atol/rtol from
        
        Returns:
            RK45Config instance with defaults
        """
        config = RK45Config()
        
        # Override from model_spec if available
        if model_spec is not None:
            import dataclasses
            updates = {}
            if hasattr(model_spec.sim, 'atol'):
                updates['atol'] = float(model_spec.sim.atol)
            if hasattr(model_spec.sim, 'rtol'):
                updates['rtol'] = float(model_spec.sim.rtol)
            
            if updates:
                config = dataclasses.replace(config, **updates)
        
        return config
    
    def pack_config(self, config: RK45Config | None) -> np.ndarray:
        """
        Pack RK45Config into float64 array.
        
        Layout (7 floats):
            [0] atol
            [1] rtol
            [2] safety
            [3] min_factor
            [4] max_factor
            [5] max_tries (as float)
            [6] min_step
        
        Args:
            config: RK45Config instance (or None)
        
        Returns:
            float64 array with packed values
        """
        if config is None:
            config = RK45Config()
        
        return np.array([
            config.atol,
            config.rtol,
            config.safety,
            config.min_factor,
            config.max_factor,
            float(config.max_tries),
            config.min_step,
        ], dtype=np.float64)

    def emit(self, rhs_fn: Callable, struct: StructSpec, model_spec=None) -> Callable:
        """
        Generate a jittable RK45 stepper function with adaptive stepping.
        
        Runtime configuration is passed via stepper_config array (7 floats):
            [0] atol
            [1] rtol
            [2] safety
            [3] min_factor
            [4] max_factor
            [5] max_tries (as float)
            [6] min_step
        
        If stepper_config is empty or all zeros, defaults are used from closure.
        
        Signature (per ABI):
            status = stepper(
                t: float, dt: float,
                y_curr: float[:], rhs,
                params: float[:] | int[:],
                sp: float[:], ss: float[:],
                sw0: float[:], sw1: float[:], sw2: float[:], sw3: float[:],
                iw0: int32[:], bw0: uint8[:],
                stepper_config: float64[:],
                y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:]
            ) -> int32
        
        Returns:
            A callable Python function implementing the RK45 stepper with
            internal accept/reject loop.
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
            t, dt,
            y_curr, rhs,
            params,
            sp, ss,
            sw0, sw1, sw2, sw3,
            iw0, bw0,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # RK45: Dormand-Prince adaptive method (DOPRI5(4))
            # 7 RHS evaluations per step (k1-k7)
            n = y_curr.size
            
            # Lane-packed k vectors
            k1 = sw0[:n];      k2 = sw0[n:2*n]
            k3 = sw1[:n];      k4 = sw1[n:2*n]
            k5 = sw2[:n];      k6 = sw2[n:2*n]
            k7 = sw3[:n]                      # embedded error term
            y_stage = sp[:n]                  # intermediate state buffer
            
            # Read runtime config with fallback to defaults
            # Config array format: [atol, rtol, safety, min_factor, max_factor, max_tries, min_step]
            if stepper_config.size >= 7:
                atol = stepper_config[0]
                rtol = stepper_config[1]
                safety = stepper_config[2]
                min_factor = stepper_config[3]
                max_factor = stepper_config[4]
                max_tries = int(stepper_config[5])
                min_step = stepper_config[6]
            else:
                # Fallback to closure defaults
                atol = default_atol
                rtol = default_rtol
                safety = default_safety
                min_factor = default_min_factor
                max_factor = default_max_factor
                max_tries = default_max_tries
                min_step = default_min_step
            
            # Adaptive loop: keep trying until accept or fail
            h = dt
            error = 0.0  # Initialize error outside loop
            
            for attempt in range(max_tries):
                # Stage 1: k1 = f(t, y)
                rhs(t, y_curr, k1, params)
                
                # Stage 2: k2 = f(t + c2*h, y + h*(a21*k1))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * a21 * k1[i]
                rhs(t + c2 * h, y_stage, k2, params)
                
                # Stage 3: k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (a31 * k1[i] + a32 * k2[i])
                rhs(t + c3 * h, y_stage, k3, params)
                
                # Stage 4: k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
                rhs(t + c4 * h, y_stage, k4, params)
                
                # Stage 5: k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (
                        a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]
                    )
                rhs(t + c5 * h, y_stage, k5, params)
                
                # Stage 6: k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
                for i in range(n):
                    y_stage[i] = y_curr[i] + h * (
                        a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + 
                        a64 * k4[i] + a65 * k5[i]
                    )
                rhs(t + c6 * h, y_stage, k6, params)
                
                # Compute 5th order solution (y_prop = y5)
                for i in range(n):
                    y_prop[i] = y_curr[i] + h * (
                        b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + 
                        b5 * k5[i] + b6 * k6[i]
                    )
                
                # Stage 7: k7 = f(t + h, y5) for embedded error estimate
                rhs(t + h, y_prop, k7, params)
                
                # Error estimate: e_i = h * |(b1-bs1)*k1 + ... + (b6-bs6)*k6 - bs7*k7|
                error = 0.0
                for i in range(n):
                    e_i = h * abs(
                        (b1 - bs1) * k1[i] + 
                        (b3 - bs3) * k3[i] + 
                        (b4 - bs4) * k4[i] + 
                        (b5 - bs5) * k5[i] + 
                        (b6 - bs6) * k6[i] - 
                        bs7 * k7[i]
                    )
                    # Scale by tolerance
                    scale_i = atol + rtol * max(abs(y_curr[i]), abs(y_prop[i]))
                    error += (e_i / scale_i) ** 2
                
                error = (error / n) ** 0.5  # RMS error
                
                # Check for NaN
                if error != error:  # NaN check
                    err_est[0] = error
                    return NAN_DETECTED
                
                # Accept or reject
                if error <= 1.0 or h <= min_step:
                    # Accept step
                    t_prop[0] = t + h
                    err_est[0] = error
                    
                    # Compute next step size
                    if error > 0.0:
                        factor = safety * (1.0 / error) ** 0.2  # 1/(embedded_order+1) = 1/5
                        factor = max(min_factor, min(factor, max_factor))
                        dt_next[0] = h * factor
                    else:
                        dt_next[0] = h * max_factor
                    
                    return OK
                else:
                    # Reject step, reduce h
                    factor = safety * (1.0 / error) ** 0.25  # 1/embedded_order = 1/4
                    factor = max(min_factor, factor)
                    h = h * factor
                    
                    if h < min_step:
                        # Step too small, fail
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
