# src/dynlib/steppers/ode/rk4.py
"""
RK4 (Runge-Kutta 4th order, explicit, fixed-step) stepper implementation.

Classic fixed-step RK4 without touching wrapper/results/ABI.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from ..base import StepperMeta, StructSpec
from dynlib.runtime.runner_api import OK

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["RK4Spec"]


class RK4Spec:
    """
    Classic 4th-order Runge-Kutta stepper (explicit, fixed-step).
    
    Formula:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    Fixed-step, order 4, explicit scheme for ODEs.
    """
    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="rk4",
                kind="ode",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="runge-kutta",
                order=4,
                embedded_order=None,
                dense_output=False,
                stiff_ok=False,
                aliases=("rk4_classic", "classical_rk4"),
            )
        self.meta = meta

    def struct_spec(self) -> StructSpec:
        """
        RK4 workspace (lane-based allocation):
          - sw0: 2 lanes for k1, k2
          - sw1: 2 lanes for k3, k4
          - sp:  1 lane for y_stage (intermediate state evaluations)
          - All other banks unused (0 lanes/elements)
        
        Lane-based sizes: sp/ss/sw* sizes are lane counts (multiples of n_state).
        """
        return StructSpec(
            sp_size=1,    # y_stage scratch (1 lane = n_state floats)
            ss_size=0,    # unused
            sw0_size=2,   # k1, k2 (2 lanes = 2*n_state floats)
            sw1_size=2,   # k3, k4 (2 lanes = 2*n_state floats)
            sw2_size=0,   # unused
            sw3_size=0,   # unused
            iw0_size=0,   # unused
            bw0_size=0,   # unused
            use_history=False,
            use_f_history=False,
            dense_output=False,
            needs_jacobian=False,
            embedded_order=None,
            stiff_ok=False,
        )

    def config_spec(self) -> type | None:
        """RK4 has no runtime-configurable parameters."""
        return None
    
    def default_config(self, model_spec=None):
        """RK4 has no config."""
        return None
    
    def pack_config(self, config) -> np.ndarray:
        """RK4 has no config - return empty array."""
        return np.array([], dtype=np.float64)

    def emit(self, rhs_fn: Callable, struct: StructSpec, model_spec=None) -> Callable:
        """
        Generate a jittable RK4 stepper function.
        
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
            A callable Python function implementing the RK4 stepper.
        """
        def rk4_stepper(
            t, dt,
            y_curr, rhs,
            params,
            sp, ss,
            sw0, sw1, sw2, sw3,
            iw0, bw0,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # RK4: classic 4-stage explicit method
            # stepper_config is ignored (RK4 has no runtime config)
            n = y_curr.size
            
            # Lane-packed k vectors (2 lanes per bank)
            k1 = sw0[:n]
            k2 = sw0[n:2*n]
            k3 = sw1[:n]
            k4 = sw1[n:2*n]
            y_stage = sp[:n]
            
            # Stage 1: k1 = f(t, y)
            rhs(t, y_curr, k1, params, ss, iw0)
            
            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k1[i]
            rhs(t + 0.5 * dt, y_stage, k2, params, ss, iw0)
            
            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k2[i]
            rhs(t + 0.5 * dt, y_stage, k3, params, ss, iw0)
            
            # Stage 4: k4 = f(t + dt, y + dt * k3)
            for i in range(n):
                y_stage[i] = y_curr[i] + dt * k3[i]
            rhs(t + dt, y_stage, k4, params, ss, iw0)
            
            # Combine: y_prop = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            for i in range(n):
                y_prop[i] = y_curr[i] + (dt / 6.0) * (
                    k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
                )
            
            # Fixed step: dt_next = dt
            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0
            
            return OK
        
        return rk4_stepper


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = RK4Spec()
    register(spec)

_auto_register()
