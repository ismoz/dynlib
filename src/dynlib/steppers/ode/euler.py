# src/dynlib/steppers/euler.py
"""
Euler (explicit, fixed-step) stepper implementation.

First real stepper. Minimal StructSpec (mostly size-1 banks).
"""
from __future__ import annotations
import ast
from typing import TYPE_CHECKING
import numpy as np

from ..base import StepperMeta, StructSpec
from dynlib.runtime.runner_api import OK

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["EulerSpec"]

# NOTE: Never add NaN / Inf checks in fixed-step steppers!
# Runners handle that globally for all steppers.
# Adaptive steppers may have these checks because they need them 
# when determining the step size.

class EulerSpec:
    """
    Explicit Euler stepper: y_{n+1} = y_n + dt * f(t_n, y_n)

    Fixed-step, order 1, explicit scheme for ODEs.
    """
    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="euler",
                kind="ode",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="euler",
                order=1,
                embedded_order=None,
                dense_output=False,
                stiff_ok=False,
                aliases=("fwd_euler", "forward_euler"),
            )
        self.meta = meta

    def struct_spec(self) -> StructSpec:
        """
        Euler requires minimal workspace:
          - sw0: 1 lane for RHS evaluation (n_state floats)
          - All other banks are unused (0 lanes/elements)
        
        Lane-based sizes: sp/ss/sw* sizes are lane counts (multiples of n_state).
        """
        return StructSpec(
            sp_size=0,    # unused
            ss_size=0,    # unused
            sw0_size=1,   # 1 lane for dy (RHS scratch)
            sw1_size=0,   # unused
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
        """Euler has no runtime-configurable parameters."""
        return None
    
    def default_config(self, model_spec=None):
        """Euler has no config."""
        return None
    
    def pack_config(self, config) -> np.ndarray:
        """Euler has no config - return empty array."""
        return np.array([], dtype=np.float64)

    def emit(self, rhs_fn: Callable, struct: StructSpec, model_spec=None) -> Callable:
        """
        Generate a jittable Euler stepper function.
        
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
            A callable Python function implementing the Euler stepper.
        """
        # Define the stepper function directly (not as a string)
        # This function will be JIT-compiled later if jit=True
        def euler_stepper(
            t, dt,
            y_curr, rhs,
            params,
            sp, ss,
            sw0, sw1, sw2, sw3,
            iw0, bw0,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # Euler: y_prop = y_curr + dt * f(t, y_curr)
            # stepper_config is ignored (Euler has no runtime config)
            n = y_curr.size
            
            # Use sw0 as scratch for dy (RHS evaluation)
            # Wrapper ensures sw0.size >= n
            dy = sw0[:n]
            
            # Evaluate RHS
            rhs(t, y_curr, dy, params, ss, iw0)
            
            # Propose next state
            for i in range(n):
                y_prop[i] = y_curr[i] + dt * dy[i]
            
            # Fixed step: dt_next = dt
            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0
            
            return OK
        
        return euler_stepper


# Auto-register on module import (optional; can also register explicitly in __init__.py)
def _auto_register():
    from ..registry import register
    spec = EulerSpec()
    register(spec)

_auto_register()
