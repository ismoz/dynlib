# src/dynlib/steppers/euler.py
"""
Euler (explicit, fixed-step) stepper implementation.

First real stepper with minimal workspace (single RHS buffer).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import numpy as np

from ..base import StepperMeta, StepperCaps
from ..config_base import ConfigMixin
from dynlib.runtime.runner_api import OK

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["EulerSpec"]

# NOTE: Never add NaN / Inf checks in fixed-step steppers!
# Runners handle that globally for all steppers.
# Adaptive steppers may have these checks because they need them 
# when determining the step size.

class EulerSpec(ConfigMixin):
    """
    Explicit Euler stepper: y_{n+1} = y_n + dt * f(t_n, y_n)

    Fixed-step, order 1, explicit scheme for ODEs.
    """
    Config = None  # No runtime configuration
    
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
                stiff=False,
                aliases=("fwd_euler", "forward_euler"),
                caps=StepperCaps(variational_stepping=True),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        dy: np.ndarray
        kv: np.ndarray

    class VariationalWorkspace(NamedTuple):
        kv: np.ndarray

    def workspace_type(self) -> type | None:
        return EulerSpec.Workspace

    def variational_workspace(self, n_state: int, model_spec=None):
        """
        Return the size and factory for the variational workspace.
        
        Returns:
            (size, factory_fn)
            factory_fn(analysis_ws, start, n_state) -> VariationalWorkspace
        """
        size = 1 * n_state

        def factory(analysis_ws, start, n_state):
            kv = analysis_ws[start : start + n_state]
            return EulerSpec.VariationalWorkspace(kv)

        return size, factory

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        return EulerSpec.Workspace(
            dy=np.zeros((n_state,), dtype=dtype),
            kv=np.zeros((n_state,), dtype=dtype),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable Euler stepper function.
        
        Signature (per ABI):
            status = stepper(
                t: float, dt: float,
                y_curr: float[:], rhs,
                params: float[:] | int[:],
                runtime_ws,
                stepper_ws,
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
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # Euler: y_prop = y_curr + dt * f(t, y_curr)
            # stepper_config is ignored (Euler has no runtime config)
            n = y_curr.size
            dy = ws.dy
            
            # Evaluate RHS
            rhs(t, y_curr, dy, params, runtime_ws)
            
            # Propose next state
            for i in range(n):
                y_prop[i] = y_curr[i] + dt * dy[i]
            
            # Fixed step: dt_next = dt
            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0
            
            return OK
        
        return euler_stepper

    def emit_tangent_step(
        self,
        jvp_fn: Callable,
        model_spec=None
    ) -> Callable:
        """
        Generate an Euler tangent-only stepping function.
        """
        def euler_tangent_step(
            t, dt,
            y_curr,
            v_curr,
            v_prop,
            params,
            runtime_ws,
            ws,
        ):
            kv = ws.kv if hasattr(ws, "kv") else ws[0]
            jvp_fn(t, y_curr, params, v_curr, kv, runtime_ws)
            n = v_curr.size
            for i in range(n):
                v_prop[i] = v_curr[i] + dt * kv[i]
        return euler_tangent_step

    def emit_step_with_variational(
        self,
        rhs_fn: Callable,
        jvp_fn: Callable,
        model_spec=None
    ) -> Callable:
        """
        Generate combined state + tangent Euler stepper.
        """
        def euler_step_with_variational(
            t, dt,
            y_curr, v_curr,
            y_prop, v_prop,
            params, runtime_ws, ws,
        ):
            dy = ws.dy
            kv = ws.kv

            # State update
            rhs_fn(t, y_curr, dy, params, runtime_ws)
            n = y_curr.size
            for i in range(n):
                y_prop[i] = y_curr[i] + dt * dy[i]

            # Tangent update
            jvp_fn(t, y_curr, params, v_curr, kv, runtime_ws)
            for i in range(n):
                v_prop[i] = v_curr[i] + dt * kv[i]

        return euler_step_with_variational


# Auto-register on module import (optional; can also register explicitly in __init__.py)
def _auto_register():
    from ..registry import register
    spec = EulerSpec()
    register(spec)

_auto_register()
