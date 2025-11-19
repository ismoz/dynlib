# src/dynlib/steppers/ode/rk4.py
"""
RK4 (Runge-Kutta 4th order, explicit, fixed-step) stepper implementation.

Classic fixed-step RK4 without touching wrapper/results/ABI.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import numpy as np

from ..base import StepperMeta
from ..config_base import ConfigMixin
from dynlib.runtime.runner_api import OK

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["RK4Spec"]


class RK4Spec(ConfigMixin):
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
    Config = None  # No runtime configuration
    
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
                stiff=False,
                aliases=("rk4_classic", "classical_rk4"),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        y_stage: np.ndarray
        k1: np.ndarray
        k2: np.ndarray
        k3: np.ndarray
        k4: np.ndarray

    def workspace_type(self) -> type | None:
        return RK4Spec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        zeros = lambda: np.zeros((n_state,), dtype=dtype)
        return RK4Spec.Workspace(
            y_stage=zeros(),
            k1=zeros(),
            k2=zeros(),
            k3=zeros(),
            k4=zeros(),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable RK4 stepper function.
        
        Returns:
            A callable Python function implementing the RK4 stepper.
        """
        def rk4_stepper(
            t, dt,
            y_curr, rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # RK4: classic 4-stage explicit method
            # stepper_config is ignored (RK4 has no runtime config)
            n = y_curr.size
            
            k1 = ws.k1
            k2 = ws.k2
            k3 = ws.k3
            k4 = ws.k4
            y_stage = ws.y_stage
            
            # Stage 1: k1 = f(t, y)
            rhs(t, y_curr, k1, params, runtime_ws)
            
            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k1[i]
            rhs(t + 0.5 * dt, y_stage, k2, params, runtime_ws)
            
            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k2[i]
            rhs(t + 0.5 * dt, y_stage, k3, params, runtime_ws)
            
            # Stage 4: k4 = f(t + dt, y + dt * k3)
            for i in range(n):
                y_stage[i] = y_curr[i] + dt * k3[i]
            rhs(t + dt, y_stage, k4, params, runtime_ws)
            
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
