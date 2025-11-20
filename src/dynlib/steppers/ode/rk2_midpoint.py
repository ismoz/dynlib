# src/dynlib/steppers/ode/rk2_midpoint.py
"""
RK2 (Explicit Midpoint, 2nd-order, explicit, fixed-step) stepper implementation.

Classic fixed-step explicit midpoint method:

    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    y_{n+1} = y_n + dt * k2

This is often called "RK2 midpoint".
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import numpy as np

from ..base import StepperMeta
from ..config_base import ConfigMixin
from dynlib.runtime.runner_api import OK

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["RK2Spec"]


class RK2Spec(ConfigMixin):
    """
    Explicit midpoint method (RK2, order 2, fixed-step, explicit).
    """
    Config = None  # No runtime configuration

    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="rk2",
                kind="ode",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="runge-kutta",
                order=2,
                embedded_order=None,
                stiff=False,
                aliases=("rk2_midpoint",),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        y_stage: np.ndarray
        k1: np.ndarray
        k2: np.ndarray

    def workspace_type(self) -> type | None:
        return RK2Spec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        zeros = lambda: np.zeros((n_state,), dtype=dtype)
        return RK2Spec.Workspace(
            y_stage=zeros(),
            k1=zeros(),
            k2=zeros(),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable RK2 (explicit midpoint) stepper function.

        Returns:
            A callable Python function implementing the RK2 stepper with the
            standard dynlib stepper ABI:

                stepper(t, dt, y_curr, rhs, params, runtime_ws,
                        stepper_ws, stepper_config,
                        y_prop, t_prop, dt_next, err_est) -> int32
        """
        def rk2_stepper(
            t, dt,
            y_curr, rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est,
        ):
            # Explicit midpoint (RK2)
            n = y_curr.size

            k1 = ws.k1
            k2 = ws.k2
            y_stage = ws.y_stage

            # k1 = f(t, y)
            rhs(t, y_curr, k1, params, runtime_ws)

            # y_stage = y + dt/2 * k1
            half_dt = 0.5 * dt
            for i in range(n):
                y_stage[i] = y_curr[i] + half_dt * k1[i]

            # k2 = f(t + dt/2, y_stage)
            rhs(t + half_dt, y_stage, k2, params, runtime_ws)

            # y_{n+1} = y + dt * k2
            for i in range(n):
                y_prop[i] = y_curr[i] + dt * k2[i]

            # Fixed step: dt_next = dt
            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0  # no embedded error estimate

            return OK

        return rk2_stepper


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = RK2Spec()
    register(spec)

_auto_register()
