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

from ..base import StepperMeta, StepperCaps
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
                caps=StepperCaps(variational_stepping=True),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        y_stage: np.ndarray
        k1: np.ndarray
        k2: np.ndarray
        v_stage: np.ndarray
        kv1: np.ndarray
        kv2: np.ndarray

    class VariationalWorkspace(NamedTuple):
        v_stage: np.ndarray
        kv1: np.ndarray
        kv2: np.ndarray

    def workspace_type(self) -> type | None:
        return RK2Spec.Workspace

    def variational_workspace(self, n_state: int, model_spec=None):
        """
        Return the size and factory for the variational workspace.

        Returns:
            (size, factory_fn)
            factory_fn(analysis_ws, start, n_state) -> VariationalWorkspace
        """
        size = 3 * n_state

        def factory(analysis_ws, start, n_state):
            v_stage = analysis_ws[start : start + n_state]
            kv1 = analysis_ws[start + n_state : start + 2 * n_state]
            kv2 = analysis_ws[start + 2 * n_state : start + 3 * n_state]
            return RK2Spec.VariationalWorkspace(v_stage, kv1, kv2)

        return size, factory

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
            v_stage=zeros(),
            kv1=zeros(),
            kv2=zeros(),
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

    def emit_tangent_step(
        self,
        jvp_fn: Callable,
        model_spec=None
    ) -> Callable:
        """
        Generate a tangent-only RK2-style stepping function.

        NOTE: This evaluates the Jacobian at y_curr for both stages (not at
        intermediate RK stage states), so it is not fully consistent with RK2
        state integration. Use emit_step_with_variational for full consistency.
        """
        def rk2_tangent_step(
            t, dt,
            y_curr,
            v_curr,
            v_prop,
            params,
            runtime_ws,
            ws,
        ):
            n = v_curr.size
            if hasattr(ws, "v_stage"):
                v_stage = ws.v_stage
                kv1 = ws.kv1
                kv2 = ws.kv2
            else:
                v_stage = ws[0]
                kv1 = ws[1]
                kv2 = ws[2]

            # Stage 1: kv1 = J(y) * v
            jvp_fn(t, y_curr, params, v_curr, kv1, runtime_ws)

            # Stage 2: kv2 = J(y) * (v + dt/2 * kv1)
            half_dt = 0.5 * dt
            for i in range(n):
                v_stage[i] = v_curr[i] + half_dt * kv1[i]
            jvp_fn(t, y_curr, params, v_stage, kv2, runtime_ws)

            # v_{n+1} = v + dt * kv2
            for i in range(n):
                v_prop[i] = v_curr[i] + dt * kv2[i]
        return rk2_tangent_step

    def emit_step_with_variational(
        self,
        rhs_fn: Callable,
        jvp_fn: Callable,
        model_spec=None
    ) -> Callable:
        """
        Generate combined state + tangent RK2 stepper.

        Evaluates J at the RK2 midpoint stage for consistent state/tangent
        integration.
        """
        def rk2_step_with_variational(
            t, dt,
            y_curr, v_curr,
            y_prop, v_prop,
            params, runtime_ws, ws,
        ):
            n = y_curr.size

            k1 = getattr(ws, "k1", None)
            k2 = getattr(ws, "k2", None)
            kv1 = ws.kv1
            kv2 = ws.kv2
            y_stage = getattr(ws, "y_stage", None)
            v_stage = ws.v_stage

            if k1 is None:
                k1 = np.zeros((n,), dtype=y_curr.dtype)
            if k2 is None:
                k2 = np.zeros((n,), dtype=y_curr.dtype)
            if y_stage is None:
                y_stage = np.zeros((n,), dtype=y_curr.dtype)

            # Stage 1: at (t, y)
            rhs_fn(t, y_curr, k1, params, runtime_ws)
            jvp_fn(t, y_curr, params, v_curr, kv1, runtime_ws)

            # Stage 2: at (t + dt/2, y + dt/2 * k1)
            half_dt = 0.5 * dt
            for i in range(n):
                y_stage[i] = y_curr[i] + half_dt * k1[i]
                v_stage[i] = v_curr[i] + half_dt * kv1[i]
            rhs_fn(t + half_dt, y_stage, k2, params, runtime_ws)
            jvp_fn(t + half_dt, y_stage, params, v_stage, kv2, runtime_ws)

            # Final combination (RK2 midpoint)
            for i in range(n):
                y_prop[i] = y_curr[i] + dt * k2[i]
                v_prop[i] = v_curr[i] + dt * kv2[i]

        return rk2_step_with_variational


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = RK2Spec()
    register(spec)

_auto_register()
