# src/dynlib/steppers/ode/ab3.py
"""
AB3 (Adams-Bashforth 3rd order, explicit, fixed-step) stepper implementation.

Three-step explicit multistep method with startup:

  - Step 0 (n = 0): Heun (2nd-order) from t0 -> t1, initializes f0, f1.
  - Step 1 (n = 1): AB2 from t1 -> t2, then rotates history to f0, f1, f2.
  - Steps n >= 2: AB3:

        y_{n+1} = y_n + h * (
            23/12 * f_n
          - 16/12 * f_{n-1}
          +  5/12 * f_{n-2}
        )
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import math
import numpy as np

from ..base import StepperMeta, StepperCaps
from dynlib.runtime.runner_api import OK
from ..config_base import ConfigMixin

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["AB3Spec"]


class AB3Spec(ConfigMixin):
    """
    Explicit 3-step Adams–Bashforth method (order 3, fixed-step).

    Main formula for n >= 2:
        y_{n+1} = y_n + h * (
            23/12 f_n
          - 16/12 f_{n-1}
          +  5/12 f_{n-2}
        )

    Startup:
      - Step 0 (n = 0): Heun's method (improved Euler, order 2) to get y1,
        initialize history with f0 and f1.
      - Step 1 (n = 1): AB2 from t1 to t2 using f0, f1, then rotate history
        to [f0, f1, f2] so AB3 is ready from step 2 onward.
    """
    CONFIG = None  # No runtime configuration

    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="ab3",
                kind="ode",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="adams-bashforth",
                order=3,
                embedded_order=None,
                stiff=False,
                aliases=("adams_bashforth_3",),
                caps=StepperCaps(variational_stepping=True),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        f_nm2: np.ndarray
        f_nm1: np.ndarray
        f_n: np.ndarray
        f_next: np.ndarray
        y_stage: np.ndarray
        step_index: np.ndarray

    class VariationalWorkspace(NamedTuple):
        v_stage: np.ndarray
        g_nm2: np.ndarray
        g_nm1: np.ndarray
        g_curr: np.ndarray
        step_index: np.ndarray

    def workspace_type(self) -> type | None:
        return AB3Spec.Workspace

    def variational_workspace(self, n_state: int, model_spec=None):
        """
        Return the size and factory for the variational workspace.

        Returns:
            (size, factory_fn)
            factory_fn(analysis_ws, start, n_state) -> VariationalWorkspace
        """
        size = 4 * n_state + 1

        def factory(analysis_ws, start, n_state):
            v_stage = analysis_ws[start : start + n_state]
            g_nm2 = analysis_ws[start + n_state : start + 2 * n_state]
            g_nm1 = analysis_ws[start + 2 * n_state : start + 3 * n_state]
            g_curr = analysis_ws[start + 3 * n_state : start + 4 * n_state]
            step_index = analysis_ws[start + 4 * n_state : start + 4 * n_state + 1]
            return AB3Spec.VariationalWorkspace(v_stage, g_nm2, g_nm1, g_curr, step_index)

        return size, factory

    def make_workspace(self, n_state: int, dtype: np.dtype, model_spec=None) -> Workspace:
        zeros = lambda: np.zeros((n_state,), dtype=dtype)
        return AB3Spec.Workspace(
            f_nm2=zeros(),
            f_nm1=zeros(),
            f_n=zeros(),
            f_next=zeros(),
            y_stage=zeros(),
            step_index=np.zeros((1,), dtype=np.int64),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable AB3 stepper backed by the NamedTuple workspace.
        """
        def ab3_stepper(
            t, dt,
            y_curr, rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):

            # Number of states
            n = y_curr.size

            f_nm2 = ws.f_nm2
            f_nm1 = ws.f_nm1
            f_n = ws.f_n
            f_next = ws.f_next
            y_stage = ws.y_stage
            step_idx_arr = ws.step_index
            step_idx = int(step_idx_arr[0])

            # --- Startup step 0 (Heun, order 2) ---
            if step_idx == 0:
                # Compute f0 = f(t0, y0) into f_nm1
                rhs(t, y_curr, f_nm1, params, runtime_ws)

                # Predictor: y_stage = y0 + dt * f0
                for i in range(n):
                    y_stage[i] = y_curr[i] + dt * f_nm1[i]

                # Predictor derivative (at y_stage) into f_n (temporary)
                rhs(t + dt, y_stage, f_n, params, runtime_ws)

                # Heun update: y1 = y0 + dt/2 * (f0 + f_pred)
                for i in range(n):
                    y_prop[i] = y_curr[i] + 0.5 * dt * (f_nm1[i] + f_n[i])

                # Refresh f_n with derivative at accepted state y1
                rhs(t + dt, y_prop, f_n, params, runtime_ws)

                # At this point:
                #   f_nm1 = f0
                #   f_n   = f1
                #   f_nm2 is unused (will be filled after step 1)
                t_prop[0] = t + dt
                dt_next[0] = dt
                err_est[0] = 0.0

                step_idx_arr[0] = step_idx + 1
                return OK

            # --- Startup step 1 (AB2 using f0, f1) ---
            if step_idx == 1:
                # We have:
                #   f_nm1 = f0
                #   f_n   = f1
                # AB2 step from t1 -> t2:
                #   y2 = y1 + dt * (3/2 f1 - 1/2 f0)
                for i in range(n):
                    y_prop[i] = y_curr[i] + dt * (1.5 * f_n[i] - 0.5 * f_nm1[i])

                # Compute f2 = f(t2, y2) into f_next
                rhs(t + dt, y_prop, f_next, params, runtime_ws)

                # Rotate history to prepare for AB3:
                #   f_nm2 <- f0
                #   f_nm1 <- f1
                #   f_n   <- f2
                for i in range(n):
                    f_nm2[i] = f_nm1[i]
                    f_nm1[i] = f_n[i]
                    f_n[i]   = f_next[i]

                t_prop[0] = t + dt
                dt_next[0] = dt
                err_est[0] = 0.0

                step_idx_arr[0] = step_idx + 1
                return OK

            # --- Main AB3 step (step_idx >= 2) ---
            # History at step start:
            #   f_nm2 = f_{n-2}
            #   f_nm1 = f_{n-1}
            #   f_n   = f_n (current)
            # AB3:
            #   y_{n+1} = y_n + dt * (
            #       23/12 f_n - 16/12 f_{n-1} + 5/12 f_{n-2}
            #   )
            for i in range(n):
                y_prop[i] = (
                    y_curr[i]
                    + dt * (
                        (23.0 / 12.0) * f_n[i]
                        - (16.0 / 12.0) * f_nm1[i]
                        + (5.0 / 12.0) * f_nm2[i]
                    )
                )

            # Compute f_{n+1} at proposed state; store temporarily in f_next
            rhs(t + dt, y_prop, f_next, params, runtime_ws)

            # Rotate history:
            #   new f_{n-2} = old f_{n-1}
            #   new f_{n-1} = old f_n
            #   new f_n     = f_{n+1}
            for i in range(n):
                f_nm2[i] = f_nm1[i]
                f_nm1[i] = f_n[i]
                f_n[i]   = f_next[i]

            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0

            step_idx_arr[0] = step_idx + 1
            return OK

        return ab3_stepper

    def emit_tangent_step(
        self,
        jvp_fn: Callable,
        model_spec=None
    ) -> Callable:
        """
        Generate a tangent-only AB3 stepping function.

        NOTE: This uses true AB3 history on J*v (g-history). Jacobian is
        evaluated at y_curr for the startup RK2 midpoint step.
        """
        def ab3_tangent_step(
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
                g_nm2 = ws.g_nm2
                g_nm1 = ws.g_nm1
                g_curr = ws.g_curr
                step_idx_arr = ws.step_index
            else:
                v_stage = ws[0]
                g_nm2 = ws[1]
                g_nm1 = ws[2]
                g_curr = ws[3]
                step_idx_arr = ws[4]
            step_idx = int(step_idx_arr[0])

            jvp_fn(t, y_curr, params, v_curr, g_curr, runtime_ws)

            if step_idx == 0:
                half_dt = 0.5 * dt
                for i in range(n):
                    v_stage[i] = v_curr[i] + half_dt * g_curr[i]
                    g_nm1[i] = g_curr[i]
                jvp_fn(t, y_curr, params, v_stage, g_curr, runtime_ws)
                for i in range(n):
                    v_prop[i] = v_curr[i] + dt * g_curr[i]
                step_idx_arr[0] = step_idx + 1
                return

            if step_idx == 1:
                for i in range(n):
                    v_prop[i] = v_curr[i] + dt * (1.5 * g_curr[i] - 0.5 * g_nm1[i])
                    g_nm2[i] = g_nm1[i]
                    g_nm1[i] = g_curr[i]
                step_idx_arr[0] = step_idx + 1
                return

            for i in range(n):
                v_prop[i] = (
                    v_curr[i]
                    + dt * (
                        (23.0 / 12.0) * g_curr[i]
                        - (16.0 / 12.0) * g_nm1[i]
                        + (5.0 / 12.0) * g_nm2[i]
                    )
                )
                g_nm2[i] = g_nm1[i]
                g_nm1[i] = g_curr[i]

            step_idx_arr[0] = step_idx + 1

        return ab3_tangent_step


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = AB3Spec()
    register(spec)

_auto_register()
