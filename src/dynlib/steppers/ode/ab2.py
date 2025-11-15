"""
AB2 (Adams-Bashforth 2nd order, explicit, fixed-step) stepper implementation.

Two-step explicit multistep method with simple 2nd-order startup (Heun).
Persists f-history in `ss` and a step counter in `iw0`.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import math
import numpy as np

from ..base import StepperMeta
from dynlib.runtime.runner_api import OK, NAN_DETECTED

# Import guards for NaN/Inf detection
from dynlib.compiler.guards import allfinite1d, allfinite_scalar

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["AB2Spec"]


class AB2Spec:
    """
    Explicit 2-step Adams–Bashforth method (order 2, fixed-step).

    Main formula for n >= 1:
        y_{n+1} = y_n + h * (3/2 f_n - 1/2 f_{n-1})

    Startup (n = 0):
        Use Heun's method (improved Euler, order 2):
            k1 = f(t, y)
            y_pred = y + h * k1
            k2 = f(t + h, y_pred)
            y_1 = y + h/2 * (k1 + k2)
        and initialize history:
            f_{-1} = k1 (at t_0, y_0)
            f_0    = k2 (at t_1, y_1)
    """
    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="ab2",
                kind="ode",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="adams-bashforth",
                order=2,
                embedded_order=None,
                dense_output=False,
                stiff_ok=False,
                aliases=("adams_bashforth_2",),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        f_prev: np.ndarray
        f_curr: np.ndarray
        f_next: np.ndarray
        y_stage: np.ndarray
        step_index: np.ndarray  # shape (1,), int64

    def workspace_type(self) -> type | None:
        return AB2Spec.Workspace

    def make_workspace(self, n_state: int, dtype: np.dtype, model_spec=None) -> Workspace:
        zeros = lambda: np.zeros((n_state,), dtype=dtype)
        return AB2Spec.Workspace(
            f_prev=zeros(),
            f_curr=zeros(),
            f_next=zeros(),
            y_stage=zeros(),
            step_index=np.zeros((1,), dtype=np.int64),
        )

    def config_spec(self) -> type | None:
        """AB2 has no runtime-configurable parameters."""
        return None

    def default_config(self, model_spec=None):
        """AB2 has no config."""
        return None

    def pack_config(self, config) -> np.ndarray:
        """AB2 has no config - return empty array."""
        return np.array([], dtype=np.float64)

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate the AB2 stepper closure that uses the NamedTuple workspace.
        """
        def ab2_stepper(
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
            f_prev = ws.f_prev
            f_curr = ws.f_curr
            f_next = ws.f_next
            y_stage = ws.y_stage
            step_idx_arr = ws.step_index
            step_idx = int(step_idx_arr[0])

            # --- Startup step (Heun, order 2) ---
            if step_idx == 0:
                # k1 = f(t, y)
                rhs(t, y_curr, f_prev, params, runtime_ws)
                if not allfinite1d(f_prev):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # Predictor: y_stage = y + dt * k1
                for i in range(n):
                    y_stage[i] = y_curr[i] + dt * f_prev[i]

                # k2 = f(t + dt, y_stage)
                rhs(t + dt, y_stage, f_curr, params, runtime_ws)
                if not allfinite1d(f_curr):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # Heun update: y_{1} = y + dt/2 * (k1 + k2)
                for i in range(n):
                    y_prop[i] = y_curr[i] + 0.5 * dt * (f_prev[i] + f_curr[i])

                if not allfinite1d(y_prop):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # AB2 requires derivative history at accepted states.
                # Refresh f_curr with f(t1, y1) instead of the predictor derivative.
                rhs(t + dt, y_prop, f_curr, params, runtime_ws)
                if not allfinite1d(f_curr):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # Initialize history:
                #   f_prev = f(t0, y0)  (already in f_prev)
                #   f_curr = f(t1, y1)  (just recomputed)
                t_prop[0] = t + dt
                dt_next[0] = dt
                err_est[0] = 0.0

                step_idx_arr[0] = step_idx + 1
                return OK

            # --- Main AB2 step (n >= 1) ---
            # Use stored f_prev = f_{n-1}, f_curr = f_n at current y_curr.
            # y_{n+1} = y_n + dt * (3/2 f_n - 1/2 f_{n-1})
            for i in range(n):
                y_prop[i] = y_curr[i] + dt * (1.5 * f_curr[i] - 0.5 * f_prev[i])

            if not allfinite1d(y_prop):
                err_est[0] = float("inf")
                return NAN_DETECTED

            # Compute f_{n+1} at proposed state; store in f_next temporarily
            rhs(t + dt, y_prop, f_next, params, runtime_ws)
            if not allfinite1d(f_next):
                err_est[0] = float("inf")
                return NAN_DETECTED

            # Rotate history:
            #   new f_prev = old f_curr
            #   new f_curr = f_next
            for i in range(n):
                f_prev[i] = f_curr[i]
                f_curr[i] = f_next[i]

            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0

            step_idx_arr[0] = step_idx + 1
            return OK

        return ab2_stepper


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = AB2Spec()
    register(spec)

_auto_register()
