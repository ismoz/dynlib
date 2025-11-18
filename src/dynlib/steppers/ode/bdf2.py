# src/dynlib/steppers/ode/bdf2_scipy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
import numpy as np

from ..base import StepperMeta, StepperCaps
from ..config_base import ConfigMixin
from dynlib.runtime.runner_api import OK, STEPFAIL

if TYPE_CHECKING:
    from typing import Callable


__all__ = ["BDF2ScipySpec"]


class BDF2ScipySpec(ConfigMixin):
    """
    Scipy-based Implicit 2-step BDF method (order 2, fixed-step).
    
    Uses scipy.optimize.root (MINPACK/hybr) to solve the nonlinear system.
    
    - Startup: backward Euler (BDF1).
    - Main steps: BDF2.
    - Not compatible with Numba nopython mode (Object mode only).
    """

    @dataclass
    class Config:
        """Runtime configuration for BDF2 scipy solver."""
        # Tolerance passed to scipy.optimize.root
        tol: float = 1e-8
        # Max iterations for the solver
        max_iter: int = 50
        # Method for root finding ('hybr', 'lm', 'broyden1', etc.)
        method: str = 'hybr'
        
        # Enum mappings co-located with config definition
        __enums__ = {
            "method": {"hybr": 0, "lm": 1, "broyden1": 2}
        }

    class Workspace(NamedTuple):
        # multistep history
        y_nm1: np.ndarray          # y_{n-1} at committed state
        step_index: np.ndarray     # shape (1,), int64, counts accepted steps
        
        # Scratch space for the RHS call within the residual wrapper
        f_tmp: np.ndarray
        # Scratch space for predictor calculation
        f_pred: np.ndarray

    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="bdf2",
                kind="ode",
                time_control="fixed",
                scheme="implicit",
                geometry=frozenset(),
                family="bdf",
                order=2,
                embedded_order=None,
                stiff_ok=True,
                aliases=("bdf2_scipy",),
                caps=StepperCaps(
                    jacobian="internal",
                    requires_scipy=True,
                    jit_capable=False,
                ),
            )
        self.meta = meta

    # --- StepperSpec protocol hooks ---------------------------------------

    def workspace_type(self) -> type | None:
        return BDF2ScipySpec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        def vec():
            return np.zeros((n_state,), dtype=dtype)

        return BDF2ScipySpec.Workspace(
            y_nm1=vec(),
            step_index=np.zeros((1,), dtype=np.int64),
            f_tmp=vec(),
            f_pred=vec(),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate the BDF2 stepper closure using scipy.optimize.root.

        Signature (frozen ABI):
            stepper(t, dt, y_curr, rhs, params, runtime_ws,
                    stepper_ws, stepper_config,
                    y_prop, t_prop, dt_next, err_est) -> int32
        """
        from scipy.optimize import root

        def bdf2_stepper(
            t, dt,
            y_curr, rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # Unpack workspace
            y_nm1 = ws.y_nm1
            step_idx_arr = ws.step_index
            f_tmp = ws.f_tmp # Scratch for residual calc
            f_pred = ws.f_pred # Scratch for predictor

            step_idx = int(step_idx_arr[0])
            t_new = t + dt

            # Config extraction
            root_tol = stepper_config[0]
            max_iter = int(stepper_config[1])
            method_id = int(stepper_config[2]) if stepper_config.size >= 3 else 0
            if method_id == 0:
                method = "hybr"
            elif method_id == 1:
                method = "lm"
            elif method_id == 2:
                method = "broyden1"
            else:
                method = "hybr"

            # ----------------- Predictor (Initial Guess) -----------------
            # We need an initial guess (x0) for the root solver.
            # rhs(t, y, out, ...)
            rhs(t, y_curr, f_pred, params, runtime_ws)
            
            if step_idx == 0:
                # Forward Euler prediction
                y_guess = y_curr + dt * f_pred
            else:
                # BDF2 predictor / extrapolation
                # y_guess = (4/3)y_n - (1/3)y_{n-1} + (2/3)dt * f_n
                y_guess = (4.0/3.0)*y_curr - (1.0/3.0)*y_nm1 + (2.0/3.0)*dt * f_pred

            # ----------------- Residual Definitions -----------------
            
            def residual_bdf1(y_val):
                # BDF1: y - y_n - dt * f(t+dt, y) = 0
                rhs(t_new, y_val, f_tmp, params, runtime_ws)
                return y_val - y_curr - dt * f_tmp

            def residual_bdf2(y_val):
                # BDF2: y - (4/3)y_n + (1/3)y_{n-1} - (2/3)dt * f(t+dt, y) = 0
                rhs(t_new, y_val, f_tmp, params, runtime_ws)
                return y_val - (4.0/3.0)*y_curr + (1.0/3.0)*y_nm1 - (2.0/3.0)*dt * f_tmp

            # ----------------- Solve -----------------
            
            current_residual = residual_bdf1 if step_idx == 0 else residual_bdf2

            sol = root(
                current_residual,
                y_guess,
                method=method,
                tol=root_tol,
                options={'maxfev': max_iter * (y_curr.size + 1)} # approx mapping for hybrid
            )

            if not sol.success:
                return STEPFAIL

            y_result = sol.x

            # Accept step: y_prop <- y_result
            y_prop[:] = y_result[:]

            # Rotate history: y_nm1 <- current committed y_n (y_curr)
            y_nm1[:] = y_curr[:]

            t_prop[0] = t_new
            dt_next[0] = dt
            err_est[0] = 0.0 # No embedded error estimation in fixed step BDF2

            step_idx_arr[0] = step_idx + 1
            return OK

        return bdf2_stepper


def _auto_register():
    from ..registry import register
    spec = BDF2ScipySpec()
    register(spec)


_auto_register()