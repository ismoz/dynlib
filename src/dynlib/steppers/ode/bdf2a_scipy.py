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


__all__ = ["BDF2AdaptiveScipySpec"]


class BDF2AdaptiveScipySpec(ConfigMixin):
    """
    Adaptive 2-step BDF method (order 2, variable step, SciPy-based).

    - Startup: backward Euler (BDF1) with implicit solve.
    - Main steps: variable-step BDF2 with embedded BDF1 for error estimation.
    - Error control: classic controller using (BDF2 - BDF1) difference.
    - Nonlinear systems solved with scipy.optimize.root (e.g. 'hybr').

    Not compatible with Numba nopython mode (Object mode only).
    """

    @dataclass
    class Config:
        """
        Runtime configuration for adaptive BDF2 (SciPy).

        - atol, rtol: error tolerances for the embedded BDF1/BDF2 pair.
        - safety, min_factor, max_factor: standard step-size controller params.
        - max_tries: max number of internal retries per outer step call.
        - min_step: lower bound on step size (absolute).
        - root_tol, max_iter, method: passed through to scipy.optimize.root.
        """
        # Error control
        atol: float = 1e-8
        rtol: float = 1e-5
        safety: float = 0.9
        min_factor: float = 0.2
        max_factor: float = 10.0
        max_tries: int = 8
        min_step: float = 1e-12

        # Nonlinear solver (Powell hybrid etc.)
        root_tol: float = 1e-8
        max_iter: int = 50
        method: str = "hybr"

        __enums__ = {
            "method": {"hybr": 0, "lm": 1, "broyden1": 2}
        }

    class Workspace(NamedTuple):
        # multistep history
        y_nm1: np.ndarray          # y_{n-1} at committed state
        step_index: np.ndarray     # shape (1,), int64, counts accepted steps
        h_prev: np.ndarray         # shape (1,), previous accepted step size

        # Scratch for RHS and predictor
        f_tmp: np.ndarray
        f_pred: np.ndarray

    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="bdf2a_scipy",
                kind="ode",
                time_control="adaptive",
                scheme="implicit",
                geometry=frozenset(),
                family="bdf",
                order=2,
                embedded_order=1,  # BDF1 used as embedded method
                stiff=True,
                aliases=("bdf2_adaptive_scipy",),
                caps=StepperCaps(
                    jacobian="internal",
                    requires_scipy=True,
                    jit_capable=False,
                ),
            )
        self.meta = meta

    # --- StepperSpec protocol hooks ---------------------------------------

    def workspace_type(self) -> type | None:
        return BDF2AdaptiveScipySpec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        def vec():
            return np.zeros((n_state,), dtype=dtype)

        return BDF2AdaptiveScipySpec.Workspace(
            y_nm1=vec(),
            step_index=np.zeros((1,), dtype=np.int64),
            h_prev=np.zeros((1,), dtype=dtype),
            f_tmp=vec(),
            f_pred=vec(),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate adaptive BDF2 stepper closure using scipy.optimize.root.

        Signature (frozen ABI):
            stepper(t, dt, y_curr, rhs, params, runtime_ws,
                    stepper_ws, stepper_config,
                    y_prop, t_prop, dt_next, err_est) -> int32
        """
        from scipy.optimize import root

        def bdf2_adaptive_stepper(
            t, dt,
            y_curr, rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # --- Unpack workspace ---
            y_nm1 = ws.y_nm1
            step_idx_arr = ws.step_index
            h_prev_arr = ws.h_prev
            f_tmp = ws.f_tmp
            f_pred = ws.f_pred

            step_idx = int(step_idx_arr[0])
            h_prev = float(h_prev_arr[0])

            # --- Unpack config (packed float64 array) ---
            # Order must match Config dataclass fields
            if stepper_config.size >= 11:
                atol = float(stepper_config[0])
                rtol = float(stepper_config[1])
                safety = float(stepper_config[2])
                min_factor = float(stepper_config[3])
                max_factor = float(stepper_config[4])
                max_tries = int(stepper_config[5])
                min_step = float(stepper_config[6])
                root_tol = float(stepper_config[7])
                max_iter = int(stepper_config[8])
                method_id = int(stepper_config[9])
                dt_max = float(stepper_config[10])
            else:
                atol = 1e-6
                rtol = 1e-5
                safety = 0.9
                min_factor = 0.2
                max_factor = 10.0
                max_tries = 8
                min_step = 1e-12
                root_tol = 1e-8
                max_iter = 50
                method_id = 0
                dt_max = np.inf
            if method_id == 0:
                method = "hybr"
            elif method_id == 1:
                method = "lm"
            elif method_id == 2:
                method = "broyden1"
            else:
                method = "hybr"

            # Ensure positive initial step guess
            h = float(dt)
            if h > dt_max:
                h = dt_max
            if not np.isfinite(h) or h <= 0.0:
                h = min_step

            # --- Helpers --------------------------------------------------

            def solve_bdf1(t_new, h_local, y_start, y_guess):
                """
                Implicit Euler (BDF1) solve:
                    y - y_start - h * f(t_new, y) = 0
                """
                def residual_bdf1(y_val):
                    rhs(t_new, y_val, f_tmp, params, runtime_ws)
                    return y_val - y_start - h_local * f_tmp

                sol = root(
                    residual_bdf1,
                    y_guess,
                    method=method,
                    tol=root_tol,
                    options={'maxfev': max_iter * (y_curr.size + 1)},
                )
                if not sol.success or not np.all(np.isfinite(sol.x)):
                    return None
                return sol.x

            def solve_bdf2_varstep(t_new, h_local, h_prev_local, y_nm1_local, y_n_local, y_guess):
                """
                Variable-step BDF2 solve using the 3-point backward differentiation
                formula on nonuniform steps.

                Derived formula:
                    [ h^2 y_{n-1} - (h + h_prev)^2 y_n + h_prev (2h + h_prev) y_{n+1} ]
                    / [ h_prev (h + h_prev) ] = h f(t_{n+1}, y_{n+1})
                """
                if h_prev_local <= 0.0:
                    # No previous step size: fall back to BDF1
                    return solve_bdf1(t_new, h_local, y_n_local, y_guess)

                def residual_bdf2(y_val):
                    rhs(t_new, y_val, f_tmp, params, runtime_ws)

                    num = (
                        (h_local * h_local) * y_nm1_local
                        - (h_local + h_prev_local) ** 2 * y_n_local
                        + h_prev_local * (2.0 * h_local + h_prev_local) * y_val
                    )
                    denom = h_prev_local * (h_local + h_prev_local)
                    return num / denom - h_local * f_tmp

                sol = root(
                    residual_bdf2,
                    y_guess,
                    method=method,
                    tol=root_tol,
                    options={'maxfev': max_iter * (y_curr.size + 1)},
                )
                if not sol.success or not np.all(np.isfinite(sol.x)):
                    return None
                return sol.x

            # --- Startup: first step (adaptive BDF1 with explicit Euler) ----
            if step_idx == 0:
                order_start = 1
                last_err_norm = np.inf

                for _try in range(max_tries):
                    if h < min_step or not np.isfinite(h):
                        # Can't take a sensible step
                        dt_next[0] = h
                        err_est[0] = float(last_err_norm) if np.isfinite(last_err_norm) else np.inf
                        return STEPFAIL

                    t_new = t + h

                    # Explicit Euler predictor y_exp = y_n + h f(t_n, y_n)
                    rhs(t, y_curr, f_pred, params, runtime_ws)
                    y_explicit = y_curr + h * f_pred  # reuse f_pred as scratch if you like

                    # Implicit Euler (BDF1) solve with y_explicit as initial guess
                    y_be = solve_bdf1(t_new, h, y_curr, y_explicit)
                    if y_be is None or not np.all(np.isfinite(y_be)):
                        # Nonlinear solve failed: shrink and retry
                        h *= max(min_factor, 0.5)
                        continue

                    # Error estimate: embedded pair (BDF1 - explicit Euler)
                    err_vec = y_be - y_explicit
                    scale = atol + rtol * np.maximum(np.abs(y_curr), np.abs(y_be))
                    scale[scale == 0.0] = 1.0

                    err_norm = np.sqrt(np.mean((err_vec / scale) ** 2))
                    last_err_norm = err_norm

                    if err_norm <= 1.0:
                        # Accept startup step
                        y_prop[:] = y_be[:]
                        t_prop[0] = t_new
                        err_est[0] = err_norm

                        # Propose next step size (p = 1 -> exponent 1/(p+1) = 1/2)
                        if err_norm == 0.0:
                            factor_next = max_factor
                        else:
                            factor_next = safety * err_norm ** (-0.5)
                            factor_next = max(min_factor, min(max_factor, factor_next))
                        dt_next[0] = h * factor_next
                        
                        # Cap at dt_max
                        if dt_next[0] > dt_max:
                            dt_next[0] = dt_max

                        # Initialize history for multistep
                        y_nm1[:] = y_curr[:]
                        h_prev_arr[0] = h
                        step_idx_arr[0] = 1
                        return OK

                    # Reject: shrink and retry
                    factor = safety * err_norm ** (-0.5)
                    if factor < min_factor:
                        factor = min_factor
                    h *= factor

                # If we exit the loop, give up on this step
                dt_next[0] = h
                err_est[0] = float(last_err_norm) if np.isfinite(last_err_norm) else np.inf
                return STEPFAIL

            # --- Main adaptive loop (BDF2 + embedded BDF1) ----------------

            order = 2  # BDF2
            last_err_norm = np.inf

            for _try in range(max_tries):
                if h < min_step or not np.isfinite(h):
                    return STEPFAIL

                t_new = t + h

                # Predictor: explicit Euler from (t, y_curr)
                rhs(t, y_curr, f_pred, params, runtime_ws)
                y_guess = y_curr + h * f_pred

                # 1) Solve BDF1
                y_be = solve_bdf1(t_new, h, y_curr, y_guess)
                if y_be is None or not np.all(np.isfinite(y_be)):
                    # Try smaller step
                    h *= max(min_factor, 0.5)
                    continue

                # 2) Solve variable-step BDF2, using y_be as initial guess
                y_bdf2 = solve_bdf2_varstep(
                    t_new, h, h_prev, y_nm1, y_curr, y_be
                )
                if y_bdf2 is None or not np.all(np.isfinite(y_bdf2)):
                    # Try smaller step
                    h *= max(min_factor, 0.5)
                    continue

                # 3) Error estimate: embedded pair (BDF2 - BDF1)
                err_vec = y_bdf2 - y_be
                scale = atol + rtol * np.maximum(np.abs(y_curr), np.abs(y_bdf2))
                # Avoid zero scale entirely:
                scale[scale == 0.0] = 1.0

                err_norm = np.sqrt(np.mean((err_vec / scale) ** 2))
                last_err_norm = err_norm

                if err_norm <= 1.0:
                    # Accept step
                    y_prop[:] = y_bdf2[:]
                    t_prop[0] = t_new
                    err_est[0] = err_norm

                    # Propose next step size
                    if err_norm == 0.0:
                        factor_next = max_factor
                    else:
                        factor_next = safety * err_norm ** (-1.0 / (order + 1.0))  # 1/(p+1) = 1/3
                        factor_next = max(min_factor, min(max_factor, factor_next))

                    dt_next[0] = h * factor_next
                    
                    # Cap at dt_max
                    if dt_next[0] > dt_max:
                        dt_next[0] = dt_max

                    # Update history for multistep
                    y_nm1[:] = y_curr[:]
                    h_prev_arr[0] = h
                    step_idx_arr[0] = step_idx + 1
                    return OK

                # Reject step: shrink h and retry
                factor = safety * err_norm ** (-1.0 / (order + 1.0))
                if factor < min_factor:
                    factor = min_factor
                h *= factor

            # If we exit loop without success, report failure
            # Runner may decide what to do with dt_next / err_est.
            dt_next[0] = h
            err_est[0] = float(last_err_norm) if np.isfinite(last_err_norm) else np.inf
            return STEPFAIL

        return bdf2_adaptive_stepper


def _auto_register():
    from ..registry import register
    register(BDF2AdaptiveScipySpec())

_auto_register()