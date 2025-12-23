# src/dynlib/steppers/ode/tr_bdf2a.py
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
import math
import numpy as np

from ..base import StepperMeta, StepperCaps
from ..config_base import ConfigMixin
from dynlib.runtime.runner_api import OK, STEPFAIL
from dynlib.compiler.guards import allfinite1d, register_guards_consumer

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["TRBDF2AdaptiveJITSpec"]

# Let the guards be inlined into the generated source for Numba
register_guards_consumer(globals(), mapping={"allfinite1d": "allfinite1d"})


class TRBDF2AdaptiveJITSpec(ConfigMixin):
    """
    Adaptive TR-BDF2 method (trapezoidal rule + BDF2), Numba-compatible.

    Scheme (one-step, L-stable for γ = 2 - sqrt(2)):

        Stage 1 (TR, Crank–Nicolson over [t_n, t_n + γ h]):

            y_{n+γ} - (γ h / 2) f(t_n + γ h, y_{n+γ})
              = y_n + (γ h / 2) f(t_n, y_n)

        Stage 2 (BDF2-like over [t_n, t_n + h]):

            y_{n+1} - γ2 h f(t_{n+1}, y_{n+1})
              = (1 - γ3) y_n + γ3 y_{n+γ}

        with γ2 = (1 - γ) / (2 - γ),
             γ3 = 1 / (γ (2 - γ)).

    Error estimate:
        - Main solution: TR-BDF2 (order 2).
        - Embedded solution: Backward Euler on [t_n, t_n + h] (order 1).
        - Local error ~ || y_TRBDF2 - y_BE ||_WRMS, same norm as bdf2a.
    """

    @dataclass
    class Config:
        atol: float = 1e-6
        rtol: float = 1e-3
        safety: float = 0.9
        min_factor: float = 0.2
        max_factor: float = 5.0
        max_tries: int = 10
        min_step: float = 1e-12
        newton_tol: float = 1e-8
        newton_max_iter: int = 50
        jac_eps: float = 1e-8
        dt_max: float = np.inf
        jacobian_mode: int = 0  # 0=internal (FD), 1=external (analytic)
        __enums__ = {"jacobian_mode": {"internal": 0, "external": 1}}

    class Workspace(NamedTuple):
        # Step counter (kept for symmetry with bdf2a, though not strictly needed)
        step_index: np.ndarray  # shape (1,)

        # Newton iterate / generic work vector
        y_guess: np.ndarray

        # TR stage value y_{n+γ}
        y_stage: np.ndarray

        # Main TR-BDF2 result y_{n+1}
        y_trbdf2: np.ndarray

        # Embedded Backward Euler result y_{n+1}^{BE}
        y_be: np.ndarray

        # f(t_n, y_n) for TR stage RHS
        f_n: np.ndarray

        # Scratch for current f(t, y_guess)
        f_val: np.ndarray

        # Scratch for finite-diff Jacobian columns
        f_tmp: np.ndarray

        # Residual vector (also reused to store Newton correction)
        residual: np.ndarray

        # Dense Jacobian matrix (frozen FD Jacobian for Modified Newton)
        J: np.ndarray

        # Work matrix for LU factorization (destroyed each solve)
        J_work: np.ndarray

    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="tr-bdf2a",
                kind="ode",
                time_control="adaptive",
                scheme="implicit",
                geometry=frozenset(),
                family="tr-bdf2",
                order=2,
                embedded_order=1,  # BE partner
                stiff=True,
                aliases=("trbdf2a",),
                caps=StepperCaps(
                    dense_output=False,
                    jacobian="optional",
                ),
            )
        self.meta = meta

    def workspace_type(self) -> type | None:
        return TRBDF2AdaptiveJITSpec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        def vec():
            return np.zeros((n_state,), dtype=dtype)

        return TRBDF2AdaptiveJITSpec.Workspace(
            step_index=np.zeros((1,), dtype=np.int64),
            y_guess=vec(),
            y_stage=vec(),
            y_trbdf2=vec(),
            y_be=vec(),
            f_n=vec(),
            f_val=vec(),
            f_tmp=vec(),
            residual=vec(),
            J=np.zeros((n_state, n_state), dtype=dtype),
            J_work=np.zeros((n_state, n_state), dtype=dtype),
        )

    def emit(self, rhs_fn: Callable, model_spec=None, jacobian_fn=None, jvp_fn=None) -> Callable:
        # Default config values (baked in as constants)
        default_cfg = self.default_config(model_spec)
        default_atol = default_cfg.atol
        default_rtol = default_cfg.rtol
        default_safety = default_cfg.safety
        default_min_factor = default_cfg.min_factor
        default_max_factor = default_cfg.max_factor
        default_max_tries = default_cfg.max_tries
        default_min_step = default_cfg.min_step
        default_newton_tol = default_cfg.newton_tol
        default_newton_max_iter = default_cfg.newton_max_iter
        default_jac_eps = default_cfg.jac_eps
        default_dt_max = default_cfg.dt_max
        default_jac_mode = default_cfg.jacobian_mode

        # TR-BDF2 parameter (L-stable choice)
        gamma = 2.0 - math.sqrt(2.0)
        gamma2 = (1.0 - gamma) / (2.0 - gamma)
        gamma3 = 1.0 / (gamma * (2.0 - gamma))
        theta_tr = 0.5 * gamma  # γ/2 factor in TR stage

        def tr_bdf2a_stepper(
            t,
            dt,
            y_curr,
            rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop,
            t_prop,
            dt_next,
            err_est,
        ):
            n = y_curr.size

            step_idx_arr = ws.step_index
            y_guess = ws.y_guess
            y_stage = ws.y_stage
            y_trbdf2 = ws.y_trbdf2
            y_be = ws.y_be
            f_n = ws.f_n
            f_val = ws.f_val
            f_tmp = ws.f_tmp
            residual = ws.residual
            # J_base: frozen FD Jacobian (Modified Newton)
            # J: work matrix for LU factorization
            J_base = ws.J
            J = ws.J_work

            # Just for potential diagnostics; not required by the scheme
            step_idx = int(step_idx_arr[0])

            # Read runtime config if provided (same layout as bdf2a)
            if stepper_config.size >= 12:
                atol = stepper_config[0]
                rtol = stepper_config[1]
                safety = stepper_config[2]
                min_factor = stepper_config[3]
                max_factor = stepper_config[4]
                max_tries = int(stepper_config[5])
                min_step = stepper_config[6]
                newton_tol = stepper_config[7]
                newton_max_iter = int(stepper_config[8])
                jac_eps = stepper_config[9]
                dt_max = stepper_config[10]
                jac_mode = int(stepper_config[11])
                if jac_mode != 1:
                    jac_mode = 0
            else:
                atol = default_atol
                rtol = default_rtol
                safety = default_safety
                min_factor = default_min_factor
                max_factor = default_max_factor
                max_tries = default_max_tries
                min_step = default_min_step
                newton_tol = default_newton_tol
                newton_max_iter = default_newton_max_iter
                jac_eps = default_jac_eps
                dt_max = default_dt_max
                jac_mode = default_jac_mode

            if max_tries < 1:
                max_tries = 1
            if newton_max_iter < 1:
                newton_max_iter = 1
            prefer_external_jac = bool(jac_mode == 1 and jacobian_fn is not None)

            h = dt
            if h > dt_max:
                h = dt_max
            error = float("inf")

            # mode:
            #   0 -> TR stage (y_{n+γ})
            #   1 -> TR-BDF2 final stage (y_{n+1})
            #   2 -> Backward Euler (embedded, y_{n+1}^{BE})
            def newton_solve(mode, t_local, h_local, c1, c2):
                """
                Modified Newton:
                - Build FD Jacobian once per solve into J_base.
                - Reuse that frozen Jacobian for all iterations, only recomputing
                  the residual and refactoring a work copy J each time.

                c1, c2 carry scalar coefficients:
                  mode 0: c1 = theta_tr
                  mode 1: c1 = gamma2, c2 = gamma3
                  mode 2: Backward Euler (c1, c2 unused)
                """

                # --- Build Jacobian once at initial guess (analytic if requested/available, else FD) ---
                rhs(t_local, y_guess, f_val, params, runtime_ws)
                if not allfinite1d(f_val):
                    return False

                # Determine scaling factor for this mode
                if mode == 0:
                    factor_fd = c1 * h_local
                elif mode == 1:
                    factor_fd = c1 * h_local
                else:
                    factor_fd = h_local

                if prefer_external_jac and jacobian_fn is not None:
                    jacobian_fn(t_local, y_guess, params, J_base, runtime_ws)
                    # Transform J_base in-place to (I - factor * df/dy)
                    for i in range(n):
                        for j in range(n):
                            J_base[i, j] = -factor_fd * J_base[i, j]
                        J_base[i, i] = J_base[i, i] + 1.0
                else:
                    for j in range(n):
                        orig_val = y_guess[j]
                        abs_base = orig_val if orig_val >= 0.0 else -orig_val
                        scale = abs_base if abs_base > 1.0 else 1.0
                        eps = jac_eps * scale

                        y_guess[j] = orig_val + eps
                        rhs(t_local, y_guess, f_tmp, params, runtime_ws)
                        y_guess[j] = orig_val  # restore

                        if not allfinite1d(f_tmp):
                            return False

                        inv_eps = 1.0 / eps

                        for i in range(n):
                            df_ij = (f_tmp[i] - f_val[i]) * inv_eps
                            if i == j:
                                J_base[i, j] = 1.0 - factor_fd * df_ij
                            else:
                                J_base[i, j] = -factor_fd * df_ij

                if not allfinite1d(J_base.ravel()):
                    return False

                # --- Newton iteration with frozen Jacobian J_base ---
                for _it in range(newton_max_iter):
                    rhs(t_local, y_guess, f_val, params, runtime_ws)
                    if not allfinite1d(f_val):
                        return False

                    max_r = 0.0
                    max_scale = 0.0

                    # Build residual
                    if mode == 0:
                        # TR stage:
                        #   y_stage - θ h f(t_n+γh, y_stage)
                        #     = y_n + θ h f(t_n, y_n)
                        # Residual: y_guess - θ h f_val - (y_n + θ h f_n) = 0
                        theta_loc = c1
                        for i in range(n):
                            base_i = y_curr[i] + theta_loc * h_local * f_n[i]
                            r_i = y_guess[i] - theta_loc * h_local * f_val[i] - base_i
                            residual[i] = r_i

                            abs_r = r_i if r_i >= 0.0 else -r_i
                            if abs_r > max_r:
                                max_r = abs_r

                            abs_y = y_guess[i] if y_guess[i] >= 0.0 else -y_guess[i]
                            if abs_y > max_scale:
                                max_scale = abs_y
                            abs_curr = y_curr[i] if y_curr[i] >= 0.0 else -y_curr[i]
                            if abs_curr > max_scale:
                                max_scale = abs_curr
                            abs_fn = f_n[i] if f_n[i] >= 0.0 else -f_n[i]
                            if abs_fn > max_scale:
                                max_scale = abs_fn

                    elif mode == 1:
                        # TR-BDF2 final stage:
                        #   y_{n+1} - γ2 h f(t_{n+1}, y_{n+1})
                        #     = (1 - γ3) y_n + γ3 y_{n+γ}
                        gamma2_loc = c1
                        gamma3_loc = c2
                        for i in range(n):
                            base_i = (
                                (1.0 - gamma3_loc) * y_curr[i]
                                + gamma3_loc * y_stage[i]
                            )
                            r_i = y_guess[i] - gamma2_loc * h_local * f_val[i] - base_i
                            residual[i] = r_i

                            abs_r = r_i if r_i >= 0.0 else -r_i
                            if abs_r > max_r:
                                max_r = abs_r

                            abs_y = y_guess[i] if y_guess[i] >= 0.0 else -y_guess[i]
                            if abs_y > max_scale:
                                max_scale = abs_y
                            abs_curr = y_curr[i] if y_curr[i] >= 0.0 else -y_curr[i]
                            if abs_curr > max_scale:
                                max_scale = abs_curr
                            abs_stage = y_stage[i] if y_stage[i] >= 0.0 else -y_stage[i]
                            if abs_stage > max_scale:
                                max_scale = abs_stage

                    else:
                        # Backward Euler embedded:
                        #   y - h f(t_{n+1}, y) = y_n
                        # Residual: y_guess - h f_val - y_n = 0
                        for i in range(n):
                            r_i = y_guess[i] - h_local * f_val[i] - y_curr[i]
                            residual[i] = r_i

                            abs_r = r_i if r_i >= 0.0 else -r_i
                            if abs_r > max_r:
                                max_r = abs_r

                            abs_y = y_guess[i] if y_guess[i] >= 0.0 else -y_guess[i]
                            if abs_y > max_scale:
                                max_scale = abs_y
                            abs_curr = y_curr[i] if y_curr[i] >= 0.0 else -y_curr[i]
                            if abs_curr > max_scale:
                                max_scale = abs_curr

                    if not allfinite1d(residual):
                        return False

                    # Copy frozen Jacobian into work matrix
                    for i in range(n):
                        for j in range(n):
                            J[i, j] = J_base[i, j]

                    # Solve J * delta = -residual via Gaussian elimination
                    for i in range(n):
                        residual[i] = -residual[i]

                    k = 0
                    singular = False
                    while k < n:
                        pivot_row = k
                        val = J[k, k]
                        pivot_val = val if val >= 0.0 else -val

                        i2 = k + 1
                        while i2 < n:
                            v = J[i2, k]
                            av = v if v >= 0.0 else -v
                            if av > pivot_val:
                                pivot_val = av
                                pivot_row = i2
                            i2 += 1

                        if pivot_val <= 1e-14:
                            singular = True
                            break

                        if pivot_row != k:
                            j2 = 0
                            while j2 < n:
                                tmp = J[k, j2]
                                J[k, j2] = J[pivot_row, j2]
                                J[pivot_row, j2] = tmp
                                j2 += 1
                            tmpb = residual[k]
                            residual[k] = residual[pivot_row]
                            residual[pivot_row] = tmpb

                        diag = J[k, k]
                        inv_diag = 1.0 / diag

                        i2 = k + 1
                        while i2 < n:
                            factor = J[i2, k] * inv_diag
                            J[i2, k] = 0.0
                            j2 = k + 1
                            while j2 < n:
                                J[i2, j2] = J[i2, j2] - factor * J[k, j2]
                                j2 += 1
                            residual[i2] = residual[i2] - factor * residual[k]
                            i2 += 1

                        k += 1

                    if singular:
                        return False

                    # Back substitution
                    i2 = n - 1
                    while i2 >= 0:
                        sum_val = residual[i2]
                        j2 = i2 + 1
                        while j2 < n:
                            sum_val = sum_val - J[i2, j2] * residual[j2]
                            j2 += 1
                        residual[i2] = sum_val / J[i2, i2]
                        i2 -= 1

                    # Update y_guess and check Newton convergence
                    max_delta_scaled = 0.0
                    for i in range(n):
                        delta_i = residual[i]
                        old_y = y_guess[i]
                        y_guess[i] = old_y + delta_i

                        abs_d = delta_i if delta_i >= 0.0 else -delta_i
                        abs_s = old_y if old_y >= 0.0 else -old_y
                        if abs_s < 1.0:
                            abs_s = 1.0
                        sc = abs_d / abs_s
                        if sc > max_delta_scaled:
                            max_delta_scaled = sc

                    scale_tol = max_scale if max_scale > 1.0 else 1.0
                    if max_r <= newton_tol * scale_tol and max_delta_scaled <= newton_tol:
                        return True

                return False

            # --- Adaptive step loop ---
            for _attempt in range(max_tries):
                t_new = t + h
                t_stage = t + gamma * h

                # Stage 0: f(t_n, y_n) (used by TR stage RHS)
                rhs(t, y_curr, f_n, params, runtime_ws)
                if not allfinite1d(f_n):
                    err_est[0] = float("inf")
                    return STEPFAIL

                # --- Stage 1: TR (Crank–Nicolson) to y_{n+γ} ---

                # Explicit Euler prediction over γ h
                for i in range(n):
                    y_guess[i] = y_curr[i] + gamma * h * f_n[i]

                if not newton_solve(0, t_stage, h, theta_tr, 0.0):
                    # Try smaller step
                    if h <= min_step:
                        err_est[0] = float("inf")
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # Save y_{n+γ}
                for i in range(n):
                    y_stage[i] = y_guess[i]

                # --- Stage 2: BDF2-like final stage y_{n+1} ---

                # Use y_stage as initial guess for final state
                for i in range(n):
                    y_guess[i] = y_stage[i]

                if not newton_solve(1, t_new, h, gamma2, gamma3):
                    if h <= min_step:
                        err_est[0] = float("inf")
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                # TR-BDF2 main solution
                for i in range(n):
                    y_trbdf2[i] = y_guess[i]

                # --- Embedded Backward Euler on same step for error estimate ---

                for i in range(n):
                    y_guess[i] = y_trbdf2[i]

                if not newton_solve(2, t_new, h, 0.0, 0.0):
                    if h <= min_step:
                        err_est[0] = float("inf")
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step:
                        h = min_step
                    continue

                for i in range(n):
                    y_be[i] = y_guess[i]

                # --- WRMS error between TR-BDF2 and BE ---

                err_acc = 0.0
                for i in range(n):
                    diff = y_trbdf2[i] - y_be[i]
                    if diff < 0.0:
                        diff = -diff

                    y_scale = y_curr[i] if y_curr[i] >= 0.0 else -y_curr[i]
                    y2 = y_trbdf2[i] if y_trbdf2[i] >= 0.0 else -y_trbdf2[i]
                    if y2 > y_scale:
                        y_scale = y2

                    scale_i = atol + rtol * y_scale
                    ratio = diff / scale_i
                    err_acc += ratio * ratio

                error = math.sqrt(err_acc / n)
                if not math.isfinite(error):
                    error = float("inf")

                if error <= 1.0 or h <= min_step:
                    # Accept step
                    for i in range(n):
                        y_prop[i] = y_trbdf2[i]

                    t_prop[0] = t_new
                    err_est[0] = error

                    # Step-size update, same pattern as bdf2a
                    if error > 0.0:
                        # Order-2 main method with order-1 embedded:
                        # local error ~ O(h^2) => exponent 1 / (1 + 1) = 1/2
                        factor = safety * math.sqrt(1.0 / error)
                        if factor < min_factor:
                            factor = min_factor
                        if factor > max_factor:
                            factor = max_factor
                        dt_next[0] = h * factor
                    else:
                        dt_next[0] = h * max_factor
                    
                    # Cap at dt_max
                    if dt_next[0] > dt_max:
                        dt_next[0] = dt_max

                    step_idx_arr[0] = step_idx + 1
                    return OK

                # Reject step, shrink h and retry
                factor = safety * math.sqrt(1.0 / error)
                if factor < min_factor:
                    factor = min_factor
                h = h * factor
                if h < min_step:
                    dt_next[0] = h
                    err_est[0] = error
                    return STEPFAIL

            err_est[0] = error
            return STEPFAIL

        return tr_bdf2a_stepper


def _auto_register():
    from ..registry import register
    spec = TRBDF2AdaptiveJITSpec()
    register(spec)


_auto_register()
