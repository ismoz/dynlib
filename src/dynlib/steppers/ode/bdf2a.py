# src/dynlib/steppers/ode/bdf2a.py
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

__all__ = ["BDF2AdaptiveJITSpec"]

register_guards_consumer(globals(), mapping={"allfinite1d": "allfinite1d"})


class BDF2AdaptiveJITSpec(ConfigMixin):
    """
    Adaptive 2-step BDF method with Variable Step Size support.
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
        jacobian_mode: int = 0  # 0=internal FD, 1=external analytic
        __enums__ = {"jacobian_mode": {"internal": 0, "external": 1}}

    class Workspace(NamedTuple):
        y_nm1: np.ndarray          
        step_index: np.ndarray     
        dt_nm1: np.ndarray      # shape (1,)
        y_guess: np.ndarray        
        y_bdf2: np.ndarray         
        f_val: np.ndarray
        f_tmp: np.ndarray
        residual: np.ndarray
        J: np.ndarray              

    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="bdf2a",
                kind="ode",
                time_control="adaptive",
                scheme="implicit",
                geometry=frozenset(),  # match rk45 / bdf2_jit
                family="bdf",
                order=2,
                embedded_order=1,
                stiff=True,
                aliases=("bdf2a_jit",),
                caps=StepperCaps(
                    dense_output=False,
                    jacobian="optional",
                ),
            )
        self.meta = meta

    def workspace_type(self) -> type | None:
        return BDF2AdaptiveJITSpec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        def vec():
            return np.zeros((n_state,), dtype=dtype)

        return BDF2AdaptiveJITSpec.Workspace(
            y_nm1=vec(),
            step_index=np.zeros((1,), dtype=np.int64),
            dt_nm1=np.zeros((1,), dtype=np.float64), # Initialize dt_nm1
            y_guess=vec(),
            y_bdf2=vec(),
            f_val=vec(),
            f_tmp=vec(),
            residual=vec(),
            J=np.zeros((n_state, n_state), dtype=dtype),
        )

    def emit(self, rhs_fn: Callable, model_spec=None, jacobian_fn=None, jvp_fn=None) -> Callable:
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

        def bdf2a_stepper(
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

            y_nm1 = ws.y_nm1
            step_idx_arr = ws.step_index
            dt_nm1_arr = ws.dt_nm1 # Access history dt

            y_guess = ws.y_guess
            y_bdf2 = ws.y_bdf2
            f_val = ws.f_val
            f_tmp = ws.f_tmp
            residual = ws.residual
            J = ws.J

            step_idx = int(step_idx_arr[0])
            # For BDF1 solves we use a configurable "previous state" y_back.
            # Normally this is y_curr, but during the step-0 Richardson start
            # it will point to an intermediate half-step state.
            y_back = y_curr
            
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

            if max_tries < 1: max_tries = 1
            if newton_max_iter < 1: newton_max_iter = 1
            prefer_external_jac = bool(jac_mode == 1 and jacobian_fn is not None)

            h = dt
            if h > dt_max:
                h = dt_max
            error = float("inf")

            # --- Newton Solver with Variable Coefficients ---
            def newton_solve(mode, t_new_local, h_local, bdf2_coeffs):
                # mode 1: BDF1 (Backward Euler)
                # mode 2: BDF2 (Variable Step)
                # bdf2_coeffs: tuple(alpha_curr, alpha_nm1, beta_h)
                
                # Unpack BDF2 specific coefficients
                # We solve: y - alpha_curr * y_n + alpha_nm1 * y_{n-1} - beta_h * h * f(y) = 0
                c_curr = 0.0
                c_prev = 0.0
                c_beta = 0.0
                
                if mode == 2:
                    c_curr = bdf2_coeffs[0]
                    c_prev = bdf2_coeffs[1]
                    c_beta = bdf2_coeffs[2]

                for it in range(newton_max_iter):
                    rhs(t_new_local, y_guess, f_val, params, runtime_ws)

                    max_r = 0.0
                    max_scale = 0.0

                    # Build Residual
                    if mode == 1:
                        # BDF1: y - y_back - h*f = 0
                        for i in range(n):
                            r_i = y_guess[i] - y_back[i] - h_local * f_val[i]
                            residual[i] = r_i
                            # ... tracking max_r/scale ...
                            abs_r = r_i if r_i >= 0.0 else -r_i
                            if abs_r > max_r: max_r = abs_r
                            abs_y = y_guess[i] if y_guess[i] >= 0.0 else -y_guess[i]
                            if abs_y > max_scale: max_scale = abs_y
                            abs_curr = y_back[i] if y_back[i] >= 0.0 else -y_back[i]
                            if abs_curr > max_scale: max_scale = abs_curr

                    else:
                        # BDF2 Variable: y - c_curr*y_n + c_prev*y_{n-1} - c_beta*h*f = 0
                        for i in range(n):
                            r_i = (
                                y_guess[i]
                                - c_curr * y_curr[i]
                                + c_prev * y_nm1[i]
                                - c_beta * h_local * f_val[i]
                            )
                            residual[i] = r_i
                            # ... tracking max_r/scale ...
                            abs_r = r_i if r_i >= 0.0 else -r_i
                            if abs_r > max_r: max_r = abs_r
                            
                            # Check scale against all 3 points
                            abs_y = y_guess[i] if y_guess[i] >= 0.0 else -y_guess[i]
                            if abs_y > max_scale: max_scale = abs_y
                            abs_curr = y_curr[i] if y_curr[i] >= 0.0 else -y_curr[i]
                            if abs_curr > max_scale: max_scale = abs_curr
                            abs_prev = y_nm1[i] if y_nm1[i] >= 0.0 else -y_nm1[i]
                            if abs_prev > max_scale: max_scale = abs_prev

                    if not allfinite1d(residual):
                        return False

                    # Build Jacobian (analytic if requested/available, else FD)
                    if mode == 1:
                        factor_fd = h_local
                    else:
                        factor_fd = c_beta * h_local

                    if prefer_external_jac and jacobian_fn is not None:
                        jacobian_fn(t_new_local, y_guess, params, J, runtime_ws)
                        for i in range(n):
                            for j in range(n):
                                J[i, j] = -factor_fd * J[i, j]
                            J[i, i] = J[i, i] + 1.0
                    else:
                        for j in range(n):
                            # Finite diff column j
                            orig_val = y_guess[j]
                            abs_base = orig_val if orig_val >= 0 else -orig_val
                            scale = abs_base if abs_base > 1.0 else 1.0
                            eps = jac_eps * scale
                            y_guess[j] = orig_val + eps
                            
                            rhs(t_new_local, y_guess, f_tmp, params, runtime_ws)
                            
                            y_guess[j] = orig_val # restore
                            
                            if not allfinite1d(f_tmp): return False
                            inv_eps = 1.0 / eps

                            # J = I - factor * df/dy
                            for i in range(n):
                                df_ij = (f_tmp[i] - f_val[i]) * inv_eps
                                if i == j:
                                    J[i, j] = 1.0 - factor_fd * df_ij
                                else:
                                    J[i, j] = -factor_fd * df_ij

                    # ... Linear Solve  ...
                    for i in range(n): residual[i] = -residual[i]
                    
                    # Gaussian Elimination
                    k = 0
                    singular = False
                    while k < n:
                        pivot_row = k
                        val = J[k, k]
                        pivot_val = val if val >= 0 else -val
                        i2 = k + 1
                        while i2 < n:
                            v = J[i2, k]
                            av = v if v >= 0 else -v
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

                    if singular: return False

                    # Back sub
                    i2 = n - 1
                    while i2 >= 0:
                        sum_val = residual[i2]
                        j2 = i2 + 1
                        while j2 < n:
                            sum_val = sum_val - J[i2, j2] * residual[j2]
                            j2 += 1
                        residual[i2] = sum_val / J[i2, i2]
                        i2 -= 1

                    # Update
                    max_delta_scaled = 0.0
                    for i in range(n):
                        delta_i = residual[i]
                        old_y = y_guess[i]
                        y_guess[i] = old_y + delta_i
                        
                        abs_d = delta_i if delta_i >= 0 else -delta_i
                        abs_s = old_y if old_y >= 0 else -old_y
                        if abs_s < 1.0: abs_s = 1.0
                        sc = abs_d / abs_s
                        if sc > max_delta_scaled: max_delta_scaled = sc

                    scale_tol = max_scale if max_scale > 1.0 else 1.0
                    if max_r <= newton_tol * scale_tol and max_delta_scaled <= newton_tol:
                        return True
                
                return False

            # --- Adaptive Loop ---
            for attempt in range(max_tries):
                t_new = t + h
                
                # Dummy coeffs for mode 1
                dummy_coeffs = (0.0, 0.0, 0.0)

                if step_idx == 0:
                    # --- Adaptive BDF1 startup using Richardson (h vs 2×h/2) ---
                    #
                    # Coarse: one BE step of size h  from (t, y_curr) -> y_coarse
                    # Fine:   two BE steps of size h/2 via an intermediate y_mid
                    # Error ~ || y_fine - y_coarse ||_wrms

                    # 1) Coarse BE step (step size = h)
                    rhs(t, y_curr, f_tmp, params, runtime_ws)
                    if not allfinite1d(f_tmp):
                        err_est[0] = float("inf")
                        return STEPFAIL

                    for i in range(n):
                        y_guess[i] = y_curr[i] + h * f_tmp[i]

                    y_back = y_curr
                    if not newton_solve(1, t_new, h, dummy_coeffs):
                        if h <= min_step:
                            err_est[0] = float("inf")
                            return STEPFAIL
                        h = h * min_factor
                        if h < min_step:
                            h = min_step
                        continue

                    # Store coarse solution in y_bdf2
                    for i in range(n):
                        y_bdf2[i] = y_guess[i]

                    # 2) Fine BE solution via two half-steps (h/2)
                    h_half = 0.5 * h
                    t_half = t + h_half

                    # First half-step: (t, y_curr) -> (t + h/2, y_mid)
                    rhs(t, y_curr, f_tmp, params, runtime_ws)
                    if not allfinite1d(f_tmp):
                        err_est[0] = float("inf")
                        return STEPFAIL

                    for i in range(n):
                        y_guess[i] = y_curr[i] + h_half * f_tmp[i]

                    # y_back is still y_curr here
                    if not newton_solve(1, t_half, h_half, dummy_coeffs):
                        if h <= min_step:
                            err_est[0] = float("inf")
                            return STEPFAIL
                        h = h * min_factor
                        if h < min_step:
                            h = min_step
                        continue

                    # Save y_mid in y_nm1 workspace
                    for i in range(n):
                        y_nm1[i] = y_guess[i]

                    # Second half-step: (t + h/2, y_mid) -> (t + h, y_fine)
                    rhs(t_half, y_nm1, f_tmp, params, runtime_ws)
                    if not allfinite1d(f_tmp):
                        err_est[0] = float("inf")
                        return STEPFAIL

                    for i in range(n):
                        y_guess[i] = y_nm1[i] + h_half * f_tmp[i]

                    # Now previous-state for BE is y_mid
                    y_back = y_nm1
                    if not newton_solve(1, t_new, h_half, dummy_coeffs):
                        if h <= min_step:
                            err_est[0] = float("inf")
                            return STEPFAIL
                        h = h * min_factor
                        if h < min_step:
                            h = min_step
                        continue

                    # y_guess now holds the fine solution y_fine.

                    # 3) Error estimate: WRMS(y_fine - y_coarse)
                    err_acc = 0.0
                    for i in range(n):
                        diff = y_guess[i] - y_bdf2[i]
                        if diff < 0.0:
                            diff = -diff

                        y_scale = y_curr[i] if y_curr[i] >= 0.0 else -y_curr[i]
                        y2 = y_guess[i] if y_guess[i] >= 0.0 else -y_guess[i]
                        if y2 > y_scale:
                            y_scale = y2

                        scale_i = atol + rtol * y_scale
                        ratio = diff / scale_i
                        err_acc += ratio * ratio

                    error = math.sqrt(err_acc / n)
                    if not math.isfinite(error):
                        error = float("inf")

                    if error <= 1.0 or h <= min_step:
                        # Accept startup step: use fine solution as y_1
                        for i in range(n):
                            y_nm1[i] = y_curr[i]   # history for BDF2
                        for i in range(n):
                            y_prop[i] = y_guess[i]  # fine solution y_fine

                        t_prop[0] = t_new
                        err_est[0] = error

                        if error > 0.0:
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

                        step_idx_arr[0] = 1
                        dt_nm1_arr[0] = h  # Save dt for next (BDF2) step
                        return OK

                    # Reject startup step, shrink h and retry
                    factor = safety * math.sqrt(1.0 / error)
                    if factor < min_factor:
                        factor = min_factor
                    h = h * factor
                    if h < min_step:
                        dt_next[0] = h
                        err_est[0] = error
                        return STEPFAIL
                    continue

                # --- BDF2 Main Step (Variable) ---
                
                # 1. Calculate Variable Coefficients based on rho = h_new / h_old
                dt_old = dt_nm1_arr[0]
                # Guard: if dt_old is not sensible (e.g. 0 on a resume),
                # fall back to constant-step coefficients (rho = 1).
                if dt_old <= 0.0:
                    rho = 1.0
                else:
                    rho = h / dt_old
                
                # Standard Variable BDF2 Coefficients (normalized for y_{n+1}):
                # y - c_curr * y_n + c_prev * y_{n-1} - c_beta * h * f(y) = 0
                #
                # Denominator D = 1 + 2*rho
                # c_curr = (1 + rho)^2 / D
                # c_prev = rho^2 / D
                # c_beta = (1 + rho) / D
                
                denom = 1.0 + 2.0 * rho
                inv_denom = 1.0 / denom
                
                one_plus_rho = 1.0 + rho
                c_curr = (one_plus_rho * one_plus_rho) * inv_denom
                c_prev = (rho * rho) * inv_denom
                c_beta = one_plus_rho * inv_denom
                
                coeffs = (c_curr, c_prev, c_beta)

                # 2. Variable Step Predictor (Linear Extrapolation)
                # y_pred = (1 + rho) * y_n - rho * y_{n-1}
                for i in range(n):
                    y_guess[i] = (1.0 + rho) * y_curr[i] - rho * y_nm1[i]

                if not newton_solve(2, t_new, h, coeffs):
                    if h <= min_step:
                        err_est[0] = float("inf")
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step: h = min_step
                    continue

                # Save BDF2 result
                for i in range(n):
                    y_bdf2[i] = y_guess[i]
                
                # 3. Error Estimate: Solve BDF1 on same step
                # Use BDF2 result as warm start; previous state is y_curr
                for i in range(n):
                    y_guess[i] = y_bdf2[i]
                y_back = y_curr

                if not newton_solve(1, t_new, h, dummy_coeffs):
                    if h <= min_step:
                        err_est[0] = float("inf")
                        return STEPFAIL
                    h = h * min_factor
                    if h < min_step: h = min_step
                    continue
                
                # Calculate Error ||y_bdf2 - y_bdf1||_wrms
                err_acc = 0.0
                for i in range(n):
                    diff = y_bdf2[i] - y_guess[i]
                    if diff < 0.0: diff = -diff
                    
                    y_scale = y_curr[i] if y_curr[i] >= 0 else -y_curr[i]
                    y2 = y_bdf2[i] if y_bdf2[i] >= 0 else -y_bdf2[i]
                    if y2 > y_scale: y_scale = y2
                    
                    scale_i = atol + rtol * y_scale
                    ratio = diff / scale_i
                    err_acc += ratio * ratio
                
                error = math.sqrt(err_acc / n)
                if not math.isfinite(error): error = float("inf")

                if error <= 1.0 or h <= min_step:
                    # Accept
                    for i in range(n): y_nm1[i] = y_curr[i]
                    for i in range(n): y_prop[i] = y_bdf2[i]
                    
                    t_prop[0] = t_new
                    err_est[0] = error
                    
                    if error > 0.0:
                        factor = safety * math.sqrt(1.0 / error)
                        if factor < min_factor: factor = min_factor
                        if factor > max_factor: factor = max_factor
                        dt_next[0] = h * factor
                    else:
                        dt_next[0] = h * max_factor
                    
                    # Cap at dt_max
                    if dt_next[0] > dt_max:
                        dt_next[0] = dt_max
                    
                    step_idx_arr[0] = step_idx + 1
                    dt_nm1_arr[0] = h # Update history dt
                    return OK
                
                # Reject
                factor = safety * math.sqrt(1.0 / error)
                if factor < min_factor: factor = min_factor
                h = h * factor
                if h < min_step:
                    dt_next[0] = h
                    err_est[0] = error
                    return STEPFAIL

            err_est[0] = error
            return STEPFAIL

        return bdf2a_stepper

def _auto_register():
    from ..registry import register
    spec = BDF2AdaptiveJITSpec()
    register(spec)

_auto_register()
