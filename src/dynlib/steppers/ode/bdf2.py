# src/dynlib/steppers/ode/bdf2.py
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
import numpy as np

from ..base import StepperMeta, StepperCaps
from dynlib.runtime.runner_api import OK, STEPFAIL
from dynlib.compiler.guards import allfinite1d, register_guards_consumer
from ..config_base import ConfigMixin

if TYPE_CHECKING:
    from typing import Callable


__all__ = ["BDF2JITSpec"]

register_guards_consumer(globals(), mapping={"allfinite1d": "allfinite1d"})


class BDF2JITSpec(ConfigMixin):
    """
    Numba Compatible Implicit 2-step BDF method (order 2, fixed-step).

    - Startup: backward Euler (BDF1) using full Newton with numerical Jacobian.
    - Main steps: BDF2 using full Newton with numerical Jacobian.
    - All scratch space is stored in a NamedTuple workspace, allocated once.
    - Since it does not rely on minpack or similar libraries, it is not as robust.
      Use your own discretion when applying to stiff systems.
    - DO NOT USE SOLVERS REQUIRING CONTINUOUS JACOBIAN FOR PROBLEMS WITH DISCONTINUITIES
      LIKE NEURON MODELS WITH THRESHOLDS AND RESET.
    """

    @dataclass
    class Config:
        # Absolute residual tolerance (on max-norm of F)
        newton_tol: float = 1e-8
        # Iteration budget (chaotic / stiff systems usually need a bit more)
        newton_max_iter: int = 50
        # Base finite-difference scale for df/dy
        jac_eps: float = 1e-8
        jacobian_mode: int = 0  # 0=internal FD, 1=external analytic
        __enums__ = {"jacobian_mode": {"internal": 0, "external": 1}}

    class Workspace(NamedTuple):
        # multistep history (not used in BDF1)
        y_nm1: np.ndarray          # y_{n-1} at committed state
        step_index: np.ndarray     # shape (1,), int64, counts accepted steps

        # Newton / Jacobian scratch
        y_guess: np.ndarray
        y_tmp: np.ndarray
        f_val: np.ndarray
        f_tmp: np.ndarray
        residual: np.ndarray
        J: np.ndarray              # (n_state, n_state)

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
                stiff=True,
                aliases=("bdf2_jit",),
                caps=StepperCaps(
                    dense_output=False,
                    jacobian="optional",  # external analytic Jacobian if provided
                ),
            )
        self.meta = meta

    # --- StepperSpec protocol hooks ---------------------------------------

    def workspace_type(self) -> type | None:
        return BDF2JITSpec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        def vec():
            return np.zeros((n_state,), dtype=dtype)

        return BDF2JITSpec.Workspace(
            y_nm1=vec(),
            step_index=np.zeros((1,), dtype=np.int64),
            y_guess=vec(),
            y_tmp=vec(),
            f_val=vec(),
            f_tmp=vec(),
            residual=vec(),
            J=np.zeros((n_state, n_state), dtype=dtype),
        )

    def emit(self, rhs_fn: Callable, model_spec=None, jacobian_fn=None, jvp_fn=None) -> Callable:
        """
        Generate the BDF2 stepper closure that uses the NamedTuple workspace.

        Signature (frozen ABI):

            stepper(t, dt, y_curr, rhs, params, runtime_ws,
                    stepper_ws, stepper_config,
                    y_prop, t_prop, dt_next, err_est) -> int32
        """
        def bdf2_stepper(
            t, dt,
            y_curr, rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            n = y_curr.size

            # Unpack workspace
            y_nm1 = ws.y_nm1
            step_idx_arr = ws.step_index

            y_guess = ws.y_guess
            y_tmp = ws.y_tmp
            f_val = ws.f_val
            f_tmp = ws.f_tmp
            residual = ws.residual
            J = ws.J

            step_idx = int(step_idx_arr[0])
            t_new = t + dt

            # Unpack config (or use hard-coded defaults if empty)
            if stepper_config.shape[0] >= 4:
                newton_tol = stepper_config[0]
                newton_max_iter = int(stepper_config[1])
                jac_eps = stepper_config[2]
                jac_mode = int(stepper_config[3])
                if jac_mode != 1:
                    jac_mode = 0
            else:
                newton_tol = 1e-8
                newton_max_iter = 50
                jac_eps = 1e-8
                jac_mode = 0
            prefer_external_jac = bool(jac_mode == 1 and jacobian_fn is not None)

            # ----------------- Predictor for Newton -----------------
            # Use explicit Euler for BDF1, explicit BDF2 for subsequent steps

            rhs(t, y_curr, f_tmp, params, runtime_ws)
            if step_idx == 0:
                for i in range(n):
                    y_guess[i] = y_curr[i] + dt * f_tmp[i]
            else:
                for i in range(n):
                    y_guess[i] = (4.0/3.0)*y_curr[i] - (1.0/3.0)*y_nm1[i] + (2.0/3.0)*dt * f_tmp[i]

            # ----------------- Newton loop (BDF1 or BDF2) -----------------
            converged = False

            for it in range(newton_max_iter):
                # f_val = f(t_new, y_guess)
                rhs(t_new, y_guess, f_val, params, runtime_ws)

                # residual and its max norm
                max_r = 0.0
                max_scale = 0.0
                if step_idx == 0:
                    # BDF1: F(y) = y - y0 - dt * f
                    for i in range(n):
                        r_i = y_guess[i] - y_curr[i] - dt * f_val[i]
                        residual[i] = r_i
                        abs_r = r_i if r_i >= 0.0 else -r_i
                        if abs_r > max_r:
                            max_r = abs_r
                        abs_y = y_guess[i]
                        if abs_y < 0.0:
                            abs_y = -abs_y
                        if abs_y > max_scale:
                            max_scale = abs_y
                        abs_curr = y_curr[i]
                        if abs_curr < 0.0:
                            abs_curr = -abs_curr
                        if abs_curr > max_scale:
                            max_scale = abs_curr
                else:
                    # BDF2: F(y) = y - (4/3)y_n + (1/3)y_{n-1} - (2/3)dt f
                    for i in range(n):
                        r_i = y_guess[i] - (4.0/3.0)*y_curr[i] + (1.0/3.0)*y_nm1[i] - (2.0/3.0)*dt * f_val[i]
                        residual[i] = r_i
                        abs_r = r_i if r_i >= 0.0 else -r_i
                        if abs_r > max_r:
                            max_r = abs_r
                        abs_y = y_guess[i]
                        if abs_y < 0.0:
                            abs_y = -abs_y
                        if abs_y > max_scale:
                            max_scale = abs_y
                        abs_curr = y_curr[i]
                        if abs_curr < 0.0:
                            abs_curr = -abs_curr
                        if abs_curr > max_scale:
                            max_scale = abs_curr
                        abs_prev = y_nm1[i]
                        if abs_prev < 0.0:
                            abs_prev = -abs_prev
                        if abs_prev > max_scale:
                            max_scale = abs_prev

                # If residual contains NaN/Inf, bail out early instead of
                # wasting work on Newton iterations with invalid data.
                if not allfinite1d(residual):
                    return STEPFAIL

                # Build Jacobian J = dF/dy
                factor = dt if step_idx == 0 else (2.0 / 3.0) * dt
                if prefer_external_jac and jacobian_fn is not None:
                    jacobian_fn(t_new, y_guess, params, J, runtime_ws)
                    for i in range(n):
                        for j in range(n):
                            J[i, j] = -factor * J[i, j]
                        J[i, i] = J[i, i] + 1.0
                else:
                    for j in range(n):
                        # y_tmp = y_guess
                        for i in range(n):
                            y_tmp[i] = y_guess[i]

                        base = y_guess[j]
                        abs_base = base if base >= 0.0 else -base
                        # Scale eps with max(1, |y_j|) to improve numeric Jacobian.
                        scale = abs_base if abs_base > 1.0 else 1.0
                        eps = jac_eps * scale
                        if eps == 0.0:
                            eps = jac_eps
                        y_tmp[j] = y_tmp[j] + eps

                        rhs(t_new, y_tmp, f_tmp, params, runtime_ws)
                        # If the perturbed RHS is non-finite, abort this step.
                        if not allfinite1d(f_tmp):
                            return STEPFAIL
                        inv_eps = 1.0 / eps

                        for i in range(n):
                            df_ij = (f_tmp[i] - f_val[i]) * inv_eps
                            if i == j:
                                J[i, j] = 1.0 - factor * df_ij
                            else:
                                J[i, j] = -factor * df_ij

                # Solve J * delta = -residual using in-place Gaussian elimination
                # We overwrite residual with delta.
                for i in range(n):
                    residual[i] = -residual[i]

                # Forward elimination with simple partial pivoting
                k = 0
                singular = False
                # Heuristic singularity threshold on pivot (absolute value)
                pivot_tol = 1e-14
                while k < n:
                    # Find pivot row
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

                    if pivot_val <= pivot_tol:
                        singular = True
                        break

                    # Swap rows if needed
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
                    return STEPFAIL

                # Back substitution
                i2 = n - 1
                while i2 >= 0:
                    sum_val = residual[i2]
                    j2 = i2 + 1
                    while j2 < n:
                        sum_val = sum_val - J[i2, j2] * residual[j2]
                        j2 += 1
                    diag = J[i2, i2]
                    if diag == 0.0:
                        return STEPFAIL
                    residual[i2] = sum_val / diag
                    i2 -= 1

                # Update y_guess += delta (delta is in residual) and track scaled correction
                max_delta_scaled = 0.0
                for i in range(n):
                    delta_i = residual[i]
                    y_old = y_guess[i]
                    y_guess[i] = y_old + delta_i

                    abs_delta = delta_i if delta_i >= 0.0 else -delta_i
                    abs_scale = y_old if y_old >= 0.0 else -y_old
                    if abs_scale < 1.0:
                        abs_scale = 1.0
                    scaled = abs_delta / abs_scale
                    if scaled > max_delta_scaled:
                        max_delta_scaled = scaled

                scale_tol = max_scale if max_scale > 1.0 else 1.0
                if max_r <= newton_tol * scale_tol and max_delta_scaled <= newton_tol:
                    converged = True
                    break

            if not converged:
                return STEPFAIL

            # Rotate history FIRST: y_nm1 <- current committed y_n (y_curr).
            # This must happen before we write to y_prop, because y_curr and
            # y_prop may alias (share the same underlying buffer) in the
            # runtime. If we wrote y_prop first, y_curr would also contain
            # y_{n+1}, and we'd lose y_n needed for the next BDF2 step.
            for i in range(n):
                y_nm1[i] = y_curr[i]

            # Accept step: y_prop <- y_guess (may alias y_curr)
            for i in range(n):
                y_prop[i] = y_guess[i]

            t_prop[0] = t_new
            dt_next[0] = dt
            err_est[0] = 0.0

            step_idx_arr[0] = step_idx + 1
            return OK

        return bdf2_stepper


def _auto_register():
    from ..registry import register
    spec = BDF2JITSpec()
    register(spec)


_auto_register()
