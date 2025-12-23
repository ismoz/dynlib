"""
SDIRK2 (Alexander, L-stable, order 2, fixed-step, JIT-compatible).

Singly Diagonally Implicit Runge–Kutta method:

    gamma = (2 - sqrt(2)) / 2

Butcher tableau:

    [ gamma      | gamma      0      ]
    [ 1          | 1 - gamma  gamma  ]
    [--------------------------------]
    [            | 1 - gamma  gamma  ]

Stage equations in state form (Y_i as unknown):

    Y_1 = y_n + dt * gamma * f(t_n + c_1 dt, Y_1)
    Y_2 = y_n + dt * ( (1-gamma) k_1 + gamma * f(t_n + c_2 dt, Y_2) )

with
    c_1 = gamma
    c_2 = 1
    k_i = f(t_n + c_i dt, Y_i)

Final update:

    y_{n+1} = y_n + dt * [ (1-gamma) k_1 + gamma k_2 ]

For Alexander 2 (stiffly accurate), Y_2 = y_{n+1}, so we use y_{n+1} = Y_2.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
import math
import numpy as np

from ..base import StepperMeta, StepperCaps
from dynlib.runtime.runner_api import OK, STEPFAIL
from dynlib.compiler.guards import allfinite1d, register_guards_consumer
from ..config_base import ConfigMixin

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["SDIRK2JITSpec"]

# Register guards so they can be inlined into jitted kernels
register_guards_consumer(globals(), mapping={"allfinite1d": "allfinite1d"})

# Alexander parameter
GAMMA = 0.5 * (2.0 - math.sqrt(2.0))


class SDIRK2JITSpec(ConfigMixin):
    """
    Numba-compatible SDIRK2 (Alexander) method (order 2, fixed-step).

    - 2-stage singly diagonally implicit RK
    - L-stable, stiffly accurate
    - Newton solve on stage states Y_i with numeric Jacobian
    - Internal Jacobian only (no external Jacobian API)

    DO NOT USE on problems with discontinuous RHS (e.g. threshold/reset neuron
    models) that violate continuous-Jacobian assumptions.
    """

    @dataclass
    class Config:
        # Absolute residual tolerance (on max-norm of F)
        newton_tol: float = 1e-8
        # Iteration budget
        newton_max_iter: int = 50
        # Base finite-difference scale for df/dy
        jac_eps: float = 1e-8
        jacobian_mode: int = 0  # 0=internal FD, 1=external analytic
        __enums__ = {"jacobian_mode": {"internal": 0, "external": 1}}

    class Workspace(NamedTuple):
        # Stage derivatives
        k1: np.ndarray
        k2: np.ndarray

        # Newton / Jacobian scratch
        y_stage: np.ndarray
        y_tmp: np.ndarray
        f_val: np.ndarray
        f_tmp: np.ndarray
        residual: np.ndarray
        # Modified Newton: store FD Jacobian once (J_base),
        # and a working copy J for the linear solve.
        J_base: np.ndarray  # (n_state, n_state)
        J: np.ndarray       # (n_state, n_state)

    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="sdirk2",
                kind="ode",
                time_control="fixed",
                scheme="implicit",
                geometry=frozenset(),
                family="dirk",
                order=2,
                embedded_order=None,
                stiff=True,
                aliases=("sdirk2_jit", "alexander2"),
                caps=StepperCaps(
                    dense_output=False,
                    jacobian="optional",
                ),
            )
        self.meta = meta

    # --- StepperSpec protocol hooks ---------------------------------------

    def workspace_type(self) -> type | None:
        return SDIRK2JITSpec.Workspace

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        def vec():
            return np.zeros((n_state,), dtype=dtype)

        mat = lambda: np.zeros((n_state, n_state), dtype=dtype)

        return SDIRK2JITSpec.Workspace(
            k1=vec(),
            k2=vec(),
            y_stage=vec(),
            y_tmp=vec(),
            f_val=vec(),
            f_tmp=vec(),
            residual=vec(),
            J_base=mat(),
            J=mat(),
        )

    def emit(self, rhs_fn: Callable, model_spec=None, jacobian_fn=None, jvp_fn=None) -> Callable:
        """
        Generate the SDIRK2 stepper closure using the NamedTuple workspace.

        Signature (frozen ABI):

            stepper(t, dt, y_curr, rhs, params, runtime_ws,
                    stepper_ws, stepper_config,
                    y_prop, t_prop, dt_next, err_est) -> int32
        """

        gamma = GAMMA
        one_minus_gamma = 1.0 - gamma

        def sdirk2_stepper(
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
            k1 = ws.k1
            k2 = ws.k2
            y_stage = ws.y_stage
            y_tmp = ws.y_tmp
            f_val = ws.f_val
            f_tmp = ws.f_tmp
            residual = ws.residual
            J_base = ws.J_base
            J = ws.J

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

            dt_gamma = dt * gamma

            # -------------------- Stage 1: Y1 ---------------------
            # Equation: F1(Y) = Y - y_n - dt*gamma*f(t + c1*dt, Y) = 0
            t1 = t + gamma * dt

            # Initial guess: Y1 ~ y_n
            for i in range(n):
                y_stage[i] = y_curr[i]

            converged = False

            # --- Build Jacobian once for Stage 1 (Modified Newton) ---
            rhs(t1, y_stage, f_val, params, runtime_ws)
            if prefer_external_jac and jacobian_fn is not None:
                jacobian_fn(t1, y_stage, params, J_base, runtime_ws)
                for i in range(n):
                    for j in range(n):
                        J_base[i, j] = -dt_gamma * J_base[i, j]
                    J_base[i, i] = J_base[i, i] + 1.0
            else:
                for j in range(n):
                    # y_tmp = y_stage (copy)
                    for i in range(n):
                        y_tmp[i] = y_stage[i]

                    base = y_stage[j]
                    abs_base = base if base >= 0.0 else -base
                    scale = abs_base if abs_base > 1.0 else 1.0
                    eps = jac_eps * scale
                    if eps == 0.0:
                        eps = jac_eps
                    y_tmp[j] = y_tmp[j] + eps

                    rhs(t1, y_tmp, f_tmp, params, runtime_ws)
                    if not allfinite1d(f_tmp):
                        return STEPFAIL

                    inv_eps = 1.0 / eps

                    for i in range(n):
                        df_ij = (f_tmp[i] - f_val[i]) * inv_eps
                        if i == j:
                            J_base[i, j] = 1.0 - dt_gamma * df_ij
                        else:
                            J_base[i, j] = -dt_gamma * df_ij

            # --- Newton iterations using frozen J_base ---
            for _it in range(newton_max_iter):
                # Working copy of Jacobian for this solve
                for i in range(n):
                    for j in range(n):
                        J[i, j] = J_base[i, j]

                # f_val = f(t1, Y1)
                rhs(t1, y_stage, f_val, params, runtime_ws)

                # Residual and scaling
                max_r = 0.0
                max_scale = 0.0

                for i in range(n):
                    r_i = y_stage[i] - y_curr[i] - dt_gamma * f_val[i]
                    residual[i] = r_i

                    abs_r = r_i if r_i >= 0.0 else -r_i
                    if abs_r > max_r:
                        max_r = abs_r

                    # scale: use |Y| and |y_n|
                    abs_y = y_stage[i]
                    if abs_y < 0.0:
                        abs_y = -abs_y
                    if abs_y > max_scale:
                        max_scale = abs_y

                    abs_curr = y_curr[i]
                    if abs_curr < 0.0:
                        abs_curr = -abs_curr
                    if abs_curr > max_scale:
                        max_scale = abs_curr

                # NaN / Inf guard on residual
                if not allfinite1d(residual):
                    return STEPFAIL

                # Solve J * delta = -residual (in-place Gaussian elimination)
                for i in range(n):
                    residual[i] = -residual[i]

                pivot_tol = 1e-14
                k_idx = 0
                singular = False

                # Forward elimination with simple partial pivoting
                while k_idx < n:
                    # Find pivot row
                    pivot_row = k_idx
                    val = J[k_idx, k_idx]
                    pivot_val = val if val >= 0.0 else -val
                    i2 = k_idx + 1
                    while i2 < n:
                        v = J[i2, k_idx]
                        av = v if v >= 0.0 else -v
                        if av > pivot_val:
                            pivot_val = av
                            pivot_row = i2
                        i2 += 1

                    if pivot_val <= pivot_tol:
                        singular = True
                        break

                    # Swap rows if needed
                    if pivot_row != k_idx:
                        j2 = 0
                        while j2 < n:
                            tmp = J[k_idx, j2]
                            J[k_idx, j2] = J[pivot_row, j2]
                            J[pivot_row, j2] = tmp
                            j2 += 1
                        tmpb = residual[k_idx]
                        residual[k_idx] = residual[pivot_row]
                        residual[pivot_row] = tmpb

                    diag = J[k_idx, k_idx]
                    inv_diag = 1.0 / diag
                    i2 = k_idx + 1
                    while i2 < n:
                        factor = J[i2, k_idx] * inv_diag
                        J[i2, k_idx] = 0.0
                        j2 = k_idx + 1
                        while j2 < n:
                            J[i2, j2] = J[i2, j2] - factor * J[k_idx, j2]
                            j2 += 1
                        residual[i2] = residual[i2] - factor * residual[k_idx]
                        i2 += 1

                    k_idx += 1

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

                # Update Y1: y_stage += delta; track scaled correction
                max_delta_scaled = 0.0
                for i in range(n):
                    delta_i = residual[i]
                    y_old = y_stage[i]
                    y_stage[i] = y_old + delta_i

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

            # Final stage-1 derivative: k1 = f(t1, Y1) at converged Y1
            rhs(t1, y_stage, f_val, params, runtime_ws)
            for i in range(n):
                k1[i] = f_val[i]

            # -------------------- Stage 2: Y2 ---------------------
            # Equation:
            #   F2(Y) = Y - y_n - dt * ( (1-gamma) k1 + gamma f(t + dt, Y) ) = 0
            t2 = t + dt

            # Initial guess: Y2 ~ y_n + dt * (1-gamma) k1 (ignore implicit part)
            for i in range(n):
                y_stage[i] = y_curr[i] + dt * one_minus_gamma * k1[i]

            converged = False

            # --- Build Jacobian once for Stage 2 (Modified Newton) ---
            rhs(t2, y_stage, f_val, params, runtime_ws)
            if prefer_external_jac and jacobian_fn is not None:
                jacobian_fn(t2, y_stage, params, J_base, runtime_ws)
                for i in range(n):
                    for j in range(n):
                        J_base[i, j] = -dt_gamma * J_base[i, j]
                    J_base[i, i] = J_base[i, i] + 1.0
            else:
                for j in range(n):
                    # y_tmp = y_stage
                    for i in range(n):
                        y_tmp[i] = y_stage[i]

                    base = y_stage[j]
                    abs_base = base if base >= 0.0 else -base
                    scale = abs_base if abs_base > 1.0 else 1.0
                    eps = jac_eps * scale
                    if eps == 0.0:
                        eps = jac_eps
                    y_tmp[j] = y_tmp[j] + eps

                    rhs(t2, y_tmp, f_tmp, params, runtime_ws)
                    if not allfinite1d(f_tmp):
                        return STEPFAIL

                    inv_eps = 1.0 / eps

                    for i in range(n):
                        df_ij = (f_tmp[i] - f_val[i]) * inv_eps
                        if i == j:
                            J_base[i, j] = 1.0 - dt_gamma * df_ij
                        else:
                            J_base[i, j] = -dt_gamma * df_ij

            # --- Newton iterations using frozen J_base ---
            for _it in range(newton_max_iter):
                # Working copy of Jacobian for this solve
                for i in range(n):
                    for j in range(n):
                        J[i, j] = J_base[i, j]

                # f_val = f(t2, Y2)
                rhs(t2, y_stage, f_val, params, runtime_ws)

                # Residual and scaling
                max_r = 0.0
                max_scale = 0.0

                for i in range(n):
                    base_i = y_curr[i] + dt * one_minus_gamma * k1[i]
                    r_i = y_stage[i] - base_i - dt_gamma * f_val[i]
                    residual[i] = r_i

                    abs_r = r_i if r_i >= 0.0 else -r_i
                    if abs_r > max_r:
                        max_r = abs_r

                    abs_y = y_stage[i]
                    if abs_y < 0.0:
                        abs_y = -abs_y
                    if abs_y > max_scale:
                        max_scale = abs_y

                    abs_curr = y_curr[i]
                    if abs_curr < 0.0:
                        abs_curr = -abs_curr
                    if abs_curr > max_scale:
                        max_scale = abs_curr

                    abs_base = base_i
                    if abs_base < 0.0:
                        abs_base = -abs_base
                    if abs_base > max_scale:
                        max_scale = abs_base

                if not allfinite1d(residual):
                    return STEPFAIL

                # Solve J * delta = -residual
                for i in range(n):
                    residual[i] = -residual[i]

                pivot_tol = 1e-14
                k_idx = 0
                singular = False

                while k_idx < n:
                    # Find pivot row
                    pivot_row = k_idx
                    val = J[k_idx, k_idx]
                    pivot_val = val if val >= 0.0 else -val
                    i2 = k_idx + 1
                    while i2 < n:
                        v = J[i2, k_idx]
                        av = v if v >= 0.0 else -v
                        if av > pivot_val:
                            pivot_val = av
                            pivot_row = i2
                        i2 += 1

                    if pivot_val <= pivot_tol:
                        singular = True
                        break

                    # Swap rows if needed
                    if pivot_row != k_idx:
                        j2 = 0
                        while j2 < n:
                            tmp = J[k_idx, j2]
                            J[k_idx, j2] = J[pivot_row, j2]
                            J[pivot_row, j2] = tmp
                            j2 += 1
                        tmpb = residual[k_idx]
                        residual[k_idx] = residual[pivot_row]
                        residual[pivot_row] = tmpb

                    diag = J[k_idx, k_idx]
                    inv_diag = 1.0 / diag
                    i2 = k_idx + 1
                    while i2 < n:
                        factor = J[i2, k_idx] * inv_diag
                        J[i2, k_idx] = 0.0
                        j2 = k_idx + 1
                        while j2 < n:
                            J[i2, j2] = J[i2, j2] - factor * J[k_idx, j2]
                            j2 += 1
                        residual[i2] = residual[i2] - factor * residual[k_idx]
                        i2 += 1

                    k_idx += 1

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

                # Update Y2 and track correction
                max_delta_scaled = 0.0
                for i in range(n):
                    delta_i = residual[i]
                    y_old = y_stage[i]
                    y_stage[i] = y_old + delta_i

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

            # Final stage-2 derivative: k2 = f(t2, Y2) at converged Y2
            rhs(t2, y_stage, f_val, params, runtime_ws)
            for i in range(n):
                k2[i] = f_val[i]

            # -------------------- Final update (stiffly accurate) ---------------------
            # Alexander 2 is stiffly accurate: Y2 = y_{n+1}
            for i in range(n):
                y_prop[i] = y_stage[i]

            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0

            return OK

        return sdirk2_stepper


def _auto_register():
    from ..registry import register
    spec = SDIRK2JITSpec()
    register(spec)


_auto_register()
