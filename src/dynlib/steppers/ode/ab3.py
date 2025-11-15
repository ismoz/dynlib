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

History layout in `ss` (after lag prefix):

    lane 0: f_{n-2}
    lane 1: f_{n-1}
    lane 2: f_n

Persists f-history in `ss` and a step counter in `iw0`.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np

from ..base import StepperMeta, StructSpec
from dynlib.runtime.runner_api import OK, NAN_DETECTED, STEPFAIL

# Import guards for NaN/Inf detection
from dynlib.compiler.guards import allfinite1d, allfinite_scalar

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["AB3Spec"]


class AB3Spec:
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
                dense_output=False,
                stiff_ok=False,
                aliases=("adams_bashforth_3",),
            )
        self.meta = meta

    def struct_spec(self) -> StructSpec:
        """
        AB3 workspace:

          - ss: 3 lanes for persistent f-history (after lag prefix):
                lane 0: f_{n-2}
                lane 1: f_{n-1}
                lane 2: f_n
          - sp: 1 lane for y_stage (Heun predictor)
          - sw0: 1 lane for temporary f_{n+1} during rotation
          - iw0: 1 slot for step_count (offset by iw0_lag_reserved slots)

        Lane-based sizes: sp/ss/sw* sizes are lane counts (multiples of n_state).
        """
        return StructSpec(
            sp_size=1,    # y_stage (Heun predictor)
            ss_size=3,    # f_{n-2}, f_{n-1}, f_n history (3 lanes)
            sw0_size=1,   # tmp lane for f_{n+1} when rotating history
            sw1_size=0,
            sw2_size=0,
            sw3_size=0,
            iw0_size=1,   # step_count
            bw0_size=0,
            use_history=False,    # stepper manages its own history
            use_f_history=True,   # this stepper uses derivative history
            dense_output=False,
            needs_jacobian=False,
            embedded_order=None,
            stiff_ok=False,
        )

    def config_spec(self) -> type | None:
        """AB3 has no runtime-configurable parameters."""
        return None

    def default_config(self, model_spec=None):
        """AB3 has no config."""
        return None

    def pack_config(self, config) -> np.ndarray:
        """AB3 has no config - return empty array."""
        return np.array([], dtype=np.float64)

    def emit(self, rhs_fn: Callable, struct: StructSpec, model_spec=None) -> Callable:
        """
        Generate a jittable AB3 stepper function.

        Signature (per ABI):
            status = stepper(
                t: float, dt: float,
                y_curr: float[:], rhs,
                params: float[:] | int[:],
                sp: float[:], ss: float[:],
                sw0: float[:], sw1: float[:], sw2: float[:], sw3: float[:],
                iw0: int32[:], bw0: uint8[:],
                stepper_config: float64[:],
                y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:]
            ) -> int32

        Returns:
            A callable Python function implementing AB3 with startup.
        """
        # Lag-reserved metadata from StructSpec (in lanes / slots)
        ss_lag_reserved_lanes = struct.ss_lag_reserved
        iw0_lag_reserved = struct.iw0_lag_reserved

        def ab3_stepper(
            t, dt,
            y_curr, rhs,
            params,
            sp, ss,
            sw0, sw1, sw2, sw3,
            iw0, bw0,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # Basic sanity check on dt (STEPFAIL if obviously invalid).
            if not allfinite_scalar(dt) or dt <= 0.0:
                err_est[0] = float("inf")
                return STEPFAIL

            # Number of states
            n = y_curr.size

            # Offsets for lag-reserved prefix in ss (lanes -> elements)
            ss_offset = ss_lag_reserved_lanes * n

            # f-history lanes in ss:
            #   f_nm2 = f_{n-2}
            #   f_nm1 = f_{n-1}
            #   f_n   = f_n     (derivative at current y_curr)
            f_nm2 = ss[ss_offset : ss_offset + n]
            f_nm1 = ss[ss_offset + n : ss_offset + 2 * n]
            f_n   = ss[ss_offset + 2 * n : ss_offset + 3 * n]

            # Temporary lane for f_{n+1} during history rotation
            f_next = sw0[:n]

            # Step counter (accepted steps so far)
            step_idx = iw0[iw0_lag_reserved + 0]

            # Scratch for predictor (Heun startup)
            y_stage = sp[:n]

            # --- Startup step 0 (Heun, order 2) ---
            if step_idx == 0:
                # Compute f0 = f(t0, y0) into f_nm1
                rhs(t, y_curr, f_nm1, params, ss, iw0)
                if not allfinite1d(f_nm1):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # Predictor: y_stage = y0 + dt * f0
                for i in range(n):
                    y_stage[i] = y_curr[i] + dt * f_nm1[i]

                # Predictor derivative (at y_stage) into f_n (temporary)
                rhs(t + dt, y_stage, f_n, params, ss, iw0)
                if not allfinite1d(f_n):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # Heun update: y1 = y0 + dt/2 * (f0 + f_pred)
                for i in range(n):
                    y_prop[i] = y_curr[i] + 0.5 * dt * (f_nm1[i] + f_n[i])

                if not allfinite1d(y_prop):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # Refresh f_n with derivative at accepted state y1
                rhs(t + dt, y_prop, f_n, params, ss, iw0)
                if not allfinite1d(f_n):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # At this point:
                #   f_nm1 = f0
                #   f_n   = f1
                #   f_nm2 is unused (will be filled after step 1)
                t_prop[0] = t + dt
                dt_next[0] = dt
                err_est[0] = 0.0

                iw0[iw0_lag_reserved + 0] = step_idx + 1
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

                if not allfinite1d(y_prop):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

                # Compute f2 = f(t2, y2) into f_next
                rhs(t + dt, y_prop, f_next, params, ss, iw0)
                if not allfinite1d(f_next):
                    err_est[0] = float("inf")
                    return NAN_DETECTED

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

                iw0[iw0_lag_reserved + 0] = step_idx + 1
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

            if not allfinite1d(y_prop):
                err_est[0] = float("inf")
                return NAN_DETECTED

            # Compute f_{n+1} at proposed state; store temporarily in f_next
            rhs(t + dt, y_prop, f_next, params, ss, iw0)
            if not allfinite1d(f_next):
                err_est[0] = float("inf")
                return NAN_DETECTED

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

            iw0[iw0_lag_reserved + 0] = step_idx + 1
            return OK

        return ab3_stepper


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = AB3Spec()
    register(spec)

_auto_register()
