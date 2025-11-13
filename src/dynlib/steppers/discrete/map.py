# src/dynlib/steppers/map.py
"""
Map (discrete-time, fixed-step) stepper implementation.

This stepper assumes the compiled callable computes the NEXT STATE directly:
    map_fn(t, y_curr, out, params)  -> writes y_{n+1} into 'out'

It never multiplies by dt; dt is only the label spacing for t.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from ..base import StepperMeta, StructSpec
from dynlib.runtime.runner_api import OK

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["MapSpec"]


class MapSpec:
    """
    Discrete-time map stepper: y_{n+1} = F(t_n, y_n; params)

    - Fixed-step (time_control="fixed")
    - Single proposal per call (no accept/reject loop)
    - dt is a label spacing only; not used in dynamics
    """
    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="map",
                kind="map",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="iter",
                order=1,
                embedded_order=None,
                dense_output=False,
                stiff_ok=False,
                aliases=("iter", "discrete"),
            )
        self.meta = meta

    def struct_spec(self) -> StructSpec:
        """
        Workspace: one lane of scratch to hold the proposed next state.
        """
        return StructSpec(
            sp_size=0,    # unused
            ss_size=0,    # unused
            sw0_size=1,   # 1 lane for y_next scratch (n_state floats)
            sw1_size=0,   # unused
            sw2_size=0,   # unused
            sw3_size=0,   # unused
            iw0_size=0,   # unused (runner can track step index if desired)
            bw0_size=0,   # unused
            use_history=False,
            use_f_history=False,
            dense_output=False,
            needs_jacobian=False,
            embedded_order=None,
            stiff_ok=False,
        )

    def config_spec(self) -> type | None:
        """Maps have no runtime-configurable parameters."""
        return None

    def default_config(self, model_spec=None):
        """No config."""
        return None

    def pack_config(self, config) -> np.ndarray:
        """No config -> empty array."""
        return np.array([], dtype=np.float64)

    def emit(self, map_fn: Callable, struct: StructSpec, model_spec=None) -> Callable:
        """
        Generate a jittable map stepper.

        Signature (per ABI):
            status = stepper(
                t: float, dt: float,
                y_curr: float[:], rhs,          # rhs is actually the compiled map_fn
                params: float[:] | int[:],
                sp: float[:], ss: float[:],
                sw0: float[:], sw1: float[:], sw2: float[:], sw3: float[:],
                iw0: int32[:], bw0: uint8[:],
                stepper_config: float64[:],
                y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:]
            ) -> int32
        """
        def map_stepper(
            t, dt,
            y_curr, rhs,   # rhs == map_fn
            params,
            sp, ss,
            sw0, sw1, sw2, sw3,
            iw0, bw0,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            n = y_curr.size
            y_next = sw0[:n]

            # Compute next state directly into scratch
            rhs(t, y_curr, y_next, params, ss, iw0)

            # Copy proposal to output buffer; runner guard enforces finiteness
            for i in range(n):
                y_prop[i] = y_next[i]

            # Fixed label advance
            t_prop[0]  = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0
            return OK

        return map_stepper


# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = MapSpec()
    register(spec)

_auto_register()
