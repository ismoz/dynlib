# src/dynlib/steppers/map.py
"""
Map (discrete-time, fixed-step) stepper implementation.

This stepper assumes the compiled callable computes the NEXT STATE directly:
    map_fn(t, y_curr, out, params)  -> writes y_{n+1} into 'out'

It never multiplies by dt; dt is only the label spacing for t.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..base import StepperMeta
from dynlib.runtime.runner_api import OK
from ..config_base import ConfigMixin

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["MapSpec"]


class MapWorkspace(NamedTuple):
    y_next: np.ndarray


class MapSpec(ConfigMixin):
    """
    Discrete-time map stepper: y_{n+1} = F(t_n, y_n; params)

    - Fixed-step (time_control="fixed")
    - Single proposal per call (no accept/reject loop)
    - dt is a label spacing only; not used in dynamics
    """
    CONFIG = None  # No runtime configuration

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
                stiff=False,
                aliases=("iter", "discrete"),
        )
        self.meta = meta

    def workspace_type(self) -> type | None:
        return MapWorkspace

    def make_workspace(self, n_state: int, dtype: np.dtype, model_spec=None) -> MapWorkspace:
        return MapWorkspace(
            y_next=np.zeros((n_state,), dtype=dtype),
        )

    def emit(self, map_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable map stepper using the workspace scratch buffer.
        """

        def map_stepper(
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
            # rhs == compiled map function
            y_next = ws.y_next
            rhs(t, y_curr, y_next, params, runtime_ws)

            # Copy proposal to output buffer; runner guards enforce finiteness
            y_prop[:] = y_next[:]

            # Fixed label advance
            t_prop[0] = t + dt
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
