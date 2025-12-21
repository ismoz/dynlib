"""Reference Lyapunov runtime analysis."""
from __future__ import annotations

import math
from typing import Callable, Optional
import numpy as np

from dynlib.runtime.fastpath.plans import FixedTracePlan
from .core import AnalysisHooks, AnalysisModule, AnalysisRequirements, TraceSpec

__all__ = ["lyapunov_mle"]


def _default_tangent(n_state: int) -> np.ndarray:
    vec = np.zeros((n_state,), dtype=float)
    if n_state > 0:
        vec[0] = 1.0
    return vec


def lyapunov_mle(
    *,
    jacobian_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    n_state: int,
    trace_plan: Optional[FixedTracePlan] = None,
    analysis_kind: int = 1,
) -> AnalysisModule:
    """
    Reference Lyapunov maximum exponent analysis.

    Assumes the model exposes a Jacobian callable. The running log growth is
    accumulated in ``analysis_out[0]`` with the step count in ``analysis_out[1]``.
    The trace (when enabled) stores the instantaneous exponent estimate.
    """

    init_vec = _default_tangent(n_state)
    plan = trace_plan if trace_plan is not None else FixedTracePlan(stride=1)
    trace_spec = TraceSpec(width=1, plan=plan, allow_growth=False)

    def _pre_step(
        t: float,
        dt: float,
        step: int,
        y_curr: np.ndarray,
        y_prev: np.ndarray,
        params: np.ndarray,
        runtime_ws,
        analysis_ws: np.ndarray,
        analysis_out: np.ndarray,
        trace_buf: np.ndarray,
        trace_count: np.ndarray,
        trace_cap: int,
        trace_stride: int,
    ) -> None:
        # One-time tangent initialization
        if analysis_ws.shape[0] >= n_state and step == 0 and analysis_out.shape[0] > 1:
            if not np.any(analysis_ws[:n_state]):
                analysis_ws[:n_state] = init_vec
            analysis_out[0] = 0.0
            analysis_out[1] = 0.0

    def _post_step(
        t: float,
        dt: float,
        step: int,
        y_curr: np.ndarray,
        y_prev: np.ndarray,
        params: np.ndarray,
        runtime_ws,
        analysis_ws: np.ndarray,
        analysis_out: np.ndarray,
        trace_buf: np.ndarray,
        trace_count: np.ndarray,
        trace_cap: int,
        trace_stride: int,
    ) -> None:
        if analysis_ws.shape[0] < n_state or analysis_out.shape[0] < 2:
            return
        jac = jacobian_fn(t, y_curr, params)
        vec = analysis_ws[:n_state]
        new_vec = jac @ vec
        norm = float(np.linalg.norm(new_vec))
        if norm == 0.0:
            return
        analysis_ws[:n_state] = new_vec / norm
        analysis_out[0] += math.log(norm)
        analysis_out[1] += 1.0

        if trace_buf.shape[0] > 0 and trace_stride > 0 and (step % trace_stride == 0):
            idx = int(trace_count[0])
            if idx < trace_cap:
                trace_buf[idx, 0] = analysis_out[0] / max(analysis_out[1], 1.0)
                trace_count[0] = idx + 1
            else:
                trace_count[0] = trace_cap

    hooks = AnalysisHooks(pre_step=_pre_step, post_step=_post_step)
    reqs = AnalysisRequirements(fixed_step=True, need_jacobian=True, mutates_state=False)
    return AnalysisModule(
        name="lyapunov_mle",
        requirements=reqs,
        workspace_size=n_state,
        output_size=2,
        output_names=("log_growth", "steps"),
        trace_names=("mle",),
        trace=trace_spec,
        python_hooks=hooks,
        jit_hooks=hooks,
        analysis_kind=analysis_kind,
    )
