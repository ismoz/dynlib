# src/dynlib/analysis/runtime/lyapunov.py
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


def _make_hooks(
    *,
    jvp_fn: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
    init_vec: np.ndarray,
    n_state: int,
) -> AnalysisHooks:
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
        if analysis_ws.shape[0] >= 2 * n_state and step == 0 and analysis_out.shape[0] > 1:
            if not np.any(analysis_ws[:n_state]):
                analysis_ws[:n_state] = init_vec
            analysis_ws[n_state : 2 * n_state] = 0.0
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
        if analysis_ws.shape[0] < 2 * n_state or analysis_out.shape[0] < 2:
            return
        vec = analysis_ws[:n_state]
        out_vec = analysis_ws[n_state : 2 * n_state]
        jvp_fn(t, y_curr, params, vec, out_vec, runtime_ws)
        norm_sq = 0.0
        for i in range(n_state):
            val = out_vec[i]
            norm_sq += float(val * val)
        norm = math.sqrt(norm_sq)
        if norm == 0.0:
            return
        analysis_ws[:n_state] = out_vec / norm
        analysis_out[0] += math.log(norm)
        analysis_out[1] += 1.0

        if trace_buf.shape[0] > 0 and trace_stride > 0 and (step % trace_stride == 0):
            idx = int(trace_count[0])
            if idx < trace_cap:
                trace_buf[idx, 0] = analysis_out[0] / max(analysis_out[1], 1.0)
                trace_count[0] = idx + 1
            else:
                trace_count[0] = trace_cap + 1

    return AnalysisHooks(pre_step=_pre_step, post_step=_post_step)


class _LyapunovModule(AnalysisModule):
    def __init__(
        self,
        *,
        jvp: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
        n_state: int,
        trace_plan: FixedTracePlan,
        analysis_kind: int,
    ) -> None:
        self._jvp_py = jvp
        self._jvp_jit = None
        init_vec = _default_tangent(n_state)
        hooks = _make_hooks(jvp_fn=jvp, init_vec=init_vec, n_state=n_state)
        reqs = AnalysisRequirements(fixed_step=True, need_jvp=True, mutates_state=False)
        super().__init__(
            name="lyapunov_mle",
            requirements=reqs,
            workspace_size=2 * n_state,
            output_size=2,
            output_names=("log_growth", "steps"),
            trace_names=("mle",),
            trace=TraceSpec(width=1, plan=trace_plan),
            hooks=hooks,
            analysis_kind=analysis_kind,
        )
        self._init_vec = init_vec
        self._n_state = int(n_state)

    def _ensure_jit_jvp(self):
        if self._jvp_jit is not None:
            return self._jvp_jit
        # Allow pre-jitted JVP callables (CPUDispatcher) directly.
        dispatcher = getattr(self._jvp_py, "signatures", None)
        if dispatcher is not None:
            return self._jvp_py
        py_target = getattr(self._jvp_py, "py_func", self._jvp_py)
        try:  # pragma: no cover - import guard
            from numba import njit  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("lyapunov_mle requires numba for jit hooks") from exc
        try:
            self._jvp_jit = njit(cache=False)(py_target)
        except Exception as exc:
            raise RuntimeError(
                "lyapunov_mle requires a numba-compatible Jacobian-vector product for JIT execution"
            ) from exc
        return self._jvp_jit

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> AnalysisHooks:
        if not jit:
            return self.hooks
        jvp_jit = self._ensure_jit_jvp()
        jit_hooks = _make_hooks(jvp_fn=jvp_jit, init_vec=self._init_vec, n_state=self._n_state)
        return self._compile_hooks(jit_hooks, dtype)


def lyapunov_mle(
    *,
    jvp: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
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
    plan = trace_plan if trace_plan is not None else FixedTracePlan(stride=1)
    return _LyapunovModule(jvp=jvp, n_state=n_state, trace_plan=plan, analysis_kind=analysis_kind)
