# src/dynlib/analysis/runtime/lyapunov.py
"""Reference Lyapunov runtime analysis."""
from __future__ import annotations

import math
from typing import Callable, Optional, TYPE_CHECKING
import numpy as np

from dynlib.runtime.fastpath.plans import FixedTracePlan
from .core import AnalysisHooks, AnalysisModule, AnalysisRequirements, TraceSpec

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from dynlib.runtime.model import Model

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


def _coerce_model(model_like) -> "Model" | None:
    """Extract a Model instance from a Model or Sim-like object."""
    if model_like is None:
        return None
    # Sim exposes the compiled model via ``.model``; accept both forms for convenience.
    model = getattr(model_like, "model", model_like)
    if getattr(model, "spec", None) is None:
        return None
    return model  # type: ignore[return-value]


def _resolve_trace_plan(
    *, trace_plan: Optional[FixedTracePlan], record_interval: Optional[int]
) -> FixedTracePlan:
    if trace_plan is not None and record_interval is not None:
        if int(record_interval) != int(trace_plan.record_interval()):
            raise ValueError(
                "record_interval must match the provided trace_plan stride for lyapunov_mle"
            )
    if record_interval is not None:
        stride = int(record_interval)
        if stride <= 0:
            raise ValueError("record_interval for lyapunov_mle must be positive")
        return FixedTracePlan(stride=stride)
    if trace_plan is not None:
        return trace_plan
    return FixedTracePlan(stride=1)


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
    model=None,
    jvp: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None]
    ] = None,
    n_state: Optional[int] = None,
    trace_plan: Optional[FixedTracePlan] = None,
    record_interval: Optional[int] = None,
    analysis_kind: int = 1,
):
    """
    Factory for Lyapunov maximum exponent analysis.
    
    Returns a factory function for dependency injection, or an AnalysisModule if model is provided.
    The factory extracts ``jvp`` and ``n_state`` from the model if not explicitly provided.
    
    Parameters
    ----------
    model : Model or Sim, optional
        Model object. If provided, returns AnalysisModule directly. If None, returns factory.
    jvp : callable, optional
        Jacobian-vector product function. If None, extracted from model.
    n_state : int, optional
        Number of state variables. If None, inferred from model.spec.states.
    trace_plan : FixedTracePlan, optional
        Trace sampling plan. If None, created from record_interval.
    record_interval : int, optional
        Recording stride for trace sampling. Defaults to 1 when not provided.
    analysis_kind : int, default=1
        Analysis algorithm variant selector.
        
    Returns
    -------
    factory or module
        If model is None: factory function with signature ``factory(model) -> _LyapunovModule``
        If model is provided: ``_LyapunovModule`` instance
        
    Examples
    --------
    >>> # Factory mode - Sim injects model
    >>> sim.run(analysis=lyapunov_mle())
    >>> sim.run(analysis=lyapunov_mle(record_interval=2))
    
    >>> # Direct mode - model provided explicitly
    >>> module = lyapunov_mle(model=sim.model, record_interval=1)
    >>> sim.run(analysis=module)
    """
    
    def _infer_n_state(target) -> int | None:
        """Extract state count from model spec."""
        spec = getattr(target, "spec", None)
        if spec is None or getattr(spec, "states", None) is None:
            return None
        return len(spec.states)

    def _build_with_model(model_obj: object) -> _LyapunovModule:
        """Build AnalysisModule using a provided model-like object."""
        model_coerced = _coerce_model(model_obj)
        if model_coerced is None:
            raise ValueError("lyapunov_mle factory requires a model")

        jvp_use = jvp if jvp is not None else getattr(model_coerced, "jvp", None)
        n_state_use = n_state if n_state is not None else _infer_n_state(model_coerced)

        if jvp_use is None:
            raise ValueError(
                "lyapunov_mle requires a JVP; provide model with jvp or pass jvp= explicitly"
            )
        if n_state_use is None:
            raise ValueError(
                "lyapunov_mle requires n_state; provide model or pass n_state= explicitly"
            )

        plan_use = _resolve_trace_plan(
            trace_plan=trace_plan,
            record_interval=record_interval,
        )

        return _LyapunovModule(
            jvp=jvp_use,
            n_state=int(n_state_use),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
        )

    # Build immediately when model or jvp/n_state is supplied.
    if model is not None:
        return _build_with_model(model)
    if jvp is not None:
        if n_state is None:
            raise ValueError("lyapunov_mle requires n_state when jvp is provided without a model")
        plan_use = _resolve_trace_plan(trace_plan=trace_plan, record_interval=record_interval)
        return _LyapunovModule(
            jvp=jvp,
            n_state=int(n_state),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
        )

    # Without model/jvp, only the factory path remains; require callers to opt-in
    # to the factory mode by providing some configuration (e.g., record_interval).
    if trace_plan is None and record_interval is None:
        raise ValueError(
            "lyapunov_mle requires a model or jvp; pass analysis=lyapunov_mle without calling for factory mode"
        )

    def _factory(model: object) -> AnalysisModule:
        """Factory function invoked by Sim with model injected."""
        return _build_with_model(model)

    # Otherwise return factory for Sim to call
    return _factory
