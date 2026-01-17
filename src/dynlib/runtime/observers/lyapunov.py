# src/dynlib/runtime/observers/lyapunov.py
"""Reference Lyapunov runtime observers."""
from __future__ import annotations

import math
from typing import Callable, Literal, Optional, TYPE_CHECKING, NamedTuple
import numpy as np

from dynlib.runtime.fastpath.plans import FixedTracePlan
from dynlib.runtime.runner_api import OK
from dynlib.steppers.registry import list_steppers
from .core import ObserverHooks, ObserverModule, ObserverRequirements, TraceSpec
from dynlib.errors import JITUnavailableError

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from dynlib.compiler.build import FullModel

__all__ = ["lyapunov_mle_observer", "lyapunov_spectrum_observer"]


def _default_tangent(n_state: int) -> np.ndarray:
    vec = np.zeros((n_state,), dtype=float)
    if n_state > 0:
        vec[0] = 1.0
    return vec


def _default_basis(n_state: int, k: int) -> np.ndarray:
    """Canonical basis (n_state, k) with k <= n_state."""
    Q = np.zeros((n_state, k), dtype=float)
    for j in range(k):
        Q[j, j] = 1.0
    return Q


def _supported_variational_steppers() -> str:
    names = list_steppers(variational_stepping=True)
    if not names:
        return "<none>"
    return ", ".join(sorted(set(names)))


def _infer_variational_ws_slices(
    factory: Optional[Callable],
    size: int,
    n_state: int,
) -> Optional[tuple[int, ...]]:
    if factory is None or size <= 0:
        return None
    scratch = np.zeros((size,), dtype=float)
    ws = factory(scratch, 0, n_state)
    try:
        items = tuple(ws)
    except TypeError:
        items = (ws,)
    sizes_raw = tuple(int(item.shape[0]) for item in items)
    if sum(sizes_raw) != int(size):
        raise ValueError("Variational workspace layout mismatch")
    if len(sizes_raw) > 5:
        raise ValueError("Variational workspace layout is too large")
    sizes = sizes_raw + (0,) * (5 - len(sizes_raw))
    return sizes


def _raise_variational_unsupported() -> None:
    supported = _supported_variational_steppers()
    raise ValueError(
        "Selected stepper does not support variational stepping; "
        f"results would be inconsistent. Use {supported}"
    )


def _resolve_mode(
    *,
    mode: Literal["flow", "map", "auto"],
    model_like: object | None,
    who: str,
) -> Literal["flow", "map"]:
    if mode in ("flow", "map"):
        return mode
    if mode != "auto":
        raise ValueError(f"{who} mode must be 'flow', 'map', or 'auto'")
    model = _coerce_model(model_like)
    spec = getattr(model, "spec", None)
    kind = getattr(spec, "kind", None)
    if kind == "ode":
        return "flow"
    if kind == "map":
        return "map"
    raise ValueError(f"{who} mode='auto' requires model.spec.kind in {{'ode','map'}}")


def _make_hooks(
    *,
    jvp_fn: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
    init_vec: np.ndarray,
    n_state: int,
    mode: int,  # 0: flow, 1: map (J*v)
    variational_step_fn: Optional[Callable] = None,  # Optional RK4/RK2/etc variational step
    runner_handles_variational: bool = False,
    variational_mode: int = 0,  # 0: euler, 1: combined, 2: tangent_only
    variational_ws_size: int = 0,
    variational_ws_factory: Optional[Callable] = None,
    variational_ws_slices: Optional[tuple[int, ...]] = None,
) -> ObserverHooks:
    """
    Create analysis hooks for Lyapunov MLE computation.
    """
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
        if runner_handles_variational:
            min_ws_size = 2 * n_state
        elif variational_step_fn is not None:
            min_ws_size = 2 * n_state + variational_ws_size
        else:
            min_ws_size = 2 * n_state
        if analysis_ws.shape[0] < min_ws_size or analysis_out.shape[0] < 2:
            return

        vec = analysis_ws[:n_state]
        out_vec = analysis_ws[n_state : 2 * n_state]

        if step == 0 and analysis_out.shape[0] > 3:
            any_nonzero = False
            for i in range(n_state):
                if vec[i] != 0.0:
                    any_nonzero = True
                    break
            if not any_nonzero:
                for i in range(n_state):
                    vec[i] = init_vec[i]
            for i in range(n_state):
                out_vec[i] = 0.0
            analysis_out[0] = 0.0
            analysis_out[1] = 0.0
            analysis_out[2] = 0.0
            analysis_out[3] = float(variational_mode)  # Store mode metadata

        if mode == 0 and (not runner_handles_variational) and variational_step_fn is not None:
            start = 2 * n_state
            if variational_ws_slices is None:
                if variational_ws_factory is None:
                    raise RuntimeError("Variational workspace factory is missing")
                var_ws = variational_ws_factory(analysis_ws, start, n_state)
            else:
                s0, s1, s2, s3, s4 = variational_ws_slices
                o1 = start + s0
                o2 = o1 + s1
                o3 = o2 + s2
                o4 = o3 + s3
                o5 = o4 + s4
                var_ws = (
                    analysis_ws[start:o1],
                    analysis_ws[o1:o2],
                    analysis_ws[o2:o3],
                    analysis_ws[o3:o4],
                    analysis_ws[o4:o5],
                )

            # Call tangent step: integrates vec → out_vec
            # Signature: (t, dt, y_curr, v_curr, v_prop, params, runtime_ws, ws)
            variational_step_fn(t, dt, y_curr, vec, out_vec, params, runtime_ws, var_ws)

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
        if runner_handles_variational:
            min_ws_size = 2 * n_state
        elif variational_step_fn is not None:
            min_ws_size = 2 * n_state + variational_ws_size
        else:
            min_ws_size = 2 * n_state
        if analysis_ws.shape[0] < min_ws_size or analysis_out.shape[0] < 2:
            return
            
        vec = analysis_ws[:n_state]
        out_vec = analysis_ws[n_state : 2 * n_state]
        
        if mode == 0:  # flow mode
            if runner_handles_variational:
                # Runner already propagated out_vec in-step.
                pass
            elif variational_step_fn is None:
                raise RuntimeError("Variational stepping requested without stepper support")

        else:
            # map: out_vec = J*v
            jvp_fn(t, y_curr, params, vec, out_vec, runtime_ws)

        norm_sq = 0.0
        for i in range(n_state):
            val = out_vec[i]
            norm_sq += float(val * val)
        norm = math.sqrt(norm_sq)
        if norm == 0.0:
            return
        inv = 1.0 / norm
        for i in range(n_state):
            analysis_ws[i] = out_vec[i] * inv
        analysis_out[0] += math.log(norm)
        analysis_out[1] += dt if mode == 0 else 1.0
        if analysis_out.shape[0] > 2:
            analysis_out[2] += 1.0

        if trace_buf.shape[0] > 0 and trace_stride > 0 and (step % trace_stride == 0):
            idx = int(trace_count[0])
            if idx < trace_cap:
                denom = analysis_out[1]
                if denom <= 0.0:
                    denom = 1.0
                trace_buf[idx, 0] = analysis_out[0] / denom
                trace_count[0] = idx + 1
            else:
                trace_count[0] = trace_cap + 1

    return ObserverHooks(pre_step=_pre_step, post_step=_post_step)


def _coerce_model(model_like) -> "FullModel" | None:
    """Extract a FullModel instance from a FullModel or Sim-like object."""
    if model_like is None:
        return None
    # Sim exposes the compiled model via ``.model``; accept both forms for convenience.
    model = getattr(model_like, "model", model_like)
    if getattr(model, "spec", None) is None:
        return None
    return model  # type: ignore[return-value]


def _resolve_trace_plan(
    *, trace_plan: Optional[FixedTracePlan], record_interval: Optional[int], who: str
) -> FixedTracePlan:
    if trace_plan is not None and record_interval is not None:
        if int(record_interval) != int(trace_plan.record_interval()):
            raise ValueError(f"record_interval must match provided trace_plan stride for {who}")
    if record_interval is not None:
        stride = int(record_interval)
        if stride <= 0:
            raise ValueError(f"record_interval for {who} must be positive")
        return FixedTracePlan(stride=stride)
    if trace_plan is not None:
        return trace_plan
    return FixedTracePlan(stride=1)


class _LyapunovModule(ObserverModule):
    def __init__(
        self,
        *,
        jvp: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
        model_spec,
        stepper_spec,
        rhs_fn,
        n_state: int,
        trace_plan: FixedTracePlan,
        analysis_kind: int,
        mode: Literal["flow", "map"],
        variational_step_fn: Optional[Callable] = None,
        use_variational_stepping: bool = False,
        variational_combined_step_fn: Optional[Callable] = None,
        use_variational_combined: bool = False,
    ) -> None:
        if mode not in ("flow", "map"):
            raise ValueError("lyapunov_mle mode must be 'flow' or 'map'")
        self._jvp_py = jvp
        self._jvp_jit = None
        self._variational_step_py = variational_step_fn
        self._variational_step_jit = None
        self._variational_step_combined_py = variational_combined_step_fn
        self._variational_step_combined_jit = None
        self._runner_variational_step_py = None
        self._runner_variational_step_jit = None
        self._use_variational_combined = use_variational_combined and variational_combined_step_fn is not None
        self._use_variational = (
            use_variational_stepping and variational_step_fn is not None and not self._use_variational_combined
        )
        self._stepper_spec = stepper_spec
        self._model_spec = model_spec
        self._rhs_fn = rhs_fn
        self._analysis_kind = analysis_kind
        
        # Determine variational mode for metadata
        if use_variational_combined and variational_combined_step_fn is not None:
            variational_mode = 1  # combined
        elif use_variational_stepping and variational_step_fn is not None and not use_variational_combined:
            variational_mode = 2  # tangent_only
        else:
            variational_mode = 0  # euler
        
        # Resolve variational workspace
        if self._use_variational and stepper_spec is not None:
            if hasattr(stepper_spec, "variational_workspace"):
                self._var_ws_size, self._var_ws_factory = stepper_spec.variational_workspace(n_state, model_spec)
            else:
                raise ValueError(f"Stepper {stepper_spec} does not support variational workspace contract")
        else:
            self._var_ws_size, self._var_ws_factory = 0, None
        self._var_ws_slices = _infer_variational_ws_slices(
            self._var_ws_factory,
            self._var_ws_size,
            n_state,
        ) if self._use_variational else None
        
        init_vec = _default_tangent(n_state)
        hooks = _make_hooks(
            jvp_fn=jvp,
            init_vec=init_vec,
            n_state=n_state,
            mode=0 if mode == "flow" else 1,
            variational_step_fn=variational_step_fn if self._use_variational else None,
            runner_handles_variational=self._use_variational_combined,
            variational_mode=variational_mode,
            variational_ws_size=self._var_ws_size,
            variational_ws_factory=self._var_ws_factory,
        )
        
        # Workspace layout:
        # [0:n_state] - current tangent vector
        # [n_state:2*n_state] - output/work tangent vector  
        # If using variational stepping:
        # [2*n_state : 2*n_state + var_ws_size] - variational buffers
        if self._use_variational_combined:
            workspace_size = 2 * n_state
        elif self._use_variational:
            workspace_size = 2 * n_state + self._var_ws_size
        else:
            workspace_size = 2 * n_state
        
        reqs = ObserverRequirements(
            fixed_step=True,
            need_jvp=True,
            mutates_state=False,
            variational_in_step=self._use_variational_combined,
        )
        
        super().__init__(
            key="lyapunov_mle",
            name="lyapunov_mle",
            requirements=reqs,
            workspace_size=workspace_size,
            output_size=4,
            output_names=("log_growth", "denom", "steps", "variational_mode"),
            trace_names=("mle",),
            trace=TraceSpec(width=1, plan=trace_plan),
            hooks=hooks,
            analysis_kind=analysis_kind,
        )
        self._init_vec = init_vec
        self._n_state = int(n_state)
        self._mode = 0 if mode == "flow" else 1

    def validate_stepper(self, stepper_spec) -> None:
        if self._mode != 0:
            return
        if not (self._use_variational or self._use_variational_combined):
            _raise_variational_unsupported()
        spec = self._stepper_spec or stepper_spec
        if spec is None:
            _raise_variational_unsupported()
        stepper_meta = getattr(spec, "meta", None)
        caps = getattr(stepper_meta, "caps", None) if stepper_meta is not None else None
        if caps is None or not getattr(caps, "variational_stepping", False):
            _raise_variational_unsupported()
        if not (hasattr(spec, "emit_step_with_variational") or hasattr(spec, "emit_tangent_step")):
            _raise_variational_unsupported()

    def signature(self, dtype: np.dtype) -> tuple:
        """
        Return a hashable signature including mode and state dimension.
        """
        trace_width = self.trace.width if self.trace else 0
        trace_stride = self.trace_stride
        return (
            "lyapunov_mle",
            self._n_state,
            self._mode,
            int(self._use_variational),
            int(self._use_variational_combined),
            self._analysis_kind,
            trace_width,
            trace_stride,
            str(np.dtype(dtype)),
        )

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
            raise JITUnavailableError("lyapunov_mle requires numba for jit hooks") from exc
        try:
            self._jvp_jit = njit(cache=False)(py_target)
        except Exception as exc:
            raise RuntimeError(
                "lyapunov_mle requires a numba-compatible Jacobian-vector product for JIT execution"
            ) from exc
        return self._jvp_jit
    
    def _ensure_jit_variational_step(self):
        """JIT compile the variational step function if needed."""
        if self._variational_step_jit is not None:
            return self._variational_step_jit
        if self._variational_step_py is None or not self._use_variational:
            return None
        
        # Re-emit tangent step with the jitted JVP to avoid closures over Python JVP.
        step_fn = None
        jvp_jit = self._ensure_jit_jvp()
        if self._stepper_spec is not None and hasattr(self._stepper_spec, "emit_tangent_step"):
            try:
                step_fn = self._stepper_spec.emit_tangent_step(
                    jvp_fn=jvp_jit,
                    model_spec=self._model_spec,
                )
            except Exception as exc:
                raise RuntimeError("Failed to emit tangent step for JIT compilation") from exc
        if step_fn is None:
            step_fn = self._variational_step_py
        
        # Check if already jitted
        dispatcher = getattr(step_fn, "signatures", None)
        if dispatcher is not None:
            self._variational_step_jit = step_fn
            return self._variational_step_jit
        
        py_target = getattr(step_fn, "py_func", step_fn)
        try:
            from numba import njit
        except Exception as exc:
            raise JITUnavailableError("lyapunov_mle requires numba for variational stepping") from exc
        try:
            self._variational_step_jit = njit(cache=False)(py_target)
        except Exception as exc:
            raise RuntimeError("Failed to JIT compile variational step function") from exc
        return self._variational_step_jit

    def _ensure_variational_combined(self, jit: bool):
        if not self._use_variational_combined:
            return None
        if jit and self._variational_step_combined_jit is not None:
            return self._variational_step_combined_jit
        if not jit and self._variational_step_combined_py is not None:
            return self._variational_step_combined_py

        base_fn = self._variational_step_combined_py
        if base_fn is None:
            return None

        target_fn = base_fn
        dispatcher = getattr(base_fn, "signatures", None)

        if jit:
            if dispatcher is None and self._stepper_spec is not None and hasattr(self._stepper_spec, "emit_step_with_variational"):
                try:
                    target_fn = self._stepper_spec.emit_step_with_variational(
                        rhs_fn=self._rhs_fn,
                        jvp_fn=self._ensure_jit_jvp(),
                        model_spec=self._model_spec,
                    )
                    dispatcher = getattr(target_fn, "signatures", None)
                except Exception as exc:
                    raise RuntimeError("Failed to emit combined variational step for JIT compilation") from exc

            if dispatcher is not None:
                self._variational_step_combined_jit = target_fn
                return target_fn

            try:
                from numba import njit
                self._variational_step_combined_jit = njit(cache=False)(getattr(target_fn, "py_func", target_fn))
            except Exception as exc:
                raise RuntimeError("Failed to JIT compile combined variational step function") from exc
            return self._variational_step_combined_jit

        # non-jit path
        self._variational_step_combined_py = target_fn
        return target_fn

    def _make_runner_variational_step(self, combined_fn: Callable) -> Callable:
        n_state = self._n_state

        def _runner_var_step(
            t, dt,
            y_curr,
            rhs,
            params,
            runtime_ws,
            stepper_ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est,
            analysis_ws,
        ):
            v_curr = analysis_ws[:n_state]
            v_prop = analysis_ws[n_state:2 * n_state]
            combined_fn(t, dt, y_curr, v_curr, y_prop, v_prop, params, runtime_ws, stepper_ws)
            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0
            return OK

        return _runner_var_step

    def runner_variational_step(self, *, jit: bool):
        if not self._use_variational_combined:
            return None
        if jit:
            if self._runner_variational_step_jit is not None:
                return self._runner_variational_step_jit
            combined = self._ensure_variational_combined(jit=True)
            if combined is None:
                return None
            wrapper = self._make_runner_variational_step(combined)
            try:
                from numba import njit
            except Exception as exc:
                raise JITUnavailableError(
                    "jit=True requires numba for variational stepping."
                ) from exc
            try:
                self._runner_variational_step_jit = njit(cache=False)(wrapper)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to JIT compile variational step wrapper."
                ) from exc
            return self._runner_variational_step_jit

        if self._runner_variational_step_py is None:
            combined = self._ensure_variational_combined(jit=False)
            if combined is None:
                return None
            self._runner_variational_step_py = self._make_runner_variational_step(combined)
        return self._runner_variational_step_py

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> ObserverHooks:
        if not jit:
            return self.hooks

        # Cache compiled hooks per (dtype, trace shape) to avoid re-jitting on every run.
        key = self._jit_key(dtype)
        cached = self._jit_cache.get(key)
        if cached is not None:
            return cached

        jvp_jit = self._ensure_jit_jvp()
        variational_jit = self._ensure_jit_variational_step() if self._use_variational else None
        
        # Determine variational mode for JIT path
        if self._use_variational_combined:
            variational_mode = 1
        elif self._use_variational:
            variational_mode = 2
        else:
            variational_mode = 0
        
        jit_hooks = _make_hooks(
            jvp_fn=jvp_jit,
            init_vec=self._init_vec,
            n_state=self._n_state,
            mode=self._mode,
            variational_step_fn=variational_jit,
            runner_handles_variational=self._use_variational_combined,
            variational_mode=variational_mode,
            variational_ws_size=self._var_ws_size,
            variational_ws_factory=self._var_ws_factory,
            variational_ws_slices=self._var_ws_slices,
        )
        return self._compile_hooks(jit_hooks, dtype)


def lyapunov_mle_observer(
    model=None,
    sim=None,
    record_interval: Optional[int] = None,
    *,
    jvp: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None]
    ] = None,
    n_state: Optional[int] = None,
    trace_plan: Optional[FixedTracePlan] = None,
    analysis_kind: int = 1,
    mode: Literal["flow", "map", "auto"] = "auto",
    prefer_variational_combined: bool = True,
):
    """
    Factory for Lyapunov maximum exponent analysis.
    
    Returns a factory function for dependency injection, or an ObserverModule if model is provided.
    The factory extracts ``jvp`` and ``n_state`` from the model if not explicitly provided.
    
    Parameters
    ----------
    model : FullModel or Sim, optional
        Compiled model. If provided, returns ObserverModule directly. If None, returns factory.
    sim : Sim, optional
        Sim instance providing the compiled model when model is not provided.
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
    mode : {"flow","map","auto"}, default="auto"
        "flow": flow-style update, denom accumulates time.
        "map":  map-style update v <- Jv, denom accumulates steps.
        "auto": infer from model.spec.kind ("ode" -> flow, "map" -> map).
    prefer_variational_combined : bool, default=True
        If True, prefer combined state+tangent stepping if supported by the stepper.
        
    Returns
    -------
    factory or module
        If model is None: factory function with signature ``factory(model) -> _LyapunovModule``
        If model is provided: ``_LyapunovModule`` instance
        
    Examples
    --------
    >>> # Factory mode - Sim injects model
    >>> sim.run(observers=lyapunov_mle_observer())
    >>> sim.run(observers=lyapunov_mle_observer(record_interval=2))
    
    >>> # Direct mode - model provided explicitly
    >>> module = lyapunov_mle_observer(model=sim.model, record_interval=1)
    >>> sim.run(observers=module)
    """

    if model is None and sim is not None:
        model = getattr(sim, "model", sim)
    record_interval_override = record_interval
    
    def _infer_n_state(target) -> int | None:
        """Extract state count from model spec."""
        spec = getattr(target, "spec", None)
        if spec is None or getattr(spec, "states", None) is None:
            return None
        return len(spec.states)

    def _build_with_model(model_obj: object) -> _LyapunovModule:
        """Build ObserverModule using a provided model-like object."""
        model_coerced = _coerce_model(model_obj)
        if model_coerced is None:
            raise ValueError("lyapunov_mle factory requires a model")
        mode_use = _resolve_mode(mode=mode, model_like=model_coerced, who="lyapunov_mle_observer")

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
            who="lyapunov_mle_observer",
        )
        
        # Check if stepper supports variational stepping (for flow mode only)
        stepper_spec = getattr(model_coerced, "stepper_spec", None)
        variational_step_fn = None
        use_variational = False
        variational_combined_fn = None
        use_variational_combined = False
        if mode_use == "flow":
            supported = _supported_variational_steppers()
            if stepper_spec is None:
                raise ValueError(
                    f"Selected stepper does not support variational stepping; results would be inconsistent. Use {supported}"
                )
            stepper_meta = getattr(stepper_spec, "meta", None)
            caps = getattr(stepper_meta, "caps", None) if stepper_meta is not None else None
            if caps is None or not getattr(caps, "variational_stepping", False):
                raise ValueError(
                    f"Selected stepper does not support variational stepping; results would be inconsistent. Use {supported}"
                )
            # Prefer combined state+tangent stepping when available
            if prefer_variational_combined and hasattr(stepper_spec, "emit_step_with_variational"):
                variational_combined_fn = stepper_spec.emit_step_with_variational(
                    rhs_fn=getattr(model_coerced, "rhs", None),
                    jvp_fn=jvp_use,
                    model_spec=getattr(model_coerced, "spec", None),
                )
                use_variational_combined = variational_combined_fn is not None
            # Fallback to tangent-only variational stepping (still within stepper)
            if not use_variational_combined and hasattr(stepper_spec, "emit_tangent_step"):
                variational_step_fn = stepper_spec.emit_tangent_step(
                    jvp_fn=jvp_use,
                    model_spec=getattr(model_coerced, "spec", None)
                )
                use_variational = variational_step_fn is not None
            if not use_variational and not use_variational_combined:
                raise ValueError(
                    f"Selected stepper does not support variational stepping; results would be inconsistent. Use {supported}"
                )

        return _LyapunovModule(
            jvp=jvp_use,
            model_spec=getattr(model_coerced, "spec", None),
            stepper_spec=stepper_spec if (use_variational or use_variational_combined) else None,
            rhs_fn=getattr(model_coerced, "rhs", None),
            n_state=int(n_state_use),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            mode=mode_use,
            variational_step_fn=variational_step_fn,
            use_variational_stepping=use_variational,
            variational_combined_step_fn=variational_combined_fn,
            use_variational_combined=use_variational_combined,
        )

    # Build immediately when model or jvp/n_state is supplied.
    if model is not None:
        return _build_with_model(model)
    if jvp is not None:
        if n_state is None:
            raise ValueError("lyapunov_mle requires n_state when jvp is provided without a model")
        if mode == "auto":
            raise ValueError("lyapunov_mle mode='auto' requires a model to infer kind")
        if mode == "flow":
             raise ValueError("lyapunov_mle mode='flow' requires a model to provide stepper support")
        plan_use = _resolve_trace_plan(
            trace_plan=trace_plan, record_interval=record_interval, who="lyapunov_mle_observer"
        )
        return _LyapunovModule(
            jvp=jvp,
            model_spec=None,
            stepper_spec=None,
            rhs_fn=None,
            n_state=int(n_state),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            mode=_resolve_mode(mode=mode, model_like=None, who="lyapunov_mle_observer"),
        )

    def _factory(model: object, sim: object | None = None, record_interval: int | None = None) -> ObserverModule:
        """Factory function invoked by Sim with model injected."""
        ri = record_interval_override if record_interval_override is not None else record_interval
        model_obj = getattr(sim, "model", model) if sim is not None else model
        return lyapunov_mle_observer(
            model=model_obj,
            record_interval=ri,
            jvp=jvp,
            n_state=n_state,
            trace_plan=trace_plan,
            analysis_kind=analysis_kind,
            mode=mode,
            prefer_variational_combined=prefer_variational_combined,
        )

    _factory.__observer_factory__ = True

    # Otherwise return factory for Sim to call
    return _factory


# -------------------------- spectrum hooks -----------------------------------

def _make_hooks_spectrum(
    *,
    jvp_fn: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
    init_basis: np.ndarray,  # shape (n_state, k)
    n_state: int,
    k: int,
    mode: int,  # 0: flow, 1: map (J*v)
    variational_step_fn: Optional[Callable] = None,
    runner_handles_variational: bool = False,
    variational_mode: int = 0,  # 0: euler, 1: combined, 2: tangent_only
    variational_ws_size_per_vec: int = 0,
    variational_ws_factory: Optional[Callable] = None,
    variational_ws_slices: Optional[tuple[int, ...]] = None,
) -> ObserverHooks:
    """
    Workspace layout:
        analysis_ws[0 : n_state*k]               -> V (current orthonormal basis), shape (n_state, k)
        analysis_ws[n_state*k : 2*n_state*k]     -> W (work), shape (n_state, k)
        if variational_step_fn:
        analysis_ws[2*n_state*k : 2*n_state*k + var_ws_size_per_vec*k] -> variational buffers

    Output layout (length k+3):
        analysis_out[0:k]     -> accum_log_diag[j] = sum log(R_jj)
        analysis_out[k]       -> denom (total_time for flow, steps for map)
        analysis_out[k+1]     -> steps (integer-ish stored as float)
        analysis_out[k+2]     -> variational_mode (metadata)
    """

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
        need_ws = 2 * n_state * k + (
            variational_ws_size_per_vec * k if variational_step_fn is not None else 0
        )
        need_out = k + 3
        if analysis_ws.shape[0] < need_ws or analysis_out.shape[0] < need_out:
            return

        V = analysis_ws[: n_state * k].reshape((n_state, k))
        W = analysis_ws[n_state * k : 2 * n_state * k].reshape((n_state, k))
        # var_ws is constructed per-column when needed.

        if step == 0:
            # Initialize only if V is all zeros (allows user to pre-seed).
            any_nonzero = False
            for i in range(n_state):
                for j in range(k):
                    if V[i, j] != 0.0:
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if not any_nonzero:
                for i in range(n_state):
                    for j in range(k):
                        V[i, j] = init_basis[i, j]

            # Clear work + outputs
            for i in range(n_state):
                for j in range(k):
                    W[i, j] = 0.0

            for j in range(k):
                analysis_out[j] = 0.0
            analysis_out[k] = 0.0
            analysis_out[k + 1] = 0.0
            analysis_out[k + 2] = float(variational_mode)  # Store mode metadata

        if mode == 0 and (not runner_handles_variational) and variational_step_fn is not None:
            for j in range(k):
                v_col = V[:, j]
                w_col = W[:, j]
                start = 2 * n_state * k + j * variational_ws_size_per_vec
                if variational_ws_slices is None:
                    if variational_ws_factory is None:
                        raise RuntimeError("Variational workspace factory is missing")
                    var_ws = variational_ws_factory(analysis_ws, start, n_state)
                else:
                    s0, s1, s2, s3, s4 = variational_ws_slices
                    o1 = start + s0
                    o2 = o1 + s1
                    o3 = o2 + s2
                    o4 = o3 + s3
                    o5 = o4 + s4
                    var_ws = (
                        analysis_ws[start:o1],
                        analysis_ws[o1:o2],
                        analysis_ws[o2:o3],
                        analysis_ws[o3:o4],
                        analysis_ws[o4:o5],
                    )
                variational_step_fn(t, dt, y_curr, v_col, w_col, params, runtime_ws, var_ws)

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
        need_ws = 2 * n_state * k + (
            variational_ws_size_per_vec * k if variational_step_fn is not None else 0
        )
        need_out = k + 3
        if analysis_ws.shape[0] < need_ws or analysis_out.shape[0] < need_out:
            return

        V = analysis_ws[: n_state * k].reshape((n_state, k))
        W = analysis_ws[n_state * k : 2 * n_state * k].reshape((n_state, k))

        # -------- propagate tangent basis columns into W --------
        # tmp_out is W[:, j], reused per column.
        if not runner_handles_variational:
            if mode == 0:
                if variational_step_fn is None:
                    raise RuntimeError("Variational stepping requested without stepper support")
            else:
                for j in range(k):
                    # Compute J*v into W[:, j]
                    v_col = V[:, j]
                    w_col = W[:, j]
                    jvp_fn(t, y_curr, params, v_col, w_col, runtime_ws)

        # -------- Modified Gram–Schmidt (QR) on W (in place) --------
        # We only need diag(R); Q overwrites W and then we copy back to V.
        # If a column collapses to zero, re-seed deterministically to a canonical axis.
        for j in range(k):
            # subtract projections onto previous q_i (stored in W[:, i])
            for i_prev in range(j):
                dot = 0.0
                for r in range(n_state):
                    dot += W[r, i_prev] * W[r, j]
                for r in range(n_state):
                    W[r, j] -= dot * W[r, i_prev]

            # norm of the orthogonalized vector
            norm_sq = 0.0
            for r in range(n_state):
                val = W[r, j]
                norm_sq += val * val
            norm = math.sqrt(float(norm_sq))

            if norm == 0.0:
                # deterministic reseed: e_{j mod n_state}
                axis = j
                if axis >= n_state:
                    axis = axis % n_state
                for r in range(n_state):
                    W[r, j] = 0.0
                W[axis, j] = 1.0
                norm = 1.0

            # accumulate log(diag(R))
            analysis_out[j] += math.log(float(norm))

            # normalize -> Q column
            inv = 1.0 / norm
            for r in range(n_state):
                W[r, j] *= inv

        # Copy orthonormal basis back to V
        for i in range(n_state):
            for j in range(k):
                V[i, j] = W[i, j]

        # Update denominators and step count
        analysis_out[k + 1] += 1.0  # steps
        if mode == 0:
            # flow: denom is time
            analysis_out[k] += dt
        else:
            # map: denom is steps
            analysis_out[k] += 1.0

        # -------- trace sampling --------
        if trace_buf.shape[0] > 0 and trace_stride > 0 and (step % trace_stride == 0):
            idx = int(trace_count[0])
            if idx < trace_cap:
                denom = float(analysis_out[k])
                if denom <= 0.0:
                    denom = 1.0
                for j in range(k):
                    trace_buf[idx, j] = analysis_out[j] / denom
                trace_count[0] = idx + 1
            else:
                trace_count[0] = trace_cap + 1

    return ObserverHooks(pre_step=_pre_step, post_step=_post_step)


# --------------------------- spectrum module ---------------------------------

class _LyapunovSpectrumModule(ObserverModule):
    def __init__(
        self,
        *,
        jvp: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
        model_spec,
        stepper_spec,
        rhs_fn,
        n_state: int,
        k: int,
        mode: Literal["flow", "map"],
        trace_plan: FixedTracePlan,
        analysis_kind: int,
        init_basis: Optional[np.ndarray] = None,
        variational_step_fn: Optional[Callable] = None,
        use_variational_stepping: bool = False,
        variational_combined_step_fn: Optional[Callable] = None,
        use_variational_combined: bool = False,
    ) -> None:
        if k <= 0:
            raise ValueError("lyapunov_spectrum requires k >= 1")
        if n_state <= 0:
            raise ValueError("lyapunov_spectrum requires n_state >= 1")
        if k > n_state:
            raise ValueError("lyapunov_spectrum requires k <= n_state")
        if mode not in ("flow", "map"):
            raise ValueError("lyapunov_spectrum mode must be 'flow' or 'map'")

        self._jvp_py = jvp
        self._jvp_jit = None
        self._variational_step_py = variational_step_fn
        self._variational_step_jit = None
        self._variational_step_combined_py = None
        self._variational_step_combined_jit = None
        self._runner_variational_step_py = None
        self._runner_variational_step_jit = None
        self._use_variational_combined = False
        self._use_variational = (
            use_variational_stepping and variational_step_fn is not None and mode == "flow"
        )
        self._stepper_spec = stepper_spec
        self._model_spec = model_spec
        self._rhs_fn = rhs_fn
        self._n_state = int(n_state)
        self._k = int(k)
        self._mode = 0 if mode == "flow" else 1
        self._analysis_kind = analysis_kind

        if init_basis is None:
            B = _default_basis(n_state, k)
        else:
            B = np.asarray(init_basis, dtype=float)
            if B.shape != (n_state, k):
                raise ValueError(f"init_basis must have shape ({n_state}, {k})")
        self._init_basis = B

        # Determine variational mode for metadata
        if self._use_variational:
            variational_mode = 2  # tangent_only
        else:
            variational_mode = 0  # euler

        # Resolve variational workspace
        if self._use_variational and stepper_spec is not None:
            if hasattr(stepper_spec, "variational_workspace"):
                self._var_ws_size_per_vec, self._var_ws_factory = stepper_spec.variational_workspace(n_state, model_spec)
            else:
                raise ValueError(f"Stepper {stepper_spec} does not support variational workspace contract")
        else:
            self._var_ws_size_per_vec, self._var_ws_factory = 0, None
        self._var_ws_slices_per_vec = _infer_variational_ws_slices(
            self._var_ws_factory,
            self._var_ws_size_per_vec,
            n_state,
        ) if self._use_variational else None

        hooks = _make_hooks_spectrum(
            jvp_fn=jvp,
            init_basis=B,
            n_state=n_state,
            k=k,
            mode=self._mode,
            variational_step_fn=variational_step_fn if self._use_variational else None,
            runner_handles_variational=self._use_variational_combined,
            variational_mode=variational_mode,
            variational_ws_size_per_vec=self._var_ws_size_per_vec,
            variational_ws_factory=self._var_ws_factory,
        )

        reqs = ObserverRequirements(
            fixed_step=True,
            need_jvp=True,
            mutates_state=False,
            variational_in_step=self._use_variational_combined,
        )

        # outputs: accum logs + denom + steps + variational_mode
        output_names = tuple([f"log_r{i}" for i in range(k)] + ["denom", "steps", "variational_mode"])
        trace_names = tuple([f"lyap{i}" for i in range(k)])

        super().__init__(
            key="lyapunov_spectrum",
            name="lyapunov_spectrum",
            requirements=reqs,
            workspace_size=(2 * n_state * k) + (self._var_ws_size_per_vec * k if self._use_variational else 0),
            output_size=k + 3,
            output_names=output_names,
            trace_names=trace_names,
            trace=TraceSpec(width=k, plan=trace_plan),
            hooks=hooks,
            analysis_kind=analysis_kind,
        )

    def validate_stepper(self, stepper_spec) -> None:
        if self._mode != 0:
            return
        if not (self._use_variational or self._use_variational_combined):
            _raise_variational_unsupported()
        spec = self._stepper_spec or stepper_spec
        if spec is None:
            _raise_variational_unsupported()
        stepper_meta = getattr(spec, "meta", None)
        caps = getattr(stepper_meta, "caps", None) if stepper_meta is not None else None
        if caps is None or not getattr(caps, "variational_stepping", False):
            _raise_variational_unsupported()
        if not (hasattr(spec, "emit_step_with_variational") or hasattr(spec, "emit_tangent_step")):
            _raise_variational_unsupported()

    def signature(self, dtype: np.dtype) -> tuple:
        """
        Return a hashable signature including k, mode, and state dimension.
        """
        trace_width = self.trace.width if self.trace else 0
        trace_stride = self.trace_stride
        return (
            "lyapunov_spectrum",
            self._n_state,
            self._k,
            self._mode,
            int(self._use_variational),
            int(self._use_variational_combined),
            self._analysis_kind,
            trace_width,
            trace_stride,
            str(np.dtype(dtype)),
        )

    def _ensure_jit_jvp(self):
        if self._jvp_jit is not None:
            return self._jvp_jit
        dispatcher = getattr(self._jvp_py, "signatures", None)
        if dispatcher is not None:
            return self._jvp_py
        py_target = getattr(self._jvp_py, "py_func", self._jvp_py)
        try:  # pragma: no cover
            from numba import njit  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise JITUnavailableError("lyapunov_spectrum requires numba for jit hooks") from exc
        try:
            self._jvp_jit = njit(cache=False)(py_target)
        except Exception as exc:
            raise RuntimeError(
                "lyapunov_spectrum requires a numba-compatible JVP for JIT execution"
            ) from exc
        return self._jvp_jit
    
    def _ensure_variational_combined(self, jit: bool):
        if not self._use_variational_combined:
            return None
        if jit and self._variational_step_combined_jit is not None:
            return self._variational_step_combined_jit
        if not jit and self._variational_step_combined_py is not None:
            return self._variational_step_combined_py

        base_fn = self._variational_step_combined_py
        if base_fn is None:
            return None

        target_fn = base_fn
        dispatcher = getattr(base_fn, "signatures", None)

        if jit:
            if dispatcher is None and self._stepper_spec is not None and hasattr(self._stepper_spec, "emit_step_with_variational"):
                try:
                    target_fn = self._stepper_spec.emit_step_with_variational(
                        rhs_fn=self._rhs_fn,
                        jvp_fn=self._ensure_jit_jvp(),
                        model_spec=self._model_spec,
                    )
                    dispatcher = getattr(target_fn, "signatures", None)
                except Exception as exc:
                    raise RuntimeError("Failed to emit combined variational step for JIT compilation") from exc

            if dispatcher is not None:
                self._variational_step_combined_jit = target_fn
                return target_fn

            try:
                from numba import njit
                self._variational_step_combined_jit = njit(cache=False)(getattr(target_fn, "py_func", target_fn))
            except Exception as exc:
                raise RuntimeError("Failed to JIT compile combined variational step function") from exc
            return self._variational_step_combined_jit

        self._variational_step_combined_py = target_fn
        return target_fn

    def _make_runner_variational_step(self, combined_fn: Callable) -> Callable:
        n_state = self._n_state
        k = self._k

        def _runner_var_step(
            t, dt,
            y_curr,
            rhs,
            params,
            runtime_ws,
            stepper_ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est,
            analysis_ws,
        ):
            V = analysis_ws[: n_state * k].reshape((n_state, k))
            W = analysis_ws[n_state * k : 2 * n_state * k].reshape((n_state, k))
            for j in range(k):
                combined_fn(t, dt, y_curr, V[:, j], y_prop, W[:, j], params, runtime_ws, stepper_ws)
            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0
            return OK

        return _runner_var_step

    def runner_variational_step(self, *, jit: bool):
        if not self._use_variational_combined:
            return None
        if jit:
            if self._runner_variational_step_jit is not None:
                return self._runner_variational_step_jit
            combined = self._ensure_variational_combined(jit=True)
            if combined is None:
                return None
            wrapper = self._make_runner_variational_step(combined)
            try:
                from numba import njit
            except Exception as exc:
                raise JITUnavailableError(
                    "jit=True requires numba for variational stepping."
                ) from exc
            try:
                self._runner_variational_step_jit = njit(cache=False)(wrapper)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to JIT compile variational step wrapper."
                ) from exc
            return self._runner_variational_step_jit

        if self._runner_variational_step_py is None:
            combined = self._ensure_variational_combined(jit=False)
            if combined is None:
                return None
            self._runner_variational_step_py = self._make_runner_variational_step(combined)
        return self._runner_variational_step_py
    
    def _ensure_jit_variational_step(self):
        """JIT compile tangent step if available; disable on failure."""
        if self._variational_step_jit is not None or not self._use_variational:
            return self._variational_step_jit
        if self._variational_step_py is None:
            return None
        
        step_fn = None
        jvp_jit = self._ensure_jit_jvp()
        if self._stepper_spec is not None and hasattr(self._stepper_spec, "emit_tangent_step"):
            try:
                step_fn = self._stepper_spec.emit_tangent_step(
                    jvp_fn=jvp_jit,
                    model_spec=self._model_spec,
                )
            except Exception as exc:
                raise RuntimeError("Failed to emit tangent step for JIT compilation") from exc
        if step_fn is None:
            step_fn = self._variational_step_py
        
        dispatcher = getattr(step_fn, "signatures", None)
        if dispatcher is not None:
            self._variational_step_jit = step_fn
            return self._variational_step_jit
        
        py_target = getattr(step_fn, "py_func", step_fn)
        try:
            from numba import njit
        except Exception as exc:
            raise JITUnavailableError("lyapunov_spectrum requires numba for variational stepping") from exc
        try:
            self._variational_step_jit = njit(cache=False)(py_target)
        except Exception as exc:
            raise RuntimeError("Failed to JIT compile variational step function") from exc
        return self._variational_step_jit

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> ObserverHooks:
        if not jit:
            return self.hooks
        key = self._jit_key(dtype)
        cached = self._jit_cache.get(key)
        if cached is not None:
            return cached

        jvp_jit = self._ensure_jit_jvp()
        variational_jit = self._ensure_jit_variational_step() if self._use_variational else None
        
        # Determine variational mode for JIT path
        if self._use_variational_combined:
            variational_mode = 1
        elif self._use_variational:
            variational_mode = 2
        else:
            variational_mode = 0
        
        jit_hooks = _make_hooks_spectrum(
            jvp_fn=jvp_jit,
            init_basis=self._init_basis,
            n_state=self._n_state,
            k=self._k,
            mode=self._mode,
            variational_step_fn=variational_jit,
            runner_handles_variational=self._use_variational_combined,
            variational_mode=variational_mode,
            variational_ws_size_per_vec=self._var_ws_size_per_vec,
            variational_ws_factory=self._var_ws_factory,
            variational_ws_slices=self._var_ws_slices_per_vec,
        )
        return self._compile_hooks(jit_hooks, dtype)


# ---------------------------- spectrum factory -------------------------------

def lyapunov_spectrum_observer(
    model=None,
    sim=None,
    record_interval: Optional[int] = None,
    *,
    jvp: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None]
    ] = None,
    n_state: Optional[int] = None,
    k: int = 2,
    mode: Literal["flow", "map", "auto"] = "auto",
    init_basis: Optional[np.ndarray] = None,
    trace_plan: Optional[FixedTracePlan] = None,
    analysis_kind: int = 1,
    prefer_variational_combined: bool = True,
):
    """
    Factory for Lyapunov spectrum analysis (Benettin / Shimada–Nagashima QR method).

    Output:
      - out[0:k]   : accumulated log(diag(R)) for each exponent
      - out[k]     : denom (total_time for mode='flow', steps for mode='map')
      - out[k+1]   : steps

    Trace (if enabled): running estimates lyap0..lyap{k-1}.
    
    mode : {"flow","map","auto"}, default="auto"
        "flow": Euler update v <- v + dt*(Jv), denom is time.
        "map":  map-style update v <- Jv, denom is steps.
        "auto": infer from model.spec.kind ("ode" -> flow, "map" -> map).
    prefer_variational_combined : bool, default=True
        Reserved for future use; combined stepping is disabled for spectrum in
        the current architecture.
    """

    if model is None and sim is not None:
        model = getattr(sim, "model", sim)
    record_interval_override = record_interval

    def _infer_n_state(target) -> int | None:
        spec = getattr(target, "spec", None)
        if spec is None or getattr(spec, "states", None) is None:
            return None
        return len(spec.states)

    def _build_with_model(model_obj: object) -> _LyapunovSpectrumModule:
        model_coerced = _coerce_model(model_obj)
        if model_coerced is None:
            raise ValueError("lyapunov_spectrum factory requires a model")
        mode_use = _resolve_mode(mode=mode, model_like=model_coerced, who="lyapunov_spectrum_observer")

        jvp_use = jvp if jvp is not None else getattr(model_coerced, "jvp", None)
        n_state_use = n_state if n_state is not None else _infer_n_state(model_coerced)

        if jvp_use is None:
            raise ValueError(
                "lyapunov_spectrum requires a JVP; provide model with jvp or pass jvp="
            )
        if n_state_use is None:
            raise ValueError("lyapunov_spectrum requires n_state; provide model or pass n_state=")

        plan_use = _resolve_trace_plan(
            trace_plan=trace_plan,
            record_interval=record_interval,
            who="lyapunov_spectrum_observer",
        )

        # Optional variational stepping (flow only)
        variational_step_fn = None
        use_variational = False
        variational_combined_fn = None
        use_variational_combined = False
        stepper_spec = getattr(model_coerced, "stepper_spec", None)
        if mode_use == "flow":
            supported = _supported_variational_steppers()
            if stepper_spec is None:
                raise ValueError(
                    f"Selected stepper does not support variational stepping; results would be inconsistent. Use {supported}"
                )
            stepper_meta = getattr(stepper_spec, "meta", None)
            caps = getattr(stepper_meta, "caps", None) if stepper_meta is not None else None
            if caps is None or not getattr(caps, "variational_stepping", False):
                raise ValueError(
                    f"Selected stepper does not support variational stepping; results would be inconsistent. Use {supported}"
                )
            if hasattr(stepper_spec, "emit_tangent_step"):
                variational_step_fn = stepper_spec.emit_tangent_step(
                    jvp_fn=jvp_use,
                    model_spec=getattr(model_coerced, "spec", None),
                )
                use_variational = variational_step_fn is not None
            if not use_variational:
                raise ValueError(
                    f"Selected stepper does not support variational stepping; results would be inconsistent. Use {supported}"
                )

        return _LyapunovSpectrumModule(
            jvp=jvp_use,
            model_spec=getattr(model_coerced, "spec", None),
            stepper_spec=stepper_spec if (use_variational or use_variational_combined) else None,
            rhs_fn=getattr(model_coerced, "rhs", None),
            n_state=int(n_state_use),
            k=int(k),
            mode=mode_use,
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            init_basis=init_basis,
            variational_step_fn=variational_step_fn,
            use_variational_stepping=use_variational,
            variational_combined_step_fn=variational_combined_fn,
            use_variational_combined=use_variational_combined,
        )

    if model is not None:
        return _build_with_model(model)

    if jvp is not None:
        if n_state is None:
            raise ValueError("lyapunov_spectrum requires n_state when jvp is provided without a model")
        if mode == "auto":
            raise ValueError("lyapunov_spectrum mode='auto' requires a model to infer kind")
        if mode == "flow":
             raise ValueError("lyapunov_spectrum mode='flow' requires a model to provide stepper support")
        plan_use = _resolve_trace_plan(
            trace_plan=trace_plan,
            record_interval=record_interval,
            who="lyapunov_spectrum_observer",
        )
        return _LyapunovSpectrumModule(
            jvp=jvp,
            model_spec=None,
            stepper_spec=None,
            rhs_fn=None,
            n_state=int(n_state),
            k=int(k),
            mode=_resolve_mode(mode=mode, model_like=None, who="lyapunov_spectrum_observer"),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            init_basis=init_basis,
        )

    # Factory path (model injected by Sim)
    def _factory(model: object, sim: object | None = None, record_interval: int | None = None) -> ObserverModule:
        ri = record_interval_override if record_interval_override is not None else record_interval
        model_obj = getattr(sim, "model", model) if sim is not None else model
        return lyapunov_spectrum_observer(
            model=model_obj,
            record_interval=ri,
            jvp=jvp,
            n_state=n_state,
            k=k,
            mode=mode,
            init_basis=init_basis,
            trace_plan=trace_plan,
            analysis_kind=analysis_kind,
            prefer_variational_combined=prefer_variational_combined,
        )

    _factory.__observer_factory__ = True

    return _factory


lyapunov_mle_observer.__observer_factory__ = True
lyapunov_spectrum_observer.__observer_factory__ = True
