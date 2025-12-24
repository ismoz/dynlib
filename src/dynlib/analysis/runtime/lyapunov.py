# src/dynlib/analysis/runtime/lyapunov.py
"""Reference Lyapunov runtime analysis."""
from __future__ import annotations

import math
from typing import Callable, Literal, Optional, TYPE_CHECKING
import numpy as np

from dynlib.runtime.fastpath.plans import FixedTracePlan
from .core import AnalysisHooks, AnalysisModule, AnalysisRequirements, TraceSpec

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from dynlib.compiler.build import FullModel

__all__ = ["lyapunov_mle", "lyapunov_spectrum"]


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
    mode: int,  # 0: flow (Euler), 1: map (J*v)
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
            any_nonzero = False
            for i in range(n_state):
                if analysis_ws[i] != 0.0:
                    any_nonzero = True
                    break
            if not any_nonzero:
                for i in range(n_state):
                    analysis_ws[i] = init_vec[i]
            analysis_ws[n_state : 2 * n_state] = 0.0
            analysis_out[0] = 0.0
            analysis_out[1] = 0.0
            if analysis_out.shape[0] > 2:
                analysis_out[2] = 0.0

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

        if mode == 0:
            # flow: Euler step on variational equation
            for i in range(n_state):
                out_vec[i] = vec[i] + dt * out_vec[i]
        else:
            # map: out_vec already holds J*v
            pass

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

    return AnalysisHooks(pre_step=_pre_step, post_step=_post_step)


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


class _LyapunovModule(AnalysisModule):
    def __init__(
        self,
        *,
        jvp: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
        n_state: int,
        trace_plan: FixedTracePlan,
        analysis_kind: int,
        mode: Literal["flow", "map"],
    ) -> None:
        if mode not in ("flow", "map"):
            raise ValueError("lyapunov_mle mode must be 'flow' or 'map'")
        self._jvp_py = jvp
        self._jvp_jit = None
        init_vec = _default_tangent(n_state)
        hooks = _make_hooks(
            jvp_fn=jvp,
            init_vec=init_vec,
            n_state=n_state,
            mode=0 if mode == "flow" else 1,
        )
        reqs = AnalysisRequirements(fixed_step=True, need_jvp=True, mutates_state=False)
        super().__init__(
            name="lyapunov_mle",
            requirements=reqs,
            workspace_size=2 * n_state,
            output_size=3,
            output_names=("log_growth", "denom", "steps"),
            trace_names=("mle",),
            trace=TraceSpec(width=1, plan=trace_plan),
            hooks=hooks,
            analysis_kind=analysis_kind,
        )
        self._init_vec = init_vec
        self._n_state = int(n_state)
        self._mode = 0 if mode == "flow" else 1

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

        # Cache compiled hooks per (dtype, trace shape) to avoid re-jitting on every run.
        key = self._jit_key(dtype)
        cached = self._jit_cache.get(key)
        if cached is not None:
            return cached

        jvp_jit = self._ensure_jit_jvp()
        jit_hooks = _make_hooks(
            jvp_fn=jvp_jit,
            init_vec=self._init_vec,
            n_state=self._n_state,
            mode=self._mode,
        )
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
    mode: Literal["flow", "map", "auto"] = "auto",
):
    """
    Factory for Lyapunov maximum exponent analysis.
    
    Returns a factory function for dependency injection, or an AnalysisModule if model is provided.
    The factory extracts ``jvp`` and ``n_state`` from the model if not explicitly provided.
    
    Parameters
    ----------
    model : FullModel or Sim, optional
        Compiled model. If provided, returns AnalysisModule directly. If None, returns factory.
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
        "flow": Euler update v <- v + dt*(Jv), denom accumulates time.
        "map":  map-style update v <- Jv, denom accumulates steps.
        "auto": infer from model.spec.kind ("ode" -> flow, "map" -> map).
        
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
        mode_use = _resolve_mode(mode=mode, model_like=model_coerced, who="lyapunov_mle")

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
            who="lyapunov_mle",
        )

        return _LyapunovModule(
            jvp=jvp_use,
            n_state=int(n_state_use),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            mode=mode_use,
        )

    # Build immediately when model or jvp/n_state is supplied.
    if model is not None:
        return _build_with_model(model)
    if jvp is not None:
        if n_state is None:
            raise ValueError("lyapunov_mle requires n_state when jvp is provided without a model")
        if mode == "auto":
            raise ValueError("lyapunov_mle mode='auto' requires a model to infer kind")
        plan_use = _resolve_trace_plan(
            trace_plan=trace_plan, record_interval=record_interval, who="lyapunov_mle"
        )
        return _LyapunovModule(
            jvp=jvp,
            n_state=int(n_state),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            mode=_resolve_mode(mode=mode, model_like=None, who="lyapunov_mle"),
        )

    def _factory(model: object) -> AnalysisModule:
        """Factory function invoked by Sim with model injected."""
        return _build_with_model(model)

    # Otherwise return factory for Sim to call
    return _factory


# -------------------------- spectrum hooks -----------------------------------

def _make_hooks_spectrum(
    *,
    jvp_fn: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
    init_basis: np.ndarray,  # shape (n_state, k)
    n_state: int,
    k: int,
    mode: int,  # 0: flow (Euler), 1: map (J*v)
) -> AnalysisHooks:
    """
    Workspace layout:
        analysis_ws[0 : n_state*k]               -> V (current orthonormal basis), shape (n_state, k)
        analysis_ws[n_state*k : 2*n_state*k]     -> W (work), shape (n_state, k)

    Output layout (length k+2):
        analysis_out[0:k]     -> accum_log_diag[j] = sum log(R_jj)
        analysis_out[k]       -> denom (total_time for flow, steps for map)
        analysis_out[k+1]     -> steps (integer-ish stored as float)
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
        if step != 0:
            return
        need_ws = 2 * n_state * k
        need_out = k + 2
        if analysis_ws.shape[0] < need_ws or analysis_out.shape[0] < need_out:
            return

        V = analysis_ws[: n_state * k].reshape((n_state, k))
        W = analysis_ws[n_state * k : 2 * n_state * k].reshape((n_state, k))

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
        need_ws = 2 * n_state * k
        need_out = k + 2
        if analysis_ws.shape[0] < need_ws or analysis_out.shape[0] < need_out:
            return

        V = analysis_ws[: n_state * k].reshape((n_state, k))
        W = analysis_ws[n_state * k : 2 * n_state * k].reshape((n_state, k))

        # -------- propagate tangent basis columns into W --------
        # tmp_out is W[:, j], reused per column.
        for j in range(k):
            # Compute J*v into W[:, j]
            v_col = V[:, j]
            w_col = W[:, j]
            jvp_fn(t, y_curr, params, v_col, w_col, runtime_ws)

            if mode == 0:
                # flow: Euler step for variational equation: v <- v + dt * (J*v)
                # Use W as the "new vectors"
                for i in range(n_state):
                    w_col[i] = v_col[i] + dt * w_col[i]
            else:
                # map: W already holds J*v
                # nothing else to do
                pass

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

    return AnalysisHooks(pre_step=_pre_step, post_step=_post_step)


# --------------------------- spectrum module ---------------------------------

class _LyapunovSpectrumModule(AnalysisModule):
    def __init__(
        self,
        *,
        jvp: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None],
        n_state: int,
        k: int,
        mode: Literal["flow", "map"],
        trace_plan: FixedTracePlan,
        analysis_kind: int,
        init_basis: Optional[np.ndarray] = None,
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
        self._n_state = int(n_state)
        self._k = int(k)
        self._mode = 0 if mode == "flow" else 1

        if init_basis is None:
            B = _default_basis(n_state, k)
        else:
            B = np.asarray(init_basis, dtype=float)
            if B.shape != (n_state, k):
                raise ValueError(f"init_basis must have shape ({n_state}, {k})")
        self._init_basis = B

        hooks = _make_hooks_spectrum(
            jvp_fn=jvp,
            init_basis=B,
            n_state=n_state,
            k=k,
            mode=self._mode,
        )

        reqs = AnalysisRequirements(fixed_step=True, need_jvp=True, mutates_state=False)

        # outputs: accum logs + denom + steps
        output_names = tuple([f"log_r{i}" for i in range(k)] + ["denom", "steps"])
        trace_names = tuple([f"lyap{i}" for i in range(k)])

        super().__init__(
            name="lyapunov_spectrum",
            requirements=reqs,
            workspace_size=2 * n_state * k,
            output_size=k + 2,
            output_names=output_names,
            trace_names=trace_names,
            trace=TraceSpec(width=k, plan=trace_plan),
            hooks=hooks,
            analysis_kind=analysis_kind,
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
            raise RuntimeError("lyapunov_spectrum requires numba for jit hooks") from exc
        try:
            self._jvp_jit = njit(cache=False)(py_target)
        except Exception as exc:
            raise RuntimeError(
                "lyapunov_spectrum requires a numba-compatible JVP for JIT execution"
            ) from exc
        return self._jvp_jit

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> AnalysisHooks:
        if not jit:
            return self.hooks
        key = self._jit_key(dtype)
        cached = self._jit_cache.get(key)
        if cached is not None:
            return cached

        jvp_jit = self._ensure_jit_jvp()
        jit_hooks = _make_hooks_spectrum(
            jvp_fn=jvp_jit,
            init_basis=self._init_basis,
            n_state=self._n_state,
            k=self._k,
            mode=self._mode,
        )
        return self._compile_hooks(jit_hooks, dtype)


# ---------------------------- spectrum factory -------------------------------

def lyapunov_spectrum(
    *,
    model=None,
    jvp: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object], None]
    ] = None,
    n_state: Optional[int] = None,
    k: int = 2,
    mode: Literal["flow", "map", "auto"] = "auto",
    init_basis: Optional[np.ndarray] = None,
    trace_plan: Optional[FixedTracePlan] = None,
    record_interval: Optional[int] = None,
    analysis_kind: int = 1,
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
    """

    def _infer_n_state(target) -> int | None:
        spec = getattr(target, "spec", None)
        if spec is None or getattr(spec, "states", None) is None:
            return None
        return len(spec.states)

    def _build_with_model(model_obj: object) -> _LyapunovSpectrumModule:
        model_coerced = _coerce_model(model_obj)
        if model_coerced is None:
            raise ValueError("lyapunov_spectrum factory requires a model")
        mode_use = _resolve_mode(mode=mode, model_like=model_coerced, who="lyapunov_spectrum")

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
            who="lyapunov_spectrum",
        )

        return _LyapunovSpectrumModule(
            jvp=jvp_use,
            n_state=int(n_state_use),
            k=int(k),
            mode=mode_use,
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            init_basis=init_basis,
        )

    if model is not None:
        return _build_with_model(model)

    if jvp is not None:
        if n_state is None:
            raise ValueError("lyapunov_spectrum requires n_state when jvp is provided without a model")
        if mode == "auto":
            raise ValueError("lyapunov_spectrum mode='auto' requires a model to infer kind")
        plan_use = _resolve_trace_plan(
            trace_plan=trace_plan,
            record_interval=record_interval,
            who="lyapunov_spectrum",
        )
        return _LyapunovSpectrumModule(
            jvp=jvp,
            n_state=int(n_state),
            k=int(k),
            mode=_resolve_mode(mode=mode, model_like=None, who="lyapunov_spectrum"),
            trace_plan=plan_use,
            analysis_kind=analysis_kind,
            init_basis=init_basis,
        )

    # Factory path (model injected by Sim)
    def _factory(model: object) -> AnalysisModule:
        return _build_with_model(model)

    return _factory
