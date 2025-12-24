# src/dynlib/analysis/runtime/core.py
"""Runtime analysis infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from dynlib.runtime.fastpath.plans import TracePlan

__all__ = [
    "AnalysisRequirements",
    "AnalysisHooks",
    "TraceSpec",
    "AnalysisModule",
    "CombinedAnalysis",
    "analysis_noop_hook",
]


@dataclass(frozen=True)
class AnalysisRequirements:
    """Declarative requirements for a runtime analysis module."""

    fixed_step: bool = False
    need_jvp: bool = False
    need_dense_jacobian: bool = False
    need_jacobian: bool = False  # legacy flag; treated as dense requirement
    requires_event_log: bool = False
    accept_reject: bool = False
    mutates_state: bool = False


@dataclass(frozen=True)
class AnalysisHooks:
    """
    Pre/post hooks invoked by the runner or wrapper.

    Signature (for both pre_step and post_step):
        hook(
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
        ) -> None
    """

    pre_step: Optional[Callable[..., None]] = None
    post_step: Optional[Callable[..., None]] = None


@dataclass(frozen=True)
class TraceSpec:
    """Trace layout for an analysis module."""

    width: int
    plan: Optional[TracePlan] = None

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError("trace width must be non-negative")
        if self.width > 0 and self.plan is None:
            raise ValueError("TraceSpec requires a TracePlan when width > 0")
        if self.plan is not None and self.plan.record_interval() <= 0:
            raise ValueError("TracePlan stride must be positive")

    def capacity(self, *, total_steps: int | None) -> int:
        if self.plan is None:
            return 0
        return int(self.plan.capacity(total_steps=total_steps))

    def record_interval(self) -> int:
        if self.plan is None:
            return 0
        return int(self.plan.record_interval())

    def finalize_index(self, filled: int) -> slice | None:
        return self.plan.finalize_index(filled) if self.plan else None

    def hit_limit(self) -> int | None:
        return self.plan.hit_limit() if self.plan else None


@dataclass(frozen=True)
class AnalysisModule:
    """Single runtime analysis with optional JIT dispatch."""

    name: str
    requirements: AnalysisRequirements
    workspace_size: int
    output_size: int
    output_names: Tuple[str, ...] | None = None
    trace: Optional[TraceSpec] = None
    trace_names: Tuple[str, ...] | None = None
    hooks: AnalysisHooks = AnalysisHooks()
    analysis_kind: int = 1
    _jit_cache: dict[tuple[int, str, int, int], AnalysisHooks] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    @property
    def needs_trace(self) -> bool:
        return self.trace is not None and self.trace.width > 0

    @property
    def trace_stride(self) -> int:
        return self.trace.record_interval() if self.trace else 0

    def trace_capacity(self, *, total_steps: int | None) -> int:
        return self.trace.capacity(total_steps=total_steps) if self.trace else 0

    def finalize_trace(self, filled: int) -> slice | None:
        return self.trace.finalize_index(filled) if self.trace else None

    def hit_cap(self) -> int | None:
        return self.trace.hit_limit() if self.trace else None

    def supports_fastpath(
        self,
        *,
        adaptive: bool,
        has_event_logs: bool,
        has_jvp: bool,
        has_dense_jacobian: bool,
    ) -> tuple[bool, str | None]:
        """
        Lightweight capability gate used by fast-path assess_capability().
        """
        if self.requirements.fixed_step and adaptive:
            return False, "analysis requires fixed-step"
        if self.requirements.requires_event_log:
            return False, "analysis requires event logs"
        if self.requirements.need_jvp and not has_jvp:
            return False, "analysis requires a model Jacobian-vector product"
        if self.requirements.need_dense_jacobian and not has_dense_jacobian:
            return False, "analysis requires a model Jacobian"
        if self.requirements.need_jacobian and not (has_dense_jacobian or has_jvp):
            return False, "analysis requires a model Jacobian"
        if self.requirements.accept_reject:
            return False, "analysis with accept/reject hooks not supported on fast path"
        if self.needs_trace and self.trace is None:
            return False, "analysis trace missing plan"
        if self.requirements.mutates_state:
            return False, "analysis mutates state"
        return True, None

    def _jit_key(self, dtype: np.dtype) -> tuple[int, str, int, int]:
        trace_width = self.trace.width if self.trace else 0
        trace_stride = self.trace_stride
        return (id(self), str(np.dtype(dtype)), trace_width, trace_stride)

    def _compile_hooks(self, hooks: AnalysisHooks, dtype: np.dtype) -> AnalysisHooks:
        try:  # pragma: no cover - numba may be missing
            from numba import njit  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                f"Analysis '{self.name}' requested jit hooks but numba is not installed"
            ) from exc

        def _jit(fn: Optional[Callable[..., None]]) -> Optional[Callable[..., None]]:
            if fn is None:
                return None
            try:
                return njit(cache=True)(fn)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to njit analysis hook '{self.name}.{getattr(fn, '__name__', 'hook')}' "
                    f"in nopython mode"
                ) from exc

        compiled = AnalysisHooks(
            pre_step=_jit(hooks.pre_step),
            post_step=_jit(hooks.post_step),
        )
        key = self._jit_key(dtype)
        self._jit_cache[key] = compiled
        return compiled

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> AnalysisHooks:
        """
        Return dispatch hooks for the requested execution mode.

        jit=True compiles the authored hooks with numba (nopython) and caches the
        result per (module, dtype, trace shape). jit=False returns the Python hooks.
        """
        if not jit:
            return self.hooks
        key = self._jit_key(dtype)
        cached = self._jit_cache.get(key)
        if cached is not None:
            return cached
        return self._compile_hooks(self.hooks, dtype)


class CombinedAnalysis(AnalysisModule):
    """Pack multiple analyses into a single analysis_kind with merged buffers."""

    def __init__(self, modules: Sequence[AnalysisModule], *, analysis_kind: int = 1):
        if not modules:
            raise ValueError("CombinedAnalysis requires at least one module")
        self.modules: tuple[AnalysisModule, ...] = tuple(modules)
        req = self._merge_requirements(modules)
        trace_spec = self._merge_trace_specs(modules)
        workspace = sum(mod.workspace_size for mod in modules)
        outputs = sum(mod.output_size for mod in modules)
        output_names = self._merge_names(tuple(mod.output_names or tuple() for mod in modules))
        trace_names = self._merge_names(tuple(mod.trace_names or tuple() for mod in modules))

        python_hooks = AnalysisHooks(
            pre_step=self._compose_hook(modules, hooks=[m.hooks for m in modules], phase="pre"),
            post_step=self._compose_hook(modules, hooks=[m.hooks for m in modules], phase="post"),
        )

        super().__init__(
            name="combined",
            requirements=req,
            workspace_size=workspace,
            output_size=outputs,
            trace=trace_spec,
            output_names=output_names,
            trace_names=trace_names,
            hooks=python_hooks,
            analysis_kind=analysis_kind,
        )

    @staticmethod
    def _merge_requirements(modules: Sequence[AnalysisModule]) -> AnalysisRequirements:
        fixed_step = any(mod.requirements.fixed_step for mod in modules)
        need_jvp = any(mod.requirements.need_jvp for mod in modules)
        need_dense_jacobian = any(mod.requirements.need_dense_jacobian for mod in modules)
        need_jacobian = any(mod.requirements.need_jacobian for mod in modules)
        requires_event_log = any(mod.requirements.requires_event_log for mod in modules)
        accept_reject = any(mod.requirements.accept_reject for mod in modules)
        mutates_state = any(mod.requirements.mutates_state for mod in modules)
        return AnalysisRequirements(
            fixed_step=fixed_step,
            need_jvp=need_jvp,
            need_dense_jacobian=need_dense_jacobian,
            need_jacobian=need_jacobian,
            requires_event_log=requires_event_log,
            accept_reject=accept_reject,
            mutates_state=mutates_state,
        )

    @staticmethod
    def _merge_trace_specs(modules: Sequence[AnalysisModule]) -> Optional[TraceSpec]:
        specs = [mod.trace for mod in modules if mod.trace is not None]
        if not specs:
            return None
        plan = specs[0].plan
        if any(spec.plan != plan for spec in specs):
            raise ValueError("CombinedAnalysis requires all trace plans to match")
        width = sum(spec.width for spec in specs)
        return TraceSpec(width=width, plan=plan)

    @staticmethod
    def _merge_names(name_sets: Tuple[Tuple[str, ...], ...]) -> Tuple[str, ...] | None:
        if not name_sets:
            return None
        flat = tuple(name for names in name_sets for name in names)
        return flat if flat else None

    @staticmethod
    def _compute_offsets(modules: Sequence[AnalysisModule]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute workspace/output/trace offsets per module for jit-safe composition.
        """
        ws_offsets = [0]
        out_offsets = [0]
        trace_offsets = [0]
        trace_widths = []
        for mod in modules:
            ws_offsets.append(ws_offsets[-1] + mod.workspace_size)
            out_offsets.append(out_offsets[-1] + mod.output_size)
            width = mod.trace.width if mod.trace else 0
            trace_offsets.append(trace_offsets[-1] + width)
            trace_widths.append(width)
        return (
            np.asarray(ws_offsets, dtype=np.int64),
            np.asarray(out_offsets, dtype=np.int64),
            np.asarray(trace_offsets, dtype=np.int64),
            np.asarray(trace_widths, dtype=np.int64),
        )

    @staticmethod
    def _compose_hook(
        modules: Sequence[AnalysisModule],
        *,
        hooks: Sequence[AnalysisHooks],
        phase: str,
    ):
        hook_name = "pre_step" if phase == "pre" else "post_step"

        def _hook(
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
            ws_offset = 0
            out_offset = 0
            trace_offset = 0
            for mod, hook_set in zip(modules, hooks):
                fn = getattr(hook_set, hook_name) if hook_set else None
                if fn is None:
                    ws_offset += mod.workspace_size
                    out_offset += mod.output_size
                    if mod.trace is not None:
                        trace_offset += mod.trace.width
                    continue

                ws_view = analysis_ws[ws_offset : ws_offset + mod.workspace_size]
                out_view = analysis_out[out_offset : out_offset + mod.output_size]
                if mod.trace is not None and trace_buf.size:
                    trace_view = trace_buf[:, trace_offset : trace_offset + mod.trace.width]
                else:
                    trace_view = np.zeros((0, 0), dtype=trace_buf.dtype)

                fn(
                    t,
                    dt,
                    step,
                    y_curr,
                    y_prev,
                    params,
                    runtime_ws,
                    ws_view,
                    out_view,
                    trace_view,
                    trace_count,
                    trace_cap,
                    trace_stride,
                )
                ws_offset += mod.workspace_size
                out_offset += mod.output_size
                if mod.trace is not None:
                    trace_offset += mod.trace.width

        return _hook

    @staticmethod
    def _compose_hook_jit(
        *,
        hooks: Sequence[AnalysisHooks],
        ws_offsets: np.ndarray,
        out_offsets: np.ndarray,
        trace_offsets: np.ndarray,
        trace_widths: np.ndarray,
    ):
        """
        JIT-friendly hook that only closes over primitive offsets and compiled callables.
        """
        try:  # pragma: no cover - import guard for numba tooling
            from numba import literal_unroll  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CombinedAnalysis with jit=True requires numba") from exc

        noop = analysis_noop_hook()
        hook_funcs = tuple(h.pre_step or noop for h in hooks)

        def _hook(
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
            idx = 0
            for fn in literal_unroll(hook_funcs):
                i = idx
                ws0 = int(ws_offsets[i])
                ws1 = int(ws_offsets[i + 1])
                out0 = int(out_offsets[i])
                out1 = int(out_offsets[i + 1])
                trace_width = int(trace_widths[i])
                trace0 = int(trace_offsets[i])
                if trace_width > 0 and trace_buf.shape[0] > 0:
                    trace_view = trace_buf[:, trace0 : trace0 + trace_width]
                else:
                    trace_view = trace_buf[:0, :0]
                fn(
                    t,
                    dt,
                    step,
                    y_curr,
                    y_prev,
                    params,
                    runtime_ws,
                    analysis_ws[ws0:ws1],
                    analysis_out[out0:out1],
                    trace_view,
                    trace_count,
                    trace_cap,
                    trace_stride,
                )
                idx += 1

        return _hook

    @staticmethod
    def _compose_hook_jit_post(
        *,
        hooks: Sequence[AnalysisHooks],
        ws_offsets: np.ndarray,
        out_offsets: np.ndarray,
        trace_offsets: np.ndarray,
        trace_widths: np.ndarray,
    ):
        try:  # pragma: no cover - import guard for numba tooling
            from numba import literal_unroll  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CombinedAnalysis with jit=True requires numba") from exc

        noop = analysis_noop_hook()
        hook_funcs = tuple(h.post_step or noop for h in hooks)

        def _hook(
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
            idx = 0
            for fn in literal_unroll(hook_funcs):
                i = idx
                ws0 = int(ws_offsets[i])
                ws1 = int(ws_offsets[i + 1])
                out0 = int(out_offsets[i])
                out1 = int(out_offsets[i + 1])
                trace_width = int(trace_widths[i])
                trace0 = int(trace_offsets[i])
                if trace_width > 0 and trace_buf.shape[0] > 0:
                    trace_view = trace_buf[:, trace0 : trace0 + trace_width]
                else:
                    trace_view = trace_buf[:0, :0]
                fn(
                    t,
                    dt,
                    step,
                    y_curr,
                    y_prev,
                    params,
                    runtime_ws,
                    analysis_ws[ws0:ws1],
                    analysis_out[out0:out1],
                    trace_view,
                    trace_count,
                    trace_cap,
                    trace_stride,
                )
                idx += 1

        return _hook

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> AnalysisHooks:
        if not jit:
            return self.hooks
        ws_offsets, out_offsets, trace_offsets, trace_widths = self._compute_offsets(self.modules)
        compiled_children = [mod.resolve_hooks(jit=True, dtype=dtype) for mod in self.modules]
        composed = AnalysisHooks(
            pre_step=self._compose_hook_jit(
                hooks=compiled_children,
                ws_offsets=ws_offsets,
                out_offsets=out_offsets,
                trace_offsets=trace_offsets,
                trace_widths=trace_widths,
            ),
            post_step=self._compose_hook_jit_post(
                hooks=compiled_children,
                ws_offsets=ws_offsets,
                out_offsets=out_offsets,
                trace_offsets=trace_offsets,
                trace_widths=trace_widths,
            ),
        )
        key = self._jit_key(dtype)
        cached = self._jit_cache.get(key)
        if cached is not None:
            return cached
        return self._compile_hooks(composed, dtype)


@lru_cache(maxsize=1)
def analysis_noop_hook():
    """
    Return a no-op hook compatible with JIT runners.

    Compiled with numba when available to keep runner typing happy.
    """
    try:  # pragma: no cover - numba may be missing
        from numba import njit  # type: ignore

        @njit(inline="always")
        def _noop(
            t: float,
            dt: float,
            step: int,
            y_curr,
            y_prev,
            params,
            runtime_ws,
            analysis_ws,
            analysis_out,
            trace_buf,
            trace_count,
            trace_cap: int,
            trace_stride: int,
        ) -> None:
            return None

        return _noop
    except Exception:  # pragma: no cover - fallback when numba absent
        def _noop(
            t: float,
            dt: float,
            step: int,
            y_curr,
            y_prev,
            params,
            runtime_ws,
            analysis_ws,
            analysis_out,
            trace_buf,
            trace_count,
            trace_cap: int,
            trace_stride: int,
        ) -> None:
            return None

        return _noop
