"""Runtime analysis infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
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
    need_events: bool = False
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
    allow_growth: bool = True

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
    """Single runtime analysis with optional Python and JIT hooks."""

    name: str
    requirements: AnalysisRequirements
    workspace_size: int
    output_size: int
    output_names: Tuple[str, ...] | None = None
    trace: Optional[TraceSpec] = None
    trace_names: Tuple[str, ...] | None = None
    python_hooks: AnalysisHooks = AnalysisHooks()
    jit_hooks: Optional[AnalysisHooks] = None
    analysis_kind: int = 1

    @property
    def needs_trace(self) -> bool:
        return self.trace is not None and self.trace.width > 0

    @property
    def trace_stride(self) -> int:
        return self.trace.record_interval() if self.trace else 0

    @property
    def allows_growth(self) -> bool:
        if self.trace is None:
            return True
        return bool(self.trace.allow_growth)

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
        if self.requirements.need_events and has_event_logs:
            return False, "analysis incompatible with event logging"
        if self.requirements.need_jvp and not has_jvp:
            return False, "analysis requires a model Jacobian-vector product"
        if self.requirements.need_dense_jacobian and not has_dense_jacobian:
            return False, "analysis requires a model Jacobian"
        if self.requirements.need_jacobian and not (has_dense_jacobian or has_jvp):
            return False, "analysis requires a model Jacobian"
        if self.requirements.accept_reject:
            return False, "analysis with accept/reject hooks not supported on fast path"
        if self.jit_hooks is None:
            return False, "analysis lacks JIT hooks"
        if self.needs_trace and self.trace is None:
            return False, "analysis trace missing plan"
        return True, None


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
            pre_step=self._compose_hook(modules, phase="pre"),
            post_step=self._compose_hook(modules, phase="post"),
        )
        jit_hooks = None
        if all(mod.jit_hooks is not None for mod in modules):
            jit_hooks = AnalysisHooks(
                pre_step=self._compose_hook(modules, phase="pre", use_jit=True),
                post_step=self._compose_hook(modules, phase="post", use_jit=True),
            )

        super().__init__(
            name="combined",
            requirements=req,
            workspace_size=workspace,
            output_size=outputs,
            trace=trace_spec,
            output_names=output_names,
            trace_names=trace_names,
            python_hooks=python_hooks,
            jit_hooks=jit_hooks,
            analysis_kind=analysis_kind,
        )

    @staticmethod
    def _merge_requirements(modules: Sequence[AnalysisModule]) -> AnalysisRequirements:
        fixed_step = any(mod.requirements.fixed_step for mod in modules)
        need_jvp = any(mod.requirements.need_jvp for mod in modules)
        need_dense_jacobian = any(mod.requirements.need_dense_jacobian for mod in modules)
        need_jacobian = any(mod.requirements.need_jacobian for mod in modules)
        need_events = any(mod.requirements.need_events for mod in modules)
        accept_reject = any(mod.requirements.accept_reject for mod in modules)
        mutates_state = any(mod.requirements.mutates_state for mod in modules)
        return AnalysisRequirements(
            fixed_step=fixed_step,
            need_jvp=need_jvp,
            need_dense_jacobian=need_dense_jacobian,
            need_jacobian=need_jacobian,
            need_events=need_events,
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
        allow_growth = all(spec.allow_growth for spec in specs)
        return TraceSpec(width=width, plan=plan, allow_growth=allow_growth)

    @staticmethod
    def _merge_names(name_sets: Tuple[Tuple[str, ...], ...]) -> Tuple[str, ...] | None:
        if not name_sets:
            return None
        flat = tuple(name for names in name_sets for name in names)
        return flat if flat else None

    @staticmethod
    def _compose_hook(
        modules: Sequence[AnalysisModule],
        *,
        phase: str,
        use_jit: bool = False,
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
            for mod in modules:
                hooks = mod.jit_hooks if use_jit else mod.python_hooks
                fn = getattr(hooks, hook_name) if hooks else None
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
