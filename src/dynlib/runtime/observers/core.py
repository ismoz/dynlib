# src/dynlib/runtime/observers/core.py
"""Runtime observer infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from dynlib.runtime.fastpath.plans import TracePlan
from dynlib.runtime.runner_api import OK

__all__ = [
    "ObserverRequirements",
    "ObserverHooks",
    "TraceSpec",
    "ObserverModule",
    "CombinedObserver",
    "observer_noop_hook",
    "observer_noop_variational_step",
]


@dataclass(frozen=True)
class ObserverRequirements:
    """Declarative requirements for a runtime observer module."""

    fixed_step: bool = False
    need_jvp: bool = False
    need_dense_jacobian: bool = False
    need_jacobian: bool = False  # legacy flag; treated as dense requirement
    requires_event_log: bool = False
    accept_reject: bool = False
    mutates_state: bool = False
    variational_in_step: bool = False


@dataclass(frozen=True)
class ObserverHooks:
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
    """Trace layout for an observer module."""

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
class ObserverModule:
    """Single runtime observer with optional JIT dispatch."""

    key: str
    name: str
    requirements: ObserverRequirements
    workspace_size: int
    output_size: int
    output_names: Tuple[str, ...] | None = None
    trace: Optional[TraceSpec] = None
    trace_names: Tuple[str, ...] | None = None
    hooks: ObserverHooks = ObserverHooks()
    analysis_kind: int = 1
    stop_phase_mask: int = 0
    _jit_cache: dict[tuple[int, str, int, int], ObserverHooks] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    @property
    def needs_trace(self) -> bool:
        return self.trace is not None and self.trace.width > 0

    @property
    def trace_stride(self) -> int:
        return self.trace.record_interval() if self.trace else 0

    def signature(self, dtype: np.dtype) -> tuple:
        """
        Return a hashable signature for cache keying.
        
        The signature must include all parameters that affect the compiled hook
        machine code or semantics. Subclasses should override to include
        observer-specific compile-time constants.
        """
        trace_width = self.trace.width if self.trace else 0
        trace_stride = self.trace_stride
        return (
            self.key,
            self.workspace_size,
            self.output_size,
            trace_width,
            trace_stride,
            str(np.dtype(dtype)),
        )

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
            return False, "observer requires fixed-step"
        if self.requirements.requires_event_log:
            return False, "observer requires event logs"
        if self.requirements.need_jvp and not has_jvp:
            return False, "observer requires a model Jacobian-vector product"
        if self.requirements.need_dense_jacobian and not has_dense_jacobian:
            return False, "observer requires a model Jacobian"
        if self.requirements.need_jacobian and not (has_dense_jacobian or has_jvp):
            return False, "observer requires a model Jacobian"
        if self.requirements.accept_reject:
            return False, "observer with accept/reject hooks not supported on fast path"
        if self.needs_trace and self.trace is None:
            return False, "observer trace missing plan"
        if self.requirements.mutates_state:
            return False, "observer mutates state"
        return True, None
    
    def validate_stepper(self, stepper_spec) -> None:
        """Optional stepper capability guard (override in subclasses)."""
        return None

    def _jit_key(self, dtype: np.dtype) -> tuple[int, str, int, int]:
        trace_width = self.trace.width if self.trace else 0
        trace_stride = self.trace_stride
        return (id(self), str(np.dtype(dtype)), trace_width, trace_stride)

    def _compile_hooks(self, hooks: ObserverHooks, dtype: np.dtype) -> ObserverHooks:
        try:  # pragma: no cover - numba may be missing
            from numba import njit  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                f"Observer '{self.name}' requested jit hooks but numba is not installed"
            ) from exc

        def _jit(fn: Optional[Callable[..., None]]) -> Optional[Callable[..., None]]:
            if fn is None:
                return None
            try:
                return njit(cache=True)(fn)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to njit observer hook '{self.name}.{getattr(fn, '__name__', 'hook')}' "
                    f"in nopython mode"
                ) from exc

        compiled = ObserverHooks(
            pre_step=_jit(hooks.pre_step),
            post_step=_jit(hooks.post_step),
        )
        key = self._jit_key(dtype)
        self._jit_cache[key] = compiled
        return compiled

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> ObserverHooks:
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


def _duplicate_keys(modules: Sequence[ObserverModule]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for mod in modules:
        if mod.key in seen:
            dupes.add(mod.key)
        seen.add(mod.key)
    return dupes


class CombinedObserver(ObserverModule):
    """Pack multiple observers into a single analysis_kind with merged buffers."""

    def __init__(self, modules: Sequence[ObserverModule], *, analysis_kind: int = 1):
        if not modules:
            raise ValueError("CombinedObserver requires at least one module")
        self.modules: tuple[ObserverModule, ...] = tuple(modules)

        dupes = _duplicate_keys(self.modules)
        if dupes:
            dupes_str = ", ".join(sorted(dupes))
            raise ValueError(f"CombinedObserver requires unique observer keys; duplicates: {dupes_str}")

        if any(mod.requirements.mutates_state for mod in self.modules):
            raise ValueError("CombinedObserver does not support observers that mutate state")

        def _wants_runner_variational(mod: ObserverModule) -> bool:
            if getattr(mod.requirements, "variational_in_step", False):
                return True
            runner_var = getattr(mod, "runner_variational_step", None)
            if callable(runner_var):
                try:
                    return runner_var(jit=False) is not None
                except Exception:
                    return False
            return False

        variational_children = sum(1 for mod in self.modules if _wants_runner_variational(mod))
        if variational_children > 1:
            raise ValueError(
                "Multiple observers require runner-level variational stepping (state+tangent integration). "
                "The runner can accept only one variational integrator per step. Run these observers separately."
            )

        req = self._merge_requirements(modules)
        trace_spec = self._merge_trace_specs(modules)
        workspace = sum(mod.workspace_size for mod in modules)
        outputs = sum(mod.output_size for mod in modules)
        output_names = self._merge_names(tuple(mod.output_names or tuple() for mod in modules))
        trace_names = self._merge_names(tuple(mod.trace_names or tuple() for mod in modules))
        stop_phase_mask = 0
        for mod in modules:
            stop_phase_mask |= int(getattr(mod, "stop_phase_mask", 0))

        python_hooks = ObserverHooks(
            pre_step=self._compose_hook(modules, hooks=[m.hooks for m in modules], phase="pre"),
            post_step=self._compose_hook(modules, hooks=[m.hooks for m in modules], phase="post"),
        )

        super().__init__(
            key="combined",
            name="combined",
            requirements=req,
            workspace_size=workspace,
            output_size=outputs,
            trace=trace_spec,
            output_names=output_names,
            trace_names=trace_names,
            hooks=python_hooks,
            analysis_kind=analysis_kind,
            stop_phase_mask=stop_phase_mask,
        )

    def signature(self, dtype: np.dtype) -> tuple:
        """
        Return a hashable signature including all child module signatures.
        """
        child_sigs = tuple(mod.signature(dtype) for mod in self.modules)
        return ("combined", child_sigs, str(np.dtype(dtype)))
    
    def validate_stepper(self, stepper_spec) -> None:
        for mod in self.modules:
            mod.validate_stepper(stepper_spec)

    @staticmethod
    def _merge_requirements(modules: Sequence[ObserverModule]) -> ObserverRequirements:
        fixed_step = any(mod.requirements.fixed_step for mod in modules)
        need_jvp = any(mod.requirements.need_jvp for mod in modules)
        need_dense_jacobian = any(mod.requirements.need_dense_jacobian for mod in modules)
        need_jacobian = any(mod.requirements.need_jacobian for mod in modules)
        requires_event_log = any(mod.requirements.requires_event_log for mod in modules)
        accept_reject = any(mod.requirements.accept_reject for mod in modules)
        mutates_state = any(mod.requirements.mutates_state for mod in modules)
        variational_in_step = any(mod.requirements.variational_in_step for mod in modules)
        return ObserverRequirements(
            fixed_step=fixed_step,
            need_jvp=need_jvp,
            need_dense_jacobian=need_dense_jacobian,
            need_jacobian=need_jacobian,
            requires_event_log=requires_event_log,
            accept_reject=accept_reject,
            mutates_state=mutates_state,
            variational_in_step=variational_in_step,
        )

    @staticmethod
    def _merge_trace_specs(modules: Sequence[ObserverModule]) -> Optional[TraceSpec]:
        specs = [mod.trace for mod in modules if mod.trace is not None]
        if not specs:
            return None
        plan = specs[0].plan
        if any(spec.plan != plan for spec in specs):
            raise ValueError("CombinedObserver requires all trace plans to match")
        width = sum(spec.width for spec in specs)
        return TraceSpec(width=width, plan=plan)

    @staticmethod
    def _merge_names(name_sets: Tuple[Tuple[str, ...], ...]) -> Tuple[str, ...] | None:
        if not name_sets:
            return None
        flat = tuple(name for names in name_sets for name in names)
        return flat if flat else None

    @staticmethod
    def _compute_offsets(modules: Sequence[ObserverModule]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        modules: Sequence[ObserverModule],
        *,
        hooks: Sequence[ObserverHooks],
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
            # Allow multiple analyses to share one trace buffer without double-counting.
            base_count = int(trace_count[0])
            curr_count = base_count
            tmp_count = np.empty_like(trace_count)
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

                # Reuse the last slot if a prior module already wrote this step.
                start_count = curr_count - 1 if curr_count > base_count else curr_count
                if start_count < 0:
                    start_count = 0
                tmp_count[0] = start_count

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
                    tmp_count,
                    trace_cap,
                    trace_stride,
                )
                new_count = int(tmp_count[0])
                if new_count > trace_cap:
                    curr_count = new_count
                    break
                if new_count > start_count:
                    curr_count = max(curr_count, start_count + 1)
                ws_offset += mod.workspace_size
                out_offset += mod.output_size
                if mod.trace is not None:
                    trace_offset += mod.trace.width
            trace_count[0] = curr_count

        return _hook

    @staticmethod
    def _generate_combined_hook_source(
        n_modules: int,
        ws_offsets: Sequence[int],
        out_offsets: Sequence[int],
        trace_offsets: Sequence[int],
        trace_widths: Sequence[int],
        phase: str,
    ) -> str:
        """
        Generate source code for a combined hook with explicit sequential calls.
        
        This avoids containers of callables - each hook is a separate global symbol
        (HOOK_0, HOOK_1, ...) that is injected at compile time.
        """
        func_name = f"combined_{phase}_hook"
        lines = [
            f"def {func_name}(",
            "    t, dt, step,",
            "    y_curr, y_prev, params,",
            "    runtime_ws,",
            "    analysis_ws, analysis_out, trace_buf,",
            "    trace_count, trace_cap, trace_stride,",
            "):",
            "    base_count = int(trace_count[0])",
            "    curr_count = base_count",
            "    tmp_count = np.empty_like(trace_count)",
        ]
        
        for i in range(n_modules):
            ws0 = ws_offsets[i]
            ws1 = ws_offsets[i + 1]
            out0 = out_offsets[i]
            out1 = out_offsets[i + 1]
            trace_width = trace_widths[i]
            trace0 = trace_offsets[i]
            
            # Generate the slice expressions
            ws_slice = f"analysis_ws[{ws0}:{ws1}]"
            out_slice = f"analysis_out[{out0}:{out1}]"
            
            if trace_width > 0:
                trace_slice = f"trace_buf[:, {trace0}:{trace0 + trace_width}] if trace_buf.shape[0] > 0 else trace_buf[:0, :0]"
            else:
                trace_slice = "trace_buf[:0, :0]"
            lines.append("    if curr_count <= trace_cap:")
            lines.append("        start_count = curr_count - 1 if curr_count > base_count else curr_count")
            lines.append("        if start_count < 0:")
            lines.append("            start_count = 0")
            lines.append("        tmp_count[0] = start_count")
            lines.append(f"        HOOK_{i}(")
            lines.append("            t, dt, step,")
            lines.append("            y_curr, y_prev, params,")
            lines.append("            runtime_ws,")
            lines.append(f"            {ws_slice},")
            lines.append(f"            {out_slice},")
            lines.append(f"            {trace_slice},")
            lines.append("            tmp_count, trace_cap, trace_stride,")
            lines.append("        )")
            lines.append("        new_count = int(tmp_count[0])")
            lines.append("        if new_count > trace_cap:")
            lines.append("            curr_count = new_count")
            lines.append("        elif new_count > start_count and start_count + 1 > curr_count:")
            lines.append("            curr_count = start_count + 1")
            lines.append("")
        lines.append("    trace_count[0] = curr_count")

        return "\n".join(lines)

    @staticmethod
    def _compile_combined_hook(
        source: str,
        func_name: str,
        hooks: Sequence[Callable],
        jit: bool,
    ) -> Callable:
        """
        Compile a combined hook from generated source with hook functions as globals.
        """
        namespace = {"np": np}
        for i, hook in enumerate(hooks):
            namespace[f"HOOK_{i}"] = hook
        
        exec(source, namespace)
        fn = namespace[func_name]
        
        if jit:
            try:
                from numba import njit
                return njit(cache=False)(fn)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to compile combined {func_name} in nopython mode"
                ) from exc
        
        return fn

    def resolve_hooks(self, *, jit: bool, dtype: np.dtype) -> ObserverHooks:
        if not jit:
            return self.hooks
        
        # Check cache first
        key = self._jit_key(dtype)
        cached = self._jit_cache.get(key)
        if cached is not None:
            return cached
        
        # Compute offsets for buffer slicing
        ws_offsets, out_offsets, trace_offsets, trace_widths = self._compute_offsets(self.modules)
        
        # Resolve hooks for all child modules (compiles them if needed)
        compiled_children = [mod.resolve_hooks(jit=True, dtype=dtype) for mod in self.modules]
        
        # Get the noop hook for modules without hooks
        noop = observer_noop_hook()
        
        # Extract pre and post hooks from children
        pre_hooks = [h.pre_step or noop for h in compiled_children]
        post_hooks = [h.post_step or noop for h in compiled_children]
        
        # Generate and compile the combined hooks using codegen (no callable containers)
        pre_source = self._generate_combined_hook_source(
            n_modules=len(self.modules),
            ws_offsets=ws_offsets.tolist(),
            out_offsets=out_offsets.tolist(),
            trace_offsets=trace_offsets.tolist(),
            trace_widths=trace_widths.tolist(),
            phase="pre",
        )
        post_source = self._generate_combined_hook_source(
            n_modules=len(self.modules),
            ws_offsets=ws_offsets.tolist(),
            out_offsets=out_offsets.tolist(),
            trace_offsets=trace_offsets.tolist(),
            trace_widths=trace_widths.tolist(),
            phase="post",
        )
        
        compiled_pre = self._compile_combined_hook(
            pre_source,
            "combined_pre_hook",
            pre_hooks,
            jit=True,
        )
        compiled_post = self._compile_combined_hook(
            post_source,
            "combined_post_hook",
            post_hooks,
            jit=True,
        )
        
        composed = ObserverHooks(pre_step=compiled_pre, post_step=compiled_post)
        self._jit_cache[key] = composed
        return composed


@lru_cache(maxsize=1)
def observer_noop_hook():
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


@lru_cache(maxsize=1)
def observer_noop_variational_step():
    """
    No-op variational stepper matching runner ABI.
    """
    try:  # pragma: no cover - numba may be missing
        from numba import njit  # type: ignore

        @njit(inline="always")
        def _noop(
            t: float,
            dt: float,
            y_curr,
            rhs,
            params,
            runtime_ws,
            stepper_ws,
            stepper_config,
            y_prop,
            t_prop,
            dt_next,
            err_est,
            analysis_ws,
        ):
            return OK

        return _noop
    except Exception:  # pragma: no cover - fallback when numba absent
        def _noop(
            t: float,
            dt: float,
            y_curr,
            rhs,
            params,
            runtime_ws,
            stepper_ws,
            stepper_config,
            y_prop,
            t_prop,
            dt_next,
            err_est,
            analysis_ws,
        ):
            return OK

        return _noop
