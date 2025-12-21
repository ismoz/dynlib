# src/dynlib/compiler/codegen/runner_discrete.py
"""
Discrete runner for maps and difference equations.

Similar to the ODE runner but adapted for discrete systems:
- Terminates on max_steps (N) instead of time horizon (T)
- Time advances exactly as t = t0 + step * dt (no accumulation error)
- Post-events/logging happen before committing state to avoid dropped logs
- No adaptive stepping support (maps use fixed iteration counts)
"""
from __future__ import annotations
import math
import math

import inspect
import platform
import textwrap
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple
import tomllib

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata  # type: ignore

# Import status codes from canonical source
from dynlib.runtime.runner_api import (
    OK, STEPFAIL, NAN_DETECTED,
    DONE, GROW_REC, GROW_EVT, USER_BREAK
)

# Import centralized JIT compilation helper
from dynlib.compiler.jit.compile import jit_compile
from dynlib.compiler.codegen._runner_cache import (
    RunnerCacheRequest,
    RunnerCacheConfig,
    RunnerDiskCache,
    DiskCacheUnavailable,
    gather_env_pins,
)

__all__ = [
    "runner_discrete",
    "get_runner_discrete",
    "configure_runner_disk_cache_discrete",
    "last_runner_cache_hit_discrete",
]

# Import guards from centralized module for pure Python mode
# When JIT-compiled, guards are inlined in the rendered source
from dynlib.compiler.guards import allfinite1d as _allfinite1d_py, allfinite_scalar as _allfinite_scalar_py
from dynlib.runtime.softdeps import softdeps

_SOFTDEPS = softdeps()
_NUMBA_AVAILABLE = _SOFTDEPS.numba
_NUMBA_VERSION = _SOFTDEPS.numba_version
_LLVMLITE_VERSION = _SOFTDEPS.llvmlite_version

if _NUMBA_AVAILABLE:
    from numba import njit as _njit_guards  # type: ignore

    # Pre-compile guards for in-memory JIT fallback
    allfinite1d = _njit_guards(inline='always')(_allfinite1d_py)
    allfinite_scalar = _njit_guards(inline='always')(_allfinite_scalar_py)
    import numba  # type: ignore
else:
    # Numba not available, use pure Python versions
    allfinite1d = _allfinite1d_py
    allfinite_scalar = _allfinite_scalar_py
    numba = None  # type: ignore


def _discover_dynlib_version() -> str:
    """Best-effort dynlib version lookup."""
    version = _read_pyproject_version()
    if version is not None:
        return version
    try:
        return importlib_metadata.version("dynlib")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0+local"


def _read_pyproject_version() -> Optional[str]:
    root = Path(__file__).resolve()
    for parent in root.parents:
        candidate = parent / "pyproject.toml"
        if not candidate.exists():
            continue
        try:
            with open(candidate, "rb") as fh:
                data = tomllib.load(fh)
        except Exception:
            continue
        project = data.get("project", {})
        version = project.get("version")
        if isinstance(version, str):
            return version
    return None


_DYNLIB_VERSION = _discover_dynlib_version()


def runner_discrete(
    # scalars
    t0, N, dt_init,
    max_steps, n_state, record_interval,
    # state/params
    y_curr, y_prev, params,
    # workspaces
    runtime_ws,
    stepper_ws,
    # stepper configuration (read-only)
    stepper_config,
    # proposals/outs (len-1 arrays where applicable)
    y_prop, t_prop, dt_next, err_est,
    # recording
    T, Y, AUX, STEP, FLAGS,
    # event log (present; cap may be 1 if disabled)
    EVT_CODE, EVT_INDEX, EVT_LOG_DATA,
    # event log scratch (for writing log values before copying)
    evt_log_scratch,
    # analysis buffers/dispatch
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    analysis_kind,
    analysis_dispatch_pre, analysis_dispatch_post,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # NEW: selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Discrete-time runner: iteration-based execution with events and recording.
    
    Frozen ABI signature - mirrors continuous runner but uses N instead of t_end.
    
    Key behavior:
      - Loop terminates at step >= N (not time-based)
      - Time is computed exactly: t = t0 + step * dt (no accumulation drift)
      - dt never clipped (always constant label spacing)
      - Supports all features: events, recording, workspace, stepper config
    
    Returns status code (int32).
    """
    has_analysis = analysis_kind != 0
    pre_hook = analysis_dispatch_pre
    post_hook = analysis_dispatch_post
    trace_cap_int = int(analysis_trace_cap) if has_analysis else 0
    trace_stride_int = int(analysis_trace_stride) if has_analysis else 0
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)         # record cursor
    step = int(step_start)   # global step counter (iteration count)
    
    # Event log cursor: hint_out[0] is used to pass m between re-entries
    # On first call, hint_out[0] is 0; on re-entry after GROW_EVT, it contains the saved m
    m = int(hint_out[0])     # event log cursor (resume from hint)
    
    # Refresh aux values before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0:
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0 (if record_interval > 0)
    if record_interval > 0 and step == 0:
        # Record initial condition
        if i >= cap_rec:
            # Need growth before recording
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = GROW_REC
            hint_out[0] = m
            return GROW_REC
        
        T[i] = t
        # Record selected states
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        # Record selected aux (if any)
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1
    
    # Main iteration loop - DISCRETE: only step count matters, not time
    while step < N:
        # Check if we need to record a pending step from before growth
        # This happens when step_start > i_start (we've advanced steps but not recorded)
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            # Re-entering after GROW_REC: attempt the pending record first
            if i >= cap_rec:
                # Still not enough space (should not happen with geometric growth)
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            # Record selected states
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            # Record selected aux (if any)
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1
            
        # 1. Pre-events on committed state
        event_code_pre, log_width_pre = events_pre(t, y_curr, params, evt_log_scratch, runtime_ws)
        
        # Record pre-event if it fired and has log data
        if event_code_pre >= 0 and log_width_pre > 0:
            if m >= cap_evt:
                # Need event buffer growth
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            # Copy log data to buffers
            for log_idx in range(log_width_pre):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            
            EVT_CODE[m] = event_code_pre
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        if has_analysis:
            pre_hook(
                t, dt, step,
                y_curr, y_prev, params,
                runtime_ws,
                analysis_ws, analysis_out, analysis_trace,
                analysis_trace_count, trace_cap_int, trace_stride_int,
            )
        
        # 2. NO dt clipping for discrete systems - dt is constant label spacing
        # Time is a derived label, not a termination criterion
        
        # 3. Stepper attempt (map: single evaluation, no internal iteration)
        step_status = stepper(
            t, dt, y_curr, rhs, params,
            runtime_ws,
            stepper_ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        )
        
        # Check for stepper failure/termination
        # Steppers return: OK (accepted step) or terminal codes (STEPFAIL, NAN_DETECTED)
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        # Universal guard: never commit non-finite proposals
        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED
        
        # Guard scalar outputs (t_prop, dt_next)
        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED
        
        # 4. Post-events on proposed state/time (may mutate y_prop in-place)
        next_step = step + 1
        t_post = t0 + next_step * dt
        event_code_post, log_width_post = events_post(
            t_post, y_prop, params, evt_log_scratch, runtime_ws
        )
        
        # Record post-event if it fired and has log data
        if event_code_post >= 0 and log_width_post > 0:
            if m >= cap_evt:
                # Need event buffer growth; keep committed state untouched
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            # Copy log data to buffers
            for log_idx in range(log_width_post):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            
            EVT_CODE[m] = event_code_post
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1
        
        # 5. Commit: y_prev <- y_curr, y_curr <- y_prop, step++
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step = next_step
        
        # Update aux values from committed state (if any aux variables exist)
        if runtime_ws.aux_values.shape[0] > 0:
            update_aux(t_post, y_curr, params, runtime_ws.aux_values, runtime_ws)
        
        lag_info = runtime_ws.lag_info
        if lag_info.shape[0] > 0:
            lag_ring = runtime_ws.lag_ring
            lag_head = runtime_ws.lag_head
            for j in range(lag_info.shape[0]):
                state_idx = lag_info[j, 0]
                depth = lag_info[j, 1]
                offset = lag_info[j, 2]
                head = int(lag_head[j]) + 1
                if head >= depth:
                    head = 0
                lag_head[j] = head
                lag_ring[offset + head] = y_curr[state_idx]

        if has_analysis:
            post_hook(
                t, dt, step,
                y_curr, y_prev, params,
                runtime_ws,
                analysis_ws, analysis_out, analysis_trace,
                analysis_trace_count, trace_cap_int, trace_stride_int,
            )
        
        # Compute time exactly (no accumulation) - KEY DIFFERENCE from continuous
        t = t_post
        # dt remains constant (from dt_next, but for maps it's always dt_init)
        dt = dt_next[0]
        
        # 6. Record (if enabled and step matches record_interval)
        if record_interval > 0 and (step % record_interval == 0):
            if i >= cap_rec:
                # Need growth
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            # Record selected states
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            # Record selected aux (if any)
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1
        
        # Check user break flag (if implemented)
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK
    
    # Successful completion (reached N iterations)
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE


_pending_cache_request: Optional[RunnerCacheRequest] = None
_inproc_runner_cache: Dict[str, Callable] = {}
_warned_reasons: set[str] = set()
_last_runner_cache_hit_discrete: bool = False


def _env_pins(platform_token: str) -> Dict[str, str]:
    cpu_name = platform.processor() or platform.machine()
    return gather_env_pins(
        platform_token=platform_token,
        dynlib_version=_DYNLIB_VERSION,
        python_version=platform.python_version(),
        numba_version=_NUMBA_VERSION,
        llvmlite_version=_LLVMLITE_VERSION,
        cpu_name=cpu_name,
    )


def configure_runner_disk_cache_discrete(
    *,
    spec_hash: str,
    stepper_name: str,
    structsig: Tuple[int, ...],
    dtype: str,
    cache_root: Path,
) -> None:
    """Store the cache context for the next disk-backed discrete runner build."""
    global _pending_cache_request
    _pending_cache_request = RunnerCacheRequest(
        spec_hash=spec_hash,
        stepper_name=stepper_name,
        structsig=tuple(int(x) for x in structsig),
        dtype=str(dtype),
        cache_root=Path(cache_root).expanduser().resolve(),
    )



def _consume_cache_request() -> Optional[RunnerCacheRequest]:
    global _pending_cache_request
    req = _pending_cache_request
    _pending_cache_request = None
    return req


def last_runner_cache_hit_discrete() -> bool:
    """Return whether the last discrete runner build was served from cache."""
    return _last_runner_cache_hit_discrete


def get_runner_discrete(
    *,
    jit: bool = True,
    disk_cache: bool = True,
) -> Callable:
    """
    Return the discrete runner function, optionally JIT-compiled with disk caching.
    
    Args:
        jit: If True, compile with numba. If False, return pure Python.
        disk_cache: If True and jit=True, attempt to use disk-backed cache.
    
    Returns:
        Callable runner_discrete function (jitted or pure Python).
    """
    global _last_runner_cache_hit_discrete
    _last_runner_cache_hit_discrete = False
    
    if not jit:
        return runner_discrete
    
    if not disk_cache:
        return jit_compile(runner_discrete, jit=True).fn

    if not _NUMBA_AVAILABLE:
        return jit_compile(runner_discrete, jit=True).fn

    request = _consume_cache_request()
    if request is None:
        raise RuntimeError(
            "get_runner_discrete(disk_cache=True) called without configure_runner_disk_cache_discrete()"
        )

    cache_config = RunnerCacheConfig(
        module_prefix="dynlib_runner_discrete",
        export_name="runner_discrete",
        render_module_source=_render_runner_module_source_discrete,
        env_pins_factory=_env_pins,
    )

    cache = RunnerDiskCache(
        request,
        inproc_cache=_inproc_runner_cache,
        config=cache_config,
    )
    try:
        cached, from_disk = cache.get_or_build()
        _last_runner_cache_hit_discrete = from_disk
        return cached
    except DiskCacheUnavailable as exc:
        _warn_disk_cache_disabled(str(exc))
        _last_runner_cache_hit_discrete = False
        return jit_compile(runner_discrete, jit=True).fn


def _render_runner_module_source_discrete(request: RunnerCacheRequest) -> str:
    runner_src = textwrap.dedent(inspect.getsource(runner_discrete)).lstrip()
    decorated = runner_src.replace(
        "def runner_discrete(", "@njit(cache=True)\ndef runner_discrete(", 1
    )
    
    # Import guards inline in the module
    from dynlib.compiler.guards import _render_guards_inline_source
    guards_src = _render_guards_inline_source()
    
    header = inspect.cleandoc(
        """
        # Auto-generated by dynlib.compiler.codegen.runner_discrete
        from __future__ import annotations
        import math

        from numba import njit
        from dynlib.runtime.runner_api import (
            OK, STEPFAIL, NAN_DETECTED,
            DONE, GROW_REC, GROW_EVT, USER_BREAK
        )
        __all__ = ["runner_discrete"]
        """
    )
    return f"{header}\n\n{guards_src}\n\n{decorated}\n"


def _warn_disk_cache_disabled(reason: str) -> None:
    if reason in _warned_reasons:
        return
    _warned_reasons.add(reason)
    warnings.warn(
        f"dynlib disk runner cache disabled: {reason}. Falling back to in-memory JIT.",
        RuntimeWarning,
        stacklevel=3,
    )
