# src/dynlib/compiler/codegen/runner_variants.py
"""
Runner variant generator for analysis-aware compilation.

This module generates specialized runner variants with analysis hooks baked in
at compile time as global symbols. This avoids Numba's experimental first-class
function type feature by ensuring hook calls resolve to static global references
rather than cell variables or function arguments.

Key design principles:
1. Hooks are injected as globals (ANALYSIS_PRE, ANALYSIS_POST) not as arguments
2. Runner variants are cached per (model_hash, stepper, analysis_signature)
3. Base runner (no analysis) has zero analysis overhead - no branching
4. CombinedObserver generates explicit sequential calls, not loops over containers
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from functools import lru_cache
from enum import Enum, auto
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from dynlib.compiler.guards import (
    allfinite1d as _allfinite1d_py,
    allfinite_scalar as _allfinite_scalar_py,
)
from dynlib.runtime.softdeps import softdeps
from dynlib.runtime.runner_api import (
    OK, STEPFAIL, NAN_DETECTED,
    EARLY_EXIT, DONE, GROW_REC, GROW_EVT, USER_BREAK, TRACE_OVERFLOW
)
from dynlib.errors import JITUnavailableError
from dynlib.compiler.codegen import runner_cache

if TYPE_CHECKING:
    from dynlib.runtime.observers import ObserverModule, ObserverHooks

__all__ = [
    "RunnerVariant",
    "get_runner",
    "get_runner_variant",
    "get_runner_variant_discrete",
    "clear_variant_cache",
    "analysis_signature_hash",
]

_SOFTDEPS = softdeps()
_NUMBA_AVAILABLE = _SOFTDEPS.numba


def analysis_signature_hash(analysis: Optional["ObserverModule"], dtype: np.dtype) -> str:
    """
    Compute a stable hash for an analysis configuration.
    
    Returns a short hex string suitable for cache keying.
    """
    if analysis is None:
        return "noop"
    sig = analysis.signature(dtype)
    sig_bytes = repr(sig).encode("utf-8")
    return hashlib.sha256(sig_bytes).hexdigest()[:16]


class RunnerVariant(Enum):
    BASE = auto()
    ANALYSIS = auto()
    FASTPATH = auto()
    FASTPATH_ANALYSIS = auto()


_RUNNER_TEMPLATE_VERSION = "v3"


class _LRUVariantCache:
    """
    LRU-bounded cache for runner variants.
    
    Keyed by (model_hash, stepper_name, analysis_sig_hash, runner_type).
    """
    
    def __init__(self, maxsize: int = 64):
        self._maxsize = maxsize
        self._cache: OrderedDict[tuple, Callable] = OrderedDict()
    
    def get(self, key: tuple) -> Optional[Callable]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, key: tuple, value: Callable) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value
    
    def clear(self) -> None:
        self._cache.clear()


# Global variant caches (separate for continuous and discrete)
_variant_cache_continuous = _LRUVariantCache(maxsize=64)
_variant_cache_discrete = _LRUVariantCache(maxsize=64)


def clear_variant_cache() -> None:
    """Clear all cached runner variants."""
    _variant_cache_continuous.clear()
    _variant_cache_discrete.clear()


# -----------------------------------------------------------------------------
# No-op hooks for base runner (inline-able, zero overhead when JIT'd)
# -----------------------------------------------------------------------------

def _noop_hook(
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
    """No-op hook for runners without analysis."""
    pass


@lru_cache(maxsize=1)
def _get_noop_hook_jit():
    """Get JIT-compiled noop hook."""
    if not _NUMBA_AVAILABLE:
        return _noop_hook
    from numba import njit
    return njit(inline="always")(_noop_hook)


# -----------------------------------------------------------------------------
# Runner source templates
# -----------------------------------------------------------------------------

# Base runner for continuous (ODE) systems - NO analysis
_RUNNER_CONTINUOUS_BASE_TEMPLATE = '''
def runner_base(
    # scalars
    t0, t_end, dt_init,
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
    # analysis buffers (unused in base runner but kept for ABI compatibility)
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Base continuous runner: no analysis hooks, zero overhead.
    """
    use_variational_step = bool(variational_step_enabled)
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values / stop flag before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0
    if record_interval > 0 and step == 0:
        if i >= cap_rec:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = GROW_REC
            hint_out[0] = m
            return GROW_REC
        
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1

    # Early-exit (success) at t0 (pre-phase)
    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT
    
    # Main integration loop
    while step < max_steps and t < t_end:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Pre-events
        event_code_pre, log_width_pre = events_pre(t, y_curr, params, evt_log_scratch, runtime_ws)
        
        if event_code_pre >= 0 and log_width_pre > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_pre):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_pre
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Clip dt to not overshoot t_end
        remaining = t_end - t
        if dt > remaining and remaining > 0.0:
            dt = remaining

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Post-events
        event_code_post, log_width_post = events_post(
            t_prop[0], y_prop, params, evt_log_scratch, runtime_ws
        )
        
        if event_code_post >= 0 and log_width_post > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_post):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_post
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Commit
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step += 1

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
            update_aux(t_prop[0], y_curr, params, runtime_ws.aux_values, runtime_ws)

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

        t = t_prop[0]
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''


# Continuous runner WITH analysis - hooks are globals
_RUNNER_CONTINUOUS_ANALYSIS_TEMPLATE = '''
def runner_analysis(
    # scalars
    t0, t_end, dt_init,
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
    # analysis buffers
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Continuous runner with analysis hooks (ANALYSIS_PRE, ANALYSIS_POST as globals).
    """
    trace_cap_int = int(analysis_trace_cap)
    trace_stride_int = int(analysis_trace_stride)
    use_variational_step = bool(variational_step_enabled)
    
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values / stop flag before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0
    if record_interval > 0 and step == 0:
        if i >= cap_rec:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = GROW_REC
            hint_out[0] = m
            return GROW_REC
        
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1

    # Early-exit (success) at t0 (pre-phase)
    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT
    
    # Main integration loop
    while step < max_steps and t < t_end:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Pre-events
        event_code_pre, log_width_pre = events_pre(t, y_curr, params, evt_log_scratch, runtime_ws)
        
        if event_code_pre >= 0 and log_width_pre > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_pre):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_pre
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Analysis pre-hook (global symbol)
        ANALYSIS_PRE(
            t, dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        # Clip dt to not overshoot t_end
        remaining = t_end - t
        if dt > remaining and remaining > 0.0:
            dt = remaining

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Post-events
        event_code_post, log_width_post = events_post(
            t_prop[0], y_prop, params, evt_log_scratch, runtime_ws
        )
        
        if event_code_post >= 0 and log_width_post > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_post):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_post
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Commit
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step += 1

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
            update_aux(t_prop[0], y_curr, params, runtime_ws.aux_values, runtime_ws)

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

        # Analysis post-hook (global symbol)
        ANALYSIS_POST(
            t_prop[0], dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        t = t_prop[0]
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''


# Continuous runner FASTPATH (no events, no growth checks)
_RUNNER_CONTINUOUS_FASTPATH_TEMPLATE = '''
def runner_fastpath(
    # scalars
    t0, t_end, dt_init,
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
    # analysis buffers (unused in base runner but kept for ABI compatibility)
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Fastpath continuous runner: no events, no growth checks.
    """
    use_variational_step = bool(variational_step_enabled)
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values / stop flag before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0
    if record_interval > 0 and step == 0:
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1

    # Early-exit (success) at t0 (pre-phase)
    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT
    
    # Main integration loop
    while step < max_steps and t < t_end:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Clip dt to not overshoot t_end
        remaining = t_end - t
        if dt > remaining and remaining > 0.0:
            dt = remaining

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Commit
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step += 1

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
            update_aux(t_prop[0], y_curr, params, runtime_ws.aux_values, runtime_ws)

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

        t = t_prop[0]
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''


# Continuous runner FASTPATH + analysis hooks
_RUNNER_CONTINUOUS_FASTPATH_ANALYSIS_TEMPLATE = '''
def runner_fastpath_analysis(
    # scalars
    t0, t_end, dt_init,
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
    # analysis buffers
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Fastpath continuous runner with analysis hooks (ANALYSIS_PRE/POST as globals).
    """
    trace_cap_int = int(analysis_trace_cap)
    trace_stride_int = int(analysis_trace_stride)
    use_variational_step = bool(variational_step_enabled)
    
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT

    # Recording at t0
    if record_interval > 0 and step == 0:
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1
    
    # Main integration loop
    while step < max_steps and t < t_end:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Analysis pre-hook (global symbol)
        ANALYSIS_PRE(
            t, dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        # Clip dt to not overshoot t_end
        remaining = t_end - t
        if dt > remaining and remaining > 0.0:
            dt = remaining

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Commit
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step += 1

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
            update_aux(t_prop[0], y_curr, params, runtime_ws.aux_values, runtime_ws)

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t_prop[0]
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT

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

        # Analysis post-hook (global symbol)
        ANALYSIS_POST(
            t_prop[0], dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        t = t_prop[0]
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''

# Base runner for discrete (map) systems - NO analysis
_RUNNER_DISCRETE_BASE_TEMPLATE = '''
def runner_discrete_base(
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
    # analysis buffers (unused in base runner but kept for ABI compatibility)
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Base discrete runner: no analysis hooks, zero overhead.
    """
    use_variational_step = bool(variational_step_enabled)
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values / stop flag before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0
    if record_interval > 0 and step == 0:
        if i >= cap_rec:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = GROW_REC
            hint_out[0] = m
            return GROW_REC
        
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1

    # Early-exit (success) at t0 (pre-phase)
    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT
    
    # Main iteration loop
    while step < N:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Pre-events
        event_code_pre, log_width_pre = events_pre(t, y_curr, params, evt_log_scratch, runtime_ws)
        
        if event_code_pre >= 0 and log_width_pre > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_pre):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_pre
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Post-events
        next_step = step + 1
        t_post = t0 + next_step * dt
        event_code_post, log_width_post = events_post(
            t_post, y_prop, params, evt_log_scratch, runtime_ws
        )
        
        if event_code_post >= 0 and log_width_post > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_post):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_post
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Commit
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step = next_step

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
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

        t = t_post
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''


# Discrete runner WITH analysis - hooks are globals
_RUNNER_DISCRETE_ANALYSIS_TEMPLATE = '''
def runner_discrete_analysis(
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
    # analysis buffers
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Discrete runner with analysis hooks (ANALYSIS_PRE, ANALYSIS_POST as globals).
    """
    trace_cap_int = int(analysis_trace_cap)
    trace_stride_int = int(analysis_trace_stride)
    use_variational_step = bool(variational_step_enabled)
    
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values / stop flag before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0
    if record_interval > 0 and step == 0:
        if i >= cap_rec:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = GROW_REC
            hint_out[0] = m
            return GROW_REC
        
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1

    # Early-exit (success) at t0 (pre-phase)
    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT
    
    # Main iteration loop
    while step < N:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Pre-events
        event_code_pre, log_width_pre = events_pre(t, y_curr, params, evt_log_scratch, runtime_ws)
        
        if event_code_pre >= 0 and log_width_pre > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_pre):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_pre
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Analysis pre-hook (global symbol)
        ANALYSIS_PRE(
            t, dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Post-events
        next_step = step + 1
        t_post = t0 + next_step * dt
        event_code_post, log_width_post = events_post(
            t_post, y_prop, params, evt_log_scratch, runtime_ws
        )
        
        if event_code_post >= 0 and log_width_post > 0:
            if m >= cap_evt:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            for log_idx in range(log_width_post):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            EVT_CODE[m] = event_code_post
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1

        # Commit
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step = next_step

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
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

        # Analysis post-hook (global symbol)
        ANALYSIS_POST(
            t_post, dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        t = t_post
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            if i >= cap_rec:
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''


# Discrete runner FASTPATH (no events, no growth checks)
_RUNNER_DISCRETE_FASTPATH_TEMPLATE = '''
def runner_discrete_fastpath(
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
    # analysis buffers (unused in base runner but kept for ABI compatibility)
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Fastpath discrete runner: no events, no growth checks.
    """
    use_variational_step = bool(variational_step_enabled)
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0
    if record_interval > 0 and step == 0:
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1

    # Early-exit (success) at t0 (pre-phase)
    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT
    
    # Main iteration loop
    while step < N:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Commit
        next_step = step + 1
        t_post = t0 + next_step * dt
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step = next_step

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
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

        t = t_post
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''


# Discrete runner FASTPATH + analysis hooks
_RUNNER_DISCRETE_FASTPATH_ANALYSIS_TEMPLATE = '''
def runner_discrete_fastpath_analysis(
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
    # analysis buffers
    analysis_ws, analysis_out, analysis_trace,
    analysis_trace_count, analysis_trace_cap, analysis_trace_stride,
    variational_step_enabled, variational_step_fn,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post, update_aux,
    # selective recording parameters
    state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
) -> int:
    """
    Fastpath discrete runner with analysis hooks (ANALYSIS_PRE/POST as globals).
    """
    trace_cap_int = int(analysis_trace_cap)
    trace_stride_int = int(analysis_trace_stride)
    use_variational_step = bool(variational_step_enabled)
    
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)
    step = int(step_start)
    m = int(hint_out[0])
    
    # Refresh aux values / stop flag before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0 or (
        runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 1) != 0
    ):
        update_aux(t, y_curr, params, runtime_ws.aux_values, runtime_ws)

    # Recording at t0
    if record_interval > 0 and step == 0:
        T[i] = t
        for k in range(n_rec_states):
            Y[k, i] = y_curr[state_rec_indices[k]]
        for k in range(n_rec_aux):
            AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1

    # Early-exit (success) at t0 (pre-phase)
    if (
        runtime_ws.stop_phase_mask.shape[0] > 0
        and (runtime_ws.stop_phase_mask[0] & 1) != 0
        and runtime_ws.stop_flag.shape[0] > 0
        and runtime_ws.stop_flag[0] != 0
    ):
        i_out[0] = i
        step_out[0] = step
        t_out[0] = t
        status_out[0] = EARLY_EXIT
        hint_out[0] = m
        return EARLY_EXIT
    
    # Main iteration loop
    while step < N:
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        # Analysis pre-hook (global symbol)
        ANALYSIS_PRE(
            t, dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        # Stepper attempt
        if use_variational_step:
            step_status = variational_step_fn(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est,
                analysis_ws,
            )
        else:
            step_status = stepper(
                t, dt, y_curr, rhs, params,
                runtime_ws,
                stepper_ws,
                stepper_config,
                y_prop, t_prop, dt_next, err_est
            )
        if step_status is None:
            step_status = OK
        
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        if not allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        if not allfinite_scalar(t_prop[0]) or not allfinite_scalar(dt_next[0]):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED

        # Commit
        next_step = step + 1
        t_post = t0 + next_step * dt
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        step = next_step

        if runtime_ws.aux_values.shape[0] > 0 or (
            runtime_ws.stop_phase_mask.shape[0] > 0 and (runtime_ws.stop_phase_mask[0] & 2) != 0
        ):
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

        # Analysis post-hook (global symbol)
        ANALYSIS_POST(
            t_post, dt, step,
            y_curr, y_prev, params,
            runtime_ws,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, trace_cap_int, trace_stride_int,
        )
        if trace_cap_int > 0 and analysis_trace_count[0] > trace_cap_int:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = TRACE_OVERFLOW
            hint_out[0] = m
            return TRACE_OVERFLOW

        t = t_post
        dt = dt_next[0]

        # Record
        if record_interval > 0 and (step % record_interval == 0):
            T[i] = t
            for k in range(n_rec_states):
                Y[k, i] = y_curr[state_rec_indices[k]]
            for k in range(n_rec_aux):
                AUX[k, i] = runtime_ws.aux_values[aux_rec_indices[k]]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1

        if (
            runtime_ws.stop_phase_mask.shape[0] > 0
            and (runtime_ws.stop_phase_mask[0] & 2) != 0
            and runtime_ws.stop_flag.shape[0] > 0
            and runtime_ws.stop_flag[0] != 0
        ):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = EARLY_EXIT
            hint_out[0] = m
            return EARLY_EXIT
        
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK

    # Done
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE
'''

# -----------------------------------------------------------------------------
# Template selection
# -----------------------------------------------------------------------------

_RUNNER_TEMPLATE_MAP: Dict[Tuple[RunnerVariant, bool], Tuple[str, str]] = {
    (RunnerVariant.BASE, False): (_RUNNER_CONTINUOUS_BASE_TEMPLATE, "runner_base"),
    (RunnerVariant.ANALYSIS, False): (_RUNNER_CONTINUOUS_ANALYSIS_TEMPLATE, "runner_analysis"),
    (RunnerVariant.FASTPATH, False): (_RUNNER_CONTINUOUS_FASTPATH_TEMPLATE, "runner_fastpath"),
    (RunnerVariant.FASTPATH_ANALYSIS, False): (
        _RUNNER_CONTINUOUS_FASTPATH_ANALYSIS_TEMPLATE,
        "runner_fastpath_analysis",
    ),
    (RunnerVariant.BASE, True): (_RUNNER_DISCRETE_BASE_TEMPLATE, "runner_discrete_base"),
    (RunnerVariant.ANALYSIS, True): (
        _RUNNER_DISCRETE_ANALYSIS_TEMPLATE,
        "runner_discrete_analysis",
    ),
    (RunnerVariant.FASTPATH, True): (
        _RUNNER_DISCRETE_FASTPATH_TEMPLATE,
        "runner_discrete_fastpath",
    ),
    (RunnerVariant.FASTPATH_ANALYSIS, True): (
        _RUNNER_DISCRETE_FASTPATH_ANALYSIS_TEMPLATE,
        "runner_discrete_fastpath_analysis",
    ),
}

# -----------------------------------------------------------------------------
# Runner compilation
# -----------------------------------------------------------------------------

def _build_base_namespace() -> dict:
    """Build the base namespace for runner compilation."""
    namespace = {
        "OK": OK,
        "STEPFAIL": STEPFAIL,
        "NAN_DETECTED": NAN_DETECTED,
        "EARLY_EXIT": EARLY_EXIT,
        "DONE": DONE,
        "GROW_REC": GROW_REC,
        "GROW_EVT": GROW_EVT,
        "USER_BREAK": USER_BREAK,
        "TRACE_OVERFLOW": TRACE_OVERFLOW,
        "allfinite1d": _allfinite1d_py,
        "allfinite_scalar": _allfinite_scalar_py,
    }
    return namespace


def _compile_runner(
    source: str,
    func_name: str,
    *,
    jit: bool,
    analysis_pre: Optional[Callable] = None,
    analysis_post: Optional[Callable] = None,
) -> Callable:
    """
    Compile a runner from source with hooks injected as globals.
    
    Parameters
    ----------
    source : str
        Runner source code (one of the templates)
    func_name : str
        Name of the function to extract from compiled module
    jit : bool
        Whether to JIT-compile with numba
    analysis_pre : callable, optional
        Pre-step analysis hook (injected as ANALYSIS_PRE global)
    analysis_post : callable, optional  
        Post-step analysis hook (injected as ANALYSIS_POST global)
    
    Returns
    -------
    Callable
        The compiled runner function
    """
    if jit and not _NUMBA_AVAILABLE:
        raise JITUnavailableError(
            "jit=True requires numba, but numba is not installed. "
            "Install numba or pass jit=False."
        )
    namespace = _build_base_namespace()
    
    # Inject hooks as globals if provided
    if analysis_pre is not None:
        namespace["ANALYSIS_PRE"] = analysis_pre
    if analysis_post is not None:
        namespace["ANALYSIS_POST"] = analysis_post
    
    if jit and _NUMBA_AVAILABLE:
        from numba import njit
        # JIT-compile the guards
        namespace["allfinite1d"] = njit(inline="always")(_allfinite1d_py)
        namespace["allfinite_scalar"] = njit(inline="always")(_allfinite_scalar_py)
    
    # Execute source to define the function
    exec(source, namespace)
    runner_fn = namespace[func_name]
    
    if jit and _NUMBA_AVAILABLE:
        from numba import njit
        return njit(cache=False, nogil=True)(runner_fn)
    
    return runner_fn


# -----------------------------------------------------------------------------
# Cached base runners (no analysis)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _get_base_runner_continuous(jit: bool) -> Callable:
    """Get the base continuous runner (no analysis)."""
    return _compile_runner(
        _RUNNER_CONTINUOUS_BASE_TEMPLATE,
        "runner_base",
        jit=jit,
    )


@lru_cache(maxsize=2)
def _get_base_runner_discrete(jit: bool) -> Callable:
    """Get the base discrete runner (no analysis)."""
    return _compile_runner(
        _RUNNER_DISCRETE_BASE_TEMPLATE,
        "runner_discrete_base",
        jit=jit,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compile_analysis_hooks(
    analysis: "ObserverModule",
    *,
    jit: bool,
    dtype: np.dtype,
) -> Tuple[Callable, Callable]:
    """
    Compile analysis hooks for injection into runner.
    
    This pre-compiles the hooks so they can be used as static globals
    in the runner, avoiding Numba's first-class function type.
    
    Parameters
    ----------
    analysis : ObserverModule
        The analysis module with hooks
    jit : bool
        Whether to JIT-compile the hooks
    dtype : np.dtype
        Data type for type specialization
    
    Returns
    -------
    tuple[Callable, Callable]
        (pre_hook, post_hook) ready for injection
        
    Raises
    ------
    RuntimeError
        If JIT compilation fails (fail-fast behavior)
    """
    hooks = analysis.resolve_hooks(jit=jit, dtype=dtype)
    
    # Get the actual hook functions, defaulting to noop
    noop = _get_noop_hook_jit() if jit else _noop_hook
    pre_hook = hooks.pre_step if hooks.pre_step is not None else noop
    post_hook = hooks.post_step if hooks.post_step is not None else noop
    
    return pre_hook, post_hook


def _set_analysis_globals(
    runner_fn: Callable,
    pre_hook: Optional[Callable],
    post_hook: Optional[Callable],
) -> None:
    if pre_hook is None and post_hook is None:
        return
    if hasattr(runner_fn, "py_func"):
        globals_dict = runner_fn.py_func.__globals__
    else:
        globals_dict = runner_fn.__globals__
    if pre_hook is not None:
        globals_dict["ANALYSIS_PRE"] = pre_hook
    if post_hook is not None:
        globals_dict["ANALYSIS_POST"] = post_hook


def get_runner(
    variant: RunnerVariant,
    *,
    model_hash: str,
    stepper_name: str,
    analysis: Optional["ObserverModule"],
    dtype: np.dtype,
    jit: bool,
    discrete: bool,
) -> Callable:
    """
    Get a runner variant from cache or newly compiled.
    """
    if jit and not _NUMBA_AVAILABLE:
        raise JITUnavailableError(
            "jit=True requires numba, but numba is not installed. "
            "Install numba or pass jit=False."
        )
    if not model_hash or not stepper_name:
        raise ValueError("model_hash and stepper_name are required for runner selection")

    if variant in (RunnerVariant.ANALYSIS, RunnerVariant.FASTPATH_ANALYSIS) and analysis is None:
        raise ValueError("analysis is required for analysis-capable runner variants")
    if variant in (RunnerVariant.BASE, RunnerVariant.FASTPATH) and analysis is not None:
        raise ValueError("analysis must be None for non-analysis runner variants")

    template_key = (variant, bool(discrete))
    try:
        template_source, func_name = _RUNNER_TEMPLATE_MAP[template_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported runner variant: {variant}") from exc

    analysis_sig = analysis_signature_hash(analysis, dtype)
    runner_kind = "discrete" if discrete else "continuous"
    dtype_key = np.dtype(dtype).name
    cache_token = runner_cache.get_runner_cache_token(model_hash, stepper_name)
    cache_key = (
        model_hash,
        stepper_name,
        analysis_sig,
        variant.name,
        runner_kind,
        dtype_key,
        cache_token,
        bool(jit),
        _RUNNER_TEMPLATE_VERSION,
    )
    cache = _variant_cache_discrete if discrete else _variant_cache_continuous
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    analysis_pre = None
    analysis_post = None
    if analysis is not None:
        analysis_pre, analysis_post = compile_analysis_hooks(analysis, jit=jit, dtype=dtype)

    runner = None
    if jit:
        runner = runner_cache.get_cached_runner(
            model_hash=model_hash,
            stepper_name=stepper_name,
            variant=variant.name,
            template_version=_RUNNER_TEMPLATE_VERSION,
            runner_kind=runner_kind,
            source=template_source,
            function_name=func_name,
        )
        if runner is not None:
            _set_analysis_globals(runner, analysis_pre, analysis_post)
            cache.put(cache_key, runner)
            return runner

    runner = _compile_runner(
        template_source,
        func_name,
        jit=jit,
        analysis_pre=analysis_pre,
        analysis_post=analysis_post,
    )
    cache.put(cache_key, runner)
    return runner


def get_runner_variant(
    *,
    model_hash: str,
    stepper_name: str,
    analysis: Optional["ObserverModule"],
    dtype: np.dtype,
    jit: bool,
) -> Callable:
    """Compatibility wrapper for continuous runners."""
    variant = RunnerVariant.ANALYSIS if analysis is not None else RunnerVariant.BASE
    return get_runner(
        variant,
        model_hash=model_hash,
        stepper_name=stepper_name,
        analysis=analysis,
        dtype=dtype,
        jit=jit,
        discrete=False,
    )


def get_runner_variant_discrete(
    *,
    model_hash: str,
    stepper_name: str,
    analysis: Optional["ObserverModule"],
    dtype: np.dtype,
    jit: bool,
) -> Callable:
    """Compatibility wrapper for discrete runners."""
    variant = RunnerVariant.ANALYSIS if analysis is not None else RunnerVariant.BASE
    return get_runner(
        variant,
        model_hash=model_hash,
        stepper_name=stepper_name,
        analysis=analysis,
        dtype=dtype,
        jit=jit,
        discrete=True,
    )
