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
4. CombinedAnalysis generates explicit sequential calls, not loops over containers
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from collections import OrderedDict
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from dynlib.runtime.softdeps import softdeps
from dynlib.runtime.runner_api import (
    OK, STEPFAIL, NAN_DETECTED,
    DONE, GROW_REC, GROW_EVT, USER_BREAK, TRACE_OVERFLOW
)

if TYPE_CHECKING:
    from dynlib.analysis.runtime import AnalysisModule, AnalysisHooks

__all__ = [
    "get_runner_variant",
    "get_runner_variant_discrete",
    "clear_variant_cache",
    "analysis_signature_hash",
]

_SOFTDEPS = softdeps()
_NUMBA_AVAILABLE = _SOFTDEPS.numba


def analysis_signature_hash(analysis: Optional["AnalysisModule"], dtype: np.dtype) -> str:
    """
    Compute a stable hash for an analysis configuration.
    
    Returns a short hex string suitable for cache keying.
    """
    if analysis is None:
        return "noop"
    sig = analysis.signature(dtype)
    sig_bytes = repr(sig).encode("utf-8")
    return hashlib.sha256(sig_bytes).hexdigest()[:16]


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
    
    # Refresh aux values before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0:
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

        if runtime_ws.aux_values.shape[0] > 0:
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
    
    # Refresh aux values before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0:
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

        if runtime_ws.aux_values.shape[0] > 0:
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
    
    # Refresh aux values before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0:
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
    
    # Refresh aux values before any potential recording at t0
    if runtime_ws.aux_values.shape[0] > 0:
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

        # Analysis post-hook (global symbol)
        ANALYSIS_POST(
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
# Guard functions (inlined for JIT)
# -----------------------------------------------------------------------------

def _allfinite1d_py(arr):
    """Check all elements of a 1-D array are finite."""
    for i in range(arr.shape[0]):
        v = arr[i]
        if v != v or v == v + 1.0e308:  # NaN or Inf check
            return False
    return True


def _allfinite_scalar_py(val):
    """Check a scalar is finite."""
    return val == val and val != val + 1.0e308


# -----------------------------------------------------------------------------
# Runner compilation
# -----------------------------------------------------------------------------

def _build_base_namespace() -> dict:
    """Build the base namespace for runner compilation."""
    namespace = {
        "OK": OK,
        "STEPFAIL": STEPFAIL,
        "NAN_DETECTED": NAN_DETECTED,
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
        return njit(cache=False)(runner_fn)
    
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
    analysis: "AnalysisModule",
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
    analysis : AnalysisModule
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


def get_runner_variant(
    *,
    model_hash: str,
    stepper_name: str,
    analysis: Optional["AnalysisModule"],
    dtype: np.dtype,
    jit: bool,
) -> Callable:
    """
    Get a continuous (ODE) runner variant, from cache or newly compiled.
    
    Parameters
    ----------
    model_hash : str
        Hash identifying the model
    stepper_name : str
        Name of the stepper being used
    analysis : AnalysisModule or None
        Analysis module, or None for base runner
    dtype : np.dtype
        Data type for the simulation
    jit : bool
        Whether to use JIT compilation
    
    Returns
    -------
    Callable
        Runner function with matching signature
    """
    if analysis is None:
        return _get_base_runner_continuous(jit)
    
    # Build cache key
    analysis_sig = analysis_signature_hash(analysis, dtype)
    cache_key = (model_hash, stepper_name, analysis_sig, "continuous", jit)
    
    # Check cache
    cached = _variant_cache_continuous.get(cache_key)
    if cached is not None:
        return cached
    
    # Compile hooks (fail-fast: will raise if incompatible)
    pre_hook, post_hook = compile_analysis_hooks(analysis, jit=jit, dtype=dtype)
    
    # Compile runner with hooks as globals
    runner = _compile_runner(
        _RUNNER_CONTINUOUS_ANALYSIS_TEMPLATE,
        "runner_analysis",
        jit=jit,
        analysis_pre=pre_hook,
        analysis_post=post_hook,
    )
    
    # Cache and return
    _variant_cache_continuous.put(cache_key, runner)
    return runner


def get_runner_variant_discrete(
    *,
    model_hash: str,
    stepper_name: str,
    analysis: Optional["AnalysisModule"],
    dtype: np.dtype,
    jit: bool,
) -> Callable:
    """
    Get a discrete (map) runner variant, from cache or newly compiled.
    
    Parameters
    ----------
    model_hash : str
        Hash identifying the model
    stepper_name : str
        Name of the stepper being used
    analysis : AnalysisModule or None
        Analysis module, or None for base runner
    dtype : np.dtype
        Data type for the simulation
    jit : bool
        Whether to use JIT compilation
    
    Returns
    -------
    Callable
        Runner function with matching signature
    """
    if analysis is None:
        return _get_base_runner_discrete(jit)
    
    # Build cache key
    analysis_sig = analysis_signature_hash(analysis, dtype)
    cache_key = (model_hash, stepper_name, analysis_sig, "discrete", jit)
    
    # Check cache
    cached = _variant_cache_discrete.get(cache_key)
    if cached is not None:
        return cached
    
    # Compile hooks (fail-fast: will raise if incompatible)
    pre_hook, post_hook = compile_analysis_hooks(analysis, jit=jit, dtype=dtype)
    
    # Compile runner with hooks as globals
    runner = _compile_runner(
        _RUNNER_DISCRETE_ANALYSIS_TEMPLATE,
        "runner_discrete_analysis",
        jit=jit,
        analysis_pre=pre_hook,
        analysis_post=post_hook,
    )
    
    # Cache and return
    _variant_cache_discrete.put(cache_key, runner)
    return runner
