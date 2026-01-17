# src/dynlib/runtime/sim.py
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np

from .wrapper import run_with_wrapper
from .results import Results
from .results_api import ResultsView
from .runner_api import Status
from .initial_step import WRMSConfig, make_wrms_config_from_stepper
from dynlib.steppers.registry import get_stepper
from dynlib.runtime.observers import ObserverModule, CombinedObserver
from dynlib.compiler.build import FullModel
import tomllib

try:  # pragma: no cover - available on 3.8+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore


__all__ = ["Sim"]

# ------------------------------- data records ---------------------------------

WorkspaceSnapshot = Dict[str, Dict[str, object]]


@dataclass(frozen=True)
class SessionPins:
    spec_hash: str
    stepper_name: str
    workspace_sig: Tuple[int, ...]
    dtype_token: str
    dynlib_version: str


@dataclass
class SessionState:
    t_curr: float
    y_curr: np.ndarray
    params_curr: np.ndarray
    dt_curr: float
    step_count: int
    workspace: WorkspaceSnapshot
    stepper_cfg: np.ndarray
    status: int
    pins: SessionPins

    def clone(self) -> SessionState:
        return SessionState(
            t_curr=self.t_curr,
            y_curr=np.array(self.y_curr, copy=True),
            params_curr=np.array(self.params_curr, copy=True),
            dt_curr=self.dt_curr,
            step_count=self.step_count,
            workspace=_copy_workspace_dict(self.workspace),
            stepper_cfg=np.array(self.stepper_cfg, dtype=np.float64, copy=True),
            status=self.status,
            pins=self.pins,
        )

    def to_seed(self) -> IntegratorSeed:
        return IntegratorSeed(
            t=self.t_curr,
            y=np.array(self.y_curr, copy=True),
            params=np.array(self.params_curr, copy=True),
            dt=self.dt_curr,
            step_count=self.step_count,
            workspace=_copy_workspace_dict(self.workspace),
        )


@dataclass
class Snapshot:
    name: str
    description: str
    created_at: str
    state: SessionState
    time_shift: float
    nominal_dt: float


@dataclass(frozen=True)
class Segment:
    id: int
    name: Optional[str]
    rec_start: int
    rec_len: int
    evt_start: int
    evt_len: int
    t_start: float
    t_end: float
    step_start: int
    step_end: int
    resume: bool
    cfg_hash: str
    note: str = ""


@dataclass
class IntegratorSeed:
    t: float
    y: np.ndarray
    params: np.ndarray
    dt: float
    step_count: int
    workspace: WorkspaceSnapshot


# Sim defaults that will be used with config()
@dataclass
class _RunDefaults:
    max_steps: Optional[int] = None
    record: Optional[bool] = None
    record_interval: Optional[int] = None
    cap_rec: Optional[int] = None
    cap_evt: Optional[int] = None



class _ResultAccumulator:
    """
    Mutable recording buffers that grow geometrically as stitched results append.
    
    Handles selective recording by accepting n_rec_states and n_rec_aux which may be 
    less than or equal to the full n_state and n_aux.
    """

    def __init__(self, *, n_rec_states: int, n_rec_aux: int, dtype: np.dtype, max_log_width: int) -> None:
        self.n_rec_states = int(n_rec_states)
        self.n_rec_aux = int(n_rec_aux)
        self.dtype = np.dtype(dtype)
        self.log_cols = max(1, int(max_log_width))
        self._record_cap = 0
        self._event_cap = 0
        self.n = 0
        self.m = 0
        self.T = np.zeros((0,), dtype=np.float64)
        self.Y = np.zeros((self.n_rec_states, 0), dtype=self.dtype)
        self.AUX = np.zeros((self.n_rec_aux, 0), dtype=self.dtype) if n_rec_aux > 0 else None
        self.STEP = np.zeros((0,), dtype=np.int64)
        self.FLAGS = np.zeros((0,), dtype=np.int32)
        self.EVT_CODE = np.zeros((0,), dtype=np.int32)
        self.EVT_INDEX = np.zeros((0,), dtype=np.int32)
        self.EVT_LOG_DATA = np.zeros((0, self.log_cols), dtype=self.dtype)

    def clear(self) -> None:
        self.n = 0
        self.m = 0

    def append_records(
        self,
        t_seg: np.ndarray,
        y_seg: np.ndarray,
        step_seg: np.ndarray,
        flags_seg: np.ndarray,
        aux_seg: np.ndarray | None = None,
    ) -> None:
        if t_seg.size == 0:
            return
        needed = self.n + t_seg.shape[0]
        self._ensure_record_capacity(needed)
        start = self.n
        end = needed
        self.T[start:end] = t_seg
        self.Y[:, start:end] = y_seg
        if aux_seg is not None and self.AUX is not None:
            self.AUX[:, start:end] = aux_seg
        self.STEP[start:end] = step_seg
        self.FLAGS[start:end] = flags_seg
        self.n = end

    def append_events(
        self,
        codes: np.ndarray,
        indices: np.ndarray,
        log_rows: np.ndarray,
    ) -> None:
        if codes.size == 0:
            return
        needed = self.m + codes.shape[0]
        self._ensure_event_capacity(needed)
        start = self.m
        end = needed
        self.EVT_CODE[start:end] = codes
        self.EVT_INDEX[start:end] = indices
        if log_rows.shape[1] > 0:
            self.EVT_LOG_DATA[start:end, : log_rows.shape[1]] = log_rows
        self.m = end

    def to_results(
        self,
        *,
        status: int,
        final_state: np.ndarray,
        final_params: np.ndarray,
        t_final: float,
        final_dt: float,
        step_count_final: int,
        workspace: WorkspaceSnapshot,
        state_names: list[str],
        aux_names: list[str],
        analysis_out: np.ndarray | None = None,
        analysis_trace: np.ndarray | None = None,
        analysis_trace_filled: int | None = None,
        analysis_trace_stride: int | None = None,
        analysis_trace_offset: int | None = None,
        analysis_modules: tuple[object, ...] | None = None,
        analysis_meta: Mapping[str, object] | None = None,
    ) -> Results:
        return Results(
            T=self.T,
            Y=self.Y,
            AUX=self.AUX,  # Pass accumulated AUX buffer
            STEP=self.STEP,
            FLAGS=self.FLAGS,
            EVT_CODE=self.EVT_CODE,
            EVT_INDEX=self.EVT_INDEX,
            EVT_LOG_DATA=self.EVT_LOG_DATA,
            n=self.n,
            m=self.m,
            status=int(status),
            final_state=final_state,
            final_params=final_params,
            t_final=float(t_final),
            final_dt=float(final_dt),
            step_count_final=int(step_count_final),
            final_workspace=workspace,
            state_names=state_names,
            aux_names=aux_names,
            analysis_out=analysis_out,
            analysis_trace=analysis_trace,
            analysis_trace_filled=analysis_trace_filled,
            analysis_trace_stride=analysis_trace_stride,
            analysis_trace_offset=analysis_trace_offset,
            analysis_modules=analysis_modules,
            analysis_meta=analysis_meta,
        )

    def assert_monotone_time(self) -> None:
        if self.n < 2:
            return
        t = self.T[: self.n]
        prev = t[:-1]
        nxt = t[1:]
        diffs = nxt - prev
        scale = np.maximum(np.maximum(np.abs(prev), np.abs(nxt)), 1.0)
        tol = np.spacing(scale)
        if np.any(diffs < -tol):
            idx = int(np.where(diffs < -tol)[0][0])
            raise RuntimeError(
                f"Non-monotone time axis after stitching around indices {idx}/{idx+1}: "
                f"{prev[idx]} -> {nxt[idx]}"
            )

    def _ensure_record_capacity(self, min_needed: int) -> None:
        if min_needed <= self._record_cap:
            return
        new_cap = max(1, self._record_cap or 1)
        while new_cap < min_needed:
            new_cap *= 2
        self.T = _resize_1d(self.T, new_cap)
        self.STEP = _resize_1d(self.STEP, new_cap)
        self.FLAGS = _resize_1d(self.FLAGS, new_cap)
        new_Y = np.zeros((self.n_rec_states, new_cap), dtype=self.dtype)
        if self.n:
            new_Y[:, : self.n] = self.Y[:, : self.n]
        self.Y = new_Y
        if self.AUX is not None:
            new_AUX = np.zeros((self.n_rec_aux, new_cap), dtype=self.dtype)
            if self.n:
                new_AUX[:, : self.n] = self.AUX[:, : self.n]
            self.AUX = new_AUX
        self._record_cap = new_cap

    def _ensure_event_capacity(self, min_needed: int) -> None:
        if min_needed <= self._event_cap:
            return
        new_cap = max(1, self._event_cap or 1)
        while new_cap < min_needed:
            new_cap *= 2
        self.EVT_CODE = _resize_1d(self.EVT_CODE, new_cap)
        self.EVT_INDEX = _resize_1d(self.EVT_INDEX, new_cap)
        new_logs = np.zeros((new_cap, self.log_cols), dtype=self.dtype)
        if self.m:
            new_logs[: self.m, :] = self.EVT_LOG_DATA[: self.m, :]
        self.EVT_LOG_DATA = new_logs
        self._event_cap = new_cap


# --------------------------------- facade -------------------------------------

class Sim:
    """
    Simulation facade around a compiled FullModel with resumable session state and optional snapshots.
    """

    def __init__(self, model: FullModel):
        self._logger = logging.getLogger(__name__)
        self.model = model
        self._raw_results: Optional[Results] = None
        self._results_view: Optional[ResultsView] = None
        self._dtype = model.dtype
        self._n_state = len(model.spec.states)
        self._n_aux = len(model.spec.aux)
        self._max_log_width = _max_event_log_width(model.spec.events)
        self._workspace_sig = tuple(model.workspace_sig)
        self._pins = SessionPins(
            spec_hash=model.spec_hash,
            stepper_name=model.stepper_name,
            workspace_sig=self._workspace_sig,
            dtype_token=str(np.dtype(model.dtype)),
            dynlib_version=_dynlib_version(),
        )
        stepper_spec = get_stepper(model.stepper_name)
        self._stepper_spec = stepper_spec
        self._fixed_time_control = getattr(stepper_spec.meta, "time_control", "fixed") == "fixed"
        self._stepper_config_names = _stepper_config_names(stepper_spec, self.model.spec)
        self._default_stepper_cfg = self._compute_default_stepper_config()
        self._session_state = self._bootstrap_session_state()
        self._snapshots: Dict[str, Snapshot] = {}
        self._initial_snapshot_name = "initial"
        self._initial_snapshot_created = False
        self._result_accum: Optional[_ResultAccumulator] = None
        self._event_time_columns = _event_time_column_map(self.model.spec)
        self._time_shift = 0.0
        self._nominal_dt = float(self.model.spec.sim.dt)
        self._run_defaults = _RunDefaults()
        self._segments: list[Segment] = []
        self._pending_run_tag: Optional[str] = None
        self._pending_run_cfg_hash: Optional[str] = None
        self._last_run_was_resume = False
        
        # Recording selection (for resume consistency): None = not yet determined, [] = explicit empty
        self._recording_state_names: list[str] | None = None
        self._recording_aux_names: list[str] | None = None
        
        # Initialize presets bank with inline presets from model DSL
        self._presets: Dict[str, _PresetData] = {}
        self._load_inline_presets()

    # ------------------------------ public API ---------------------------------

    def dry_run(self) -> bool:
        """Tiny helper to assert callability."""
        return (
            callable(self.model.rhs)
            and callable(self.model.events_pre)
            and callable(self.model.events_post)
            and callable(self.model.runner)
            and callable(self.model.stepper)
        )

    @property
    def stepper(self):
        """Access the stepper specification object."""
        return self._stepper_spec

    def run(
        self,
        *,
        t0: Optional[float] = None,
        T: Optional[float] = None,  # Continuous: end time
        N: Optional[int] = None,    # Discrete: number of iterations
        dt: Optional[float] = None,
        max_steps: Optional[int] = None,
        record: Optional[bool] = None,
        record_interval: Optional[int] = None,
        record_vars: list[str] | None = None,  # NEW: selective recording
        observers=None,
        ic: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        cap_rec: Optional[int] = None,
        cap_evt: Optional[int] = None,
        transient: Optional[float] = None,
        resume: bool = False,
        # Note: legacy `t_end` removed from public API; use `T` instead.
        tag: Optional[str] = None,
        **stepper_kwargs,
    ) -> None:
        """
        Run the compiled model. Set resume=True to continue from the last SessionState.
        
        Args:
            t0: Initial time (default from sim config)
            T: End time for continuous systems (ODEs, SDEs, etc.)
            N: Number of iterations for discrete systems (maps, difference equations)
            dt: Time step / label spacing (default from sim config)
            max_steps: Maximum steps (safety guard for continuous, target for discrete if N not set)
            record: Whether to record states (default from sim config)
            record_interval: Record every N steps
            record_vars: List of variable names to record.
                - None (default): Record all states (backward compatible)
                - []: Record nothing (only T, STEP, FLAGS)
                - State names: "x", "y", "z"
                - Aux names (auto-detected): "energy", "power"
                - Aux names with explicit prefix: "aux.energy", "aux.power"
                - Can mix: ["x", "energy", "z"] or ["x", "aux.energy", "z"]
                - Variables are auto-detected: states first, then aux
            observers: Optional runtime observers to run alongside integration
            ic: Initial conditions (default from model spec)
            params: Parameters (default from model spec)
            cap_rec: Initial recording buffer capacity
            cap_evt: Initial event log buffer capacity
            transient: Transient warm-up period (continuous) or iterations (discrete)
            resume: Continue from last session state
            tag: Optional name recorded for this segment when samples are kept
            **stepper_kwargs: Runtime stepper configuration parameters
        
        Default precedence:
            explicit run() arg > Sim.config() defaults > DSL sim defaults
        
        Notes:
            - For discrete systems: Specify N (iterations) or use max_steps as target
            - For continuous systems: Specify T (end time), max_steps is safety guard
            - If transient > 0: warm-up period before recording (no effect on resume)
        """
        sim_defaults = self.model.spec.sim
        if tag is not None:
            if not isinstance(tag, str):
                raise TypeError("tag must be a string or None")
            if tag == "":
                raise ValueError("tag cannot be empty")

        # Apply persistent defaults for run-level knobs
        # record
        if record is None:
            if self._run_defaults.record is not None:
                record = self._run_defaults.record
            else:
                record = sim_defaults.record

        # record_interval
        if record_interval is None:
            if self._run_defaults.record_interval is not None:
                record_interval = self._run_defaults.record_interval
            else:
                record_interval = 1

        observer_mod = self._resolve_observers(observers, record_interval=record_interval)

        # max_steps
        if max_steps is None:
            if self._run_defaults.max_steps is not None:
                max_steps = self._run_defaults.max_steps
            else:
                max_steps = sim_defaults.max_steps

        # capacities
        if cap_rec is None:
            if self._run_defaults.cap_rec is not None:
                cap_rec = self._run_defaults.cap_rec
            else:
                cap_rec = 1024

        if cap_evt is None:
            if self._run_defaults.cap_evt is not None:
                cap_evt = self._run_defaults.cap_evt
            else:
                cap_evt = 1

        run_t0 = t0 if t0 is not None else sim_defaults.t0
        
        # Determine if we're running a discrete or continuous system
        is_discrete = self.model.spec.kind == "map"
        
        # dt / nominal_dt handling
        if resume and any(arg is not None for arg in (ic, params, t0, dt)):
            raise ValueError("resume=True ignores ic/params/t0/dt overrides; omit them for clarity")

        if not resume:
            # Precedence: explicit dt > stored nominal_dt > DSL sim.dt
            if dt is not None:
                nominal_dt = float(dt)
            else:
                nominal_dt = float(self._nominal_dt if self._nominal_dt is not None else sim_defaults.dt)
            if nominal_dt <= 0.0:
                raise ValueError("dt must be positive")
            self._nominal_dt = nominal_dt
        else:
            nominal_dt = self._nominal_dt

        # Handle T/N parameter conflicts and defaults
        
        if is_discrete:
            # Discrete system: N is primary, T is derived
            if T is not None and N is not None:
                raise ValueError("For discrete systems, specify either N (iterations) or T (inferred), not both")
            
            # Determine N (number of iterations)
            if N is not None:
                target_N = int(N)
                # T is derived from N
                target_T = run_t0 + target_N * nominal_dt
            elif T is not None:
                # Infer N from T
                target_T = float(T)
                target_N = int(round((target_T - run_t0) / nominal_dt))
                if target_N < 0:
                    raise ValueError(f"T ({T}) must be >= t0 ({run_t0}) for discrete systems")
            else:
                # Neither N nor T specified: use max_steps as N
                target_N = max_steps
                target_T = run_t0 + target_N * nominal_dt
        else:
            # Continuous system: T is primary
            if N is not None:
                raise ValueError("For continuous systems, use 'T' (end time), not 'N' (iterations)")
            
            target_T = T if T is not None else sim_defaults.t_end
            target_N = None  # Not used for continuous
        
        # Transient default (time for continuous, iterations for discrete)
        # Prefer explicit run() arg; otherwise fall back to DSL sim default
        # If the DSL doesn't declare a transient default, use 0.0.
        transient = 0.0 if transient is None else float(transient)
        if transient < 0.0:
            raise ValueError("transient must be non-negative")
        if resume and transient > 0.0:
            raise ValueError("transient warm-up is not allowed during resume")
        
        # Validate record_vars consistency for resume
        if resume and record_vars is not None:
            raise ValueError(
                "record_vars cannot be changed during resume. "
                "Call reset() first to start a new recording session with different variables."
            )

         # At this point:
        #   - record, record_interval, max_steps, cap_rec, cap_evt, transient
        #     have final values from (run arg > config > default)
        #   - nominal_dt has final value from (run arg > _nominal_dt > DSL)

        if not resume:
            self._result_accum = None
            self._raw_results = None
            self._results_view = None
            self._time_shift = 0.0
            self._segments = []
            self._pending_run_tag = None
            self._pending_run_cfg_hash = None
            self._last_run_was_resume = False
            # Clear recording selection when starting fresh
            self._recording_state_names = []
            self._recording_aux_names = []
        else:
            # When resuming, use the stored record_vars from previous run
            # (record_vars parameter is disallowed for resume, enforced above)
            record_vars = None  # Will be reconstructed from stored names below
        
        # For now, disable result accumulation when using selective recording
        # (would require more complex logic to handle variable shape changes)
        if record_vars is not None and resume and self._result_accum is not None:
            raise ValueError(
                "Selective recording with resume is not supported yet. "
                "Use record_vars=None (default) for resume functionality."
            )

        seed = self._select_seed(
            resume=resume,
            t0=run_t0,
            dt=nominal_dt,
            ic=ic,
            params=params,
        )
        if resume and self._fixed_time_control:
            seed.dt = self._nominal_dt
        self._ensure_initial_snapshot(seed if not self._initial_snapshot_created else None)

        if resume:
            target_T_abs = target_T + self._time_shift
            if target_T_abs <= seed.t:
                current_time = seed.t - self._time_shift
                raise ValueError(
                    f"Resume target {'T' if not is_discrete else 'T (from N)'} ({target_T}) "
                    f"must exceed current time ({current_time})"
                )

        prev_cfg = getattr(self._session_state, "stepper_cfg", None)
        stepper_config, stepper_config_values = self._build_stepper_config(
            stepper_kwargs,
            prev_cfg,
            return_values=True,
        )
        self._sync_initial_snapshot_config(stepper_config)
        n_state = self._n_state
        max_steps_internal = int(max_steps if target_N is None else target_N)
        cap_rec = max(1, int(cap_rec))
        cap_evt = max(1, int(cap_evt))
        base_steps_for_session = seed.step_count
        step_offset_initial = seed.step_count
        run_seed = seed
        record_target_T = target_T + self._time_shift
        stepper_meta = self._stepper_spec.meta
        adaptive = getattr(stepper_meta, "time_control", "fixed") == "adaptive"
        has_event_logs = self._max_log_width > 0
        wrms_cfg: WRMSConfig | None = None
        if adaptive and not resume:
            config_source = stepper_config_values or {}
            max_dt_hint = float(dt) if dt is not None else None
            wrms_cfg = make_wrms_config_from_stepper(
                stepper_meta,
                config_source,
                max_dt=max_dt_hint,
            )
        if observer_mod is not None:
            self._validate_observer_requirements(
                observer_mod,
                adaptive=adaptive,
                has_event_logs=has_event_logs,
            )
        def _remaining_steps(recorded_completed: int) -> int:
            remaining = target_N - recorded_completed
            if remaining <= 0:
                raise ValueError(
                    "Requested discrete horizon already satisfied by current state; "
                    "increase N/T or reset() to start over."
                )
            return remaining

        if is_discrete:
            if resume:
                transient_steps = 0
                if self._nominal_dt != 0.0:
                    transient_steps = int(round(self._time_shift / self._nominal_dt))
                recorded_completed = max(0, base_steps_for_session - transient_steps)
            else:
                recorded_completed = 0
            record_target_steps = _remaining_steps(recorded_completed)
        else:
            record_target_steps = None
        
        # Optional transient warm-up (no recording, no stitching) before the recorded run.
        if transient > 0.0:
            if is_discrete:
                # transient is in iterations for discrete systems
                transient_N = int(transient)
                transient_T = seed.t + transient_N * nominal_dt
            else:
                # transient is in time for continuous systems
                transient_T = seed.t + transient
                transient_N = max_steps  # Use max_steps as guard
            
            # For transient, we don't care about recording selection - use empty arrays
            # Observers are disabled during warm-up; they should only run during the recorded segment.
            warm_result = self._execute_run(
                seed=run_seed,
                t_end=transient_T,
                target_steps=transient_N if is_discrete else None,
                max_steps=transient_N if is_discrete else max_steps,
                record=False,
                record_interval=record_interval,
                cap_rec=cap_rec,
                cap_evt=cap_evt,
                stepper_config=stepper_config,
                adaptive=adaptive,
                wrms_cfg=wrms_cfg,
                state_rec_indices=np.array([], dtype=np.int32),
                aux_rec_indices=np.array([], dtype=np.int32),
                state_names=[],
                aux_names=[],
                observers=None,
            )
            self._ensure_runner_done(warm_result, phase="transient warm-up")
            self._session_state = self._state_from_results(
                warm_result, base_steps=seed.step_count, stepper_config=stepper_config
            )
            warm_state = self._session_state
            base_steps_for_session = warm_state.step_count
            step_offset_initial = 0
            run_seed = warm_state.to_seed()
            run_seed.dt = self._nominal_dt
            
            if is_discrete:
                recorded_duration = target_T - run_t0
                if recorded_duration <= 0:
                    raise ValueError("transient exceeds or equals requested horizon; nothing left to record")
            else:
                recorded_duration = target_T - run_t0
                if recorded_duration <= 0:
                    raise ValueError("transient exceeds or equals requested horizon; nothing left to record")
            
            self._time_shift = warm_state.t_curr - run_t0
            record_target_T = target_T + self._time_shift
            if is_discrete:
                record_target_steps = _remaining_steps(recorded_completed)

        self._pending_run_tag = tag
        self._pending_run_cfg_hash = _config_digest(stepper_config)
        self._last_run_was_resume = bool(resume)
        
        # Resolve recording selection (which variables to record)
        if resume and self._recording_state_names is not None:
            # Reconstruct record_vars from stored selection for consistency
            record_vars = list(self._recording_state_names)
            record_vars.extend(f"aux.{name}" for name in self._recording_aux_names)
        
        state_rec_indices, aux_rec_indices, state_names, aux_names = self._resolve_recording_selection(record_vars)
        
        # Store the recording selection for future resume calls
        if not resume:
            self._recording_state_names = state_names
            self._recording_aux_names = aux_names

        # Recorded run (or the only run when transient==0)
        recorded_result = self._execute_run(
            seed=run_seed,
            t_end=record_target_T,
            target_steps=record_target_steps,
            max_steps=max_steps_internal,
            record=record,
            record_interval=record_interval,
            cap_rec=cap_rec,
            cap_evt=cap_evt,
            stepper_config=stepper_config,
            adaptive=adaptive,
            wrms_cfg=wrms_cfg,
            state_rec_indices=state_rec_indices,
            aux_rec_indices=aux_rec_indices,
            state_names=state_names,
            aux_names=aux_names,
            observers=observer_mod,
        )
        try:
            self._ensure_runner_done(recorded_result, phase="recorded run")
            if not is_discrete and recorded_result.step_count_final >= max_steps:
                warnings.warn(
                    f"Simulation terminated after reaching max_steps ({max_steps}) steps. "
                    f"The target end time T={target_T} may not have been reached.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            if self._time_shift != 0.0:
                self._rebase_times(recorded_result, self._time_shift)
            self._session_state = self._state_from_results(
                recorded_result,
                base_steps=base_steps_for_session,
                stepper_config=stepper_config,
            )
            self._append_results(
                recorded_result,
                step_offset_initial=step_offset_initial,
            )
        finally:
            self._pending_run_tag = None
            self._pending_run_cfg_hash = None
        self._publish_results(recorded_result)

    def fastpath(
        self,
        *,
        plan=None,
        t0: Optional[float] = None,
        T: Optional[float] = None,
        N: Optional[int] = None,
        dt: Optional[float] = None,
        record_vars: list[str] | None = None,
        transient: Optional[float] = None,
        record_interval: Optional[int] = None,
        max_steps: Optional[int] = None,
        ic: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        observers=None,
    ):
        """
        Run the model via the fastpath backend when supported.

        This does not mutate the Sim session state or history. Falls back to a
        clear error when the capability gate rejects the request.
        """
        from dynlib.runtime.fastpath import FixedStridePlan
        from dynlib.runtime.fastpath.executor import fastpath_for_sim
        from dynlib.runtime.fastpath.capability import assess_capability

        sim_defaults = self.model.spec.sim
        record_interval_use = (
            int(record_interval)
            if record_interval is not None
            else int(self._run_defaults.record_interval)
            if self._run_defaults.record_interval is not None
            else 1
        )
        max_steps_use = (
            int(max_steps)
            if max_steps is not None
            else int(self._run_defaults.max_steps)
            if self._run_defaults.max_steps is not None
            else int(sim_defaults.max_steps)
        )
        dt_use = (
            float(dt)
            if dt is not None
            else float(self._nominal_dt if self._nominal_dt is not None else sim_defaults.dt)
        )
        transient_use = float(transient) if transient is not None else 0.0

        plan_use = plan if plan is not None else FixedStridePlan(stride=record_interval_use)

        # Prepare ic/params without mutating session state
        ic_vec = np.array(ic, copy=True) if ic is not None else self.state_vector(source="session", copy=True)
        params_vec = (
            np.array(params, copy=True) if params is not None else self.param_vector(source="session", copy=True)
        )

        stepper_spec = self._stepper_spec
        adaptive = getattr(stepper_spec.meta, "time_control", "fixed") == "adaptive"
        observer_mod = self._resolve_observers(observers, record_interval=record_interval_use)
        support = assess_capability(
            self,
            plan=plan_use,
            record_vars=record_vars,
            dt=dt_use,
            transient=transient_use,
            adaptive=adaptive,
            observers=observer_mod,
        )
        if not support.ok:
            reason = support.reason or "unsupported configuration"
            raise RuntimeError(f"Fastpath unavailable: {reason}")

        res = fastpath_for_sim(
            self,
            plan=plan_use,
            t0=t0,
            T=T,
            N=N,
            dt=dt_use,
            record_vars=record_vars,
            transient=transient_use,
            record_interval=record_interval_use,
            max_steps=max_steps_use,
            ic=ic_vec,
            params=params_vec,
            support=support,
            observers=observer_mod,
        )
        if res is None:
            raise RuntimeError("Fastpath unavailable")
        return res

    def raw_results(self) -> Results:
        """Return the stitched raw results façade (raises if run() not yet called)."""
        if self._raw_results is None:
            raise RuntimeError("No simulation results available; call run() first (reset() clears history).")
        return self._raw_results

    def results(self) -> ResultsView:
        """Return a cached ResultsView wrapper over the stitched run history."""
        if self._results_view is None:
            raw = self.raw_results()
            self._results_view = ResultsView(raw, self.model.spec, segments=list(self._segments))
        return self._results_view

    def create_snapshot(self, name: str, description: str = "") -> None:
        """Capture the current SessionState into an immutable snapshot."""
        if not name:
            raise ValueError("snapshot name cannot be empty")
        if name in self._snapshots:
            raise ValueError(f"Snapshot '{name}' already exists")
        self._ensure_initial_snapshot()
        snapshot_state = self._session_state.clone()
        snapshot_state.stepper_cfg = np.array(self._session_state.stepper_cfg, dtype=np.float64, copy=True)
        self._snapshots[name] = Snapshot(
            name=name,
            description=description,
            created_at=_now_iso(),
            state=snapshot_state,
            time_shift=self._time_shift,
            nominal_dt=self._nominal_dt,
        )

    def reset(self, name: str = "initial") -> None:
        """Reset to the named snapshot (default 'initial') and clear any recorded results/history."""
        snapshot = self._resolve_snapshot(name)
        ok, diff = self.compat_check(snapshot)
        if not ok:
            raise RuntimeError(f"Snapshot '{name}' is incompatible: {diff}")
        self._session_state = snapshot.state.clone()
        self._result_accum = None
        self._raw_results = None
        self._results_view = None
        self._time_shift = snapshot.time_shift
        self._nominal_dt = snapshot.nominal_dt
        self._segments = []
        self._pending_run_tag = None
        self._pending_run_cfg_hash = None
        self._last_run_was_resume = False
        # Clear recording selection so new record_vars can be specified
        self._recording_state_names = None
        self._recording_aux_names = None

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Return metadata for all snapshots (auto-creating the initial snapshot if needed)."""
        self._ensure_initial_snapshot()
        out: list[dict[str, Any]] = []
        for snap in self._snapshots.values():
            out.append(
                {
                    "name": snap.name,
                    "t": snap.state.t_curr,
                    "step": snap.state.step_count,
                    "created_at": snap.created_at,
                    "description": snap.description,
                }
            )
        return out

    def name_last_segment(self, name: str) -> None:
        """Rename the most recently recorded segment.

        Args:
            name: New unique name to assign.
        """
        if not self._segments:
            raise RuntimeError("No recorded segments available to rename")
        self.name_segment(len(self._segments) - 1, name)

    def name_segment(self, index_or_old: int | str, new_name: str) -> None:
        """Rename a recorded segment by index or existing name/alias."""
        if not isinstance(new_name, str):
            raise TypeError("segment name must be a string")
        if new_name == "":
            raise ValueError("segment name cannot be empty")
        if not self._segments:
            raise RuntimeError("No recorded segments available to rename")
        idx = self._resolve_segment_index(index_or_old)
        unique = self._unique_segment_name(new_name, skip_index=idx)
        if self._segments[idx].name == unique:
            return
        import dataclasses

        self._segments[idx] = dataclasses.replace(self._segments[idx], name=unique)
        self._results_view = None

    def session_state_summary(self) -> dict[str, Any]:
        """Return a small diagnostic summary of the current SessionState."""
        can_resume, reason = self.can_resume()
        state = self._session_state
        summary = {
            "t": state.t_curr,
            "step": state.step_count,
            "dt": state.dt_curr,
            "status": Status(state.status).name,
            "stepper_name": self.model.stepper_name,
            "can_resume": can_resume,
            "reason": reason,
        }
        summary["stepper_config"] = self._summarize_stepper_config(state.stepper_cfg)
        return summary

    def stepper_config(self, **kwargs: Any) -> np.ndarray:
        """
        Return the stored stepper configuration, optionally applying overrides for future runs.

        Args:
            **kwargs: Stepper-specific runtime parameters.

        Returns:
            Copy of the stored stepper configuration array.
        """
        if not kwargs:
            return np.array(self._session_state.stepper_cfg, dtype=np.float64, copy=True)
        new_cfg = self._build_stepper_config(kwargs, self._session_state.stepper_cfg)
        self._session_state.stepper_cfg = np.array(new_cfg, dtype=np.float64, copy=True)
        self._sync_initial_snapshot_config(new_cfg)
        return np.array(new_cfg, dtype=np.float64, copy=True)

    def config(
        self,
        *,
        dt: Optional[float] = None,
        max_steps: Optional[int] = None,
        record: Optional[bool] = None,
        record_interval: Optional[int] = None,
        cap_rec: Optional[int] = None,
        cap_evt: Optional[int] = None,
        **stepper_kwargs: Any,
    ) -> None:
        """
        Configure persistent defaults for this Sim.

        Simulation-level defaults:
          - dt: nominal time step / label spacing
          - max_steps: default safety / iteration limit
          - record: default recording flag
          - record_interval: default recording interval
          - cap_rec: default initial record capacity
          - cap_evt: default initial event capacity
    (transient warm-up duration is configured per-run via run())

        Any remaining keyword arguments are forwarded to stepper_config()
        as stepper-specific runtime parameters.

        Explicit arguments passed to run() always override these defaults.
        """
        # dt uses the same storage as run(): _nominal_dt
        if dt is not None:
            new_dt = float(dt)
            if new_dt <= 0.0:
                raise ValueError("dt must be positive")
            self._nominal_dt = new_dt
            # Keep session state summary consistent; snapshots remain historical.
            self._session_state.dt_curr = new_dt

        if max_steps is not None:
            ms = int(max_steps)
            if ms <= 0:
                raise ValueError("max_steps must be positive")
            self._run_defaults.max_steps = ms

        if record is not None:
            self._run_defaults.record = bool(record)

        if record_interval is not None:
            ri = int(record_interval)
            if ri <= 0:
                raise ValueError("record_interval must be >= 1")
            self._run_defaults.record_interval = ri

        if cap_rec is not None:
            cr = int(cap_rec)
            if cr <= 0:
                raise ValueError("cap_rec must be positive")
            self._run_defaults.cap_rec = cr

        if cap_evt is not None:
            ce = int(cap_evt)
            if ce <= 0:
                raise ValueError("cap_evt must be positive")
            self._run_defaults.cap_evt = ce

        # Note: transient is intentionally not stored here; transient
        # warm-up duration should be provided to run() per invocation.

        if stepper_kwargs:
            self.stepper_config(**stepper_kwargs)

    def assign(
        self,
        mapping: Optional[Mapping[str, Any]] = None,
        /,
        *,
        clear_history: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Assign values to states and parameters by name on the current session.

        Accepts both mapping and keyword arguments. Values are resolved automatically:
        names are first checked against state names, then parameter names. Unknown
        names raise ValueError with suggestions.

        Args:
            mapping: Dictionary-like mapping of {name: value}
            clear_history: If True, clears results/history but leaves session state unchanged
            **kwargs: Keyword arguments override mapping for the same key

        Examples:
            >>> sim.assign({"v": -65.0, "I": 10.0})
            >>> sim.assign(v=-65.0, I=12.0)
            >>> sim.assign({"v": -65.0}, I=15.0, clear_history=True)

        Notes:
            - Names are resolved: states first, then params
            - Unknown names raise ValueError with "did you mean?" suggestions
            - Values are cast to model dtype (warnings on precision loss)
            - clear_history=False: keeps results/history
            - clear_history=True: clears results but not session state (time, workspace, etc.)
            - Does not modify snapshots or stepper config
            - Always affects the next run() unless overridden by explicit ic/params
        """
        # Merge mapping and kwargs
        updates: Dict[str, Any] = {}
        if mapping is not None:
            if not isinstance(mapping, Mapping):
                raise TypeError("mapping must be a Mapping (dict-like) or None")
            updates.update(mapping)
        updates.update(kwargs)

        if not updates:
            return

        # Build lookup tables from model spec
        state_names = list(self.model.spec.states)
        param_names = list(self.model.spec.params)
        state_idx = {name: i for i, name in enumerate(state_names)}
        param_idx = {name: i for i, name in enumerate(param_names)}

        # Partition updates into states, params, and unknown
        state_updates: Dict[str, Any] = {}
        param_updates: Dict[str, Any] = {}
        unknown: list[str] = []

        for key, val in updates.items():
            if key in state_idx:
                state_updates[key] = val
            elif key in param_idx:
                param_updates[key] = val
            else:
                unknown.append(key)

        # Handle unknown names with suggestions
        if unknown:
            candidates = state_names + param_names
            suggestions = []
            for uk in sorted(unknown):
                suggestion = _did_you_mean(uk, candidates)
                if suggestion:
                    suggestions.append(f"'{uk}' (did you mean '{suggestion}'?)")
                else:
                    suggestions.append(f"'{uk}'")
            raise ValueError(
                f"Unknown state/param name(s): {', '.join(suggestions)}. "
                f"Valid states: {state_names}, params: {param_names}"
            )

        # Cast values to model dtype
        casted_states: Dict[str, Any] = {}
        if state_updates:
            casted_states = _cast_values_to_dtype(
                {k: float(v) for k, v in state_updates.items()},
                self._dtype,
                "<assign>",
                "state",
            )

        casted_params: Dict[str, Any] = {}
        if param_updates:
            casted_params = _cast_values_to_dtype(
                {k: float(v) for k, v in param_updates.items()},
                self._dtype,
                "<assign>",
                "param",
            )

        # Apply updates in-place on current SessionState
        if casted_states:
            y = self._session_state.y_curr
            for name, val in casted_states.items():
                y[state_idx[name]] = val

        if casted_params:
            p = self._session_state.params_curr
            for name, val in casted_params.items():
                p[param_idx[name]] = val

        # Handle clear_history
        if clear_history:
            self._result_accum = None
            self._raw_results = None
            self._results_view = None
            self._segments = []
            self._pending_run_tag = None
            self._pending_run_cfg_hash = None
            self._last_run_was_resume = False

    def state_vector(
        self,
        *,
        source: Literal["session", "model", "snapshot"] = "session",
        snapshot: str = "initial",
        copy: bool = True,
    ) -> np.ndarray:
        """
        Return state values in DSL declaration order as a 1D array.

        source:
          - "session"  (default): current SessionState values
          - "model"              : DSL-declared state_ic
          - "snapshot"           : state values from a named snapshot

        snapshot:
          - Snapshot name when source="snapshot" (default "initial")

        copy:
          - If True (default), return a copy. If False, return a view into
            the underlying storage (mutating it is your responsibility).
        """
        if source == "session":
            arr = self._session_state.y_curr
        elif source == "model":
            arr = self.model.spec.state_ic
        elif source == "snapshot":
            snap = self._resolve_snapshot(snapshot)
            arr = snap.state.y_curr
        else:
            raise ValueError("source must be 'session', 'model', or 'snapshot'")

        return np.array(arr, dtype=self._dtype, copy=copy)

    def param_vector(
        self,
        *,
        source: Literal["session", "model", "snapshot"] = "session",
        snapshot: str = "initial",
        copy: bool = True,
    ) -> np.ndarray:
        """
        Return parameter values in DSL declaration order as a 1D array.

        source:
          - "session"  (default): current SessionState params
          - "model"              : DSL-declared param_vals
          - "snapshot"           : params from a named snapshot

        snapshot:
          - Snapshot name when source="snapshot" (default "initial")
        """
        if source == "session":
            arr = self._session_state.params_curr
        elif source == "model":
            arr = self.model.spec.param_vals
        elif source == "snapshot":
            snap = self._resolve_snapshot(snapshot)
            arr = snap.state.params_curr
        else:
            raise ValueError("source must be 'session', 'model', or 'snapshot'")

        return np.array(arr, dtype=self._dtype, copy=copy)

    def state_dict(
        self,
        *,
        source: Literal["session", "model", "snapshot"] = "session",
        snapshot: str = "initial",
    ) -> dict[str, float]:
        """
        Return state values as a name->value dictionary.

        source:
          - "session"  (default): current SessionState params
          - "model"              : DSL-declared param_vals
          - "snapshot"           : params from a named snapshot

        snapshot:
          - Snapshot name when source="snapshot" (default "initial")
        """
        arr = self.state_vector(source=source, snapshot=snapshot, copy=False)
        return {name: float(arr[i]) for i, name in enumerate(self.model.spec.states)}

    def param_dict(
        self,
        *,
        source: Literal["session", "model", "snapshot"] = "session",
        snapshot: str = "initial",
    ) -> dict[str, float]:
        """
        Return parameter values as a name->value dictionary.

        source:
          - "session"  (default): current SessionState params
          - "model"              : DSL-declared param_vals
          - "snapshot"           : params from a named snapshot

        snapshot:
          - Snapshot name when source="snapshot" (default "initial")
        """
        arr = self.param_vector(source=source, snapshot=snapshot, copy=False)
        return {name: float(arr[i]) for i, name in enumerate(self.model.spec.params)}

    def state(self, name: str) -> float:
        """
        Return current session state value by name (scalar float).

        Example:
            >>> sim.run(T=100, record=False)
            >>> v_final = sim.state("v")
        """
        names = list(self.model.spec.states)
        idx = {n: i for i, n in enumerate(names)}
        if name in idx:
            return float(self._session_state.y_curr[idx[name]])

        suggestion = _did_you_mean(name, names)
        if suggestion:
            raise KeyError(
                f"Unknown state {name!r}. Did you mean {suggestion!r}? "
                f"Valid states: {names}"
            )
        raise KeyError(f"Unknown state {name!r}. Valid states: {names}")

    def param(self, name: str) -> float:
        """
        Return current session parameter value by name (scalar float).

        Example:
            >>> I = sim.param("I")
        """
        names = list(self.model.spec.params)
        idx = {n: i for i, n in enumerate(names)}
        if name in idx:
            return float(self._session_state.params_curr[idx[name]])

        suggestion = _did_you_mean(name, names)
        if suggestion:
            raise KeyError(
                f"Unknown param {name!r}. Did you mean {suggestion!r}? "
                f"Valid params: {names}"
            )
        raise KeyError(f"Unknown param {name!r}. Valid params: {names}")

    def can_resume(self) -> tuple[bool, Optional[str]]:
        """Return (bool, reason) describing whether resume() may be invoked safely."""
        diff = _diff_pins(self._pins, self._session_state.pins)
        if diff:
            return False, f"pin mismatch: {diff}"
        return True, None

    def compat_check(self, snapshot: Snapshot | str) -> tuple[bool, Dict[str, Tuple[Any, Any]]]:
        """Compare the model pins against a snapshot's pins."""
        snap = self._resolve_snapshot(snapshot)
        diff = _diff_pins(self._pins, snap.state.pins)
        return (len(diff) == 0, diff)

    def export_snapshot(
        self,
        path: str | Path,
        *,
        source: Literal["current", "snapshot"] = "current",
        name: Optional[str] = None,
    ) -> None:
        """
        Export session state to disk as a strict snapshot file (.npz format).
        
        Snapshots are full session images (integrator state) and do not include results/history.
        
        Args:
            path: Target file path (will be overwritten atomically if it exists)
            source: "current" for current session state, "snapshot" for named in-memory snapshot
            name: Required when source="snapshot", ignored otherwise
        
        Raises:
            ValueError: Invalid arguments (e.g., missing name when source="snapshot")
        """
        path = Path(path)
        
        # Validate arguments
        if source == "snapshot" and name is None:
            raise ValueError("name is required when source='snapshot'")
        if source == "current" and name is not None:
            raise ValueError("name should not be provided when source='current'")
        
        # Pick source state and build metadata
        state, snap_name, description, time_shift, nominal_dt = self._snapshot_pick_state(
            source, name
        )
        meta = self._snapshot_build_meta(state, snap_name, description, time_shift, nominal_dt)
        
        # Write atomically
        self._snapshot_write_npz(path, meta, state)

    def import_snapshot(self, path: str | Path) -> None:
        """
        Import session state from a snapshot file, replacing current session entirely.
        
        Import requires compatibility (pin match) and replaces the session; history is cleared.
        
        Args:
            path: Snapshot file path to import from
        
        Raises:
            ValueError: File format issues (schema, missing keys, shape mismatches)
            RuntimeError: Pin compatibility issues
        """
        path = Path(path)
        
        # Read and validate file
        meta, y, params, workspace, stepper_config = self._snapshot_read_npz(path)

        # Restore session state
        restored_state = self._snapshot_restore(meta, y, params, workspace, stepper_config)
        
        # Replace current session and clear results/history
        self._session_state = restored_state
        self._result_accum = None
        self._raw_results = None
        self._results_view = None
        self._time_shift = float(meta["time_shift"])
        self._nominal_dt = float(meta["nominal_dt"])
        self._segments = []
        self._pending_run_tag = None
        self._pending_run_cfg_hash = None
        self._last_run_was_resume = False

    def inspect_snapshot(self, path: str | Path) -> dict[str, Any]:
        """
        Return parsed metadata from a snapshot file without modifying sim state.
        
        Args:
            path: Snapshot file path to inspect
        
        Returns:
            Dictionary containing metadata from the snapshot file
            
        Raises:
            ValueError: File format issues
        """
        path = Path(path)
        
        try:
            with np.load(path, allow_pickle=False) as npz_file:
                if "meta.json" not in npz_file.files:
                    raise ValueError("Missing 'meta.json' in snapshot file")
                
                meta_bytes = npz_file["meta.json"]
                meta_str = meta_bytes.tobytes().decode("utf-8")
                meta = json.loads(meta_str)
                
                return meta
        except (OSError, IOError) as e:
            raise ValueError(f"Cannot read snapshot file: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in meta.json: {e}")

    # ---------------------------- presets API ----------------------------------

    def list_presets(self, pattern: str = "*") -> list[str]:
        """
        Return preset names in this Sim's presets bank matching a glob pattern.
        
        Args:
            pattern: Glob pattern (supports *, ?, []). Default "*" returns all.
        
        Returns:
            List of matching preset names, sorted alphabetically.
        """
        all_names = list(self._presets.keys())
        matched = [name for name in all_names if fnmatch.fnmatch(name, pattern)]
        return sorted(matched)

    def apply_preset(self, name: str) -> None:
        """
        Apply a preset from this Sim's bank to current session.

        Updates params and/or states present in the preset. Does NOT touch time, dt,
        step_count, stepper workspace, or results/history.

        Args:
            name: Preset name (must exist in bank)

        Raises:
            KeyError: Preset not found
            ValueError: Validation failures (unknown names, type mismatches, overflow)
        
        Validation:
          - All preset param keys must exist in model params
          - All preset state keys must exist in model states
          - Preset must define at least one param or state
          - Values cast to model dtype with warnings on precision loss
          - Atomic: validates everything before applying
        """
        if name not in self._presets:
            available = sorted(self._presets.keys())
            raise KeyError(
                f"Preset '{name}' not found in bank. Available: {available}"
            )
        
        preset = self._presets[name]
        
        # Validate names
        param_names = set(self.model.spec.params)
        state_names = set(self.model.spec.states)
        _validate_preset_names(preset, param_names, state_names)
        
        # Cast values (with warnings/errors)
        casted_params = _cast_values_to_dtype(
            preset.params, self._dtype, preset.name, "param"
        )
        state_values = preset.states or {}
        casted_states = _cast_values_to_dtype(
            state_values, self._dtype, preset.name, "state"
        ) if state_values else {}

        # Atomic apply: params first, then states
        # Update params
        param_array = self._session_state.params_curr
        if casted_params:
            for i, pname in enumerate(self.model.spec.params):
                if pname in casted_params:
                    param_array[i] = casted_params[pname]

        # Update states (if present)
        if casted_states:
            state_array = self._session_state.y_curr
            for i, sname in enumerate(self.model.spec.states):
                if sname in casted_states:
                    state_array[i] = casted_states[sname]

    def load_preset(
        self,
        name_or_pattern: str,
        path: str | Path,
        *,
        on_conflict: Literal["error", "keep", "replace"] = "error",
    ) -> int:
        """
        Import preset(s) from a TOML file into this Sim's presets bank.
        
        Args:
            name_or_pattern: Exact name or glob pattern ("*", "fast_*", etc.)
            path: Path to TOML file
            on_conflict: How to handle name conflicts:
                - "error" (default): raise if name exists
                - "keep": skip file preset, keep existing (WARNING)
                - "replace": overwrite existing with file preset (WARNING)
        
        Returns:
            Number of presets successfully loaded into bank
        
        Raises:
            ValueError: File format errors, validation failures
        
        File format:
          - Must have [__presets__].schema = "dynlib-presets-v1"
          - Presets under [presets.<name>.params] and optionally [presets.<name>.states]
          - Validates structural compatibility with active model
        """
        path = Path(path)
        doc = _toml_read(path)
        
        # Validate schema header
        header = doc.get("__presets__", {})
        if header.get("schema") != "dynlib-presets-v1":
            raise ValueError(
                f"Invalid or missing schema in {path}. "
                f"Expected [__presets__].schema = 'dynlib-presets-v1'"
            )
        
        # Extract presets section
        presets_section = doc.get("presets", {})
        if not isinstance(presets_section, dict):
            raise ValueError(f"[presets] must be a table in {path}")
        
        # Find matching presets
        all_preset_names = list(presets_section.keys())
        if name_or_pattern == "*" or any(c in name_or_pattern for c in "*?[]"):
            # Glob pattern
            matched_names = [n for n in all_preset_names if fnmatch.fnmatch(n, name_or_pattern)]
        else:
            # Exact name
            matched_names = [name_or_pattern] if name_or_pattern in all_preset_names else []
        
        if not matched_names:
            raise ValueError(
                f"No presets matching '{name_or_pattern}' found in {path}. "
                f"Available: {sorted(all_preset_names)}"
            )
        
        # Parse and validate each matched preset
        param_names = set(self.model.spec.params)
        state_names = set(self.model.spec.states)
        loaded_count = 0
        
        # Track duplicates within the file
        file_presets: Dict[str, _PresetData] = {}
        
        for pname in matched_names:
            preset_table = presets_section[pname]
            if not isinstance(preset_table, dict):
                raise ValueError(f"[presets.{pname}] must be a table in {path}")
            
            # Parse params (optional, may be empty)
            params_table = preset_table.get("params")
            if params_table is None:
                params_table = {}
            elif not isinstance(params_table, dict):
                raise ValueError(f"[presets.{pname}].params must be a table in {path}")
            
            # Parse states (optional, may be empty)
            states_table = preset_table.get("states")
            if states_table is None:
                states_table = {}
            elif not isinstance(states_table, dict):
                raise ValueError(f"[presets.{pname}].states must be a table in {path}")

            if len(params_table) == 0 and len(states_table) == 0:
                raise ValueError(
                    f"[presets.{pname}] must define at least one param or state"
                )
            
            preset_data = _PresetData(
                name=pname,
                params={k: float(v) for k, v in params_table.items()},
                states=(
                    {k: float(v) for k, v in states_table.items()}
                    if states_table
                    else None
                ),
                source="file",
            )
            
            # Validate against model
            _validate_preset_names(preset_data, param_names, state_names)
            
            # Check for duplicates within file (last wins with WARNING)
            if pname in file_presets:
                warnings.warn(
                    f"Preset '{pname}' defined multiple times in {path}; using last definition",
                    RuntimeWarning,
                    stacklevel=2,
                )
            file_presets[pname] = preset_data
        
        # Merge into bank according to on_conflict policy
        for pname, preset_data in file_presets.items():
            if pname in self._presets:
                if on_conflict == "error":
                    raise ValueError(
                        f"Preset '{pname}' already exists in bank. "
                        f"Use on_conflict='keep' or 'replace' to resolve."
                    )
                elif on_conflict == "keep":
                    warnings.warn(
                        f"Skipping file preset '{pname}' (already in bank, on_conflict='keep')",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                elif on_conflict == "replace":
                    warnings.warn(
                        f"Replacing bank preset '{pname}' with file preset (on_conflict='replace')",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            
            self._presets[pname] = preset_data
            loaded_count += 1
        
        return loaded_count

    def add_preset(
        self,
        name: str,
        *,
        states: Mapping[str, float] | np.ndarray | None = None,
        params: Mapping[str, float] | np.ndarray | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add or update a preset in this Sim's in-memory preset bank.

        Args:
            name:
                Preset name (key in the bank).
            states:
                Optional state values for this preset.

                - If both `states` and `params` are None:
                      use the current session's states and params.
                - If `states` is provided:
                      use only these values for states (no fallback).
                      Can be:
                        * 1D ndarray: interpreted in state declaration order.
                        * Mapping: keys are state names (can be partial).
            params:
                Optional parameter values for this preset.

                - If both `states` and `params` are None:
                      use the current session's states and params.
                - If `params` is provided:
                      use only these values for params (no fallback).
                      Can be:
                        * 1D ndarray: interpreted in param declaration order.
                        * Mapping: keys are param names (can be partial).
            overwrite:
                If False, raise ValueError when a preset with this name
                already exists in the bank. If True, replace it.

        Behavior:
            - If both `states` and `params` are None:
                  preset = {states = current session states,
                            params = current session params}
            - If at least one of `states` / `params` is provided:
                  preset contains only those sections that are provided
                  (params-only, states-only, or both).

        Raises:
            ValueError:
                - if overwrite is False and the name already exists.
                - if neither states nor params can be determined.
            TypeError:
                - if `states` / `params` are of an unsupported type.
        """
        # ----------------------- basic conflict handling -----------------------
        bank = self._presets

        if not overwrite and name in bank:
            raise ValueError(
                f"Preset {name!r} already exists in this Sim's bank "
                "(use overwrite=True to replace it)."
            )

        state_dict: dict[str, float] | None = None
        param_dict: dict[str, float] | None = None

        # ----------------------------------------------------------------------
        # Helper: normalize ndarray or mapping into {name: float} dict
        # ----------------------------------------------------------------------
        def _normalize(
            names: list[str],
            values: Mapping[str, float] | np.ndarray,
        ) -> dict[str, float]:
            if isinstance(values, np.ndarray):
                if values.ndim != 1 or values.shape[0] != len(names):
                    raise ValueError(
                        f"Expected 1D vector of length {len(names)} for preset "
                        f"values, got shape {values.shape!r}"
                    )
                return {
                    name: float(v) for name, v in zip(names, values, strict=True)
                }

            if isinstance(values, Mapping):
                # allow partial mappings, just coerce to float
                return {str(k): float(v) for k, v in values.items()}

            raise TypeError(
                f"Preset values must be a mapping or 1D numpy array, "
                f"got {type(values)!r}"
            )

        # ----------------------------------------------------------------------
        # Case 1: both states and params None -> use current session
        # ----------------------------------------------------------------------
        if states is None and params is None:
            session = self._session_state
            state_names = list(self.model.spec.states)
            param_names = list(self.model.spec.params)

            if state_names and session.y_curr.size:
                state_dict = {
                    n: float(v)
                    for n, v in zip(state_names, np.asarray(session.y_curr), strict=True)
                }

            if param_names and session.params_curr.size:
                param_dict = {
                    n: float(v)
                    for n, v in zip(param_names, np.asarray(session.params_curr), strict=True)
                }

        # ----------------------------------------------------------------------
        # Case 2: at least one explicit argument -> use only those sections
        # ----------------------------------------------------------------------
        else:
            if states is not None:
                state_names = list(self.model.spec.states)
                state_dict = _normalize(state_names, states)

            if params is not None:
                param_names = list(self.model.spec.params)
                param_dict = _normalize(param_names, params)

        if state_dict is None and param_dict is None:
            raise ValueError(
                "Cannot create preset: neither states nor params are defined. "
                "Provide values explicitly or ensure the session has states/params."
            )

        param_values = param_dict or {}
        state_values = state_dict if state_dict else None

        preset = _PresetData(
            name=name,
            params=param_values,
            states=state_values,
            source="session",
        )

        _validate_preset_names(
            preset,
            set(self.model.spec.params),
            set(self.model.spec.states),
        )

        bank[name] = preset

    def save_preset(
        self,
        name: str,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Export a preset from this Sim's bank to a TOML file.

        Args:
            name:
                Preset name (must exist in this Sim's preset bank).
            path:
                Target TOML file path (created if missing).
            overwrite:
                If True, replace existing preset with this name in the file.
                If False, raise ValueError if a preset with this name already
                exists in the file.

        Behavior:
            - Reads the preset entry from self._presets[name].
            - Writes:
                  [presets.<name>.states]   (if present)
                  [presets.<name>.params]   (if present)
              into the TOML file at `path`, using existing TOML helpers.

        Raises:
            KeyError:
                If the preset name is not found in this Sim's bank.
            ValueError:
                On conflicts when overwrite=False, or invalid existing [presets].
        """
        path = Path(path)

        bank = self._presets
        try:
            preset = bank[name]
        except KeyError as exc:
            raise KeyError(f"Preset {name!r} not found in this Sim's bank") from exc

        # --------------------------- load existing TOML ------------------------
        if path.exists():
            data = _toml_read(path)  # type: ignore[name-defined]
        else:
            data = {}

        header = data.setdefault("__presets__", {})
        if not isinstance(header, dict):
            raise ValueError(
                f"Expected [__presets__] table in {path}, "
                f"found {type(header).__name__}"
            )
        schema = header.setdefault("schema", "dynlib-presets-v1")
        if schema != "dynlib-presets-v1":
            raise ValueError(
                f"[__presets__].schema in {path} must be 'dynlib-presets-v1', "
                f"found {schema!r}"
            )

        presets_tbl = data.setdefault("presets", {})
        if not isinstance(presets_tbl, dict):
            raise ValueError(
                f"Expected [presets] table in {path}, "
                f"found {type(presets_tbl).__name__}"
            )

        if not overwrite and name in presets_tbl:
            raise ValueError(
                f"Preset {name!r} already exists in {path} "
                "(use overwrite=True to replace it)."
            )

        # preset is expected to be dict-like with optional 'states'/'params' keys
        states = getattr(preset, "states", None)
        params = getattr(preset, "params", None)
        if isinstance(preset, dict):
            # prefer dict keys if present
            states = preset.get("states", states)
            params = preset.get("params", params)

        entry: dict[str, dict[str, float]] = {}
        if states:
            entry["states"] = dict(states)
        if params:
            entry["params"] = dict(params)

        if not entry:
            raise ValueError(
                f"Preset {name!r} has neither states nor params; "
                "nothing to save."
            )

        presets_tbl[name] = entry

        # ----------------------------- atomic write ----------------------------
        _toml_write_atomic(path, data)  # type: ignore[name-defined]

    # ---------------------------- internal helpers -----------------------------

    def _resolve_recording_selection(
        self, record_vars: list[str] | None
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Convert variable names to index arrays for selective recording.
        
        Args:
            record_vars: List of variable names to record, or None for default (all states)
        
        Returns:
            state_indices: np.ndarray of state indices to record, shape (n_rec_states,)
            aux_indices: np.ndarray of aux indices to record, shape (n_rec_aux,)
            state_names: list of state variable names being recorded
            aux_names: list of aux variable names being recorded (without "aux." prefix)
        """
        if record_vars is None:
            # Default: all states, no aux (backward compatible)
            state_indices = np.arange(self._n_state, dtype=np.int32)
            aux_indices = np.array([], dtype=np.int32)
            state_names = list(self.model.spec.states)
            aux_names = []
            return state_indices, aux_indices, state_names, aux_names
        
        all_state_names = list(self.model.spec.states)
        all_aux_names = list(self.model.spec.aux.keys()) if self.model.spec.aux else []
        
        state_indices_list = []
        aux_indices_list = []
        state_names = []
        aux_names = []
        
        for var in record_vars:
            if var.startswith("aux."):
                # Explicit aux variable with prefix
                aux_name = var[4:]  # Remove "aux." prefix
                if aux_name not in all_aux_names:
                    raise ValueError(f"Unknown aux variable: {aux_name}")
                aux_indices_list.append(all_aux_names.index(aux_name))
                aux_names.append(aux_name)
            elif var in all_state_names:
                # State variable
                state_indices_list.append(all_state_names.index(var))
                state_names.append(var)
            elif var in all_aux_names:
                # Aux variable without prefix (auto-detected)
                aux_indices_list.append(all_aux_names.index(var))
                aux_names.append(var)
            else:
                # Not found - provide helpful error message
                raise ValueError(
                    f"Unknown variable: '{var}'. "
                    f"Available states: {all_state_names}. "
                    f"Available aux: {all_aux_names}."
                )
        
        return (
            np.array(state_indices_list, dtype=np.int32),
            np.array(aux_indices_list, dtype=np.int32),
            state_names,
            aux_names
        )

    def _resolve_observers(self, observers, *, record_interval: Optional[int]):
        """
        Accept ObserverModule instances directly, ObserverFactory callables, or
        sequences of ObserverModule.

        ObserverFactory callables are invoked with (model, sim, record_interval)
        and must return an ObserverModule. Sequences are wrapped into a
        CombinedObserver to run in a single pass.
        """

        def _is_observer_factory(target: object) -> bool:
            return callable(target) and bool(getattr(target, "__observer_factory__", False))

        def _materialize(target):
            if target is None:
                return None
            if isinstance(target, ObserverModule):
                return target
            if _is_observer_factory(target):
                module = target(self.model, self, record_interval)
                if not isinstance(module, ObserverModule):
                    raise TypeError("ObserverFactory must return an ObserverModule")
                return module
            raise TypeError("observers must be an ObserverModule or ObserverFactory")

        if observers is None:
            return None

        if isinstance(observers, Sequence) and not isinstance(observers, (str, bytes)):
            if len(observers) == 0:
                raise ValueError("observers sequence cannot be empty")
            if any(not isinstance(item, ObserverModule) for item in observers):
                raise TypeError("observers sequence must contain ObserverModule instances")
            modules = tuple(observers)
            if len(modules) == 1:
                modules[0].validate_stepper(self._stepper_spec)
                return modules[0]
            combined = CombinedObserver(modules)
            combined.validate_stepper(self._stepper_spec)
            return combined

        module = _materialize(observers)
        if module is not None:
            module.validate_stepper(self._stepper_spec)
        return module

    def _load_inline_presets(self) -> None:
        """Auto-populate presets bank from model.spec.presets."""
        param_names = set(self.model.spec.params)
        state_names = set(self.model.spec.states)
        for preset_spec in self.model.spec.presets:
            params_dict = {k: float(v) for k, v in preset_spec.params.items()}
            states_dict = (
                {k: float(v) for k, v in preset_spec.states.items()}
                if preset_spec.states
                else {}
            )
            if not params_dict and not states_dict:
                raise ValueError(
                    f"Inline preset '{preset_spec.name}' is empty; "
                    "must define at least one param or state."
                )

            preset_data = _PresetData(
                name=preset_spec.name,
                params=params_dict,
                states=states_dict or None,
                source="inline",
            )

            _validate_preset_names(preset_data, param_names, state_names)
            
            # Defensive: shouldn't happen with fresh Sim, but log if it does
            if preset_data.name in self._presets:
                warnings.warn(
                    f"Inline preset '{preset_data.name}' already in bank; keeping first",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            
            self._presets[preset_data.name] = preset_data

    def _snapshot_pick_state(
        self, source: Literal["current", "snapshot"], name: Optional[str]
    ) -> tuple[SessionState, str, str, float, float]:
        """
        Returns the SessionState to serialize, plus snap_name and description.
        """
        if source == "current":
            return (
                self._session_state,
                "current",
                "Current session state",
                float(self._time_shift),
                float(self._nominal_dt),
            )
        else:  # source == "snapshot"
            if name is None:
                raise ValueError("name is required when source='snapshot'")
            snapshot = self._resolve_snapshot(name)
            return (
                snapshot.state,
                snapshot.name,
                snapshot.description,
                float(snapshot.time_shift),
                float(snapshot.nominal_dt),
            )

    def _snapshot_build_meta(
        self,
        state: SessionState,
        snap_name: str,
        description: str,
        time_shift: float,
        nominal_dt: float,
    ) -> dict[str, Any]:
        """
        Builds the dict for meta.json using pins, names from model.spec, and values from state.
        """
        spec = self.model.spec
        
        return {
            "schema": "dynlib-snapshot-v1",
            "created_at": _now_iso(),
            "name": snap_name,
            "description": description,
            "pins": {
                "spec_hash": self._pins.spec_hash,
                "stepper_name": self._pins.stepper_name,
                "workspace_sig": list(self._pins.workspace_sig),
                "dtype_token": self._pins.dtype_token,
                "dynlib_version": self._pins.dynlib_version,
            },
            "n_state": len(spec.states),
            "n_params": len(spec.params),
            "state_names": list(spec.states),
            "param_names": list(spec.params),
            "t_curr": float(state.t_curr),
            "dt_curr": float(state.dt_curr),
            "step_count": int(state.step_count),
            "status": int(state.status),
            "time_shift": float(time_shift),
            "nominal_dt": float(nominal_dt),
            "stepper_config_names": list(self._stepper_config_names),
            "stepper_config_values": state.stepper_cfg.tolist(),
        }

    def _snapshot_write_npz(self, path: Path, meta: dict[str, Any], state: SessionState) -> None:
        """
        Opens a temp file, writes meta.json and arrays, then atomically replaces path.
        """
        # Serialize metadata as JSON bytes
        meta_json = json.dumps(meta, indent=2)
        meta_bytes = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)
        
        # Prepare arrays to save
        arrays_to_save = {
            "meta.json": meta_bytes,
            "y": state.y_curr,
            "params": state.params_curr,
        }
        
        # Add workspace arrays if non-empty
        for bucket, ws_vals in state.workspace.items():
            for ws_name, ws_array in ws_vals.items():
                if isinstance(ws_array, np.ndarray) and ws_array.size > 0:
                    arrays_to_save[f"workspace/{bucket}/{ws_name}"] = ws_array

        if state.stepper_cfg.size > 0:
            arrays_to_save["stepper_config"] = state.stepper_cfg
        
        # Write atomically using temporary file
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            delete=False, 
            dir=path.parent, 
            prefix=f".{path.name}_tmp_", 
            suffix=".npz"
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            np.savez_compressed(tmp_path, **arrays_to_save)
            tmp_path.replace(path)
        except Exception:
            # Clean up temp file on error
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _snapshot_read_npz(
        self, path: Path
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """
        Reads and returns (meta, y, params, workspace).
        """
        try:
            with np.load(path, allow_pickle=False) as npz_file:
                # Read and parse metadata
                if "meta.json" not in npz_file.files:
                    raise ValueError("Missing 'meta.json' in snapshot file")
                
                meta_bytes = npz_file["meta.json"]
                meta_str = meta_bytes.tobytes().decode("utf-8")
                meta = json.loads(meta_str)
                
                # Read required arrays
                if "y" not in npz_file.files:
                    raise ValueError("Missing 'y' array in snapshot file")
                if "params" not in npz_file.files:
                    raise ValueError("Missing 'params' array in snapshot file")
                
                y = npz_file["y"]
                params = npz_file["params"]
                
                # Read workspace arrays grouped by bucket (stepper/runtime)
                workspace: WorkspaceSnapshot = {}
                workspace_prefix = "workspace/"
                for key in npz_file.files:
                    if key.startswith(workspace_prefix):
                        remainder = key[len(workspace_prefix):]
                        if not remainder:
                            continue
                        if "/" in remainder:
                            bucket, ws_name = remainder.split("/", 1)
                        else:
                            bucket, ws_name = "stepper", remainder
                        if ws_name:
                            workspace.setdefault(bucket, {})[ws_name] = npz_file[key]
                stepper_config = (
                    np.array(npz_file["stepper_config"], dtype=np.float64)
                    if "stepper_config" in npz_file.files
                    else np.array([], dtype=np.float64)
                )

                return meta, y, params, workspace, stepper_config

        except (OSError, IOError) as e:
            raise ValueError(f"Cannot read snapshot file: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in meta.json: {e}")
        except Exception as e:
            raise ValueError(f"Error reading snapshot file: {e}")

    def _snapshot_restore(
        self,
        meta: dict[str, Any],
        y: np.ndarray,
        params: np.ndarray,
        workspace: WorkspaceSnapshot,
        stepper_config: np.ndarray,
    ) -> SessionState:
        """
        Validates schema, shapes, pins; returns a new SessionState built from file content.
        """
        # Validate schema
        if meta.get("schema") != "dynlib-snapshot-v1":
            raise ValueError(f"Unsupported schema: {meta.get('schema')}")
        
        # Validate shapes
        expected_n_state = len(self.model.spec.states)
        expected_n_params = len(self.model.spec.params)
        
        if y.shape != (expected_n_state,):
            raise ValueError(
                f"State vector shape mismatch: expected ({expected_n_state},), got {y.shape}"
            )
        
        if params.shape != (expected_n_params,):
            raise ValueError(
                f"Parameters shape mismatch: expected ({expected_n_params},), got {params.shape}"
            )
        
        # Validate pins
        file_pins_data = meta.get("pins", {})
        file_pins = SessionPins(
            spec_hash=file_pins_data.get("spec_hash", ""),
            stepper_name=file_pins_data.get("stepper_name", ""),
            workspace_sig=tuple(file_pins_data.get("workspace_sig", [])),
            dtype_token=file_pins_data.get("dtype_token", ""),
            dynlib_version=file_pins_data.get("dynlib_version", ""),
        )
        
        diff = _diff_pins(self._pins, file_pins)
        if diff:
            raise RuntimeError(f"Snapshot incompatible: {diff}")
        
        # Build new SessionState
        cfg_values = meta.get("stepper_config_values", [])
        cfg_array = np.array(cfg_values, dtype=np.float64) if cfg_values else np.array([], dtype=np.float64)
        if stepper_config.size:
            cfg_array = np.array(stepper_config, dtype=np.float64, copy=True)

        return SessionState(
            t_curr=float(meta["t_curr"]),
            y_curr=np.array(y, dtype=self._dtype, copy=True),
            params_curr=np.array(params, dtype=self._dtype, copy=True),
            dt_curr=float(meta["dt_curr"]),
            step_count=int(meta["step_count"]),
            workspace=_copy_workspace_dict(workspace),
            stepper_cfg=cfg_array,
            status=int(meta["status"]),
            pins=self._pins,
        )

    def _bootstrap_session_state(self) -> SessionState:
        spec = self.model.spec
        sim_defaults = spec.sim
        y0 = np.array(spec.state_ic, dtype=self._dtype, copy=True)
        params0 = np.array(spec.param_vals, dtype=self._dtype, copy=True)
        return SessionState(
            t_curr=float(sim_defaults.t0),
            y_curr=y0,
            params_curr=params0,
            dt_curr=float(sim_defaults.dt),
            step_count=0,
            workspace={},
            stepper_cfg=np.array(self._default_stepper_cfg, dtype=np.float64, copy=True),
            status=int(Status.DONE),
            pins=self._pins,
        )

    def _ensure_initial_snapshot(self, seed: Optional[IntegratorSeed] = None) -> None:
        if self._initial_snapshot_created:
            return
        base = seed or self._session_state.to_seed()
        snapshot_state = SessionState(
            t_curr=base.t,
            y_curr=np.array(base.y, copy=True),
            params_curr=np.array(base.params, copy=True),
            dt_curr=base.dt,
            step_count=base.step_count,
            workspace=_copy_workspace_dict(base.workspace),
            stepper_cfg=np.array(self._session_state.stepper_cfg, dtype=np.float64, copy=True),
            status=int(Status.DONE),
            pins=self._pins,
        )
        self._snapshots[self._initial_snapshot_name] = Snapshot(
            name=self._initial_snapshot_name,
            description="auto-created before first run",
            created_at=_now_iso(),
            state=snapshot_state,
            time_shift=self._time_shift,
            nominal_dt=self._nominal_dt,
        )
        self._initial_snapshot_created = True

    def _select_seed(
        self,
        *,
        resume: bool,
        t0: float,
        dt: float,
        ic: Optional[np.ndarray],
        params: Optional[np.ndarray],
    ) -> IntegratorSeed:
        """
        Build IntegratorSeed for run().
        
        For resume=True: continue from current SessionState (time, workspace, etc.).
        For resume=False: restart integration from t0 with fresh workspace, but use
                         current SessionState values (y_curr, params_curr) as defaults
                         unless overridden by explicit ic/params arguments.
        """
        if resume:
            return self._session_state.to_seed()

        # NEW behavior for resume=False:
        # Use current SessionState values as defaults, overridable by ic/params
        y_source = ic if ic is not None else self._session_state.y_curr
        p_source = params if params is not None else self._session_state.params_curr

        y0 = np.array(y_source, dtype=self._dtype, copy=True)
        p0 = np.array(p_source, dtype=self._dtype, copy=True)

        return IntegratorSeed(
            t=float(t0),
            y=y0,
            params=p0,
            dt=float(dt),
            step_count=0,
            workspace={},
        )

    def _execute_run(
        self,
        *,
        seed: IntegratorSeed,
        t_end: float,
        target_steps: Optional[int],
        max_steps: int,
        record: bool,
        record_interval: int,
        cap_rec: int,
        cap_evt: int,
        stepper_config: np.ndarray,
        adaptive: bool,
        wrms_cfg: WRMSConfig | None,
        state_rec_indices: np.ndarray,
        aux_rec_indices: np.ndarray,
        state_names: list[str],
        aux_names: list[str],
        observers=None,
    ) -> Results:
        stop_phase_mask = 0
        stop_spec = getattr(self.model.spec.sim, "stop", None)
        if stop_spec is not None:
            phase = stop_spec.phase
            if phase in ("pre", "both"):
                stop_phase_mask |= 1
            if phase in ("post", "both"):
                stop_phase_mask |= 2
        return run_with_wrapper(
            runner=self.model.runner,
            stepper=self.model.stepper,
            rhs=self.model.rhs,
            events_pre=self.model.events_pre,
            events_post=self.model.events_post,
            update_aux=self.model.update_aux,
            dtype=self.model.dtype,
            n_state=self._n_state,
            n_aux=len(self.model.spec.aux or {}),
            stop_phase_mask=stop_phase_mask,
            t0=float(seed.t),
            t_end=float(t_end),
            dt_init=float(seed.dt),
            max_steps=max_steps,
            record=record,
            record_interval=int(record_interval),
            state_record_indices=state_rec_indices,
            aux_record_indices=aux_rec_indices,
            state_names=state_names,
            aux_names=aux_names,
            ic=seed.y,
            params=seed.params,
            cap_rec=cap_rec,
            cap_evt=cap_evt,
            max_log_width=self._max_log_width,
            stepper_config=stepper_config,
            workspace_seed=seed.workspace,
            discrete=(target_steps is not None),
            target_steps=target_steps,
            lag_state_info=getattr(self.model, "lag_state_info", None),
            make_stepper_workspace=getattr(self.model, "make_stepper_workspace", None),
            wrms_cfg=wrms_cfg,
            observers=observers,
            adaptive=adaptive,
            model_hash=getattr(self.model, "spec_hash", None),
            stepper_name=getattr(self.model, "stepper_name", None),
        )

    def _ensure_runner_done(self, result: Results, *, phase: str) -> None:
        """Raise when the wrapped runner did not complete successfully."""
        status_value = int(result.status)
        if status_value in (int(Status.DONE), int(Status.EARLY_EXIT), int(Status.TRACE_OVERFLOW)):
            return
        try:
            status_name = Status(status_value).name
        except ValueError:
            status_name = f"Status<{status_value}>"
        raise RuntimeError(
            f"Runner exited with status {status_name} ({status_value}) during {phase}"
        )

    def _state_from_results(
        self, result: Results, *, base_steps: int, stepper_config: np.ndarray
    ) -> SessionState:
        total_steps = base_steps + int(result.step_count_final)
        return SessionState(
            t_curr=float(result.t_final),
            y_curr=np.array(result.final_state_view, dtype=self._dtype, copy=True),
            params_curr=np.array(result.final_params_view, dtype=self._dtype, copy=True),
            dt_curr=float(result.final_dt),
            step_count=total_steps,
            workspace=_copy_workspace_dict(result.final_workspace_view),
            stepper_cfg=np.array(stepper_config, dtype=np.float64, copy=True),
            status=int(result.status),
            pins=self._pins,
        )

    def _append_results(self, chunk: Results, *, step_offset_initial: int) -> None:
        accum = self._ensure_accumulator()
        prev_n = accum.n
        prev_m = accum.m
        n_curr = chunk.n
        m_curr = chunk.m
        if prev_n == 0 and n_curr == 0 and m_curr == 0:
            return

        drop_first = False
        if prev_n > 0 and n_curr > 0:
            prev_last_t = accum.T[prev_n - 1]
            curr_first_t = float(chunk.T_view[0])
            eps = _ulp_tolerance(prev_last_t, curr_first_t)
            if abs(curr_first_t - prev_last_t) <= eps:
                drop_first = True

        start = 1 if drop_first else 0
        trimmed_n = max(n_curr - start, 0)
        rec_start = prev_n
        if trimmed_n > 0:
            if prev_n > 0:
                step_offset = accum.STEP[prev_n - 1]
            else:
                step_offset = step_offset_initial
            accum.append_records(
                chunk.T_view[start:],
                chunk.Y_view[:, start:],
                chunk.STEP_view[start:] + step_offset,
                chunk.FLAGS_view[start:],
                aux_seg=chunk.AUX_view[:, start:] if chunk.AUX is not None else None,
            )

        appended_evt_len = 0
        if chunk.m > 0:
            codes = chunk.EVT_CODE_view
            idxs = np.array(chunk.EVT_INDEX_view, dtype=np.int64, copy=True)
            logs = chunk.EVT_LOG_DATA_view

            if drop_first and idxs.size:
                keep_mask = idxs != 0
                codes = codes[keep_mask]
                logs = logs[keep_mask, :]
                idxs = idxs[keep_mask]
                shift_mask = idxs > 0
                idxs[shift_mask] -= 1

            evt_offset = (prev_n - 1) if (drop_first and prev_n > 0) else prev_n
            pos_mask = idxs >= 0
            idxs[pos_mask] = idxs[pos_mask] + evt_offset

            accum.append_events(codes, idxs.astype(np.int32, copy=False), logs)
            appended_evt_len = accum.m - prev_m

        accum.assert_monotone_time()

        if trimmed_n > 0:
            t_view = chunk.T_view
            t_start = float(t_view[start])
            t_end = float(t_view[n_curr - 1])
            step_start = int(accum.STEP[rec_start])
            step_end = int(accum.STEP[rec_start + trimmed_n - 1])
            seg_id = len(self._segments)
            cfg_hash = self._pending_run_cfg_hash or _config_digest(self._session_state.stepper_cfg)
            seg_name = self._unique_segment_name(self._pending_run_tag)
            segment = Segment(
                id=seg_id,
                name=seg_name,
                rec_start=rec_start,
                rec_len=trimmed_n,
                evt_start=prev_m,
                evt_len=appended_evt_len,
                t_start=t_start,
                t_end=t_end,
                step_start=step_start,
                step_end=step_end,
                resume=self._last_run_was_resume,
                cfg_hash=cfg_hash,
            )
            self._segments.append(segment)

    def _rebase_times(self, result: Results, shift: float) -> None:
        if shift == 0.0:
            return
        if result.n > 0:
            result.T[: result.n] = result.T[: result.n] - shift
        if result.m == 0 or not self._event_time_columns:
            return
        codes = result.EVT_CODE_view
        log_data = result.EVT_LOG_DATA
        for row in range(result.m):
            cols = self._event_time_columns.get(int(codes[row]))
            if not cols:
                continue
            for col in cols:
                log_data[row, col] -= shift

    def _publish_results(self, last_result: Results) -> None:
        accum = self._ensure_accumulator()
        state = self._session_state
        self._raw_results = accum.to_results(
            status=int(last_result.status),
            final_state=np.array(state.y_curr, copy=True),
            final_params=np.array(state.params_curr, copy=True),
            t_final=state.t_curr,
            final_dt=state.dt_curr,
            step_count_final=state.step_count,
            workspace=_copy_workspace_dict(state.workspace),
            state_names=last_result.state_names,
            aux_names=last_result.aux_names,
            analysis_out=last_result.analysis_out,
            analysis_trace=last_result.analysis_trace,
            analysis_trace_filled=last_result.analysis_trace_filled,
            analysis_trace_stride=last_result.analysis_trace_stride,
            analysis_trace_offset=getattr(last_result, "analysis_trace_offset", None),
            analysis_modules=last_result.analysis_modules,
            analysis_meta=getattr(last_result, "analysis_meta", None),
        )
        self._results_view = None

    def _ensure_accumulator(self) -> _ResultAccumulator:
        if self._result_accum is None:
            # Use the recorded states/aux count from the recording selection
            # None means "not determined yet" -> default to full recording
            # Empty list [] means "explicitly no recording"
            n_rec_states = len(self._recording_state_names) if self._recording_state_names is not None else self._n_state
            n_rec_aux = len(self._recording_aux_names) if self._recording_aux_names is not None else self._n_aux
            self._result_accum = _ResultAccumulator(
                n_rec_states=n_rec_states,
                n_rec_aux=n_rec_aux,
                dtype=self._dtype,
                max_log_width=self._max_log_width,
            )
        return self._result_accum

    def _resolve_segment_index(self, key: int | str) -> int:
        if isinstance(key, int):
            if key < 0 or key >= len(self._segments):
                raise IndexError(f"segment index {key} out of range")
            return key
        for idx, seg in enumerate(self._segments):
            if seg.name == key or self._segment_auto_name(seg) == key:
                return idx
        available = ", ".join(self._segment_available_names()) or "<none>"
        raise KeyError(f"Unknown segment '{key}'. Known names: {available}")

    def _unique_segment_name(self, base: Optional[str], *, skip_index: Optional[int] = None) -> Optional[str]:
        if base is None:
            return None
        if base == "":
            raise ValueError("segment name cannot be empty")
        reserved = self._segment_reserved_names(skip_index=skip_index)
        candidate = base
        suffix = 2
        while candidate in reserved:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _segment_available_names(self) -> list[str]:
        names: list[str] = []
        for seg in self._segments:
            effective = self._segment_effective_name(seg)
            auto = self._segment_auto_name(seg)
            names.append(effective)
            if auto not in names:
                names.append(auto)
        return names

    def _segment_reserved_names(self, *, skip_index: Optional[int] = None) -> set[str]:
        reserved: set[str] = set()
        for idx, seg in enumerate(self._segments):
            if skip_index is not None and idx == skip_index:
                continue
            reserved.add(self._segment_effective_name(seg))
            reserved.add(self._segment_auto_name(seg))
            if seg.name is not None:
                reserved.add(seg.name)
        return reserved

    def _segment_auto_name(self, segment: Segment) -> str:
        return f"run#{segment.id}"

    def _segment_effective_name(self, segment: Segment) -> str:
        return segment.name if segment.name is not None else self._segment_auto_name(segment)

    def _resolve_snapshot(self, snapshot: Snapshot | str) -> Snapshot:
        if isinstance(snapshot, Snapshot):
            return snapshot
        self._ensure_initial_snapshot()
        if snapshot not in self._snapshots:
            raise KeyError(f"Unknown snapshot '{snapshot}'")
        return self._snapshots[snapshot]

    def _compute_default_stepper_config(self) -> np.ndarray:
        default_config = self._stepper_spec.default_config(self.model.spec)
        if default_config is None:
            return np.array([], dtype=np.float64)
        return self._stepper_spec.pack_config(default_config)

    def _sync_initial_snapshot_config(self, config: np.ndarray) -> None:
        snapshot = self._snapshots.get(self._initial_snapshot_name)
        if snapshot is None:
            return
        snapshot.state.stepper_cfg = np.array(config, dtype=np.float64, copy=True)

    def _log_stepper_overrides(
        self, previous: np.ndarray, new: np.ndarray, updates: Mapping[str, Any]
    ) -> None:
        prev_digest = _config_digest(previous)
        new_digest = _config_digest(new)
        self._logger.info(
            "Stepper config overrides applied (%s → %s) with %s",
            prev_digest,
            new_digest,
            updates,
        )

    def _summarize_stepper_config(self, cfg: np.ndarray) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "size": int(cfg.size),
            "hash": _config_digest(cfg),
        }
        if cfg.size == 0:
            summary["preview"] = {}
            return summary
        preview_count = min(3, cfg.size)
        preview: Dict[str, float] = {}
        names = self._stepper_config_names
        for idx in range(preview_count):
            key = names[idx] if idx < len(names) else f"f{idx}"
            preview[key] = float(cfg[idx])
        summary["preview"] = preview
        return summary

    def _build_stepper_config(
        self,
        kwargs: dict,
        prev_config: Optional[np.ndarray],
        *,
        return_values: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, Optional[Dict[str, float]]]:
        """
        Build stepper config array from run() kwargs, session defaults, and prior runs.
        
        Handles string→int enum conversion for config fields that accept string values.
        """
        stepper_name = self.model.stepper_name
        stepper_spec = self._stepper_spec
        default_config = stepper_spec.default_config(self.model.spec)

        def _finalize(cfg: np.ndarray, values: Optional[Dict[str, float]]) -> np.ndarray | tuple[np.ndarray, Optional[Dict[str, float]]]:
            cfg_array = np.array(cfg, dtype=np.float64, copy=True)
            if return_values:
                return cfg_array, values
            return cfg_array

        if default_config is None:
            if kwargs:
                warnings.warn(
                    f"Stepper '{stepper_name}' does not accept runtime parameters. "
                    f"Ignoring: {list(kwargs.keys())}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            empty = np.array([], dtype=np.float64)
            return _finalize(empty, None)

        if not kwargs:
            if prev_config is not None and prev_config.size:
                cfg = np.array(prev_config, dtype=np.float64, copy=True)
                values = self._config_values_from_array(cfg)
                return _finalize(cfg, values)
            cfg = np.array(self._default_stepper_cfg, dtype=np.float64, copy=True)
            values = self._config_values_from_array(cfg)
            return _finalize(cfg, values)

        valid_fields = {f.name for f in dataclasses.fields(default_config)}
        config_updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        invalid = set(kwargs.keys()) - valid_fields
        if invalid:
            warnings.warn(
                f"Unknown stepper parameters for '{stepper_name}': {invalid}. "
                f"Valid parameters: {valid_fields}",
                RuntimeWarning,
                stacklevel=3,
            )
        
        # Convert string enum values to ints if stepper declares enum mappings
        enum_maps = None
        if hasattr(stepper_spec, 'config_enum_maps'):
            try:
                enum_maps = stepper_spec.config_enum_maps()
            except Exception:
                pass  # Stepper doesn't implement it or returned None
        
        if enum_maps and config_updates:
            from dynlib.steppers.config_utils import convert_config_enums
            try:
                config_updates = convert_config_enums(config_updates, enum_maps, stepper_name)
            except ValueError as e:
                # Re-raise with context about where this came from
                raise ValueError(f"Invalid stepper config: {e}") from None

        final_config = (
            dataclasses.replace(default_config, **config_updates)
            if config_updates
            else dataclasses.replace(default_config)
        )
        new_config = stepper_spec.pack_config(final_config)
        if prev_config is not None and prev_config.size and not np.array_equal(prev_config, new_config):
            self._log_stepper_overrides(prev_config, new_config, config_updates)
        values = dataclasses.asdict(final_config)
        return _finalize(new_config, values)

    def _config_values_from_array(self, cfg: np.ndarray) -> Dict[str, float]:
        values: Dict[str, float] = {}
        if cfg.size == 0:
            return values
        names = self._stepper_config_names
        limit = min(len(names), int(cfg.size))
        for idx in range(limit):
            values[names[idx]] = float(cfg[idx])
        return values

    def _validate_observer_requirements(
        self,
        observer,
        *,
        adaptive: bool | None = None,
        has_event_logs: bool | None = None,
    ) -> None:
        """Ensure the provided observer is compatible with this model."""
        req = getattr(observer, "requirements", None)
        if req is None:
            return

        has_jvp = getattr(self.model, "jvp", None) is not None
        has_dense_jac = getattr(self.model, "jacobian", None) is not None
        name = getattr(observer, "name", "observer")

        if req.need_jvp and not has_jvp:
            raise ValueError(
                f"Observer '{name}' requires a model Jacobian-vector product, but none is available."
            )
        if req.need_dense_jacobian and not has_dense_jac:
            raise ValueError(
                f"Observer '{name}' requires a dense Jacobian fill callable, but the model does not provide one."
            )
        if req.need_jacobian and not (has_jvp or has_dense_jac):
            raise ValueError(
                f"Observer '{name}' requires a model Jacobian, but the model does not provide one."
            )
        if adaptive is True and getattr(req, "fixed_step", False):
            raise ValueError(f"Observer '{name}' requires fixed-step execution")
        if req.requires_event_log:
            if has_event_logs is False:
                raise ValueError(f"Observer '{name}' requires event logging, but no event logs are configured.")
        if getattr(observer, "trace", None) is not None and observer.needs_trace:
            if observer.trace.plan is None:
                raise ValueError(f"Observer '{name}' requires a TracePlan when trace width > 0.")
            if observer.trace.record_interval() <= 0:
                raise ValueError(f"Observer '{name}' trace stride must be positive.")
            if adaptive:
                raise ValueError(f"Observer '{name}' traces require fixed-step execution.")


# ------------------------------- misc helpers ---------------------------------

@dataclass
class _PresetData:
    """Internal preset representation."""
    name: str
    params: Dict[str, float]
    states: Optional[Dict[str, float]]
    source: Literal["inline", "file", "session"]


def _did_you_mean(name: str, candidates: list[str], max_distance: int = 2) -> Optional[str]:
    """Return the closest matching candidate using Levenshtein distance."""
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    best = None
    best_dist = max_distance + 1
    for cand in candidates:
        dist = levenshtein(name.lower(), cand.lower())
        if dist < best_dist:
            best_dist = dist
            best = cand
    
    return best if best_dist <= max_distance else None


def _validate_preset_names(
    preset: _PresetData,
    param_names: set[str],
    state_names: set[str],
) -> None:
    """Validate that all param/state names in preset exist in the model.
    
    Raises ValueError with suggestions if any names are unknown.
    """
    if not preset.params and not (preset.states or {}):
        raise ValueError(
            f"Preset '{preset.name}' is empty; must define at least one param or state."
        )

    # Check params
    unknown_params = set(preset.params.keys()) - param_names
    if unknown_params:
        suggestions = []
        for uk in sorted(unknown_params):
            suggestion = _did_you_mean(uk, list(param_names))
            if suggestion:
                suggestions.append(f"'{uk}' (did you mean '{suggestion}'?)")
            else:
                suggestions.append(f"'{uk}'")
        raise ValueError(
            f"Preset '{preset.name}' has unknown param(s): {', '.join(suggestions)}. "
            f"Valid params: {sorted(param_names)}"
        )

    # Check states (subset allowed)
    state_values = preset.states or {}
    unknown_states = set(state_values.keys()) - state_names
    if unknown_states:
        suggestions = []
        for uk in sorted(unknown_states):
            suggestion = _did_you_mean(uk, list(state_names))
            if suggestion:
                suggestions.append(f"'{uk}' (did you mean '{suggestion}'?)")
            else:
                suggestions.append(f"'{uk}'")
        raise ValueError(
            f"Preset '{preset.name}' has unknown state(s): {', '.join(suggestions)}. "
            f"Valid states: {sorted(state_names)}"
        )


def _cast_values_to_dtype(
    values: Dict[str, float],
    dtype: np.dtype,
    preset_name: str,
    kind: str,  # "param" or "state"
) -> Dict[str, Any]:
    """Cast preset values to model dtype with overflow/precision warnings.
    
    Returns a dict of casted values (same keys).
    Warns on precision loss, errors on overflow.
    """
    result = {}
    
    for key, val in values.items():
        # Check for special float values (NaN/Inf are allowed as input)
        if isinstance(val, float):
            if np.isnan(val):
                result[key] = dtype.type(val)
                continue
            # Inf is allowed but we check for overflow during casting below
        
        # Cast first to check for overflow
        # Suppress numpy's overflow warning since we check for it explicitly below
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'overflow encountered in cast', RuntimeWarning)
            casted = dtype.type(val)
        
        # Check for overflow (resulted in inf when input wasn't inf)
        if np.issubdtype(dtype, np.floating):
            if np.isinf(casted) and not (isinstance(val, float) and np.isinf(val)):
                dtype_info = np.finfo(dtype)
                raise ValueError(
                    f"Preset '{preset_name}': {kind} '{key}' value {val} overflows {dtype} "
                    f"(max: {dtype_info.max})"
                )
        elif np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
            if val < dtype_info.min or val > dtype_info.max:
                raise ValueError(
                    f"Preset '{preset_name}': {kind} '{key}' value {val} out of {dtype} range "
                    f"[{dtype_info.min}, {dtype_info.max}]"
                )
        
        # Warn on precision loss
        if dtype == np.float32 and isinstance(val, (float, int)):
            # Check if the value can be represented exactly in float32
            roundtrip = float(casted)
            if val != 0.0 and roundtrip != val:
                rel_error = abs((roundtrip - val) / val)
                if rel_error > 1e-8:  # Significant precision loss (>0.00001%)
                    warnings.warn(
                        f"Preset '{preset_name}': {kind} '{key}' may lose precision "
                        f"when casting {val} to float32 (relative error: {rel_error:.2e})",
                        RuntimeWarning,
                        stacklevel=4,
                    )
        elif np.issubdtype(dtype, np.integer) and isinstance(val, float):
            if val != float(int(val)):
                warnings.warn(
                    f"Preset '{preset_name}': {kind} '{key}' loses fractional part "
                    f"when casting {val} to {dtype}",
                    RuntimeWarning,
                    stacklevel=4,
                )
        elif np.issubdtype(dtype, np.floating) and np.issubdtype(type(val), np.complexfloating):
            warnings.warn(
                f"Preset '{preset_name}': {kind} '{key}' loses imaginary part "
                f"when casting complex to {dtype}",
                RuntimeWarning,
                stacklevel=4,
            )
        
        result[key] = casted
    
    return result


def _toml_read(path: Path) -> Dict[str, Any]:
    """Read a TOML file and return parsed dict."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ValueError(f"Preset file not found: {path}")
    except Exception as e:
        raise ValueError(f"Failed to read preset file {path}: {e}")


def _toml_write_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write TOML data to file atomically using temp file + replace."""
    # Simple TOML emitter (handles basic types, dicts, nested tables)
    def _emit_value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, (int, float)):
            if isinstance(v, float):
                if np.isnan(v):
                    return '"nan"'
                elif np.isposinf(v):
                    return '"+inf"'
                elif np.isneginf(v):
                    return '"-inf"'
            return str(v)
        elif isinstance(v, str):
            # Escape quotes
            escaped = v.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(v, (list, tuple)):
            return "[" + ", ".join(_emit_value(x) for x in v) + "]"
        else:
            raise ValueError(f"Unsupported TOML value type: {type(v)}")
    
    def _emit_table(lines: list[str], table_path: str, table_data: dict) -> None:
        lines.append(f"[{table_path}]")
        for key, val in table_data.items():
            if isinstance(val, dict):
                # Nested table - skip here, emit separately
                pass
            else:
                lines.append(f"{key} = {_emit_value(val)}")
        lines.append("")  # blank line after table
    
    def _emit_nested(lines: list[str], prefix: str, data: dict) -> None:
        # First emit direct key-value pairs
        direct = {k: v for k, v in data.items() if not isinstance(v, dict)}
        if direct:
            _emit_table(lines, prefix, direct)
        
        # Then emit nested tables
        for key, val in data.items():
            if isinstance(val, dict):
                nested_path = f"{prefix}.{key}" if prefix else key
                _emit_nested(lines, nested_path, val)
    
    lines: list[str] = []
    _emit_nested(lines, "", data)
    
    content = "\n".join(lines)
    
    # Write to temp file then replace
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}_tmp_",
        suffix=".toml",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(content)
    
    try:
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _resize_1d(arr: np.ndarray, new_cap: int) -> np.ndarray:
    new_arr = np.zeros((new_cap,), dtype=arr.dtype)
    length = min(arr.shape[0], new_cap)
    if length:
        new_arr[:length] = arr[:length]
    return new_arr


def _copy_workspace_dict(ws: WorkspaceSnapshot) -> WorkspaceSnapshot:
    if not ws:
        return {}
    copied: WorkspaceSnapshot = {}
    for bucket, values in ws.items():
        copied[bucket] = {
            name: np.array(buff, copy=True)
            for name, buff in values.items()
            if isinstance(buff, np.ndarray)
        }
    return copied

def _event_time_column_map(spec) -> Dict[int, Tuple[int, ...]]:
    mapping: Dict[int, Tuple[int, ...]] = {}
    for idx, event in enumerate(spec.events):
        cols = tuple(i for i, field in enumerate(event.log) if field == "t")
        if cols:
            mapping[idx] = cols
    return mapping


def _max_event_log_width(events) -> int:
    width = 0
    for event in events:
        width = max(width, len(getattr(event, "log", ())))
    return width


def _config_digest(cfg: np.ndarray) -> str:
    if cfg.size == 0:
        return "empty"
    return hashlib.sha1(cfg.tobytes()).hexdigest()[:10]




def _stepper_config_names(stepper_spec, model_spec) -> Tuple[str, ...]:
    if not hasattr(stepper_spec, "default_config"):
        return ()
    default_config = stepper_spec.default_config(model_spec)
    if default_config is None:
        return ()
    import dataclasses

    return tuple(field.name for field in dataclasses.fields(default_config))


def _dynlib_version() -> str:
    try:
        return importlib_metadata.version("dynlib")
    except importlib_metadata.PackageNotFoundError:
        project_root = Path(__file__).resolve()
        for parent in project_root.parents:
            candidate = parent / "pyproject.toml"
            if not candidate.exists():
                continue
            try:
                with open(candidate, "rb") as fh:
                    data = tomllib.load(fh)
            except Exception:  # pragma: no cover
                continue
            project = data.get("project", {})
            version = project.get("version")
            if isinstance(version, str):
                return version
        return "0.0.0+local"


def _ulp_tolerance(a: float, b: float) -> float:
    return float(np.spacing(max(abs(a), abs(b), 1.0)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _diff_pins(pins_a: SessionPins, pins_b: SessionPins) -> Dict[str, Tuple[Any, Any]]:
    diffs: Dict[str, Tuple[Any, Any]] = {}
    if pins_a.spec_hash != pins_b.spec_hash:
        diffs["spec_hash"] = (pins_a.spec_hash, pins_b.spec_hash)
    if pins_a.stepper_name != pins_b.stepper_name:
        diffs["stepper_name"] = (pins_a.stepper_name, pins_b.stepper_name)
    if pins_a.workspace_sig != pins_b.workspace_sig:
        diffs["workspace_sig"] = (pins_a.workspace_sig, pins_b.workspace_sig)
    if pins_a.dtype_token != pins_b.dtype_token:
        diffs["dtype_token"] = (pins_a.dtype_token, pins_b.dtype_token)
    if pins_a.dynlib_version != pins_b.dynlib_version:
        diffs["dynlib_version"] = (pins_a.dynlib_version, pins_b.dynlib_version)
    return diffs
