# src/dynlib/runtime/sim.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
import warnings

import numpy as np

from .model import Model
from .wrapper import run_with_wrapper
from .results import Results
from .results_api import ResultsView
from .runner_api import Status
from dynlib.steppers.registry import get_stepper

try:  # pragma: no cover - available on 3.8+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:  # pragma: no cover - Python >=3.11
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

__all__ = ["Sim"]


# ------------------------------- data records ---------------------------------

WorkspaceSnapshot = Dict[str, np.ndarray]


@dataclass(frozen=True)
class SessionPins:
    spec_hash: str
    stepper_name: str
    structsig: Tuple[int, ...]
    dtype_token: str
    dynlib_version: str


@dataclass
class SessionState:
    t_curr: float
    y_curr: np.ndarray
    params_curr: np.ndarray
    dt_curr: float
    step_count: int
    stepper_ws: WorkspaceSnapshot
    status: int
    pins: SessionPins

    def clone(self) -> SessionState:
        return SessionState(
            t_curr=self.t_curr,
            y_curr=np.array(self.y_curr, copy=True),
            params_curr=np.array(self.params_curr, copy=True),
            dt_curr=self.dt_curr,
            step_count=self.step_count,
            stepper_ws=_copy_workspace_dict(self.stepper_ws),
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
            workspace=_copy_workspace_dict(self.stepper_ws),
        )


@dataclass
class Snapshot:
    name: str
    description: str
    created_at: str
    state: SessionState
    time_shift: float
    nominal_dt: float


@dataclass
class IntegratorSeed:
    t: float
    y: np.ndarray
    params: np.ndarray
    dt: float
    step_count: int
    workspace: WorkspaceSnapshot


class _ResultAccumulator:
    """
    Mutable recording buffers that grow geometrically as stitched results append.
    """

    def __init__(self, *, n_state: int, dtype: np.dtype, max_log_width: int) -> None:
        self.n_state = int(n_state)
        self.dtype = np.dtype(dtype)
        self.log_cols = max(1, int(max_log_width))
        self._record_cap = 0
        self._event_cap = 0
        self.n = 0
        self.m = 0
        self.T = np.zeros((0,), dtype=np.float64)
        self.Y = np.zeros((self.n_state, 0), dtype=self.dtype)
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
    ) -> None:
        if t_seg.size == 0:
            return
        needed = self.n + t_seg.shape[0]
        self._ensure_record_capacity(needed)
        start = self.n
        end = needed
        self.T[start:end] = t_seg
        self.Y[:, start:end] = y_seg
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
    ) -> Results:
        return Results(
            T=self.T,
            Y=self.Y,
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
            final_stepper_ws=workspace,
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
        new_Y = np.zeros((self.n_state, new_cap), dtype=self.dtype)
        if self.n:
            new_Y[:, : self.n] = self.Y[:, : self.n]
        self.Y = new_Y
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
    Simulation facade around a compiled Model with resumable session state and optional snapshots.
    """

    def __init__(self, model: Model):
        self.model = model
        self._raw_results: Optional[Results] = None
        self._results_view: Optional[ResultsView] = None
        self._dtype = model.dtype
        self._n_state = len(model.spec.states)
        self._max_log_width = _max_event_log_width(model.spec.events)
        self._structsig = _struct_signature(model.struct)
        self._pins = SessionPins(
            spec_hash=model.spec_hash,
            stepper_name=model.stepper_name,
            structsig=self._structsig,
            dtype_token=str(np.dtype(model.dtype)),
            dynlib_version=_dynlib_version(),
        )
        self._session_state = self._bootstrap_session_state()
        self._snapshots: Dict[str, Snapshot] = {}
        self._initial_snapshot_name = "initial"
        self._initial_snapshot_created = False
        self._result_accum: Optional[_ResultAccumulator] = None
        self._event_time_columns = _event_time_column_map(self.model.spec)
        self._time_shift = 0.0
        stepper_spec = get_stepper(model.stepper_name)
        self._fixed_time_control = getattr(stepper_spec.meta, "time_control", "fixed") == "fixed"
        self._nominal_dt = float(self.model.spec.sim.dt)

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

    def run(
        self,
        *,
        t0: Optional[float] = None,
        t_end: Optional[float] = None,
        dt: Optional[float] = None,
        max_steps: int = 100000,
        record: Optional[bool] = None,
        record_interval: int = 1,
        ic: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        cap_rec: int = 1024,
        cap_evt: int = 1,
        transient: Optional[float] = None,
        resume: bool = False,
        **stepper_kwargs,
    ) -> None:
        """
        Run the compiled model. Set resume=True to continue from the last SessionState.
        """
        sim_defaults = self.model.spec.sim
        record = record if record is not None else sim_defaults.record
        run_t0 = t0 if t0 is not None else sim_defaults.t0
        if not resume:
            nominal_dt = float(dt if dt is not None else sim_defaults.dt)
            self._nominal_dt = nominal_dt
        else:
            nominal_dt = self._nominal_dt
        transient = 0.0 if transient is None else float(transient)
        if transient < 0.0:
            raise ValueError("transient must be non-negative")
        if resume and transient > 0.0:
            raise ValueError("transient warm-up is not allowed during resume")

        if resume and any(arg is not None for arg in (ic, params, t0, dt)):
            raise ValueError("resume=True ignores ic/params/t0/dt overrides; omit them for clarity")

        if not resume:
            self._result_accum = None
            self._raw_results = None
            self._results_view = None
            self._time_shift = 0.0

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

        target_t_end = t_end if t_end is not None else sim_defaults.t_end
        if resume:
            target_t_end_abs = target_t_end + self._time_shift
            if target_t_end_abs <= seed.t:
                current_time = seed.t - self._time_shift
                raise ValueError(
                    f"Resume target t_end ({target_t_end}) must exceed current time ({current_time})"
                )

        stepper_config = self._build_stepper_config(stepper_kwargs)
        n_state = self._n_state
        max_steps = int(max_steps)
        cap_rec = max(1, int(cap_rec))
        cap_evt = max(1, int(cap_evt))
        base_steps_for_session = seed.step_count
        step_offset_initial = seed.step_count
        run_seed = seed
        record_target_t_end = target_t_end + self._time_shift
        # Optional transient warm-up (no recording, no stitching) before the recorded run.
        if transient > 0.0:
            warm_result = self._execute_run(
                seed=run_seed,
                t_end=seed.t + transient,
                max_steps=max_steps,
                record=False,
                record_interval=record_interval,
                cap_rec=cap_rec,
                cap_evt=cap_evt,
                stepper_config=stepper_config,
            )
            self._session_state = self._state_from_results(warm_result, base_steps=seed.step_count)
            warm_state = self._session_state
            base_steps_for_session = warm_state.step_count
            step_offset_initial = 0
            run_seed = warm_state.to_seed()
            run_seed.dt = self._nominal_dt
            recorded_duration = target_t_end - run_t0
            if recorded_duration <= 0:
                raise ValueError("transient exceeds or equals requested horizon; nothing left to record")
            self._time_shift = warm_state.t_curr - run_t0
            record_target_t_end = target_t_end + self._time_shift

        # Recorded run (or the only run when transient==0)
        recorded_result = self._execute_run(
            seed=run_seed,
            t_end=record_target_t_end,
            max_steps=max_steps,
            record=record,
            record_interval=record_interval,
            cap_rec=cap_rec,
            cap_evt=cap_evt,
            stepper_config=stepper_config,
        )
        if self._time_shift != 0.0:
            self._rebase_times(recorded_result, self._time_shift)
        self._session_state = self._state_from_results(
            recorded_result, base_steps=base_steps_for_session
        )
        self._append_results(recorded_result, step_offset_initial=step_offset_initial)
        self._publish_results(recorded_result)

    def raw_results(self) -> Results:
        """Return the stitched raw results façade (raises if run() not yet called)."""
        if self._raw_results is None:
            raise RuntimeError("No simulation results available; call run() first (reset() clears history).")
        return self._raw_results

    def results(self) -> ResultsView:
        """Return a cached ResultsView wrapper over the stitched run history."""
        if self._results_view is None:
            raw = self.raw_results()
            self._results_view = ResultsView(raw, self.model.spec)
        return self._results_view

    def create_snapshot(self, name: str, description: str = "") -> None:
        """Capture the current SessionState into an immutable snapshot."""
        if not name:
            raise ValueError("snapshot name cannot be empty")
        if name in self._snapshots:
            raise ValueError(f"Snapshot '{name}' already exists")
        self._ensure_initial_snapshot()
        self._snapshots[name] = Snapshot(
            name=name,
            description=description,
            created_at=_now_iso(),
            state=self._session_state.clone(),
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

    def session_state_summary(self) -> dict[str, Any]:
        """Return a small diagnostic summary of the current SessionState."""
        can_resume, reason = self.can_resume()
        state = self._session_state
        return {
            "t": state.t_curr,
            "step": state.step_count,
            "dt": state.dt_curr,
            "status": Status(state.status).name,
            "stepper_name": self.model.stepper_name,
            "can_resume": can_resume,
            "reason": reason,
        }

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

    # ---------------------------- internal helpers -----------------------------

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
            stepper_ws={},
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
            stepper_ws=_copy_workspace_dict(base.workspace),
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
        if resume:
            return self._session_state.to_seed()
        y0 = np.array(
            ic if ic is not None else self.model.spec.state_ic, dtype=self._dtype, copy=True
        )
        p0 = np.array(
            params if params is not None else self.model.spec.param_vals,
            dtype=self._dtype,
            copy=True,
        )
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
        max_steps: int,
        record: bool,
        record_interval: int,
        cap_rec: int,
        cap_evt: int,
        stepper_config: np.ndarray,
    ) -> Results:
        return run_with_wrapper(
            runner=self.model.runner,
            stepper=self.model.stepper,
            rhs=self.model.rhs,
            events_pre=self.model.events_pre,
            events_post=self.model.events_post,
            struct=self.model.struct,
            dtype=self.model.dtype,
            n_state=self._n_state,
            t0=float(seed.t),
            t_end=float(t_end),
            dt_init=float(seed.dt),
            max_steps=max_steps,
            record=record,
            record_interval=int(record_interval),
            ic=seed.y,
            params=seed.params,
            cap_rec=cap_rec,
            cap_evt=cap_evt,
            max_log_width=self._max_log_width,
            stepper_config=stepper_config,
            workspace_seed=seed.workspace,
        )

    def _state_from_results(self, result: Results, *, base_steps: int) -> SessionState:
        total_steps = base_steps + int(result.step_count_final)
        return SessionState(
            t_curr=float(result.t_final),
            y_curr=np.array(result.final_state_view, dtype=self._dtype, copy=True),
            params_curr=np.array(result.final_params_view, dtype=self._dtype, copy=True),
            dt_curr=float(result.final_dt),
            step_count=total_steps,
            stepper_ws=_copy_workspace_dict(result.final_stepper_ws),
            status=int(result.status),
            pins=self._pins,
        )

    def _append_results(self, chunk: Results, *, step_offset_initial: int) -> None:
        accum = self._ensure_accumulator()
        prev_n = accum.n
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
        if trimmed_n > 0:
            if prev_n > 0:
                step_offset = accum.STEP[prev_n - 1] + 1
            else:
                step_offset = step_offset_initial
            accum.append_records(
                chunk.T_view[start:],
                chunk.Y_view[:, start:],
                chunk.STEP_view[start:] + step_offset,
                chunk.FLAGS_view[start:],
            )

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

        accum.assert_monotone_time()

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
            workspace=_copy_workspace_dict(state.stepper_ws),
        )
        self._results_view = None

    def _ensure_accumulator(self) -> _ResultAccumulator:
        if self._result_accum is None:
            self._result_accum = _ResultAccumulator(
                n_state=self._n_state,
                dtype=self._dtype,
                max_log_width=self._max_log_width,
            )
        return self._result_accum

    def _resolve_snapshot(self, snapshot: Snapshot | str) -> Snapshot:
        if isinstance(snapshot, Snapshot):
            return snapshot
        self._ensure_initial_snapshot()
        if snapshot not in self._snapshots:
            raise KeyError(f"Unknown snapshot '{snapshot}'")
        return self._snapshots[snapshot]

    def _build_stepper_config(self, kwargs: dict) -> np.ndarray:
        """
        Build stepper config array from run() kwargs and model_spec.
        """
        from dynlib.steppers.registry import get_stepper

        stepper_name = self.model.stepper_name
        stepper_spec = get_stepper(stepper_name)
        default_config = stepper_spec.default_config(self.model.spec)

        if default_config is None:
            if kwargs:
                warnings.warn(
                    f"Stepper '{stepper_name}' does not accept runtime parameters. "
                    f"Ignoring: {list(kwargs.keys())}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return np.array([], dtype=np.float64)

        import dataclasses

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

        final_config = (
            dataclasses.replace(default_config, **config_updates) if config_updates else default_config
        )
        return stepper_spec.pack_config(final_config)


# ------------------------------- misc helpers ---------------------------------

def _resize_1d(arr: np.ndarray, new_cap: int) -> np.ndarray:
    new_arr = np.zeros((new_cap,), dtype=arr.dtype)
    length = min(arr.shape[0], new_cap)
    if length:
        new_arr[:length] = arr[:length]
    return new_arr


def _copy_workspace_dict(ws: Mapping[str, np.ndarray]) -> WorkspaceSnapshot:
    if not ws:
        return {}
    return {name: np.array(buff, copy=True) for name, buff in ws.items()}

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


def _struct_signature(struct) -> Tuple[int, ...]:
    return (
        struct.sp_size,
        struct.ss_size,
        struct.sw0_size,
        struct.sw1_size,
        struct.sw2_size,
        struct.sw3_size,
        struct.iw0_size,
        struct.bw0_size,
        int(bool(struct.use_history)),
        int(bool(struct.use_f_history)),
        int(bool(struct.dense_output)),
        int(bool(struct.needs_jacobian)),
        -1 if struct.embedded_order is None else int(struct.embedded_order),
        int(bool(struct.stiff_ok)),
    )


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
    if pins_a.structsig != pins_b.structsig:
        diffs["structsig"] = (pins_a.structsig, pins_b.structsig)
    if pins_a.dtype_token != pins_b.dtype_token:
        diffs["dtype_token"] = (pins_a.dtype_token, pins_b.dtype_token)
    if pins_a.dynlib_version != pins_b.dynlib_version:
        diffs["dynlib_version"] = (pins_a.dynlib_version, pins_b.dynlib_version)
    return diffs
