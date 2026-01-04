# src/dynlib/runtime/runner_api.py
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

__all__ = [
    "Status",
    # int constants (jit-friendly)
  "OK", "STEPFAIL", "NAN_DETECTED", "EARLY_EXIT", "DONE",
    "GROW_REC", "GROW_EVT", "USER_BREAK", "TRACE_OVERFLOW",
    "RunnerABI",
]

class Status(IntEnum):
    """Stable exit/status codes for runner/stepper."""
    OK = 0              # internal (no exit): step accepted, proceed
    STEPFAIL = 2        # exit (recoverable by wrapper or fail)
    NAN_DETECTED = 3    # exit
    EARLY_EXIT = 7      # exit (reserved for future basin analysis)
    DONE = 9            # exit
    GROW_REC = 10       # exit: request record buffer growth
    GROW_EVT = 11       # exit: request event buffer growth
    USER_BREAK = 12     # exit: user requested stop
    TRACE_OVERFLOW = 13 # exit: analysis trace exceeded capacity

# Plain int constants for JIT friendliness in tests / kernels
OK: int = int(Status.OK)
STEPFAIL: int = int(Status.STEPFAIL)
NAN_DETECTED: int = int(Status.NAN_DETECTED)
EARLY_EXIT: int = int(Status.EARLY_EXIT)
DONE: int = int(Status.DONE)
GROW_REC: int = int(Status.GROW_REC)
GROW_EVT: int = int(Status.GROW_EVT)
USER_BREAK: int = int(Status.USER_BREAK)
TRACE_OVERFLOW: int = int(Status.TRACE_OVERFLOW)


@dataclass(frozen=True)
class RunnerABI:
    """
    Discoverability-only record for the frozen runner signature.
    Never used inside the hot loop; serves as a single source of truth.
    """
    # Scalars
    t0: str = "float64"
    t_end: str = "float64"
    dt_init: str = "float64"
    max_steps: str = "int64"
    n_state: str = "int64"
    record_interval: str = "int64"

    # Main state/params (model dtype for ODEs; maps may be float or int)
    y_curr: str = "dtype[:]"   # length = n_state
    y_prev: str = "dtype[:]"   # length = n_state
    params: str = "dtype[:] | int64[:]"  # concrete single dtype per build

    # Runtime/stepper workspaces
    runtime_ws: str = "RuntimeWorkspace"
    stepper_ws: str = "StepperWorkspace"

    # Stepper configuration (read-only; float64)
    stepper_config: str = "float64[:]"

    # Proposals/outs (len-1 arrays; model dtype for t_prop/dt_next/err_est)
    y_prop: str = "dtype[:]"        # length = n_state
    t_prop: str = "dtype[1]"
    dt_next: str = "dtype[1]"
    err_est: str = "dtype[1]"

    # Recording buffers
    T: str = "float64[:]"                 # committed t
    Y: str = "dtype[:, :]"          # shape (cap_rec, n_state)
    STEP: str = "int64[:]"
    FLAGS: str = "int32[:]"

    # Event log buffers
    EVT_CODE: str = "int32[:]"
    EVT_INDEX: str = "int32[:]"
    EVT_LOG_DATA: str = "dtype[:, :]"  # shape (cap_evt, max_log_width)
    evt_log_scratch: str = "dtype[:]"   # scratch buffer for log values

    # Analysis buffers
    analysis_ws: str = "dtype[:]"
    analysis_out: str = "dtype[:]"
    analysis_trace: str = "dtype[:, :]"
    analysis_trace_count: str = "int64[1]"
    analysis_trace_cap: str = "int64"
    analysis_trace_stride: str = "int64"
    variational_step_enabled: str = "int32"
    variational_step_fn: str = "callable"

    # Cursors & caps
    i_start: str = "int64"
    step_start: str = "int64"
    cap_rec: str = "int64"
    cap_evt: str = "int64"

    # Control/outs (len-1)
    user_break_flag: str = "int32[1]"
    status_out: str = "int32[1]"
    hint_out: str = "int32[1]"
    i_out: str = "int64[1]"
    step_out: str = "int64[1]"
    t_out: str = "float64[1]"

    # Function symbols (jittable callables)
    stepper: str = "callable"
    rhs: str = "callable"
    events_pre: str = "callable"
    events_post: str = "callable"


# ---- Canonical runner signature (documentation) ------------------------------
__doc__ = (__doc__ or "") + r"""

RUNNER ABI (names/order/shapes/dtypes)

runner(
  # scalars
  t0: float64, t_end: float64, dt_init: float64,
  max_steps: int64, n_state: int64, record_interval: int64,
  # state/params
  y_curr: dtype[:], y_prev: dtype[:], params: dtype[:] | int64[:],
  # workspaces
  runtime_ws: tuple,
  stepper_ws: tuple,
  # stepper configuration (read-only)
  stepper_config: float64[:],
  # proposals/outs (len-1 arrays where applicable)
  y_prop: dtype[:], t_prop: dtype[1], dt_next: dtype[1], err_est: dtype[1],
  # recording
  T: float64[:], Y: dtype[:, :], STEP: int64[:], FLAGS: int32[:],
  # event log
  EVT_CODE: int32[:], EVT_INDEX: int32[:], EVT_LOG_DATA: dtype[:, :],
  evt_log_scratch: dtype[:],
  # analysis buffers
  analysis_ws: dtype[:], analysis_out: dtype[:], analysis_trace: dtype[:, :],
  analysis_trace_count: int64[1], analysis_trace_cap: int64, analysis_trace_stride: int64,
  variational_step_enabled: int32, variational_step_fn,
  # cursors & caps
  i_start: int64, step_start: int64, cap_rec: int64, cap_evt: int64,
  # control/outs (len-1)
  user_break_flag: int32[1], status_out: int32[1], hint_out: int32[1],
  i_out: int64[1], step_out: int64[1], t_out: float64[1],
  # function symbols (jittable callables)
  stepper, rhs, events_pre, events_post
) -> int32

Exit statuses (int32): DONE=9, GROW_REC=10, GROW_EVT=11, USER_BREAK=12, STEPFAIL=2, NAN_DETECTED=3, EARLY_EXIT=7.
Internal (no exit): OK=0 (step accepted, runner continues).

Rules:
- Runner performs capacity checks, pre/post events, commit & record, then exits only with the codes above.
- Stepper reads t, dt, y_curr, params, runtime_ws, stepper_ws, stepper_config; writes y_prop, t_prop[0], dt_next[0], err_est[0]; may mutate stepper_ws.
- stepper_config is a read-only float64 array containing runtime configuration (packed from stepper's config dataclass).
- RHS: rhs(t: float64, y_vec: dtype[:], dy_out: dtype[:], params: dtype[:] | int64[:]) -> None.
- Events: events_phase(t, y_vec, params, evt_log_scratch) -> (event_code: int32, log_width: int32)
  - event_code: -1 if no event fired, else unique event identifier (0, 1, 2...)
  - log_width: number of values written to evt_log_scratch (len(event.log))
  - Events run 'pre' on committed state before step; 'post' after commit. May only mutate states/params.
- EVT_INDEX stores the record index (0-based) that owns the event (or -1 when no record exists).
- Time can be logged using log=["t", ...] - it appears as first column in EVT_LOG_DATA
- T is float64; Y/EVT_LOG_DATA is model dtype; STEP:int64, FLAGS:int32, EVT_CODE/EVT_INDEX:int32.
- runtime_ws: NamedTuple owned by the runner/DSL (e.g., lag buffers).
- stepper_ws: NamedTuple owned by the stepper implementation.
- t_prop is model dtype; committed t written to T as float64.
- Variational: when variational_step_enabled!=0, runner calls variational_step_fn
  instead of stepper. Signature:
      variational_step_fn(t, dt, y_curr, rhs, params, runtime_ws, stepper_ws,
                          stepper_config, y_prop, t_prop, dt_next, err_est,
                          analysis_ws) -> int32|None
  It must write y_prop/t_prop/dt_next/err_est and propagate tangent data in
  analysis_ws. Returning None is treated as OK.
- t_prop is model dtype; committed t written to T as float64.
- Analysis: hooks are baked into runner variants via global symbols. Pre-step runs
  after pre-events (y_prev/y_curr ready) and post-step runs after commit + aux update.
  analysis_trace_count tracks filled trace rows; hooks may set user_break_flag to stop the run.
- Growth/resume: on GROW_REC/GROW_EVT wrapper reallocates & re-enters; runner resumes seamlessly.

"""
