# src/dynlib/runtime/buffers.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

__all__ = [
    "RecordingPools", "EventPools",
    "allocate_pools", "grow_rec_arrays", "grow_evt_arrays",
]

# ---- Data holders (lightweight; views are handled by wrapper/results) --------

@dataclass(frozen=True)
class RecordingPools:
    """
    Recording storage.
    Shapes:
      - T: (cap_rec,) float64        — committed times
      - Y: (n_state, cap_rec) dtype  — committed states per record slot
      - STEP: (cap_rec,) int64       — global step index at record time
      - FLAGS: (cap_rec,) int32      — bitmask per record (reserved)
    """
    T: np.ndarray          # float64
    Y: np.ndarray          # model dtype
    STEP: np.ndarray       # int64
    FLAGS: np.ndarray      # int32
    cap_rec: int
    n_state: int
    dtype: np.dtype


@dataclass(frozen=True)
class EventPools:
    """
    Event log storage (may be disabled with cap_evt==1).
    Shapes:
      - EVT_CODE:  (cap_evt,) int32   — event identifier (runner-defined)
      - EVT_INDEX: (cap_evt,) int32   — owning record index (or -1 if not materialized)
      - EVT_LOG_DATA: (cap_evt, max_log_width) dtype — logged signal values
    """
    EVT_CODE: np.ndarray      # int32
    EVT_INDEX: np.ndarray     # int32
    EVT_LOG_DATA: np.ndarray  # model dtype, shape (cap_evt, max_log_width)
    cap_evt: int
    max_log_width: int        # maximum number of signals logged across all events


# ---- Allocation --------------------------------------------------------------

def _zeros(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    a = np.zeros(shape, dtype=dtype)
    # We rely on C-contiguous arrays for simple slicing/copy.
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return a


def allocate_pools(
    *,
    n_state: int,
    dtype: np.dtype,
    cap_rec: int,
    cap_evt: int,
    max_log_width: int = 0,      # maximum log width across all events
) -> tuple[RecordingPools, EventPools]:
    """
    Allocate recording and event pools with frozen dtypes.

    - Model dtype: user-selected (default float64).
    - Recording T is always float64.
    - STEP:int64, FLAGS:int32, EVT_CODE:int32, EVT_INDEX:int32.
    """
    # Recording pools
    T     = _zeros((cap_rec,), np.float64)
    Y     = _zeros((n_state, cap_rec), dtype)
    STEP  = _zeros((cap_rec,), np.int64)
    FLAGS = _zeros((cap_rec,), np.int32)
    rec   = RecordingPools(T, Y, STEP, FLAGS, cap_rec, n_state, dtype)

    # Event pools (cap_evt may be 1 if disabled; max_log_width may be 0 if no logging)
    EVT_CODE     = _zeros((cap_evt,), np.int32)
    EVT_INDEX    = _zeros((cap_evt,), np.int32)
    EVT_LOG_DATA = _zeros((cap_evt, max(1, max_log_width)), dtype)  # at least (cap_evt, 1) for numba
    ev           = EventPools(EVT_CODE, EVT_INDEX, EVT_LOG_DATA, cap_evt, max_log_width)

    return rec, ev


# ---- Geometric growth (copy only filled regions) -----------------------------

def _next_cap(old_cap: int, min_needed: int) -> int:
    """
    Deterministic geometric growth: ×2 until >= min_needed.
    """
    cap = old_cap
    while cap < min_needed:
        cap *= 2
    return cap


def grow_rec_arrays(
    rec: RecordingPools,
    *,
    filled: int,           # number of filled record slots (0..filled-1 valid)
    min_needed: int,       # required capacity (e.g., filled+1)
) -> RecordingPools:
    """
    Grow recording arrays to at least min_needed capacity.
    Copies only the first `filled` columns/entries.
    """
    if min_needed <= rec.cap_rec:
        return rec

    new_cap = _next_cap(rec.cap_rec, min_needed)

    T_new     = _zeros((new_cap,), np.float64)
    Y_new     = _zeros((rec.n_state, new_cap), rec.dtype)
    STEP_new  = _zeros((new_cap,), np.int64)
    FLAGS_new = _zeros((new_cap,), np.int32)

    if filled > 0:
        T_new[:filled] = rec.T[:filled]
        Y_new[:, :filled] = rec.Y[:, :filled]
        STEP_new[:filled] = rec.STEP[:filled]
        FLAGS_new[:filled] = rec.FLAGS[:filled]

    return RecordingPools(T_new, Y_new, STEP_new, FLAGS_new, new_cap, rec.n_state, rec.dtype)


def grow_evt_arrays(
    ev: EventPools,
    *,
    filled: int,           # number of filled event slots (0..filled-1 valid)
    min_needed: int,       # required capacity (e.g., filled+1)
    dtype: np.dtype, # needed for EVT_LOG_DATA
) -> EventPools:
    """
    Grow event log arrays to at least min_needed capacity.
    Copies only the first `filled` entries.
    """
    if min_needed <= ev.cap_evt:
        return ev

    new_cap = _next_cap(ev.cap_evt, min_needed)

    EVT_CODE_new     = _zeros((new_cap,), np.int32)
    EVT_INDEX_new    = _zeros((new_cap,), np.int32)
    EVT_LOG_DATA_new = _zeros((new_cap, max(1, ev.max_log_width)), dtype)

    if filled > 0:
        EVT_CODE_new[:filled]     = ev.EVT_CODE[:filled]
        EVT_INDEX_new[:filled]    = ev.EVT_INDEX[:filled]
        EVT_LOG_DATA_new[:filled, :] = ev.EVT_LOG_DATA[:filled, :]

    return EventPools(EVT_CODE_new, EVT_INDEX_new, EVT_LOG_DATA_new, new_cap, ev.max_log_width)
