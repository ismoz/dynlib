# src/dynlib/runtime/buffers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from dynlib.steppers.base import StepperSpec  # for StructSpec sizes
from dynlib.runtime.runner_api import (
    DONE, GROW_REC, GROW_EVT, STEPFAIL, NAN_DETECTED, USER_BREAK,
)
# NOTE: We only import status codes and keep this module JIT-free.

__all__ = [
    "RecordingPools", "EventPools", "WorkBanks",
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
    model_dtype: np.dtype


@dataclass(frozen=True)
class EventPools:
    """
    Event log storage (may be disabled with cap_evt==1).
    Shapes:
      - EVT_TIME:  (cap_evt,) float64
      - EVT_CODE:  (cap_evt,) int32   — event identifier (runner-defined)
      - EVT_INDEX: (cap_evt,) int32   — additional index (runner-defined)
    """
    EVT_TIME: np.ndarray   # float64
    EVT_CODE: np.ndarray   # int32
    EVT_INDEX: np.ndarray  # int32
    cap_evt: int


@dataclass(frozen=True)
class WorkBanks:
    """
    Banks for stepper/runner work memory.
    Sizes come from the Stepper StructSpec (model dtype, except iw0/bw0).
    """
    # model dtype
    sp: np.ndarray
    ss: np.ndarray
    sw0: np.ndarray
    sw1: np.ndarray
    sw2: np.ndarray
    sw3: np.ndarray
    # int / byte banks
    iw0: np.ndarray        # int32
    bw0: np.ndarray        # uint8
    model_dtype: np.dtype


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
    struct: StepperSpec,          # only for .struct_spec() sizes
    model_dtype: np.dtype,
    cap_rec: int,
    cap_evt: int,
) -> tuple[WorkBanks, RecordingPools, EventPools]:
    """
    Allocate banks and record/log pools with the frozen dtypes.

    - Model dtype: user-selected (default float64).
    - Recording T, EVT_TIME are always float64.
    - STEP:int64, FLAGS:int32, EVT_CODE:int32, EVT_INDEX:int32.
    
    Lane-based allocation for model-dtype banks (sp, ss, sw0-sw3):
      - Size = number of lanes (multiples of n_state)
      - 0 lanes → unused (length 0)
      - 1 lane  → length = n_state
      - 2 lanes → length = 2*n_state, etc.
    
    Raw element counts for int/byte banks (iw0, bw0):
      - Size = raw element count; 0 is unused
    """
    sspec = struct.struct_spec()

    # Model-dtype work banks: lane counts (size * n_state)
    sp  = _zeros((sspec.sp_size * n_state,),  model_dtype)
    ss  = _zeros((sspec.ss_size * n_state,),  model_dtype)
    sw0 = _zeros((sspec.sw0_size * n_state,), model_dtype)
    sw1 = _zeros((sspec.sw1_size * n_state,), model_dtype)
    sw2 = _zeros((sspec.sw2_size * n_state,), model_dtype)
    sw3 = _zeros((sspec.sw3_size * n_state,), model_dtype)
    
    # Int/byte banks: raw element counts
    iw0 = _zeros((sspec.iw0_size,), np.int32)
    bw0 = _zeros((sspec.bw0_size,), np.uint8)

    banks = WorkBanks(sp, ss, sw0, sw1, sw2, sw3, iw0, bw0, model_dtype)

    # Recording pools
    T     = _zeros((cap_rec,), np.float64)
    Y     = _zeros((n_state, cap_rec), model_dtype)
    STEP  = _zeros((cap_rec,), np.int64)
    FLAGS = _zeros((cap_rec,), np.int32)
    rec   = RecordingPools(T, Y, STEP, FLAGS, cap_rec, n_state, model_dtype)

    # Event pools (cap_evt may be 1 if disabled)
    EVT_TIME  = _zeros((cap_evt,), np.float64)
    EVT_CODE  = _zeros((cap_evt,), np.int32)
    EVT_INDEX = _zeros((cap_evt,), np.int32)
    ev       = EventPools(EVT_TIME, EVT_CODE, EVT_INDEX, cap_evt)

    return banks, rec, ev


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
    Y_new     = _zeros((rec.n_state, new_cap), rec.model_dtype)
    STEP_new  = _zeros((new_cap,), np.int64)
    FLAGS_new = _zeros((new_cap,), np.int32)

    if filled > 0:
        T_new[:filled] = rec.T[:filled]
        Y_new[:, :filled] = rec.Y[:, :filled]
        STEP_new[:filled] = rec.STEP[:filled]
        FLAGS_new[:filled] = rec.FLAGS[:filled]

    return RecordingPools(T_new, Y_new, STEP_new, FLAGS_new, new_cap, rec.n_state, rec.model_dtype)


def grow_evt_arrays(
    ev: EventPools,
    *,
    filled: int,           # number of filled event slots (0..filled-1 valid)
    min_needed: int,       # required capacity (e.g., filled+1)
) -> EventPools:
    """
    Grow event log arrays to at least min_needed capacity.
    Copies only the first `filled` entries.
    """
    if min_needed <= ev.cap_evt:
        return ev

    new_cap = _next_cap(ev.cap_evt, min_needed)

    EVT_TIME_new  = _zeros((new_cap,), np.float64)
    EVT_CODE_new  = _zeros((new_cap,), np.int32)
    EVT_INDEX_new = _zeros((new_cap,), np.int32)

    if filled > 0:
        EVT_TIME_new[:filled]  = ev.EVT_TIME[:filled]
        EVT_CODE_new[:filled]  = ev.EVT_CODE[:filled]
        EVT_INDEX_new[:filled] = ev.EVT_INDEX[:filled]

    return EventPools(EVT_TIME_new, EVT_CODE_new, EVT_INDEX_new, new_cap)
