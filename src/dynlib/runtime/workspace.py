# src/dynlib/runtime/workspace.py
from __future__ import annotations

from collections import namedtuple
import zlib
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


RuntimeWorkspace = namedtuple(
    "RuntimeWorkspace",
    ["lag_ring", "lag_head", "lag_info", "aux_values", "stop_flag", "stop_phase_mask"],
)

__all__ = [
    "RuntimeWorkspace",
    "make_runtime_workspace",
    "initialize_lag_runtime_workspace",
    "snapshot_workspace",
    "restore_workspace",
    "workspace_structsig",
]


def _zeros(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Allocate contiguous zeros with dtype."""
    arr = np.zeros(shape, dtype=dtype)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def make_runtime_workspace(
    *,
    lag_state_info: Sequence[Tuple[int, int, int, int]] | None,
    dtype: np.dtype,
    n_aux: int = 0,
    stop_enabled: bool = False,
    stop_phase_mask: int = 0,
) -> RuntimeWorkspace:
    """
    Allocate the runtime workspace owned by the runner/DSL layer.

    Args:
        lag_state_info: Sequence of (state_idx, depth, offset, head_index)
            describing circular buffers for lagged states.
        dtype: Model dtype used for lag_ring and aux_values.
        n_aux: Number of auxiliary variables to allocate storage for.
    """
    lag_meta = tuple(lag_state_info or ())
    aux_values = _zeros((n_aux,), dtype)
    stop_flag = _zeros((1,), np.int32) if stop_enabled else _zeros((0,), np.int32)
    stop_phase_arr = _zeros((1,), np.int32) if stop_enabled else _zeros((0,), np.int32)
    if stop_phase_arr.size:
        stop_phase_arr[0] = int(stop_phase_mask)
    if not lag_meta:
        empty_ring = _zeros((0,), dtype)
        empty_head = _zeros((0,), np.int32)
        empty_info = _zeros((0, 3), np.int32)
        return RuntimeWorkspace(empty_ring, empty_head, empty_info, aux_values, stop_flag, stop_phase_arr)

    total_depth = int(sum(depth for _, depth, _, _ in lag_meta))
    n_lag = len(lag_meta)

    lag_ring = _zeros((total_depth,), dtype)
    lag_head = _zeros((n_lag,), np.int32)
    lag_info = _zeros((n_lag, 3), np.int32)

    for j, (state_idx, depth, offset, _) in enumerate(lag_meta):
        lag_info[j, 0] = int(state_idx)
        lag_info[j, 1] = int(depth)
        lag_info[j, 2] = int(offset)

    return RuntimeWorkspace(lag_ring, lag_head, lag_info, aux_values, stop_flag, stop_phase_arr)


def initialize_lag_runtime_workspace(
    runtime_ws: RuntimeWorkspace,
    *,
    lag_state_info: Sequence[Tuple[int, int, int, int]] | None,
    y_curr: np.ndarray,
) -> None:
    """
    Seed lag buffers with the initial condition before the first runner entry.
    """
    lag_meta = tuple(lag_state_info or ())
    if not lag_meta or runtime_ws.lag_head.size == 0:
        return

    for j, (state_idx, depth, offset, _) in enumerate(lag_meta):
        value = y_curr[state_idx]
        runtime_ws.lag_ring[offset : offset + depth] = value
        runtime_ws.lag_head[j] = depth - 1


def _workspace_items(ws: object | None) -> Iterable[Tuple[str, object]]:
    if ws is None:
        return ()
    if hasattr(ws, "_asdict"):
        return ws._asdict().items()  # type: ignore[no-any-return]
    if hasattr(ws, "_fields"):
        return ((name, getattr(ws, name)) for name in ws._fields)  # type: ignore[attr-defined]
    raise TypeError(f"Workspace object of type {type(ws).__name__} is not a NamedTuple")


def snapshot_workspace(ws: object | None) -> Dict[str, object]:
    """
    Capture a shallow snapshot of a workspace by copying ndarray fields.
    """
    snapshot: Dict[str, object] = {}
    for name, value in _workspace_items(ws):
        if isinstance(value, np.ndarray):
            snapshot[name] = np.array(value, copy=True)
    return snapshot


def restore_workspace(
    ws: object | None,
    snapshot: Mapping[str, object] | None,
) -> None:
    """
    Restore workspace arrays from a snapshot produced by snapshot_workspace().
    """
    if ws is None or not snapshot:
        return

    for name, target in _workspace_items(ws):
        if name not in snapshot:
            continue
        saved = snapshot[name]
        if isinstance(target, np.ndarray):
            data = np.asarray(saved)
            if target.shape != data.shape or target.dtype != data.dtype:
                raise ValueError(
                    f"Workspace seed mismatch for '{name}': "
                    f"expected shape={target.shape}, dtype={target.dtype}, "
                    f"got {data.shape}/{data.dtype}"
                )
            if target.size:
                target[...] = data


def workspace_structsig(ws: object | None) -> Tuple[int, ...]:
    """
    Compute a deterministic structural signature derived from array shapes/dtypes.
    """
    sig: list[int] = []
    for name_idx, (name, value) in enumerate(_workspace_items(ws)):
        token = zlib.crc32(str(name).encode("utf-8")) & 0xFFFFFFFF
        if isinstance(value, np.ndarray):
            sig.append(token)
            sig.append(value.ndim)
            sig.extend(int(dim) for dim in value.shape)
            sig.append(int(value.dtype.num))
        elif np.isscalar(value):
            sig.append(token)
            sig.append(-1)
            sig.append(int(value))
    return tuple(sig)
