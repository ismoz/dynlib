# src/dynlib/analysis/basin_auto.py
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Literal, Sequence

import os
import numpy as np

from dynlib.analysis.basin import (
    Attractor,
    BasinResult,
    BLOWUP,
    OUTSIDE,
    UNRESOLVED,
    NumbaList,
    _coerce_batch,
    _normalize_dims,
    _normalize_grid,
    _prepare_record_vars,
    _require_numba,
    _resolve_mode,
    _seq_len,
    nb_types,
    njit,
    prange,
)
from dynlib.runtime.observers import ObserverHooks, ObserverModule, ObserverRequirements, TraceSpec
from dynlib.errors import JITUnavailableError
from dynlib.runtime.fastpath.plans import FixedStridePlan, HitTracePlan
from dynlib.runtime.fastpath.capability import assess_capability, FastpathSupport
from dynlib.runtime.results_api import ResultsView
from dynlib.runtime.runner_api import NAN_DETECTED
from dynlib.runtime.sim import Sim

__all__ = ["basin_auto"]


# Internal scan results (Numba)
_SCAN_NONE = -999999
_SCAN_BLOWUP = -1
_SCAN_OUTSIDE = -2

# Online PCR-BM status codes (analysis_out[0])
_PCR_ONLINE_NONE = 0
_PCR_ONLINE_ASSIGNED = 1
_PCR_ONLINE_BLOWUP = 2
_PCR_ONLINE_OUTSIDE = 3

# Online PCR-BM analysis_out indices
_PCR_OUT_STATUS = 0
_PCR_OUT_ASSIGNED_ID = 1
_PCR_OUT_DETECT_POS = 2
_PCR_OUT_DETECTED = 3
_PCR_OUT_OBS_COUNT = 4


def _normalize_downsample(name: str, values: Sequence[int] | int, d: int) -> np.ndarray:
    if isinstance(values, (int, np.integer)):
        arr = np.full((d,), int(values), dtype=np.int64)
    else:
        arr = np.asarray(values, dtype=np.int64)
        if arr.shape != (d,):
            raise ValueError(f"{name} must have length {d}")
    if np.any(arr <= 0):
        raise ValueError(f"{name} values must be positive")
    return arr


def _hash_capacity(window: int) -> int:
    target = max(8, int(window) * 2)
    cap = 1
    while cap < target:
        cap <<= 1
    return cap


def _read_mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _estimate_pcr_memory_bytes(
    *,
    batch: int,
    max_samples: int,
    obs_dim: int,
    blowup_dim: int,
    record_dim: int,
    dtype: np.dtype,
) -> int:
    cap_rec = int(max_samples) + 2
    obs_bytes = batch * max_samples * obs_dim * 8
    blowup_bytes = batch * max_samples * blowup_dim * 8
    cell_ids_bytes = batch * max_samples * 8
    mask_bytes = batch * max_samples * np.dtype(np.bool_).itemsize * 2
    labels_bytes = batch * 8
    record_bytes = batch * cap_rec * record_dim * int(np.dtype(dtype).itemsize)
    record_meta_bytes = batch * cap_rec * (8 + 8 + 4)  # T, STEP, FLAGS
    return int(
        obs_bytes
        + blowup_bytes
        + cell_ids_bytes
        + mask_bytes
        + labels_bytes
        + record_bytes
        + record_meta_bytes
    )


def _extract_observations(
    *,
    views: Sequence[object],
    max_samples: int,
    obs_dim: int,
    blowup_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = len(views)
    obs = np.zeros((batch, max_samples, obs_dim), dtype=np.float64)
    blowup_vals = (
        np.zeros((batch, max_samples, blowup_idx.size), dtype=np.float64) if blowup_idx.size else None
    )
    lengths = np.zeros((batch,), dtype=np.int64)
    run_status = np.zeros((batch,), dtype=np.int32)

    for i, view in enumerate(views):
        raw = view._raw  # ResultsView
        y = raw.Y_view
        n_total = int(raw.n)
        n_obs = max(0, n_total - 1)
        n_obs = min(n_obs, max_samples)
        lengths[i] = n_obs
        run_status[i] = int(view.status)
        if n_obs == 0:
            continue
        obs[i, :n_obs, :] = y[:obs_dim, 1 : n_obs + 1].T
        if blowup_vals is not None:
            blowup_vals[i, :n_obs, :] = y[blowup_idx, 1 : n_obs + 1].T

    if blowup_vals is None:
        blowup_vals = np.zeros((batch, max_samples, 0), dtype=np.float64)
    return obs, blowup_vals, lengths, run_status


if njit is not None:  # pragma: no cover

    @njit
    def _hash_slot(keys: np.ndarray, key: int, mask: int) -> tuple[int, bool]:
        idx = (key * 2654435761) & mask
        first_deleted = -1
        for _ in range(keys.shape[0]):
            k = keys[idx]
            if k == -1:
                return (first_deleted if first_deleted != -1 else idx), False
            if k == key:
                return idx, True
            if k == -2 and first_deleted == -1:
                first_deleted = idx
            idx = (idx + 1) & mask
        if first_deleted != -1:
            return first_deleted, False
        return idx, False

    @njit
    def _hash_slot_float(keys: np.ndarray, key: int, mask: int) -> tuple[int, bool]:
        idx = (key * 2654435761) & mask
        first_deleted = -1
        for _ in range(keys.shape[0]):
            k = int(keys[idx])
            if k == -1:
                return (first_deleted if first_deleted != -1 else idx), False
            if k == key:
                return idx, True
            if k == -2 and first_deleted == -1:
                first_deleted = idx
            idx = (idx + 1) & mask
        if first_deleted != -1:
            return first_deleted, False
        return idx, False

    @njit(parallel=True)
    def _quantize_cells(
        obs: np.ndarray,
        lengths: np.ndarray,
        z_min: np.ndarray,
        z_max: np.ndarray,
        grid_res: np.ndarray,
        strides: np.ndarray,
        cell_ids: np.ndarray,
    ) -> None:
        batch = obs.shape[0]
        dim = obs.shape[2]
        for i in prange(batch):
            n = int(lengths[i])
            for j in range(n):
                outside = False
                cell = 0
                for k in range(dim):
                    val = obs[i, j, k]
                    if not (val >= z_min[k] and val <= z_max[k]):
                        outside = True
                        break
                    denom = z_max[k] - z_min[k]
                    if denom <= 0.0:
                        outside = True
                        break
                    frac = (val - z_min[k]) / denom
                    idx = int(frac * grid_res[k])
                    if idx < 0:
                        idx = 0
                    elif idx >= grid_res[k]:
                        idx = grid_res[k] - 1
                    cell += idx * strides[k]
                cell_ids[i, j] = -1 if outside else cell

    @njit
    def _contains_sorted(arr: np.ndarray, key: int) -> bool:
        lo = 0
        hi = arr.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            val = arr[mid]
            if val == key:
                return True
            if val < key:
                lo = mid + 1
            else:
                hi = mid
        return False

    @njit
    def _contains_sorted_span(arr: np.ndarray, count: int, key: int) -> bool:
        lo = 0
        hi = count
        while lo < hi:
            mid = (lo + hi) // 2
            val = arr[mid]
            if val == key:
                return True
            if val < key:
                lo = mid + 1
            else:
                hi = mid
        return False

    @njit
    def _scan_persistence_outside_blowup(
        cell_row: np.ndarray,
        blowup_row: np.ndarray,
        n: int,
        cell_sets: object,
        p_in: int,
        outside_limit: int,
    ) -> int:
        outside_run = 0
        n_attr = len(cell_sets)
        if n_attr > 0 and p_in > 0:
            counters = np.zeros((n_attr,), dtype=np.int64)
        else:
            counters = np.zeros((0,), dtype=np.int64)

        for s in range(n):
            if blowup_row[s]:
                return _SCAN_BLOWUP

            c = cell_row[s]
            if c < 0:
                outside_run += 1
                if outside_run >= outside_limit:
                    return _SCAN_OUTSIDE
                if counters.size:
                    for k in range(counters.size):
                        counters[k] = 0
                continue

            outside_run = 0

            if counters.size:
                for k in range(counters.size):
                    if _contains_sorted(cell_sets[k], c):
                        counters[k] += 1
                        if counters[k] >= p_in:
                            return k
                    else:
                        counters[k] = 0

        return _SCAN_NONE

    @njit
    def _find_candidate_single(
        cell_row: np.ndarray,
        blowup_row: np.ndarray,
        n: int,
        window: int,
        u_th: float,
        recur_need: int,
        outside_limit: int,
        transient_samples: int,
        hash_cap: int,
    ) -> tuple[int, int]:
        """
        Returns (status, detect_pos)
          status: 0 none, 2 blowup, 3 outside, 1 candidate
          detect_pos: s where recur_run hit recur_need, else -1
        """
        mask = hash_cap - 1
        keys = np.full((hash_cap,), -1, dtype=np.int64)
        counts = np.zeros((hash_cap,), dtype=np.int64)
        ring = np.zeros((window,), dtype=np.int64)
        ring_count = 0
        ring_pos = 0
        unique_count = 0

        outside_run = 0
        recur_run = 0

        for s in range(n):
            if blowup_row[s]:
                return 2, -1

            c = cell_row[s]
            if c < 0:
                outside_run += 1
                if outside_run >= outside_limit:
                    return 3, -1
                continue

            outside_run = 0

            if ring_count == window:
                old = ring[ring_pos]
                slot, found = _hash_slot(keys, old, mask)
                if found:
                    counts[slot] -= 1
                    if counts[slot] <= 0:
                        keys[slot] = -2
                        counts[slot] = 0
                        unique_count -= 1
            else:
                ring_count += 1

            ring[ring_pos] = c
            ring_pos += 1
            if ring_pos >= window:
                ring_pos = 0

            slot, found = _hash_slot(keys, c, mask)
            if found:
                counts[slot] += 1
            else:
                keys[slot] = c
                counts[slot] = 1
                unique_count += 1

            if ring_count < window:
                continue
            if (s + 1) < transient_samples:
                continue

            if (unique_count / float(window)) <= u_th:
                recur_run += 1
            else:
                recur_run = 0

            if recur_run >= recur_need:
                return 1, s

        return 0, -1

else:

    def _hash_slot_float(*args, **kwargs):
        raise JITUnavailableError("basin_auto requires numba for hashing")

    def _quantize_cells(*args, **kwargs):
        raise JITUnavailableError("basin_auto requires numba for quantization")

    def _scan_persistence_outside_blowup(*args, **kwargs):
        raise JITUnavailableError("basin_auto requires numba for scanning")

    def _find_candidate_single(*args, **kwargs):
        raise JITUnavailableError("basin_auto requires numba for detection")

    def _assign_persistence_batch(*args, **kwargs):
        raise JITUnavailableError("basin_auto requires numba for persistence assignment")

    def _contains_sorted_span(*args, **kwargs):
        raise JITUnavailableError("basin_auto requires numba for scanning")


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return 0.0 if union <= 0 else inter / union


def _registry_match_or_create(
    registry: list[Attractor],
    candidate_fp: set[int],
    s_merge: float,
) -> int:
    best_id = -1
    best_sim = -1.0
    for attractor in registry:
        sim = _jaccard(attractor.fingerprint, candidate_fp)
        if sim > best_sim:
            best_sim = sim
            best_id = attractor.id
    if best_sim >= s_merge and best_id >= 0:
        return best_id
    new_id = len(registry)
    registry.append(Attractor(id=new_id, fingerprint=set(candidate_fp), cells=set()))
    return new_id


def _registry_update_cells(
    registry: list[Attractor],
    attr_id: int,
    evidence_cells: set[int],
    fp_cells: set[int],
    *,
    fingerprint_cap: int,
) -> None:
    A = registry[attr_id]
    A.cells.update(evidence_cells)
    if fp_cells:
        A.fingerprint.update(fp_cells)
    if len(A.fingerprint) > fingerprint_cap:
        A.fingerprint = set(sorted(A.fingerprint)[:fingerprint_cap])


def _build_cell_sets_numba(registry: list[Attractor]) -> object:
    _require_numba("basin_auto")
    if NumbaList is None or nb_types is None:
        raise JITUnavailableError("basin_auto requires numba typed.List")
    cell_sets = NumbaList.empty_list(nb_types.int64[:])
    for attractor in registry:
        cell_sets.append(np.array(sorted(attractor.cells), dtype=np.int64))
    return cell_sets


def _sync_cell_sets_numba(cell_sets: object | None, registry: list[Attractor]) -> object:
    updated = _build_cell_sets_numba(registry)
    if cell_sets is None:
        return updated
    # Keep the list object stable for numba hooks; replace contents in-place.
    while len(cell_sets) > 0:
        cell_sets.pop()
    for arr in updated:
        cell_sets.append(arr)
    return cell_sets


def _init_cell_set_arrays(
    *,
    max_attr: int,
    max_cells_per_attr: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_attr <= 0 or max_cells_per_attr <= 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    cell_sets = np.full((max_attr, max_cells_per_attr), -1, dtype=np.int64)
    cell_counts = np.zeros((max_attr,), dtype=np.int64)
    return cell_sets, cell_counts


def _fill_cell_set_arrays(
    *,
    cell_sets: np.ndarray,
    cell_counts: np.ndarray,
    registry: list[Attractor],
    max_cells_per_attr: int,
) -> bool:
    if cell_sets.size == 0 or cell_counts.size == 0:
        return False
    cell_sets.fill(-1)
    cell_counts[:] = 0
    truncated = False
    if len(registry) > cell_sets.shape[0]:
        truncated = True
    n_attr = min(len(registry), cell_sets.shape[0])
    for i in range(n_attr):
        cells = np.array(sorted(registry[i].cells), dtype=np.int64)
        if cells.size > max_cells_per_attr:
            cells = cells[:max_cells_per_attr]
            truncated = True
        cell_counts[i] = int(cells.size)
        if cells.size > 0:
            cell_sets[i, : cells.size] = cells
    return truncated


def _make_pcr_online_analysis(
    *,
    name: str,
    obs_idx: np.ndarray,
    blowup_idx: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    grid_res: np.ndarray,
    strides: np.ndarray,
    window: int,
    u_th: float,
    recur_need: int,
    transient_samples: int,
    outside_limit: int,
    p_in: int,
    b_max: float | None,
    post_detect_samples: int,
    cell_sets: np.ndarray,
    cell_counts: np.ndarray,
    capture_evidence: bool,
) -> ObserverModule:
    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    blowup_idx = np.asarray(blowup_idx, dtype=np.int64)
    z_min = np.asarray(z_min, dtype=np.float64)
    z_max = np.asarray(z_max, dtype=np.float64)
    grid_res = np.asarray(grid_res, dtype=np.int64)
    strides = np.asarray(strides, dtype=np.int64)

    hash_cap = int(_hash_capacity(window))
    mask = hash_cap - 1
    if cell_sets.size == 0 or cell_counts.size == 0:
        max_attr = 0
    else:
        max_attr = int(cell_counts.shape[0])

    ring_offset = 0
    keys_offset = ring_offset + int(window)
    counts_offset = keys_offset + hash_cap
    counters_offset = counts_offset + hash_cap
    scalars_offset = counters_offset + max_attr
    ring_count_idx = scalars_offset
    ring_pos_idx = scalars_offset + 1
    unique_idx = scalars_offset + 2
    outside_idx = scalars_offset + 3
    recur_idx = scalars_offset + 4
    obs_idx_ws = scalars_offset + 5
    detect_flag_idx = scalars_offset + 6
    detect_pos_idx = scalars_offset + 7
    post_remaining_idx = scalars_offset + 8
    ws_size = post_remaining_idx + 1

    check_blowup = b_max is not None and blowup_idx.size > 0
    b_max_val = float(b_max) if b_max is not None else 0.0
    p_in_val = int(p_in)
    outside_limit_val = int(outside_limit)
    window_val = int(window)
    transient_val = int(transient_samples)
    recur_need_val = int(recur_need)
    post_detect_val = int(post_detect_samples)
    obs_dim = int(obs_idx.size)

    def _pre_step(
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
        if step != 0:
            return
        if analysis_ws.shape[0] < ws_size or analysis_out.shape[0] < 5:
            return
        analysis_out[_PCR_OUT_STATUS] = float(_PCR_ONLINE_NONE)
        analysis_out[_PCR_OUT_ASSIGNED_ID] = -1.0
        analysis_out[_PCR_OUT_DETECT_POS] = -1.0
        analysis_out[_PCR_OUT_DETECTED] = 0.0
        analysis_out[_PCR_OUT_OBS_COUNT] = 0.0
        trace_count[0] = 0

        keys = analysis_ws[keys_offset:counts_offset]
        counts = analysis_ws[counts_offset:counters_offset]
        for i in range(hash_cap):
            keys[i] = -1.0
            counts[i] = 0.0

        counters = analysis_ws[counters_offset:scalars_offset]
        for i in range(max_attr):
            counters[i] = 0.0

        analysis_ws[ring_count_idx] = 0.0
        analysis_ws[ring_pos_idx] = 0.0
        analysis_ws[unique_idx] = 0.0
        analysis_ws[outside_idx] = 0.0
        analysis_ws[recur_idx] = 0.0
        analysis_ws[obs_idx_ws] = 0.0
        analysis_ws[detect_flag_idx] = 0.0
        analysis_ws[detect_pos_idx] = -1.0
        analysis_ws[post_remaining_idx] = 0.0

    def _post_step(
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
        if analysis_ws.shape[0] < ws_size or analysis_out.shape[0] < 5:
            return
        if analysis_out[_PCR_OUT_STATUS] != float(_PCR_ONLINE_NONE):
            return

        obs_count = int(analysis_ws[obs_idx_ws])
        just_detected = False

        if check_blowup:
            for i in range(blowup_idx.size):
                val = y_curr[int(blowup_idx[i])]
                if abs(val) > b_max_val:
                    analysis_out[_PCR_OUT_STATUS] = float(_PCR_ONLINE_BLOWUP)
                    if runtime_ws.stop_flag.shape[0] > 0:
                        runtime_ws.stop_flag[0] = 1
                    return

        detect_flag = int(analysis_ws[detect_flag_idx])

        outside = False
        cell = 0
        for k in range(obs_dim):
            val = y_curr[int(obs_idx[k])]
            if not (val >= z_min[k] and val <= z_max[k]):
                outside = True
                break
            denom = z_max[k] - z_min[k]
            if denom <= 0.0:
                outside = True
                break
            frac = (val - z_min[k]) / denom
            idx = int(frac * grid_res[k])
            if idx < 0:
                idx = 0
            elif idx >= grid_res[k]:
                idx = int(grid_res[k] - 1)
            cell += idx * int(strides[k])

        if outside:
            outside_run = int(analysis_ws[outside_idx]) + 1
            analysis_ws[outside_idx] = float(outside_run)
            if outside_run >= outside_limit_val:
                analysis_out[_PCR_OUT_STATUS] = float(_PCR_ONLINE_OUTSIDE)
                if runtime_ws.stop_flag.shape[0] > 0:
                    runtime_ws.stop_flag[0] = 1
                return
            if p_in_val > 0 and max_attr > 0:
                counters = analysis_ws[counters_offset:scalars_offset]
                n_attr = len(cell_sets)
                if n_attr > max_attr:
                    n_attr = max_attr
                for i in range(n_attr):
                    counters[i] = 0.0
            if detect_flag != 0 and post_detect_val > 0:
                post_remaining = int(analysis_ws[post_remaining_idx])
                if post_remaining > 0 and not just_detected:
                    analysis_ws[post_remaining_idx] = float(post_remaining - 1)
            analysis_ws[obs_idx_ws] = float(obs_count + 1)
            analysis_out[_PCR_OUT_OBS_COUNT] = float(obs_count + 1)
            return

        analysis_ws[outside_idx] = 0.0

        if p_in_val > 0 and max_attr > 0 and cell_sets.size > 0:
            counters = analysis_ws[counters_offset:scalars_offset]
            for i in range(max_attr):
                count = int(cell_counts[i])
                if count <= 0:
                    continue
                if _contains_sorted_span(cell_sets[i], count, cell):
                    counters[i] += 1.0
                    if counters[i] >= float(p_in_val):
                        analysis_out[_PCR_OUT_STATUS] = float(_PCR_ONLINE_ASSIGNED)
                        analysis_out[_PCR_OUT_ASSIGNED_ID] = float(i)
                        if runtime_ws.stop_flag.shape[0] > 0:
                            runtime_ws.stop_flag[0] = 1
                        return
                else:
                    counters[i] = 0.0

        ring = analysis_ws[ring_offset:keys_offset]
        keys = analysis_ws[keys_offset:counts_offset]
        counts = analysis_ws[counts_offset:counters_offset]

        if detect_flag == 0:
            ring_count = int(analysis_ws[ring_count_idx])
            ring_pos = int(analysis_ws[ring_pos_idx])
            unique_count = int(analysis_ws[unique_idx])

            if ring_count == window_val:
                old = int(ring[ring_pos])
                slot, found = _hash_slot_float(keys, old, mask)
                if found:
                    counts[slot] -= 1.0
                    if counts[slot] <= 0.0:
                        keys[slot] = -2.0
                        counts[slot] = 0.0
                        unique_count -= 1
            else:
                ring_count += 1

            ring[ring_pos] = float(cell)
            ring_pos += 1
            if ring_pos >= window_val:
                ring_pos = 0

            slot, found = _hash_slot_float(keys, int(cell), mask)
            if found:
                counts[slot] += 1.0
            else:
                keys[slot] = float(cell)
                counts[slot] = 1.0
                unique_count += 1

            analysis_ws[ring_count_idx] = float(ring_count)
            analysis_ws[ring_pos_idx] = float(ring_pos)
            analysis_ws[unique_idx] = float(unique_count)

            if ring_count == window_val and (obs_count + 1) >= transient_val:
                if (float(unique_count) / float(window_val)) <= float(u_th):
                    analysis_ws[recur_idx] = float(int(analysis_ws[recur_idx]) + 1)
                else:
                    analysis_ws[recur_idx] = 0.0
                if int(analysis_ws[recur_idx]) >= recur_need_val:
                    detect_flag = 1
                    just_detected = True
                    analysis_ws[detect_flag_idx] = 1.0
                    analysis_out[_PCR_OUT_DETECTED] = 1.0
                    analysis_ws[detect_pos_idx] = float(obs_count)
                    analysis_out[_PCR_OUT_DETECT_POS] = float(obs_count)
                    analysis_ws[post_remaining_idx] = float(post_detect_val)
                    if trace_cap > 0 and trace_buf.shape[0] > 0:
                        if ring_count < window_val:
                            for j in range(ring_count):
                                idx = int(trace_count[0])
                                if idx >= trace_cap:
                                    trace_count[0] = trace_cap + 1
                                    break
                                trace_buf[idx, 0] = ring[j]
                                trace_count[0] = idx + 1
                        else:
                            for j in range(window_val):
                                idx = int(trace_count[0])
                                if idx >= trace_cap:
                                    trace_count[0] = trace_cap + 1
                                    break
                                ring_idx = ring_pos + j
                                if ring_idx >= window_val:
                                    ring_idx -= window_val
                                trace_buf[idx, 0] = ring[ring_idx]
                                trace_count[0] = idx + 1

        if detect_flag != 0 and post_detect_val > 0:
            post_remaining = int(analysis_ws[post_remaining_idx])
            if post_remaining > 0 and not just_detected:
                if trace_cap > 0 and trace_buf.shape[0] > 0:
                    idx = int(trace_count[0])
                    if idx < trace_cap:
                        trace_buf[idx, 0] = float(cell)
                        trace_count[0] = idx + 1
                    else:
                        trace_count[0] = trace_cap + 1
                analysis_ws[post_remaining_idx] = float(post_remaining - 1)

        analysis_ws[obs_idx_ws] = float(obs_count + 1)
        analysis_out[_PCR_OUT_OBS_COUNT] = float(obs_count + 1)

    trace = None
    trace_names = None
    if capture_evidence:
        trace = TraceSpec(width=1, plan=HitTracePlan(stride=1, max_hits=window_val + post_detect_val))
        trace_names = ("evidence_cell",)

    class _PCROnlineModule(ObserverModule):
        def _compile_hooks(self, hooks: ObserverHooks, dtype: np.dtype) -> ObserverHooks:
            if njit is None:  # pragma: no cover - numba may be missing
                raise RuntimeError(
                    f"Observer '{self.name}' requested jit hooks but numba is not installed"
                )

            def _jit(fn):
                if fn is None:
                    return None
                try:
                    return njit(cache=False)(fn)
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

    return _PCROnlineModule(
        key=name,
        name=name,
        requirements=ObserverRequirements(fixed_step=True),
        workspace_size=ws_size,
        output_size=5,
        output_names=(
            "phase_b_status",
            "assigned_id",
            "detect_pos",
            "detected",
            "obs_count",
        ),
        trace=trace,
        trace_names=trace_names,
        hooks=ObserverHooks(pre_step=_pre_step, post_step=_post_step),
        stop_phase_mask=2,
    )


def _assess_fastpath(
    sim: Sim,
    *,
    plan: FixedStridePlan,
    record_vars: Sequence[str],
    dt: float,
    transient: float,
) -> FastpathSupport:
    adaptive = getattr(sim._stepper_spec.meta, "time_control", "fixed") == "adaptive"
    return assess_capability(
        sim,
        plan=plan,
        record_vars=record_vars,
        dt=dt,
        transient=transient,
        adaptive=adaptive,
    )


def _coarsen_cells_numpy(
    cells: np.ndarray,
    grid_res: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    """
    Packed detection-cell ids -> packed merge-cell ids using integer downsample per axis.
    cells must be non-negative int64.
    """
    d = int(grid_res.shape[0])
    if d == 1:
        n0 = int(grid_res[0])
        d0 = int(down[0])
        idx0 = cells % n0
        idx0 = idx0 // d0
        return idx0.astype(np.int64, copy=False)

    if d == 2:
        n0 = int(grid_res[0])
        n1 = int(grid_res[1])
        d0 = int(down[0])
        d1 = int(down[1])
        idx0 = cells % n0
        idx1 = (cells // n0) % n1
        idx0 = idx0 // d0
        idx1 = idx1 // d1
        m0 = (n0 + d0 - 1) // d0
        return (idx0 + idx1 * m0).astype(np.int64, copy=False)

    n0 = int(grid_res[0])
    n1 = int(grid_res[1])
    n2 = int(grid_res[2])
    d0 = int(down[0])
    d1 = int(down[1])
    d2 = int(down[2])
    idx0 = cells % n0
    tmp = cells // n0
    idx1 = tmp % n1
    idx2 = tmp // n1
    idx0 = idx0 // d0
    idx1 = idx1 // d1
    idx2 = idx2 // d2
    m0 = (n0 + d0 - 1) // d0
    m1 = (n1 + d1 - 1) // d1
    return (idx0 + idx1 * m0 + idx2 * (m0 * m1)).astype(np.int64, copy=False)


def _fingerprint_from_merge_ids(
    merge_ids: np.ndarray,
    *,
    fingerprint_cap: int,
    min_count: int,
    min_fp: int,
) -> set[int]:
    if merge_ids.size == 0:
        return set()
    vals, counts = np.unique(merge_ids, return_counts=True)
    if vals.size == 0:
        return set()
    order = np.lexsort((vals, -counts))
    vals = vals[order]
    counts = counts[order]
    keep = vals[counts >= min_count]
    if keep.size < min_fp:
        take = max(min_fp, int(vals.size // 2))
        keep = vals[: min(take, vals.size)]
    if keep.size > fingerprint_cap:
        keep = keep[:fingerprint_cap]
    return set(int(x) for x in keep.tolist())


def _pcr_outcome_from_view(
    view: ResultsView,
    analysis_name: str,
) -> tuple[int, int, int, np.ndarray]:
    if view.status == NAN_DETECTED:
        return NAN_DETECTED, _PCR_ONLINE_NONE, -1, np.empty((0,), dtype=np.int64)
    res = view.observers.get(analysis_name)
    if res is None or res["out"] is None:
        return view.status, _PCR_ONLINE_NONE, -1, np.empty((0,), dtype=np.int64)

    out = res["out"]
    status = int(out[_PCR_OUT_STATUS]) if out.size > 0 else _PCR_ONLINE_NONE
    assigned_id = int(out[_PCR_OUT_ASSIGNED_ID]) if status == _PCR_ONLINE_ASSIGNED else -1
    detected = int(out[_PCR_OUT_DETECTED]) != 0 if out.size > _PCR_OUT_DETECTED else False

    if not detected:
        return view.status, status, assigned_id, np.empty((0,), dtype=np.int64)

    trace = res["trace"]
    if trace is None or trace.size == 0:
        return view.status, status, assigned_id, np.empty((0,), dtype=np.int64)

    evidence = np.asarray(trace[:, 0], dtype=np.int64)
    evidence = evidence[evidence >= 0]
    return view.status, status, assigned_id, evidence


def _apply_pcr_outcome(
    *,
    idx: int,
    outcome: tuple[int, int, int, np.ndarray],
    labels: np.ndarray,
    registry: list[Attractor],
    grid_arr: np.ndarray,
    down_arr: np.ndarray,
    fp_cap: int,
    fp_min_count: int,
    fp_min_fp: int,
    s_merge: float,
    assign_only: bool = False,
) -> bool:
    run_status, status, assigned_id, evidence = outcome

    if assign_only:
        if status == _PCR_ONLINE_ASSIGNED and assigned_id >= 0:
            labels[idx] = int(assigned_id)
        return False

    if run_status == NAN_DETECTED:
        labels[idx] = BLOWUP
        return False
    if status == _PCR_ONLINE_ASSIGNED:
        labels[idx] = int(assigned_id)
        return False
    if status == _PCR_ONLINE_BLOWUP:
        labels[idx] = BLOWUP
        return False
    if status == _PCR_ONLINE_OUTSIDE:
        labels[idx] = OUTSIDE
        return False

    if evidence.size == 0:
        labels[idx] = UNRESOLVED
        return False

    evid_unique = np.unique(evidence)
    if evid_unique.size == 0:
        labels[idx] = UNRESOLVED
        return False

    evidence_cells = set(int(x) for x in evid_unique.tolist())
    merge_ids = _coarsen_cells_numpy(evid_unique, grid_arr, down_arr)
    candidate_fp = _fingerprint_from_merge_ids(
        merge_ids,
        fingerprint_cap=fp_cap,
        min_count=fp_min_count,
        min_fp=fp_min_fp,
    )
    if not candidate_fp:
        vals = np.unique(merge_ids)
        if vals.size > fp_cap:
            vals = vals[:fp_cap]
        candidate_fp = set(int(x) for x in vals.tolist())

    attr_id = _registry_match_or_create(registry, candidate_fp, float(s_merge))
    _registry_update_cells(registry, int(attr_id), evidence_cells, candidate_fp, fingerprint_cap=fp_cap)
    labels[idx] = int(attr_id)
    return True


def _is_jitted_runner(fn) -> bool:
    """Best-effort detection of a numba-compiled runner."""
    return bool(getattr(fn, "signatures", None))


def _resolve_process_workers(max_workers: int | None) -> int:
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, 8)
    return max(1, int(max_workers))


def _chunk_ranges(total: int, n_workers: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    chunk_size = (total + n_workers - 1) // n_workers
    return [
        (start, min(start + chunk_size, total))
        for start in range(0, total, chunk_size)
    ]


# Module-level worker state for process-based PCR batches
_basin_auto_worker_sim: "Sim | None" = None
_basin_auto_worker_config: dict | None = None


def _init_basin_auto_worker(init_config: dict) -> None:
    global _basin_auto_worker_sim, _basin_auto_worker_config
    from dynlib.compiler.build import build
    from dynlib.runtime.sim import Sim as SimClass

    full_model = build(
        init_config["model_spec"],
        stepper=init_config["stepper"],
        jit=bool(init_config.get("jit", True)),
    )
    sim = SimClass(full_model)

    session_params = init_config.get("session_params")
    if session_params is not None:
        sim.assign(**{
            name: val for name, val in zip(sim.model.spec.params, session_params)
        })

    record_vars = init_config.get("record_vars", [])
    (
        state_rec_indices,
        aux_rec_indices,
        state_rec_names,
        aux_names,
    ) = sim._resolve_recording_selection(record_vars)

    init_config["state_rec_indices"] = state_rec_indices
    init_config["aux_rec_indices"] = aux_rec_indices
    init_config["state_rec_names"] = state_rec_names
    init_config["aux_names"] = aux_names
    init_config["stepper_config"] = sim.stepper_config()
    init_config["adaptive"] = getattr(sim._stepper_spec.meta, "time_control", "fixed") == "adaptive"

    _basin_auto_worker_sim = sim
    _basin_auto_worker_config = init_config


def _basin_auto_chunk_worker(args: tuple[np.ndarray, np.ndarray, dict]) -> list[tuple[int, int, int, np.ndarray]]:
    global _basin_auto_worker_sim, _basin_auto_worker_config
    ic_chunk, params_chunk, run_config = args

    if _basin_auto_worker_sim is None or _basin_auto_worker_config is None:
        raise RuntimeError("Worker not initialized. Call _init_basin_auto_worker first.")

    sim = _basin_auto_worker_sim
    cfg = _basin_auto_worker_config
    analysis_name = run_config["analysis_name"]
    analysis_mod = _make_pcr_online_analysis(
        name=analysis_name,
        obs_idx=cfg["obs_state_idx"],
        blowup_idx=cfg["blowup_state_idx"],
        z_min=cfg["obs_min_arr"],
        z_max=cfg["obs_max_arr"],
        grid_res=cfg["grid_arr"],
        strides=cfg["strides"],
        window=cfg["window"],
        u_th=cfg["u_th"],
        recur_need=cfg["recur_windows"],
        transient_samples=cfg["transient_samples"],
        outside_limit=cfg["outside_limit"],
        p_in=cfg["p_in"],
        b_max=cfg["b_max"],
        post_detect_samples=cfg["post_detect_samples"],
        cell_sets=run_config["cell_sets"],
        cell_counts=run_config["cell_counts"],
        capture_evidence=run_config["capture_evidence"],
    )

    use_fastpath = bool(run_config.get("use_fastpath", False))
    views: list[ResultsView]

    if use_fastpath:
        from dynlib.runtime.fastpath import fastpath_batch_for_sim

        plan = FixedStridePlan(stride=cfg["record_stride"])
        views = fastpath_batch_for_sim(
            sim,
            plan=plan,
            t0=cfg["t0"],
            T=cfg["T"],
            N=int(cfg["N"]) if cfg["N"] is not None else None,
            dt=cfg["dt_use"],
            record_vars=cfg["record_vars"],
            transient=0.0,
            record_interval=cfg["record_stride"],
            max_steps=cfg["max_samples"] + 1,
            ic=ic_chunk,
            params=params_chunk,
            parallel_mode="none",
            max_workers=1,
            observers=analysis_mod,
        )
        if views is None:
            use_fastpath = False

    if not use_fastpath:
        views = []
        for i in range(ic_chunk.shape[0]):
            seed = sim._select_seed(
                resume=False,
                t0=cfg["t0"],
                dt=cfg["dt_use"],
                ic=ic_chunk[i],
                params=params_chunk[i],
            )
            result = sim._execute_run(
                seed=seed,
                t_end=float(cfg["T"] if cfg["T"] is not None else seed.t + float(cfg["max_samples"]) * cfg["dt_use"]),
                target_steps=int(cfg["N"]) if cfg["N"] is not None else None,
                max_steps=int(cfg["max_samples"] + 1),
                record=True,
                record_interval=cfg["record_stride"],
                cap_rec=2,
                cap_evt=1,
                stepper_config=cfg["stepper_config"],
                adaptive=cfg["adaptive"],
                wrms_cfg=None,
                state_rec_indices=cfg["state_rec_indices"],
                aux_rec_indices=cfg["aux_rec_indices"],
                state_names=cfg["state_rec_names"],
                aux_names=cfg["aux_names"],
                observers=analysis_mod,
            )
            views.append(ResultsView(result, sim.model.spec))

    return [_pcr_outcome_from_view(view, analysis_name) for view in views]


if njit is not None:  # pragma: no cover

    @njit(parallel=True)
    def _assign_persistence_batch(
        cell_ids: np.ndarray,
        lengths: np.ndarray,
        labels: np.ndarray,
        cell_sets: object,
        p_in: int,
        unresolved_label: int,
        outside_limit: int,
    ) -> None:
        batch = cell_ids.shape[0]
        n_attr = len(cell_sets)
        for i in prange(batch):
            if labels[i] != unresolved_label:
                continue
            if n_attr == 0 or p_in <= 0:
                continue
            counters = np.zeros((n_attr,), dtype=np.int64)
            outside_run = 0
            n = int(lengths[i])
            for s in range(n):
                c = cell_ids[i, s]
                if c < 0:
                    outside_run += 1
                    if outside_run >= outside_limit:
                        break
                    for k in range(n_attr):
                        counters[k] = 0
                    continue
                outside_run = 0
                for k in range(n_attr):
                    if _contains_sorted(cell_sets[k], c):
                        counters[k] += 1
                        if counters[k] >= p_in:
                            labels[i] = k
                            break
                    else:
                        counters[k] = 0
                if labels[i] != unresolved_label:
                    break


def basin_auto(
    sim: Sim,
    *,
    ic: np.ndarray | None = None,
    ic_grid: Sequence[int] | None = None,
    ic_bounds: Sequence[tuple[float, float]] | None = None,
    params: np.ndarray | None = None,
    mode: Literal["map", "ode", "auto"] = "auto",
    observe_vars: Sequence[str | int] | None = None,
    obs_min: Sequence[float] | float | None = None,
    obs_max: Sequence[float] | float | None = None,
    grid_res: Sequence[int] | int = 64,
    max_samples: int = 1024,
    transient_samples: int = 0,
    window: int = 64,
    u_th: float = 0.6,
    recur_windows: int = 3,
    s_merge: float = 0.6,
    p_in: int = 8,
    b_max: float | None = None,
    outside_limit: int = 10,
    dt_obs: float | None = None,
    blowup_vars: Sequence[str | int] | None = None,
    parallel_mode: Literal["auto", "threads", "process", "none"] = "auto",
    max_workers: int | None = None,
    # Stabilizers:
    merge_downsample: Sequence[int] | int = 1,
    post_detect_samples: int = 0,
    # Online execution (streamed analysis) (offline mode is for debugging/experimentation)
    online: bool = True,
    # Memory guard for offline mode
    max_memory_bytes: int | None = None,
    batch_size: int | None = None,
    refine_unresolved: bool = True,
    online_max_attr: int | None = None,
    online_max_cells: int | None = None,
) -> BasinResult:
    """
    Compute basins of attraction for maps and ODEs using Persistent Cell-Recurrence Basin Mapping (PCR-BM).

    This function identifies which attractor each initial condition converges to by detecting recurrent
    patterns in the state space. It uses a grid-based cell quantization combined with temporal recurrence
    detection to efficiently map basins for large batches of initial conditions.

    Parameters
    ----------
    sim : Sim
        Simulation object containing the dynamical system model and stepper configuration.
        For ODEs, must use a fixed-step stepper (adaptive steppers are not supported).

    ic : np.ndarray | None, default=None
        Initial conditions to test. Shape: (n_points, n_states) or (n_states,) for a single point.
        Each row represents one initial condition from which to compute the basin.
        Either ic or ic_grid must be provided (but not both).

    ic_grid : Sequence[int] | None, default=None
        Auto-generate uniform grid of initial conditions. Sequence of resolution per dimension.
        Example: [512, 512] creates a 512x512 grid. Requires ic_bounds to be provided.
        Either ic or ic_grid must be provided (but not both).

    ic_bounds : Sequence[tuple[float, float]] | None, default=None
        Bounds for auto-generated IC grid. Sequence of (min, max) tuples per dimension.
        Example: [(-3, 3), (-3, 3)] for 2D grid. Required when ic_grid is provided.
        Also serves as default for obs_min/obs_max when those are not specified.

    params : np.ndarray | None, default=None
        Parameter values for each initial condition. Shape: (n_points, n_params) or (n_params,).
        If None, uses the current session parameters from sim. Can broadcast with ic.

    mode : {"map", "ode", "auto"}, default="auto"
        System type. "auto" infers from sim.model.spec.kind. "map" for discrete maps,
        "ode" for continuous systems. Affects how time stepping is interpreted.

    observe_vars : Sequence[str | int] | None, default=None
        State variables to observe for basin detection. Can be variable names (str) or indices (int).
        Defines the observation space dimension. If None, uses the first d state variables where
        d is inferred from grid_res/obs_min/obs_max dimensions. Typical: first 2-3 state variables.

    obs_min : Sequence[float] | float | None, default=None
        Minimum bounds of the observation space for each dimension. If scalar, broadcasts to all dimensions.
        Defines the lower edge of the grid where attractors are detected. Should cover the region
        where attractors reside. If None, defaults to ic_bounds lower values when ic_bounds is provided,
        otherwise defaults to 0.0. Example: [-2.0, -2.0] for a 2D system.

    obs_max : Sequence[float] | float | None, default=None
        Maximum bounds of the observation space for each dimension. Must be > obs_min for all dimensions.
        Defines the upper edge of the detection grid. If None, defaults to ic_bounds upper values when
        ic_bounds is provided, otherwise defaults to 1.0. Example: [2.0, 2.0] for a 2D system.

    grid_res : Sequence[int] | int, default=64
        Grid resolution for each observation dimension. Higher values give finer spatial resolution
        but increase memory usage and computation time. Sensible range: 32-256 per dimension.
        Total cells = product of all dimensions. Keep grid_res^d manageable (< 10^7 cells typical).

    max_samples : int, default=1024
        Maximum number of time steps to evolve each trajectory. Longer trajectories can detect
        slower-converging attractors but increase computation time. Sensible range: 500-10000.
        Should be large enough for transients to decay and recurrence to emerge.

    transient_samples : int, default=0
        Number of initial samples to skip before recurrence detection begins. Use this to
        ignore initial transient behavior. Sensible range: 0 to max_samples//4.
        Typical: 100-500 for systems with long transients.

    window : int, default=64
        Sliding window size for recurrence detection. The algorithm checks how many unique cells
        are visited within this window. Larger windows are more robust but slower to detect.
        Sensible range: 32-256. Must be <= max_samples. Typical: max_samples//8 to max_samples//4.

    u_th : float, default=0.6
        Uniqueness threshold in [0, 1]. Recurrence is detected when (unique_cells / window) <= u_th.
        Lower values require tighter recurrence (more restrictive). Higher values detect recurrence
        more easily. Sensible range: 0.3-0.8. Typical: 0.5-0.7 for chaotic systems, 0.3-0.5 for
        simple attractors.

    recur_windows : int, default=3
        Number of consecutive windows that must satisfy u_th for detection. Higher values reduce
        false positives but may miss attractors. Sensible range: 2-10. Typical: 2-5.

    s_merge : float, default=0.6
        Similarity threshold in [0, 1] for merging attractor fingerprints via Jaccard similarity.
        Higher values require more similar fingerprints to merge (more conservative).
        Sensible range: 0.4-0.9. Typical: 0.5-0.7. Use higher values (0.7-0.9) when attractors
        are well-separated, lower values (0.4-0.6) when attractors have overlapping footprints.

    p_in : int, default=8
        Persistence threshold: number of consecutive in-cell hits required to assign a trajectory
        to an attractor. Higher values reduce false assignments but may leave more unresolved.
        Sensible range: 4-20. Typical: 5-10. Set to 0 to disable persistence assignment.

    b_max : float | None, default=None
        Blowup detection threshold. If any blowup_var exceeds abs(b_max), trajectory is labeled
        BLOWUP. Use to detect diverging trajectories. Typical: 1e3 to 1e10 depending on system.
        If None, blowup detection is disabled (relies only on NaN/Inf checks).

    outside_limit : int, default=10
        Number of consecutive samples outside [obs_min, obs_max] before labeling trajectory as OUTSIDE.
        Helps detect trajectories escaping the observation region. Sensible range: 5-50.
        Typical: 10-20.

    dt_obs : float | None, default=None
        Time step between observations. For maps, this is the iteration step (default=sim dt).
        For ODEs, this is required and defines the sampling interval. Typical for ODEs: 0.01-0.1
        depending on system timescale.

    blowup_vars : Sequence[str | int] | None, default=None
        Variables to monitor for blowup detection (used with b_max). Can be names or indices.
        If None and b_max is set, defaults to observe_vars. Typical: use state variables that
        may diverge (e.g., position, velocity).

    parallel_mode : {"auto", "threads", "process", "none"}, default="auto"
        Parallelization strategy. "auto" chooses based on system. "threads" uses thread pool,
        "process" uses process pool (better for CPU-bound), "none" runs serially.

    max_workers : int | None, default=None
        Maximum number of parallel workers. If None, uses system CPU count.
        Typical: None (auto) or number of physical cores.

    merge_downsample : Sequence[int] | int, default=1
        Downsampling factor per axis for the merge fingerprint grid. Fingerprints are computed
        on a coarser grid (grid_res // merge_downsample) to make matching more robust.
        Sensible range: 1-4. Use 1 for no downsampling, 2-4 for more robust merging when
        attractors have slightly different trajectories. Higher values = more aggressive merging.

    post_detect_samples : int, default=0
        Number of additional samples to collect after recurrence detection. These samples are
        added to the evidence segment, expanding the attractor fingerprint and cell set.
        Sensible range: 0 to window. Typical: 0 (disabled) or window//4 to window//2.
        Helps grow attractor footprints for better distinction.

    online : bool, default=True
        If True, uses online (streaming) mode that analyzes trajectories during integration,
        avoiding full trajectory storage. Recommended for large batches. If False, uses offline
        mode that stores full trajectories (useful for debugging but memory-intensive).

    max_memory_bytes : int | None, default=None
        Memory limit for offline mode in bytes. If None, uses 50% of available system memory.
        Prevents out-of-memory errors. Only applies when online=False. Typical: None (auto) or
        explicit limit like 8*1024**3 for 8 GB.

    batch_size : int | None, default=None
        Number of initial conditions to process per batch in online mode. If None, auto-determines
        based on total ic count (min 1, default ~4096). Affects memory-parallelism tradeoff.
        Sensible range: 100-10000. Typical: 1000-5000 for balanced performance.

    refine_unresolved : bool, default=True
        In online mode, whether to re-run persistence assignment on UNRESOLVED points after
        the full registry is built. Helps assign more points that were encountered before
        their attractor was discovered. Recommended: True.

    online_max_attr : int | None, default=None
        Maximum number of attractors to track for persistence scans in online mode.
        If None, defaults to 64. Limits memory for large attractor counts.
        Sensible range: 32-256. Increase if you expect many distinct attractors.

    online_max_cells : int | None, default=None
        Maximum number of cells to store per attractor for persistence scans in online mode.
        If None, defaults to max(256, window*8). Limits memory per attractor.
        Sensible range: 128-2048. Increase for large, complex attractors.

    Returns
    -------
    BasinResult
        Result object containing:
        - labels: np.ndarray of shape (n_points,) with attractor IDs (non-negative integers)
          or special codes: BLOWUP (-1), OUTSIDE (-2), UNRESOLVED (-3).
        - registry: list[Attractor] containing discovered attractors with their fingerprints
          and cell sets.
        - meta: dict with algorithm parameters and execution metadata.

    Notes
    -----
    Algorithm overview:
    1. Each trajectory is quantized into cells on the observation grid.
    2. Recurrence detection identifies when a trajectory revisits the same region repeatedly.
    3. Detected attractors are fingerprinted and stored in a registry.
    4. Attractors are merged based on fingerprint similarity (s_merge).
    5. Trajectories are assigned to attractors via persistence (p_in consecutive hits).

    Parameter tuning guidelines:
    - Start with defaults and adjust based on system characteristics.
    - For simple periodic attractors: lower u_th (0.3-0.5), smaller window (32-64).
    - For chaotic attractors: higher u_th (0.6-0.8), larger window (64-256).
    - If too many UNRESOLVED: increase max_samples, decrease u_th, or increase window.
    - If attractors merge incorrectly: increase s_merge or decrease merge_downsample.
    - If attractors don't merge when they should: decrease s_merge or increase merge_downsample.

    Requires numba for JIT compilation (nopython mode).

    Examples
    --------
    >>> # 2D Henon map basin of attraction
    >>> from dynlib import Sim
    >>> import numpy as np
    >>> 
    >>> # Create grid of initial conditions
    >>> x = np.linspace(-2, 2, 200)
    >>> y = np.linspace(-2, 2, 200)
    >>> X, Y = np.meshgrid(x, y, indexing="ij")
    >>> ic = np.column_stack([X.ravel(), Y.ravel()])
    >>> 
    >>> # Compute basins (method 1: manual IC grid)
    >>> result = basin_auto(
    ...     sim,
    ...     ic=ic,
    ...     mode="map",
    ...     obs_min=[-2, -2],
    ...     obs_max=[2, 2],
    ...     grid_res=64,
    ...     max_samples=500,
    ...     window=50,
    ...     u_th=0.5,
    ... )
    >>> 
    >>> # Method 2: auto-generate IC grid
    >>> result = basin_auto(
    ...     sim,
    ...     ic_grid=[200, 200],
    ...     ic_bounds=[(-2, 2), (-2, 2)],
    ...     mode="map",
    ...     grid_res=64,
    ...     max_samples=500,
    ... )
    >>> 
    >>> # Reshape for visualization
    >>> labels_2d = result.labels.reshape(200, 200)
    """
    _require_numba("basin_auto")

    # Validate ic/ic_grid mutual exclusivity
    if ic is None and ic_grid is None:
        raise ValueError("Either ic or ic_grid must be provided")
    if ic is not None and ic_grid is not None:
        raise ValueError("Cannot specify both ic and ic_grid")
    if ic_grid is not None and ic_bounds is None:
        raise ValueError("ic_bounds must be provided when using ic_grid")

    # Auto-generate IC grid if requested
    ic_grid_meta: tuple[int, ...] | None = None
    ic_bounds_meta: tuple[tuple[float, float], ...] | None = None
    if ic_grid is not None:
        ic_grid_arr = np.asarray(ic_grid, dtype=np.int64)
        if ic_grid_arr.ndim != 1:
            raise ValueError("ic_grid must be a 1D sequence")
        n_dims = len(ic_grid_arr)
        if len(ic_bounds) != n_dims:
            raise ValueError(f"ic_bounds must have {n_dims} elements to match ic_grid")
        
        # Create meshgrid
        axes = [np.linspace(float(bmin), float(bmax), int(n)) 
                for (bmin, bmax), n in zip(ic_bounds, ic_grid_arr)]
        meshgrids = np.meshgrid(*axes, indexing="ij")
        ic = np.column_stack([g.ravel(order="C") for g in meshgrids])
        ic_grid_meta = tuple(int(x) for x in ic_grid_arr)
        ic_bounds_meta = tuple((float(bmin), float(bmax)) for bmin, bmax in ic_bounds)

    # Default obs_min/obs_max from ic_bounds if not provided
    if obs_min is None:
        if ic_bounds is not None:
            obs_min = [float(bmin) for bmin, _ in ic_bounds]
        else:
            obs_min = 0.0
    if obs_max is None:
        if ic_bounds is not None:
            obs_max = [float(bmax) for _, bmax in ic_bounds]
        else:
            obs_max = 1.0

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if recur_windows <= 0:
        raise ValueError("recur_windows must be positive")
    if outside_limit <= 0:
        raise ValueError("outside_limit must be positive")
    if not (0.0 <= u_th <= 1.0):
        raise ValueError("u_th must be in [0, 1]")
    if not (0.0 <= s_merge <= 1.0):
        raise ValueError("s_merge must be in [0, 1]")
    if transient_samples < 0:
        raise ValueError("transient_samples must be non-negative")
    if p_in < 0:
        raise ValueError("p_in must be non-negative")
    if b_max is not None and b_max <= 0.0:
        raise ValueError("b_max must be positive when provided")
    if window > max_samples:
        raise ValueError("window cannot exceed max_samples")
    if post_detect_samples < 0:
        raise ValueError("post_detect_samples must be non-negative")
    if transient_samples >= max_samples:
        raise ValueError("transient_samples must be less than max_samples")

    mode_use = _resolve_mode(mode=mode, sim=sim)
    adaptive = getattr(sim._stepper_spec.meta, "time_control", "fixed") == "adaptive"
    if mode_use == "ode" and adaptive:
        raise ValueError("basin_auto requires a fixed-step stepper for ODE mode")

    dims: list[int] = []
    if observe_vars is not None:
        obs_len = _seq_len(observe_vars)
        if obs_len is not None:
            dims.append(obs_len)
    grid_len = _seq_len(grid_res)
    if grid_len is not None:
        dims.append(grid_len)
    obs_min_len = _seq_len(obs_min)
    if obs_min_len is not None:
        dims.append(obs_min_len)
    obs_max_len = _seq_len(obs_max)
    if obs_max_len is not None:
        dims.append(obs_max_len)

    d = dims[0] if dims else 1
    if any(dim != d for dim in dims):
        raise ValueError("observe_vars, obs_min/obs_max, and grid_res must agree on dimension")

    obs_min_arr = _normalize_dims("obs_min", obs_min, d)
    obs_max_arr = _normalize_dims("obs_max", obs_max, d)
    if np.any(obs_max_arr <= obs_min_arr):
        raise ValueError("obs_max must be greater than obs_min for all dimensions")
    grid_arr = _normalize_grid(grid_res, d)

    down_arr = _normalize_downsample("merge_downsample", merge_downsample, d)
    down_arr = np.minimum(down_arr, grid_arr)

    blowup_use = blowup_vars
    if b_max is not None and blowup_use is None:
        blowup_use = observe_vars if observe_vars is not None else tuple(range(d))

    record_vars, blowup_idx = _prepare_record_vars(sim, observe_vars, blowup_use, d)

    state_names = list(sim.model.spec.states)
    state_to_idx = {name: idx for idx, name in enumerate(state_names)}
    obs_names = record_vars[:d]
    obs_state_idx = np.array([state_to_idx[name] for name in obs_names], dtype=np.int64)
    blowup_names = [record_vars[int(idx)] for idx in blowup_idx.tolist()]
    blowup_state_idx = np.array([state_to_idx[name] for name in blowup_names], dtype=np.int64)

    n_state = len(sim.model.spec.states)
    n_params = len(sim.model.spec.params)
    dtype = sim.model.dtype
    session_params = sim.param_vector(source="session", copy=True)
    if params is None:
        params = session_params
    ic_arr, params_arr = _coerce_batch(
        ic=ic,
        params=params,
        n_state=n_state,
        n_params=n_params,
        dtype=dtype,
    )

    if not online:
        guard_bytes = max_memory_bytes
        if guard_bytes is None:
            avail_bytes = _read_mem_available_bytes()
            if avail_bytes is not None:
                guard_bytes = int(avail_bytes * 0.5)
        if guard_bytes is not None and guard_bytes > 0:
            est_bytes = _estimate_pcr_memory_bytes(
                batch=int(ic_arr.shape[0]),
                max_samples=int(max_samples),
                obs_dim=int(d),
                blowup_dim=int(blowup_idx.size),
                record_dim=int(len(record_vars)),
                dtype=dtype,
            )
            if est_bytes > guard_bytes:
                raise MemoryError(
                    "basin_auto estimated allocation "
                    f"({_format_bytes(est_bytes)}) exceeds guard "
                    f"({_format_bytes(int(guard_bytes))}). "
                    "Reduce max_samples/grid size, or set online=True "
                    "(or override max_memory_bytes=0 to disable the guard)."
                )

    if mode_use == "ode":
        if dt_obs is None:
            raise ValueError("dt_obs is required for ODE mode")
        dt_use = float(dt_obs)
        t0 = float(sim.model.spec.sim.t0)
        T = t0 + float(max_samples) * dt_use
        N = None
    else:
        dt_use = float(dt_obs if dt_obs is not None else sim.model.spec.sim.dt)
        T = None
        N = int(max_samples)

    if online:
        grid_cells = int(np.prod(grid_arr))
        if grid_cells > (2**31 - 1):
            raise ValueError(
                "online mode requires grid_res product <= 2^31-1 to keep cell ids exact "
                f"(got {grid_cells}). Reduce grid_res or use offline mode."
            )
        record_stride = int(max_samples + 1)
        plan = FixedStridePlan(stride=record_stride)
        record_vars_run: list[str] = []
        batch = int(ic_arr.shape[0])
        labels = np.full((batch,), UNRESOLVED, dtype=np.int64)
        registry: list[Attractor] = []

        strides = np.ones((d,), dtype=np.int64)
        for k in range(1, d):
            strides[k] = strides[k - 1] * grid_arr[k - 1]

        fp_cap = max(32, int(window))
        fp_min_count = 2
        fp_min_fp = max(8, int(window) // 16)

        max_attr = int(online_max_attr) if online_max_attr is not None else 64
        max_cells = int(online_max_cells) if online_max_cells is not None else max(256, int(window) * 8)
        if max_attr < 0 or max_cells < 0:
            raise ValueError("online_max_attr and online_max_cells must be non-negative")
        cell_sets_arr, cell_counts = _init_cell_set_arrays(
            max_attr=max_attr,
            max_cells_per_attr=max_cells,
        )
        analysis_name = "pcr_basin_online"
        analysis_mod = _make_pcr_online_analysis(
            name=analysis_name,
            obs_idx=obs_state_idx,
            blowup_idx=blowup_state_idx,
            z_min=obs_min_arr,
            z_max=obs_max_arr,
            grid_res=grid_arr,
            strides=strides,
            window=window,
            u_th=u_th,
            recur_need=recur_windows,
            transient_samples=transient_samples,
            outside_limit=outside_limit,
            p_in=p_in,
            b_max=b_max,
            post_detect_samples=post_detect_samples,
            cell_sets=cell_sets_arr,
            cell_counts=cell_counts,
            capture_evidence=True,
        )

        support = assess_capability(
            sim,
            plan=plan,
            record_vars=record_vars_run,
            dt=dt_use,
            transient=0.0,
            adaptive=adaptive,
            observers=analysis_mod,
        )
        use_fastpath = support.ok
        fastpath_used = False

        (
            state_rec_indices,
            aux_rec_indices,
            state_rec_names,
            aux_names,
        ) = sim._resolve_recording_selection(record_vars_run)
        stepper_config = sim.stepper_config()

        if batch_size is None:
            batch_size_use = min(batch, 4096) if batch > 0 else 0
        else:
            batch_size_use = int(batch_size)
            if batch_size_use <= 0:
                raise ValueError("batch_size must be positive when provided")
            batch_size_use = min(batch_size_use, batch)

        use_process_parallel = parallel_mode in ("process", "auto") and batch > 1000
        n_workers = _resolve_process_workers(max_workers)
        if parallel_mode == "none" or n_workers == 1:
            use_process_parallel = False

        effective_parallel_mode = parallel_mode
        if parallel_mode == "process" and not use_process_parallel:
            effective_parallel_mode = "none"

        process_executor = None
        if use_process_parallel:
            init_config = dict(
                model_spec=sim.model.spec,
                stepper=sim.model.stepper_name,
                jit=_is_jitted_runner(sim.model.runner),
                session_params=session_params,
                record_vars=record_vars_run,
                record_stride=record_stride,
                dt_use=dt_use,
                T=T,
                N=N,
                t0=sim.model.spec.sim.t0,
                max_samples=max_samples,
                obs_state_idx=obs_state_idx,
                blowup_state_idx=blowup_state_idx,
                obs_min_arr=obs_min_arr,
                obs_max_arr=obs_max_arr,
                grid_arr=grid_arr,
                strides=strides,
                window=window,
                u_th=u_th,
                recur_windows=recur_windows,
                transient_samples=transient_samples,
                outside_limit=outside_limit,
                p_in=p_in,
                b_max=b_max,
                post_detect_samples=post_detect_samples,
            )
            process_executor = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_basin_auto_worker,
                initargs=(init_config,),
            )

        registry_dirty = True
        persistence_truncated = False

        for start in range(0, batch, max(1, batch_size_use)):
            stop = min(batch, start + max(1, batch_size_use))
            if registry_dirty:
                persistence_truncated = _fill_cell_set_arrays(
                    cell_sets=cell_sets_arr,
                    cell_counts=cell_counts,
                    registry=registry,
                    max_cells_per_attr=max_cells,
                )
                registry_dirty = False

            if use_process_parallel and process_executor is not None:
                batch_ic = np.ascontiguousarray(ic_arr[start:stop])
                batch_params = np.ascontiguousarray(params_arr[start:stop])
                run_config = dict(
                    analysis_name=analysis_name,
                    capture_evidence=True,
                    cell_sets=cell_sets_arr,
                    cell_counts=cell_counts,
                    use_fastpath=use_fastpath,
                )
                chunk_ranges = _chunk_ranges(batch_ic.shape[0], n_workers)
                chunk_args = [
                    (batch_ic[c_start:c_stop], batch_params[c_start:c_stop], run_config)
                    for c_start, c_stop in chunk_ranges
                ]
                outcomes: list[tuple[int, int, int, np.ndarray]] = []
                for chunk_outcomes in process_executor.map(_basin_auto_chunk_worker, chunk_args):
                    outcomes.extend(chunk_outcomes)
                if use_fastpath:
                    fastpath_used = True
            else:
                if use_fastpath:
                    from dynlib.runtime.fastpath import fastpath_batch_for_sim

                    views = fastpath_batch_for_sim(
                        sim,
                        plan=plan,
                        t0=sim.model.spec.sim.t0,
                        T=T,
                        N=N,
                        dt=dt_use,
                        record_vars=record_vars_run,
                        transient=0.0,
                        record_interval=record_stride,
                        max_steps=max_samples + 1,
                        ic=ic_arr[start:stop],
                        params=params_arr[start:stop],
                        parallel_mode=effective_parallel_mode,
                        max_workers=max_workers,
                        observers=analysis_mod,
                    )
                    fastpath_used = True
                else:
                    views = []
                    for i in range(start, stop):
                        seed = sim._select_seed(
                            resume=False,
                            t0=sim.model.spec.sim.t0,
                            dt=dt_use,
                            ic=ic_arr[i],
                            params=params_arr[i],
                        )
                        result = sim._execute_run(
                            seed=seed,
                            t_end=float(T if T is not None else seed.t + float(max_samples) * dt_use),
                            target_steps=int(N) if N is not None else None,
                            max_steps=int(max_samples + 1),
                            record=True,
                            record_interval=record_stride,
                            cap_rec=2,
                            cap_evt=1,
                            stepper_config=stepper_config,
                            adaptive=adaptive,
                            wrms_cfg=None,
                            state_rec_indices=state_rec_indices,
                            aux_rec_indices=aux_rec_indices,
                            state_names=state_rec_names,
                            aux_names=aux_names,
                            observers=analysis_mod,
                        )
                        views.append(ResultsView(result, sim.model.spec))

                outcomes = [_pcr_outcome_from_view(view, analysis_name) for view in views]

            for j, outcome in enumerate(outcomes):
                idx = start + j
                if _apply_pcr_outcome(
                    idx=idx,
                    outcome=outcome,
                    labels=labels,
                    registry=registry,
                    grid_arr=grid_arr,
                    down_arr=down_arr,
                    fp_cap=fp_cap,
                    fp_min_count=fp_min_count,
                    fp_min_fp=fp_min_fp,
                    s_merge=float(s_merge),
                ):
                    registry_dirty = True

        if refine_unresolved and registry and p_in > 0:
            unresolved_idx = np.nonzero(labels == UNRESOLVED)[0]
            if unresolved_idx.size > 0:
                persistence_truncated = _fill_cell_set_arrays(
                    cell_sets=cell_sets_arr,
                    cell_counts=cell_counts,
                    registry=registry,
                    max_cells_per_attr=max_cells,
                ) or persistence_truncated
                analysis_refine = _make_pcr_online_analysis(
                    name=f"{analysis_name}_refine",
                    obs_idx=obs_state_idx,
                    blowup_idx=blowup_state_idx,
                    z_min=obs_min_arr,
                    z_max=obs_max_arr,
                    grid_res=grid_arr,
                    strides=strides,
                    window=window,
                    u_th=u_th,
                    recur_need=recur_windows,
                    transient_samples=transient_samples,
                    outside_limit=outside_limit,
                    p_in=p_in,
                    b_max=b_max,
                    post_detect_samples=post_detect_samples,
                    cell_sets=cell_sets_arr,
                    cell_counts=cell_counts,
                    capture_evidence=False,
                )
                if use_fastpath:
                    support = assess_capability(
                        sim,
                        plan=plan,
                        record_vars=record_vars_run,
                        dt=dt_use,
                        transient=0.0,
                        adaptive=adaptive,
                        observers=analysis_refine,
                    )
                    use_fastpath = support.ok

                for start in range(0, unresolved_idx.size, max(1, batch_size_use)):
                    stop = min(unresolved_idx.size, start + max(1, batch_size_use))
                    idx_chunk = unresolved_idx[start:stop]
                    if use_process_parallel and process_executor is not None:
                        batch_ic = np.ascontiguousarray(ic_arr[idx_chunk])
                        batch_params = np.ascontiguousarray(params_arr[idx_chunk])
                        run_config = dict(
                            analysis_name=analysis_refine.name,
                            capture_evidence=False,
                            cell_sets=cell_sets_arr,
                            cell_counts=cell_counts,
                            use_fastpath=use_fastpath,
                        )
                        chunk_ranges = _chunk_ranges(batch_ic.shape[0], n_workers)
                        chunk_args = [
                            (batch_ic[c_start:c_stop], batch_params[c_start:c_stop], run_config)
                            for c_start, c_stop in chunk_ranges
                        ]
                        outcomes = []
                        for chunk_outcomes in process_executor.map(_basin_auto_chunk_worker, chunk_args):
                            outcomes.extend(chunk_outcomes)
                        if use_fastpath:
                            fastpath_used = True
                    else:
                        if use_fastpath:
                            from dynlib.runtime.fastpath import fastpath_batch_for_sim

                            views = fastpath_batch_for_sim(
                                sim,
                                plan=plan,
                                t0=sim.model.spec.sim.t0,
                                T=T,
                                N=N,
                                dt=dt_use,
                                record_vars=record_vars_run,
                                transient=0.0,
                                record_interval=record_stride,
                                max_steps=max_samples + 1,
                                ic=ic_arr[idx_chunk],
                                params=params_arr[idx_chunk],
                                parallel_mode=effective_parallel_mode,
                                max_workers=max_workers,
                                observers=analysis_refine,
                            )
                            fastpath_used = True
                        else:
                            views = []
                            for i in idx_chunk:
                                seed = sim._select_seed(
                                    resume=False,
                                    t0=sim.model.spec.sim.t0,
                                    dt=dt_use,
                                    ic=ic_arr[i],
                                    params=params_arr[i],
                                )
                                result = sim._execute_run(
                                    seed=seed,
                                    t_end=float(T if T is not None else seed.t + float(max_samples) * dt_use),
                                    target_steps=int(N) if N is not None else None,
                                    max_steps=int(max_samples + 1),
                                    record=True,
                                    record_interval=record_stride,
                                    cap_rec=2,
                                    cap_evt=1,
                                    stepper_config=stepper_config,
                                    adaptive=adaptive,
                                    wrms_cfg=None,
                                    state_rec_indices=state_rec_indices,
                                    aux_rec_indices=aux_rec_indices,
                                    state_names=state_rec_names,
                                    aux_names=aux_names,
                                    observers=analysis_refine,
                                )
                                views.append(ResultsView(result, sim.model.spec))

                        outcomes = [_pcr_outcome_from_view(view, analysis_refine.name) for view in views]

                    for j, outcome in enumerate(outcomes):
                        idx = int(idx_chunk[j])
                        _apply_pcr_outcome(
                            idx=idx,
                            outcome=outcome,
                            labels=labels,
                            registry=registry,
                            grid_arr=grid_arr,
                            down_arr=down_arr,
                            fp_cap=fp_cap,
                            fp_min_count=fp_min_count,
                            fp_min_fp=fp_min_fp,
                            s_merge=float(s_merge),
                            assign_only=True,
                        )

        if process_executor is not None:
            process_executor.shutdown()

        meta = {
            "mode": mode_use,
            "observe_vars": tuple(record_vars[:d]),
            "grid_res": tuple(int(x) for x in grid_arr),
            "merge_downsample": tuple(int(x) for x in down_arr),
            "window": int(window),
            "u_th": float(u_th),
            "recur_windows": int(recur_windows),
            "s_merge": float(s_merge),
            "p_in": int(p_in),
            "outside_limit": int(outside_limit),
            "transient_samples": int(transient_samples),
            "max_samples": int(max_samples),
            "dt_obs": float(dt_use),
            "fastpath": bool(fastpath_used),
            "fingerprint_cap": int(fp_cap),
            "post_detect_samples": int(post_detect_samples),
            "online": True,
            "batch_size": int(batch_size_use),
            "refine_unresolved": bool(refine_unresolved),
            "online_max_attr": int(max_attr),
            "online_max_cells": int(max_cells),
            "persistence_truncated": bool(persistence_truncated),
        }
        if ic_grid_meta is not None:
            meta["ic_grid"] = ic_grid_meta
        if ic_bounds_meta is not None:
            meta["ic_bounds"] = ic_bounds_meta

        return BasinResult(labels=labels, registry=registry, meta=meta)

    plan = FixedStridePlan(stride=1)
    support = _assess_fastpath(sim, plan=plan, record_vars=record_vars, dt=dt_use, transient=0.0)

    views = None
    fastpath_used = False
    if support.ok:
        try:
            from dynlib.runtime.fastpath import fastpath_batch_for_sim

            views = fastpath_batch_for_sim(
                sim,
                plan=plan,
                t0=sim.model.spec.sim.t0,
                T=T,
                N=N,
                dt=dt_use,
                record_vars=record_vars,
                transient=0.0,
                record_interval=1,
                max_steps=max_samples + 1,
                ic=ic_arr,
                params=params_arr,
                parallel_mode=parallel_mode,
                max_workers=max_workers,
            )
            if views is not None:
                fastpath_used = True
        except RuntimeError as exc:
            if "Fastpath runner exited with status" not in str(exc):
                raise
            views = None

    if views is None:
        state_rec_indices, aux_rec_indices, state_names, aux_names = sim._resolve_recording_selection(record_vars)
        stepper_config = sim.stepper_config()
        adaptive = getattr(sim._stepper_spec.meta, "time_control", "fixed") == "adaptive"
        views = []
        for i in range(ic_arr.shape[0]):
            seed = sim._select_seed(
                resume=False,
                t0=sim.model.spec.sim.t0,
                dt=dt_use,
                ic=ic_arr[i],
                params=params_arr[i],
            )
            result = sim._execute_run(
                seed=seed,
                t_end=float(T if T is not None else seed.t + float(max_samples) * dt_use),
                target_steps=int(N) if N is not None else None,
                max_steps=int(max_samples + 1),
                record=True,
                record_interval=1,
                cap_rec=int(max_samples + 2),
                cap_evt=1,
                stepper_config=stepper_config,
                adaptive=adaptive,
                wrms_cfg=None,
                state_rec_indices=state_rec_indices,
                aux_rec_indices=aux_rec_indices,
                state_names=state_names,
                aux_names=aux_names,
                observers=None,
            )
            views.append(ResultsView(result, sim.model.spec))

    obs, blowup_vals, lengths, run_status = _extract_observations(
        views=views,
        max_samples=max_samples,
        obs_dim=d,
        blowup_idx=blowup_idx,
    )

    nonfinite_mask = np.any(~np.isfinite(obs), axis=2)
    if blowup_vals.size > 0:
        nonfinite_mask |= np.any(~np.isfinite(blowup_vals), axis=2)

    blowup_mask = nonfinite_mask.copy()
    if b_max is not None and blowup_vals.size > 0:
        blowup_mask |= np.any(np.abs(blowup_vals) > float(b_max), axis=2)

    cell_ids = np.full((obs.shape[0], max_samples), -1, dtype=np.int64)
    strides = np.ones((d,), dtype=np.int64)
    for k in range(1, d):
        strides[k] = strides[k - 1] * grid_arr[k - 1]
    _quantize_cells(obs, lengths, obs_min_arr, obs_max_arr, grid_arr, strides, cell_ids)

    labels = np.full((obs.shape[0],), UNRESOLVED, dtype=np.int64)
    registry: list[Attractor] = []
    cell_sets_nb: object | None = None
    empty_cell_sets_nb = _build_cell_sets_numba([])

    hash_cap = int(_hash_capacity(window))
    fp_cap = max(32, int(window))

    # fingerprint extraction knobs (stable defaults)
    fp_min_count = 2
    fp_min_fp = max(8, window // 16)

    for i in range(obs.shape[0]):
        if run_status[i] == NAN_DETECTED:
            labels[i] = BLOWUP
            continue

        n = int(lengths[i])
        if n <= 0:
            labels[i] = UNRESOLVED
            continue

        # Phase B: early assignment via persistence on accumulated cell sets
        if registry and p_in > 0:
            if cell_sets_nb is None:
                cell_sets_nb = _build_cell_sets_numba(registry)
            res = _scan_persistence_outside_blowup(
                cell_ids[i], blowup_mask[i], n, cell_sets_nb, int(p_in), int(outside_limit)
            )
        else:
            res = _scan_persistence_outside_blowup(
                cell_ids[i], blowup_mask[i], n, empty_cell_sets_nb, 0, int(outside_limit)
            )

        if res == _SCAN_BLOWUP:
            labels[i] = BLOWUP
            continue
        if res == _SCAN_OUTSIDE:
            labels[i] = OUTSIDE
            continue
        if res != _SCAN_NONE:
            labels[i] = int(res)
            continue

        # Phase A: recurrence detection
        status, detect_pos = _find_candidate_single(
            cell_ids[i],
            blowup_mask[i],
            n,
            int(window),
            float(u_th),
            int(recur_windows),
            int(outside_limit),
            int(transient_samples),
            int(hash_cap),
        )
        if status == 2:
            labels[i] = BLOWUP
            continue
        if status == 3:
            labels[i] = OUTSIDE
            continue
        if status != 1 or detect_pos < 0:
            labels[i] = UNRESOLVED
            continue

        # Evidence segment around detection:
        # - include the last window up to detect_pos
        # - plus post_detect_samples after detect_pos
        seg_start = max(0, int(detect_pos) - int(window) + 1)
        seg_end = min(n, int(detect_pos) + 1 + int(post_detect_samples))
        seg = cell_ids[i, seg_start:seg_end]
        seg = seg[seg >= 0]
        if seg.size == 0:
            labels[i] = UNRESOLVED
            continue

        # Accumulate evidence cells on detection grid
        evid_unique = np.unique(seg)
        evidence_cells = set(int(x) for x in evid_unique.tolist())

        # Merge fingerprint built on COARSENED merge grid
        merge_ids = _coarsen_cells_numpy(evid_unique, grid_arr, down_arr)
        candidate_fp = _fingerprint_from_merge_ids(
            merge_ids,
            fingerprint_cap=fp_cap,
            min_count=fp_min_count,
            min_fp=fp_min_fp,
        )
        if not candidate_fp:
            # fallback: cap unique ids
            vals = np.unique(merge_ids)
            if vals.size > fp_cap:
                vals = vals[:fp_cap]
            candidate_fp = set(int(x) for x in vals.tolist())

        attr_id = _registry_match_or_create(registry, candidate_fp, float(s_merge))
        _registry_update_cells(registry, int(attr_id), evidence_cells, candidate_fp, fingerprint_cap=fp_cap)
        labels[i] = int(attr_id)

        # registry changed => membership cache stale
        cell_sets_nb = None

    # Refinement pass: assign remaining UNRESOLVED via persistence (using grown cell sets)
    if registry and p_in > 0:
        cell_sets_nb = _build_cell_sets_numba(registry)
        _assign_persistence_batch(
            cell_ids,
            lengths,
            labels,
            cell_sets_nb,
            int(p_in),
            int(UNRESOLVED),
            int(outside_limit),
        )

    meta = {
        "mode": mode_use,
        "observe_vars": tuple(record_vars[:d]),
        "grid_res": tuple(int(x) for x in grid_arr),
        "merge_downsample": tuple(int(x) for x in down_arr),
        "window": int(window),
        "u_th": float(u_th),
        "recur_windows": int(recur_windows),
        "s_merge": float(s_merge),
        "p_in": int(p_in),
        "outside_limit": int(outside_limit),
        "transient_samples": int(transient_samples),
        "max_samples": int(max_samples),
        "dt_obs": float(dt_use),
        "fastpath": bool(fastpath_used),
        "fingerprint_cap": int(fp_cap),
        "post_detect_samples": int(post_detect_samples),
    }
    if ic_grid_meta is not None:
        meta["ic_grid"] = ic_grid_meta
    if ic_bounds_meta is not None:
        meta["ic_bounds"] = ic_bounds_meta

    return BasinResult(labels=labels, registry=registry, meta=meta)
