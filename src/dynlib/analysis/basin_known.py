# src/dynlib/analysis/basin_known.py
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, Sequence

import numpy as np

from dynlib.analysis.basin import (
    Attractor,
    BasinResult,
    BLOWUP,
    OUTSIDE,
    UNRESOLVED,
    KnownAttractorLibrary,
    FixedPoint,
    ReferenceRun,
    build_known_attractors_psc,
    _coerce_batch,
    _prepare_record_vars,
    _require_numba,
    _resolve_mode,
    njit,
)
from dynlib.runtime.observers import ObserverHooks, ObserverModule, ObserverRequirements
from dynlib.runtime.fastpath.plans import FixedStridePlan
from dynlib.runtime.fastpath.capability import assess_capability
from dynlib.runtime.results_api import ResultsView
from dynlib.runtime.runner_api import NAN_DETECTED
from dynlib.runtime.sim import Sim

__all__ = ["basin_known"]


def _choose_psc_grid_res(obs_dim: int, target_cells: int = 4096) -> np.ndarray:
    """Internal default grid_res with ~target_cells total cells (no user tuning)."""
    d = int(obs_dim)
    if d <= 0:
        return np.zeros((0,), dtype=np.int64)
    if d == 1:
        n = int(target_cells)
    elif d == 2:
        n = 64
    elif d == 3:
        n = 16
    else:
        n = int(round(float(target_cells) ** (1.0 / float(d))))
        n = max(4, n)
    return np.full((d,), n, dtype=np.int64)


def _grid_strides(grid_res: np.ndarray) -> np.ndarray:
    grid_res = np.asarray(grid_res, dtype=np.int64)
    d = int(grid_res.size)
    strides = np.ones((d,), dtype=np.int64)
    if d <= 1:
        return strides
    s = 1
    for k in range(d - 1, -1, -1):
        strides[k] = s
        s *= int(grid_res[k])
    return strides


def _hash_capacity_from_count(count: int) -> int:
    target = max(8, int(count) * 2)
    cap = 1
    while cap < target:
        cap <<= 1
    return cap


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

else:  # pragma: no cover - numba missing

    def _hash_slot(keys: np.ndarray, key: int, mask: int) -> tuple[int, bool]:
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


def _cell_id_to_indices(cell_id: int, grid_res: np.ndarray, strides: np.ndarray) -> np.ndarray:
    d = int(grid_res.size)
    out = np.zeros((d,), dtype=np.int64)
    for k in range(d - 1, -1, -1):
        stride = int(strides[k])
        if stride <= 0:
            out[k] = 0
        else:
            out[k] = int((cell_id // stride) % int(grid_res[k]))
    return out


def _dilate_cell_counts(
    counts: dict[int, int],
    grid_res: np.ndarray,
    strides: np.ndarray,
    steps: int,
) -> dict[int, int]:
    if steps <= 0 or not counts:
        return counts
    current = dict(counts)
    d = int(grid_res.size)
    for _ in range(int(steps)):
        additions: dict[int, int] = {}
        for cell_id in list(current.keys()):
            idxs = _cell_id_to_indices(int(cell_id), grid_res, strides)
            for k in range(d):
                stride = int(strides[k])
                if idxs[k] > 0:
                    neighbor = int(cell_id - stride)
                    if neighbor not in current:
                        additions.setdefault(neighbor, 1)
                if idxs[k] + 1 < int(grid_res[k]):
                    neighbor = int(cell_id + stride)
                    if neighbor not in current:
                        additions.setdefault(neighbor, 1)
        if not additions:
            break
        current.update(additions)
    return current


def _build_psc_pairs_index(
    *,
    n_attr: int,
    ptr: np.ndarray,
    cell_ids: np.ndarray,
    cell_logp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(cell_ids.size)
    if total == 0 or n_attr <= 0:
        cap = _hash_capacity_from_count(0)
        ht_keys = np.full((cap,), -1, dtype=np.int64)
        ht_start = np.full((cap,), -1, dtype=np.int64)
        ht_len = np.zeros((cap,), dtype=np.int32)
        return (
            ht_keys,
            ht_start,
            ht_len,
            np.zeros((0,), dtype=np.int16),
            np.zeros((0,), dtype=np.float32),
        )

    k_dtype = np.int16 if n_attr <= np.iinfo(np.int16).max else np.int32
    pairs_cell = np.empty((total,), dtype=np.int64)
    pairs_k = np.empty((total,), dtype=k_dtype)
    pairs_logp = np.empty((total,), dtype=np.float32)
    pos = 0
    for k in range(n_attr):
        start = int(ptr[k])
        stop = int(ptr[k + 1])
        count = stop - start
        if count <= 0:
            continue
        pairs_cell[pos : pos + count] = cell_ids[start:stop]
        pairs_k[pos : pos + count] = k
        pairs_logp[pos : pos + count] = cell_logp[start:stop]
        pos += count

    if pos != total:
        pairs_cell = pairs_cell[:pos]
        pairs_k = pairs_k[:pos]
        pairs_logp = pairs_logp[:pos]

    order = np.argsort(pairs_cell, kind="mergesort")
    pairs_cell = pairs_cell[order]
    pairs_k = pairs_k[order]
    pairs_logp = pairs_logp[order]

    uniq_cells, start_idx, counts = np.unique(pairs_cell, return_index=True, return_counts=True)
    cap = _hash_capacity_from_count(int(uniq_cells.size))
    ht_keys = np.full((cap,), -1, dtype=np.int64)
    ht_start = np.full((cap,), -1, dtype=np.int64)
    ht_len = np.zeros((cap,), dtype=np.int32)

    mask = cap - 1
    for i in range(uniq_cells.size):
        cell = int(uniq_cells[i])
        idx = (cell * 2654435761) & mask
        while True:
            if ht_keys[idx] == -1:
                ht_keys[idx] = cell
                ht_start[idx] = int(start_idx[i])
                ht_len[idx] = int(counts[i])
                break
            idx = (idx + 1) & mask

    return ht_keys, ht_start, ht_len, pairs_k, pairs_logp


def _build_psc_signature_from_known(
    *,
    known: KnownAttractorLibrary,
    is_ref: np.ndarray,
    grid_res: np.ndarray,
    strides: np.ndarray,
    dilate_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build PSC signature arrays (ptr, cell_ids, cell_logp) from KnownAttractorLibrary trajectories.

    - Only ReferenceRun attractors (is_ref[k]==1) contribute cells.
    - FixedPoints get empty spans (ptr[k+1]==ptr[k]).
    """
    is_ref = np.asarray(is_ref, dtype=np.int64)
    grid_res = np.asarray(grid_res, dtype=np.int64)
    strides = np.asarray(strides, dtype=np.int64)

    n_attr = int(known.n_attr)
    obs_min = np.asarray(known.obs_min, dtype=np.float64)
    obs_max = np.asarray(known.obs_max, dtype=np.float64)
    d = int(obs_min.size)

    ptr = np.zeros((n_attr + 1,), dtype=np.int64)
    cell_ids_parts: list[np.ndarray] = []
    cell_logp_parts: list[np.ndarray] = []
    pos = 0

    for k in range(n_attr):
        if int(is_ref[k]) == 0:
            ptr[k + 1] = pos
            continue
        traj = known.trajectories[k]
        if traj is None or traj.size == 0:
            ptr[k + 1] = pos
            continue

        counts: dict[int, int] = {}
        # traj is (n_samples, d) in observation space
        for i in range(int(traj.shape[0])):
            cell = 0
            outside = False
            for j in range(d):
                val = float(traj[i, j])
                if not (val >= obs_min[j] and val <= obs_max[j]):
                    outside = True
                    break
                denom = float(obs_max[j] - obs_min[j])
                if denom <= 0.0:
                    outside = True
                    break
                frac = (val - obs_min[j]) / denom
                idx = int(frac * float(grid_res[j]))
                if idx < 0:
                    idx = 0
                elif idx >= int(grid_res[j]):
                    idx = int(grid_res[j] - 1)
                cell += idx * int(strides[j])
            if outside:
                continue
            counts[cell] = counts.get(cell, 0) + 1

        if dilate_steps > 0 and counts:
            counts = _dilate_cell_counts(counts, grid_res, strides, steps=int(dilate_steps))

        if not counts:
            ptr[k + 1] = pos
            continue

        keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        vals = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
        total = float(np.sum(vals))
        if total <= 0.0:
            ptr[k + 1] = pos
            continue
        logp = np.log(vals.astype(np.float64) / total).astype(np.float32)

        cell_ids_parts.append(keys)
        cell_logp_parts.append(logp)
        pos += int(keys.size)
        ptr[k + 1] = pos

    if pos > 0:
        cell_ids = np.concatenate(cell_ids_parts).astype(np.int64, copy=False)
        cell_logp = np.concatenate(cell_logp_parts).astype(np.float32, copy=False)
    else:
        cell_ids = np.zeros((0,), dtype=np.int64)
        cell_logp = np.zeros((0,), dtype=np.float32)
    return ptr, cell_ids, cell_logp


def _make_known_hybrid_psc_analysis(
    *,
    name: str,
    n_attr: int,
    obs_idx: np.ndarray,
    blowup_idx: np.ndarray,
    # PSC quantization bounds
    obs_min: np.ndarray,
    obs_max: np.ndarray,
    # Escape / outside bounds
    escape_min: np.ndarray,
    escape_max: np.ndarray,
    grid_res: np.ndarray,
    strides: np.ndarray,
    # PSC index
    ht_keys: np.ndarray,
    ht_start: np.ndarray,
    ht_len: np.ndarray,
    pairs_k: np.ndarray,
    pairs_logp: np.ndarray,
    logp_null: float,
    # Runtime policy
    transient_samples: int,
    min_evidence: int,
    confidence: float,
    check_stride: int,
    outside_limit: int,
    b_max: float | None,
    # Fixed-point settle fast-path
    fp_ids: np.ndarray,            # global attractor ids for fixed points
    fp_locs: np.ndarray,           # (n_fp, obs_dim)
    fp_radii_sq: np.ndarray,       # (n_fp,)
    fixed_point_settle_steps: int,
) -> ObserverModule:
    """
    Unified Numba-compatible online classifier:
      - FixedPoint settle-in-radius (fast-path, early exit)
      - PSC scoring for ReferenceRuns (distributional; robust for chaos & cycles)

    Automatically degenerates to:
      - FP-only when pairs are empty
      - PSC-only when fp_ids is empty
    """
    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    blowup_idx = np.asarray(blowup_idx, dtype=np.int64)
    obs_min = np.asarray(obs_min, dtype=np.float64)
    obs_max = np.asarray(obs_max, dtype=np.float64)
    escape_min = np.asarray(escape_min, dtype=np.float64)
    escape_max = np.asarray(escape_max, dtype=np.float64)
    grid_res = np.asarray(grid_res, dtype=np.int64)
    strides = np.asarray(strides, dtype=np.int64)

    ht_keys = np.asarray(ht_keys, dtype=np.int64)
    ht_start = np.asarray(ht_start, dtype=np.int64)
    ht_len = np.asarray(ht_len, dtype=np.int32)
    pairs_k = np.asarray(pairs_k)
    pairs_logp = np.asarray(pairs_logp, dtype=np.float32)

    fp_ids = np.asarray(fp_ids, dtype=np.int64)
    fp_locs = np.asarray(fp_locs, dtype=np.float64)
    fp_radii_sq = np.asarray(fp_radii_sq, dtype=np.float64)

    obs_dim = int(obs_idx.size)
    n_attr = int(n_attr)
    n_fp = int(fp_ids.size)
    ht_mask = int(ht_keys.shape[0] - 1)

    check_blowup = b_max is not None and blowup_idx.size > 0
    b_max_val = float(b_max) if b_max is not None else 0.0
    outside_limit_val = int(outside_limit)
    transient_val = int(transient_samples)
    min_evidence_val = int(min_evidence)
    check_stride_val = int(check_stride) if int(check_stride) > 0 else 1
    logp_null_val = float(logp_null)

    conf = float(confidence)
    if conf <= 1.0e-12:
        conf = 1.0e-12
    elif conf >= 1.0 - 1.0e-12:
        conf = 1.0 - 1.0e-12
    thr = float(np.log(conf) - np.log(1.0 - conf))

    # Workspace:
    # 0 obs_count
    # 1 outside_run (escape bounds)
    # 2 scored_count
    # 3 null_score
    # 4 fp_candidate_global_id
    # 5 fp_settle_count
    # 6.. scores[n_attr] (PSC deltas)
    ws_obs = 0
    ws_outside = 1
    ws_scored = 2
    ws_null = 3
    ws_fp_cand = 4
    ws_fp_run = 5
    ws_scores = 6
    ws_size = ws_scores + max(n_attr, 0)

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
        if analysis_ws.shape[0] < ws_size or analysis_out.shape[0] < 2:
            return
        analysis_out[0] = 0.0
        analysis_out[1] = -1.0
        analysis_ws[ws_obs] = 0.0
        analysis_ws[ws_outside] = 0.0
        analysis_ws[ws_scored] = 0.0
        analysis_ws[ws_null] = 0.0
        analysis_ws[ws_fp_cand] = -1.0
        analysis_ws[ws_fp_run] = 0.0
        scores = analysis_ws[ws_scores:ws_size]
        for k in range(n_attr):
            scores[k] = 0.0

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
        if analysis_ws.shape[0] < ws_size or analysis_out.shape[0] < 2:
            return
        if analysis_out[0] != 0.0:
            return

        if check_blowup:
            for i in range(blowup_idx.size):
                val = y_curr[int(blowup_idx[i])]
                if abs(val) > b_max_val:
                    analysis_out[0] = 2.0
                    if runtime_ws.stop_flag.shape[0] > 0:
                        runtime_ws.stop_flag[0] = 1
                    return

        # Escape/outside check uses escape_min/max
        escaped = False
        for k in range(obs_dim):
            v = y_curr[int(obs_idx[k])]
            if not (v >= escape_min[k] and v <= escape_max[k]):
                escaped = True
                break
        if escaped:
            outside_run = int(analysis_ws[ws_outside]) + 1
            analysis_ws[ws_outside] = float(outside_run)
            analysis_ws[ws_fp_cand] = -1.0
            analysis_ws[ws_fp_run] = 0.0
            if outside_run >= outside_limit_val:
                analysis_out[0] = 3.0
                if runtime_ws.stop_flag.shape[0] > 0:
                    runtime_ws.stop_flag[0] = 1
            analysis_ws[ws_obs] = float(int(analysis_ws[ws_obs]) + 1)
            return

        analysis_ws[ws_outside] = 0.0
        obs_count = int(analysis_ws[ws_obs]) + 1
        analysis_ws[ws_obs] = float(obs_count)
        if obs_count <= transient_val:
            return

        # ------------------------------------------------------------
        # FixedPoint settle fast-path (always first)
        # ------------------------------------------------------------
        if n_fp > 0:
            inside = -1
            for j in range(n_fp):
                dist_sq = 0.0
                for k in range(obs_dim):
                    diff = y_curr[int(obs_idx[k])] - fp_locs[j, k]
                    dist_sq += diff * diff
                if dist_sq <= fp_radii_sq[j]:
                    inside = int(fp_ids[j])
                    break

            cand = int(analysis_ws[ws_fp_cand])
            if inside >= 0:
                if inside == cand:
                    run = int(analysis_ws[ws_fp_run]) + 1
                    analysis_ws[ws_fp_run] = float(run)
                    if run >= int(fixed_point_settle_steps):
                        analysis_out[0] = 1.0
                        analysis_out[1] = float(inside)
                        if runtime_ws.stop_flag.shape[0] > 0:
                            runtime_ws.stop_flag[0] = 1
                        return
                else:
                    analysis_ws[ws_fp_cand] = float(inside)
                    analysis_ws[ws_fp_run] = 1.0
            else:
                analysis_ws[ws_fp_cand] = -1.0
                analysis_ws[ws_fp_run] = 0.0

        # ------------------------------------------------------------
        # PSC scoring fallback (only if we actually have PSC pairs)
        # ------------------------------------------------------------
        if n_attr <= 0 or pairs_logp.size == 0 or ht_keys.size == 0:
            return

        # Quantization uses obs_min/max; if outside, skip scoring for this step.
        cell = 0
        for k in range(obs_dim):
            v = y_curr[int(obs_idx[k])]
            if not (v >= obs_min[k] and v <= obs_max[k]):
                return
            denom = obs_max[k] - obs_min[k]
            if denom <= 0.0:
                return
            frac = (v - obs_min[k]) / denom
            idx = int(frac * grid_res[k])
            if idx < 0:
                idx = 0
            elif idx >= int(grid_res[k]):
                idx = int(grid_res[k] - 1)
            cell += idx * int(strides[k])

        scored = int(analysis_ws[ws_scored]) + 1
        analysis_ws[ws_scored] = float(scored)
        analysis_ws[ws_null] = analysis_ws[ws_null] + logp_null_val

        slot, found = _hash_slot(ht_keys, int(cell), ht_mask)
        if found:
            start = int(ht_start[slot])
            length = int(ht_len[slot])
            scores = analysis_ws[ws_scores:ws_size]
            for j in range(length):
                k = int(pairs_k[start + j])
                lp = pairs_logp[start + j]
                scores[k] += float(lp) - logp_null_val  # floor==null by design

        if scored < min_evidence_val:
            return
        if check_stride_val > 1 and (scored % check_stride_val) != 0:
            return

        best_id = -1
        best = -1.0e300
        second = -1.0e300
        scores = analysis_ws[ws_scores:ws_size]
        for k in range(n_attr):
            s = scores[k]
            if s > best:
                second = best
                best = s
                best_id = k
            elif s > second:
                second = s

        if best_id < 0:
            return

        # Compare against null and second-best with the same logit threshold
        if best < thr:
            return
        if (best - second) >= thr:
            analysis_out[0] = 1.0
            analysis_out[1] = float(best_id)
            if runtime_ws.stop_flag.shape[0] > 0:
                runtime_ws.stop_flag[0] = 1

    class _KnownHybridPSCModule(ObserverModule):
        def _compile_hooks(self, hooks: ObserverHooks, dtype: np.dtype) -> ObserverHooks:
            if njit is None:  # pragma: no cover
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

    return _KnownHybridPSCModule(
        key=name,
        name=name,
        requirements=ObserverRequirements(fixed_step=True),
        workspace_size=ws_size,
        output_size=2,
        output_names=("status", "matched_id"),
        hooks=ObserverHooks(pre_step=_pre_step, post_step=_post_step),
        stop_phase_mask=2,
    )


def _make_fixed_point_only_analysis(
    *,
    name: str,
    obs_idx: np.ndarray,
    blowup_idx: np.ndarray,
    escape_min: np.ndarray,
    escape_max: np.ndarray,
    transient_samples: int,
    outside_limit: int,
    b_max: float | None,
    fp_ids: np.ndarray,
    fp_locs: np.ndarray,
    fp_radii_sq: np.ndarray,
    fixed_point_settle_steps: int,
) -> ObserverModule:
    """Fixed-point-only fast path: settle-in-radius + escape/blowup, no PSC scoring."""
    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    blowup_idx = np.asarray(blowup_idx, dtype=np.int64)
    escape_min = np.asarray(escape_min, dtype=np.float64)
    escape_max = np.asarray(escape_max, dtype=np.float64)
    fp_ids = np.asarray(fp_ids, dtype=np.int64)
    fp_locs = np.asarray(fp_locs, dtype=np.float64)
    fp_radii_sq = np.asarray(fp_radii_sq, dtype=np.float64)

    obs_dim = int(obs_idx.size)
    n_fp = int(fp_ids.size)
    transient_val = int(transient_samples)
    outside_limit_val = int(outside_limit)
    check_blowup = b_max is not None and blowup_idx.size > 0
    b_max_val = float(b_max) if b_max is not None else 0.0
    settle_steps = int(fixed_point_settle_steps)

    ws_obs = 0
    ws_outside = 1
    ws_fp_cand = 2
    ws_fp_run = 3
    ws_size = 4

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
        if analysis_ws.shape[0] < ws_size or analysis_out.shape[0] < 2:
            return
        analysis_out[0] = 0.0
        analysis_out[1] = -1.0
        analysis_ws[ws_obs] = 0.0
        analysis_ws[ws_outside] = 0.0
        analysis_ws[ws_fp_cand] = -1.0
        analysis_ws[ws_fp_run] = 0.0

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
        if analysis_ws.shape[0] < ws_size or analysis_out.shape[0] < 2:
            return
        if analysis_out[0] != 0.0:
            return

        if check_blowup:
            for i in range(blowup_idx.size):
                val = y_curr[int(blowup_idx[i])]
                if abs(val) > b_max_val:
                    analysis_out[0] = 2.0
                    if runtime_ws.stop_flag.shape[0] > 0:
                        runtime_ws.stop_flag[0] = 1
                    return

        escaped = False
        for k in range(obs_dim):
            v = y_curr[int(obs_idx[k])]
            if not (v >= escape_min[k] and v <= escape_max[k]):
                escaped = True
                break
        if escaped:
            outside_run = int(analysis_ws[ws_outside]) + 1
            analysis_ws[ws_outside] = float(outside_run)
            analysis_ws[ws_fp_cand] = -1.0
            analysis_ws[ws_fp_run] = 0.0
            if outside_run >= outside_limit_val:
                analysis_out[0] = 3.0
                if runtime_ws.stop_flag.shape[0] > 0:
                    runtime_ws.stop_flag[0] = 1
            analysis_ws[ws_obs] = float(int(analysis_ws[ws_obs]) + 1)
            return

        analysis_ws[ws_outside] = 0.0
        obs_count = int(analysis_ws[ws_obs]) + 1
        analysis_ws[ws_obs] = float(obs_count)
        if obs_count <= transient_val:
            return

        if n_fp <= 0:
            return

        inside = -1
        for j in range(n_fp):
            dist_sq = 0.0
            for k in range(obs_dim):
                diff = y_curr[int(obs_idx[k])] - fp_locs[j, k]
                dist_sq += diff * diff
            if dist_sq <= fp_radii_sq[j]:
                inside = int(fp_ids[j])
                break

        cand = int(analysis_ws[ws_fp_cand])
        if inside >= 0:
            if inside == cand:
                run = int(analysis_ws[ws_fp_run]) + 1
                analysis_ws[ws_fp_run] = float(run)
                if run >= settle_steps:
                    analysis_out[0] = 1.0
                    analysis_out[1] = float(inside)
                    if runtime_ws.stop_flag.shape[0] > 0:
                        runtime_ws.stop_flag[0] = 1
            else:
                analysis_ws[ws_fp_cand] = float(inside)
                analysis_ws[ws_fp_run] = 1.0
        else:
            analysis_ws[ws_fp_cand] = -1.0
            analysis_ws[ws_fp_run] = 0.0

    class _FPOnlyModule(ObserverModule):
        def _compile_hooks(self, hooks: ObserverHooks, dtype: np.dtype) -> ObserverHooks:
            if njit is None:  # pragma: no cover
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

    return _FPOnlyModule(
        key=name,
        name=name,
        requirements=ObserverRequirements(fixed_step=True),
        workspace_size=ws_size,
        output_size=2,
        output_names=("status", "matched_id"),
        hooks=ObserverHooks(pre_step=_pre_step, post_step=_post_step),
        stop_phase_mask=2,
    )

# ---------------------------------------------------------------------------
# Coarse-to-Fine Refinement Helpers
# ---------------------------------------------------------------------------

def _dilate_mask_1d(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.bool_)
    p = np.pad(m, (1, 1), mode="constant", constant_values=False)
    out = p[0:-2] | p[1:-1] | p[2:]
    return out


def _dilate_mask_2d(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.bool_)
    ny, nx = m.shape
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)

    out = np.zeros((ny, nx), dtype=np.bool_)
    # 3x3 neighborhood OR
    for dy in (-1, 0, 1):
        ys = 1 + dy
        for dx in (-1, 0, 1):
            xs = 1 + dx
            out |= p[ys : ys + ny, xs : xs + nx]
    return out


def _dilate_mask_3d(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.bool_)
    nz, ny, nx = m.shape
    p = np.pad(m, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False)

    out = np.zeros((nz, ny, nx), dtype=np.bool_)
    # 3x3x3 neighborhood OR
    for dz in (-1, 0, 1):
        zs = 1 + dz
        for dy in (-1, 0, 1):
            ys = 1 + dy
            for dx in (-1, 0, 1):
                xs = 1 + dx
                out |= p[zs : zs + nz, ys : ys + ny, xs : xs + nx]
    return out


def _dilate_mask_nd(mask: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return np.asarray(mask, dtype=np.bool_)

    m = np.asarray(mask, dtype=np.bool_)
    for _ in range(int(steps)):
        if m.ndim == 1:
            m = _dilate_mask_1d(m)
        elif m.ndim == 2:
            m = _dilate_mask_2d(m)
        elif m.ndim == 3:
            m = _dilate_mask_3d(m)
        else:
            raise ValueError("boundary dilation only supports ndim <= 3")
    return m


def _find_boundary_mask_2d(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Find boundary cells in a 2D label grid.
    
    A cell is a boundary cell if any of its neighbors has a different label.
    Returns a boolean mask of boundary cells.
    """
    ny, nx = labels.shape
    boundary = np.zeros((ny, nx), dtype=np.bool_)
    
    # Check horizontal neighbors
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    
    # Check vertical neighbors
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    
    # Check diagonal neighbors
    boundary[:-1, :-1] |= labels[:-1, :-1] != labels[1:, 1:]
    boundary[1:, 1:] |= labels[:-1, :-1] != labels[1:, 1:]
    boundary[:-1, 1:] |= labels[:-1, 1:] != labels[1:, :-1]
    boundary[1:, :-1] |= labels[:-1, 1:] != labels[1:, :-1]
    
    return boundary


def _find_boundary_mask_nd(
    labels: np.ndarray,
    grid_shape: tuple[int, ...],
    dilation: int = 1,
) -> np.ndarray:
    """
    Find boundary cells in an N-dimensional label grid.
    
    Returns a boolean mask (flattened) of boundary cells.
    """
    labels_grid = labels.reshape(grid_shape)
    ndim = len(grid_shape)
    
    if ndim == 2:
        boundary = _find_boundary_mask_2d(labels_grid)
    else:
        # Generic N-dimensional boundary detection
        boundary = np.zeros(grid_shape, dtype=np.bool_)
        
        # Check each axis
        for axis in range(ndim):
            # Shift forward
            slices_a = [slice(None)] * ndim
            slices_b = [slice(None)] * ndim
            slices_a[axis] = slice(None, -1)
            slices_b[axis] = slice(1, None)
            
            diff = labels_grid[tuple(slices_a)] != labels_grid[tuple(slices_b)]
            
            # Mark both sides of boundary
            boundary[tuple(slices_a)] |= diff
            boundary[tuple(slices_b)] |= diff
    
    # Dilate boundary region
    if dilation > 0:
        boundary = _dilate_mask_nd(boundary, steps=int(dilation))
    
    return boundary.ravel()


def _generate_grid_ics(
    grid_shape: Sequence[int],
    bounds: Sequence[tuple[float, float]],
    dtype: np.dtype,
) -> np.ndarray:
    axes = [
        np.linspace(float(bmin), float(bmax), int(n))
        for (bmin, bmax), n in zip(bounds, grid_shape)
    ]
    meshgrids = np.meshgrid(*axes, indexing="ij")  # <-- critical
    return np.column_stack([g.ravel(order="C") for g in meshgrids]).astype(dtype, copy=False)


def _grid_points_from_flat_indices(
    flat_idx: np.ndarray,
    grid_shape: tuple[int, ...],
    bounds: Sequence[tuple[float, float]],
    dtype: np.dtype,
) -> np.ndarray:
    flat_idx = np.asarray(flat_idx, dtype=np.int64)
    if flat_idx.size == 0:
        return np.zeros((0, len(grid_shape)), dtype=dtype)

    coords = np.unravel_index(flat_idx, grid_shape)
    ndim = len(grid_shape)
    out = np.empty((flat_idx.size, ndim), dtype=np.float64)

    for d in range(ndim):
        n = int(grid_shape[d])
        bmin, bmax = float(bounds[d][0]), float(bounds[d][1])
        if n <= 1:
            out[:, d] = bmin
        else:
            out[:, d] = bmin + (bmax - bmin) * (coords[d].astype(np.float64) / float(n - 1))

    return out.astype(dtype, copy=False)


def _upscale_labels(
    coarse_labels: np.ndarray,
    coarse_shape: tuple[int, ...],
    fine_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Upscale coarse labels to fine grid using nearest-neighbor interpolation.
    """
    coarse_grid = coarse_labels.reshape(coarse_shape)

    # Fast path: integer scale factors -> repeat along each axis.
    if all(int(f) % int(c) == 0 for f, c in zip(fine_shape, coarse_shape)):
        out = coarse_grid
        for axis, (f, c) in enumerate(zip(fine_shape, coarse_shape)):
            factor = int(f) // int(c)
            if factor > 1:
                out = np.repeat(out, factor, axis=axis)
        return out.ravel()

    # General path: build per-axis index maps and use broadcasted indexing.
    idx_axes: list[np.ndarray] = []
    for f, c in zip(fine_shape, coarse_shape):
        f_int = int(f)
        c_int = int(c)
        if f_int <= 1 or c_int <= 1:
            idx_axes.append(np.zeros((f_int,), dtype=np.intp))
            continue
        idx = (np.arange(f_int, dtype=np.intp) * c_int) // f_int
        idx_axes.append(idx)

    return coarse_grid[np.ix_(*idx_axes)].ravel()


def _upscale_mask(
    coarse_mask: np.ndarray,
    coarse_shape: tuple[int, ...],
    fine_shape: tuple[int, ...],
) -> np.ndarray:
    coarse_grid = coarse_mask.reshape(coarse_shape)

    # Fast path: integer scale factors -> repeat along each axis.
    if all(int(f) % int(c) == 0 for f, c in zip(fine_shape, coarse_shape)):
        out = coarse_grid
        for axis, (f, c) in enumerate(zip(fine_shape, coarse_shape)):
            factor = int(f) // int(c)
            if factor > 1:
                out = np.repeat(out, factor, axis=axis)
        return out

    # General path: build per-axis index maps and use broadcasted indexing.
    idx_axes: list[np.ndarray] = []
    for f, c in zip(fine_shape, coarse_shape):
        f_int = int(f)
        c_int = int(c)
        if f_int <= 1 or c_int <= 1:
            idx_axes.append(np.zeros((f_int,), dtype=np.intp))
            continue
        idx = (np.arange(f_int, dtype=np.intp) * c_int) // f_int
        idx_axes.append(idx)

    return coarse_grid[np.ix_(*idx_axes)]

# ---------------------------------------------------------------------------
# Chunk-based parallel processing for process-level parallelism
# ---------------------------------------------------------------------------

# Module-level worker state - initialized once per process, reused across chunks
_worker_sim: "Sim | None" = None
_worker_known: "KnownAttractorLibrary | None" = None
_worker_prepared: dict[str, object] | None = None
_worker_prepared_key: tuple | None = None


def _resolve_process_workers(max_workers: int | None) -> int:
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, 8)
    return max(1, int(max_workers))


def _init_worker(init_config: dict) -> None:
    """
    Initialize worker process state.
    
    Called once per process when the pool is created. Sets up the Sim object
    and known library that will be reused for all chunks processed by this worker.
    """
    global _worker_sim, _worker_known, _worker_prepared, _worker_prepared_key
    
    from dynlib.compiler.build import build
    from dynlib.runtime.sim import Sim as SimClass
    
    # Build model from ModelSpec (avoids needing URI)
    model_spec = init_config["model_spec"]
    stepper_name = init_config["stepper"]
    
    # Build compiled model from spec
    full_model = build(model_spec, stepper=stepper_name, jit=True)
    
    # Create Sim instance
    _worker_sim = SimClass(full_model)
    
    # Apply session parameters
    session_params = init_config.get("session_params")
    if session_params is not None:
        _worker_sim.assign(**{
            name: val for name, val in zip(_worker_sim.model.spec.params, session_params)
        })
    
    # Build known library once (KnownAttractorLibrary is a dataclass, n_attr is a property)
    _worker_known = KnownAttractorLibrary(
        obs_idx=init_config["obs_idx"],
        names=init_config["names"],
        trajectories=init_config["trajectories"],
        obs_min=init_config["obs_min"],
        obs_max=init_config["obs_max"],
        escape_min=init_config["escape_min"],
        escape_max=init_config["escape_max"],
        attractor_radii=init_config["attractor_radii"],
    )
    _worker_prepared = None
    _worker_prepared_key = None
    

def _prepared_key(run_config: dict) -> tuple:
    blowup_idx = np.asarray(run_config["blowup_state_idx"], dtype=np.int64).ravel()
    blowup_key = tuple(int(x) for x in blowup_idx)
    b_max = run_config["b_max"]
    return (
        int(run_config["max_samples"]),
        int(run_config["transient_samples"]),
        float(run_config["dt_use"]),
        None if run_config["T"] is None else float(run_config["T"]),
        None if run_config["N"] is None else int(run_config["N"]),
        float(run_config["dist_threshold"]),
        None if b_max is None else float(b_max),
        blowup_key,
        bool(run_config["adaptive"]),
        int(run_config["fixed_point_settle_steps"]),
    )


def _classify_chunk_worker(args: tuple) -> np.ndarray:
    """
    Worker function for process-based parallelism.
    
    Uses the pre-initialized _worker_sim and _worker_known from _init_worker.
    This avoids recreating the Sim object for each chunk, saving JIT and
    memory allocation overhead.
    
    Args:
        args: Tuple containing (chunk_ic, chunk_params, run_config)
            - chunk_ic: IC array for this chunk
            - chunk_params: params array for this chunk
            - run_config: dict with classification parameters (not sim config)
    
    Returns:
        Labels array for this chunk
    """
    global _worker_sim, _worker_known, _worker_prepared, _worker_prepared_key
    
    chunk_ic, chunk_params, run_config = args
    
    if _worker_sim is None or _worker_known is None:
        raise RuntimeError("Worker not initialized. Call _init_worker first.")
    
    prepared_key = _prepared_key(run_config)
    if _worker_prepared is None or _worker_prepared_key != prepared_key:
        _worker_prepared = _prepare_classifier(
            sim=_worker_sim,
            known=_worker_known,
            max_samples=run_config["max_samples"],
            transient_samples=run_config["transient_samples"],
            dt_use=run_config["dt_use"],
            T=run_config["T"],
            N=run_config["N"],
            dist_threshold=run_config["dist_threshold"],
            b_max=run_config["b_max"],
            blowup_state_idx=run_config["blowup_state_idx"],
            adaptive=run_config["adaptive"],
            fixed_point_settle_steps=run_config["fixed_point_settle_steps"],
        )
        _worker_prepared_key = prepared_key

    # Run classification on this chunk using pre-initialized sim
    return _classify_batch_core_inner(
        sim=_worker_sim,
        ic_arr=chunk_ic,
        params_arr=chunk_params,
        known=_worker_known,
        max_samples=run_config["max_samples"],
        transient_samples=run_config["transient_samples"],
        dt_use=run_config["dt_use"],
        T=run_config["T"],
        N=run_config["N"],
        dist_threshold=run_config["dist_threshold"],
        b_max=run_config["b_max"],
        blowup_state_idx=run_config["blowup_state_idx"],
        batch_size=run_config["batch_size"],
        adaptive=run_config["adaptive"],
        fixed_point_settle_steps=run_config["fixed_point_settle_steps"],
        prepared=_worker_prepared,
    )


def _serialize_known_library(known: KnownAttractorLibrary) -> dict:
    """Serialize KnownAttractorLibrary to a dict for pickling."""
    return {
        "obs_idx": known.obs_idx,
        "trajectories": known.trajectories,  # list of arrays
        "attractor_radii": known.attractor_radii,  # list of arrays or None
        "obs_min": known.obs_min,
        "obs_max": known.obs_max,
        "escape_min": known.escape_min,
        "escape_max": known.escape_max,
        "names": known.names,
    }


def _build_process_init_config(sim: Sim, known: KnownAttractorLibrary) -> dict:
    # Pass ModelSpec directly since it's picklable (frozen dataclass with basic types)
    return {
        "model_spec": sim.model.spec,
        "stepper": sim.model.stepper_name,
        "session_params": sim.param_vector(source="session", copy=True),
        **_serialize_known_library(known),
    }


def _build_process_run_config(
    *,
    max_samples: int,
    transient_samples: int,
    dt_use: float,
    T: float | None,
    N: int | None,
    dist_threshold: float,
    b_max: float | None,
    blowup_state_idx: np.ndarray,
    batch_size: int,
    adaptive: bool,
    fixed_point_settle_steps: int,
) -> dict:
    return {
        "max_samples": max_samples,
        "transient_samples": transient_samples,
        "dt_use": dt_use,
        "T": T,
        "N": N,
        "dist_threshold": dist_threshold,
        "b_max": b_max,
        "blowup_state_idx": blowup_state_idx,
        "batch_size": batch_size,
        "adaptive": adaptive,
        "fixed_point_settle_steps": fixed_point_settle_steps,
    }


def _classify_batch_core_process(
    ic_arr: np.ndarray,
    params_arr: np.ndarray,
    *,
    run_config: dict,
    n_workers: int,
    init_config: dict | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> np.ndarray:
    total = int(ic_arr.shape[0])
    labels = np.full((total,), UNRESOLVED, dtype=np.int64)
    if total == 0:
        return labels

    chunk_size = (total + n_workers - 1) // n_workers
    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            break
        chunks.append((
            np.ascontiguousarray(ic_arr[start:end]),
            np.ascontiguousarray(params_arr[start:end]),
            run_config,
        ))

    if executor is None:
        if init_config is None:
            raise ValueError("init_config is required when executor is None")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(init_config,),
        ) as pool:
            chunk_results = list(pool.map(_classify_chunk_worker, chunks))
    else:
        chunk_results = list(executor.map(_classify_chunk_worker, chunks))

    pos = 0
    for chunk_labels in chunk_results:
        n = len(chunk_labels)
        labels[pos:pos + n] = chunk_labels
        pos += n
    return labels


def _prepare_classifier(
    sim: Sim,
    known: KnownAttractorLibrary,
    *,
    max_samples: int,
    transient_samples: int,
    dt_use: float,
    T: float | None,
    N: int | None,
    dist_threshold: float,
    b_max: float | None,
    blowup_state_idx: np.ndarray,
    adaptive: bool,
    fixed_point_settle_steps: int = 10,
) -> dict[str, object]:
    analysis_name = "known_hybrid"

    n_attr = int(known.n_attr)
    obs_dim = int(known.obs_idx.size)

    blowup_state_idx = np.asarray(blowup_state_idx, dtype=np.int64)

    # Identify fixed points (radius != None) vs reference runs (radius == None)
    is_fp = np.zeros((n_attr,), dtype=np.int64)
    is_ref = np.zeros((n_attr,), dtype=np.int64)
    for k in range(n_attr):
        if known.attractor_radii[k] is None:
            is_ref[k] = 1
        else:
            is_fp[k] = 1

    has_fp = bool(np.any(is_fp == 1))
    has_ref = bool(np.any(is_ref == 1))

    # Build fixed-point arrays (global ids + locs + radii^2), using dist_threshold as fallback
    fp_ids_list: list[int] = []
    fp_locs_list: list[np.ndarray] = []
    fp_r2_list: list[float] = []
    if has_fp:
        for k in range(n_attr):
            if int(is_fp[k]) == 0:
                continue
            loc = known.trajectories[k]
            if loc is None or loc.size == 0:
                continue
            # loc stored as (1, obs_dim)
            fp_ids_list.append(int(k))
            fp_locs_list.append(np.asarray(loc[0, :], dtype=np.float64))
            rad = known.attractor_radii[k]
            r = float(dist_threshold)
            if rad is not None:
                rr = np.asarray(rad, dtype=np.float64).reshape(-1)
                if rr.size > 0:
                    r = float(np.max(rr))
            if r <= 0.0:
                r = float(dist_threshold)
            fp_r2_list.append(r * r)

    if fp_ids_list:
        fp_ids = np.asarray(fp_ids_list, dtype=np.int64)
        fp_locs = np.vstack(fp_locs_list).astype(np.float64, copy=False).reshape(-1, obs_dim)
        fp_radii_sq = np.asarray(fp_r2_list, dtype=np.float64)
    else:
        fp_ids = np.zeros((0,), dtype=np.int64)
        fp_locs = np.zeros((0, obs_dim), dtype=np.float64)
        fp_radii_sq = np.zeros((0,), dtype=np.float64)

    outside_limit = max(10, min(50, max_samples // 5))

    # Practical defaults (no user tuning):
    # min_evidence ~ derived from max_samples
    available = max(0, int(max_samples) - int(transient_samples))
    min_evidence = min(64, max(16, max(1, available // 4)))
    confidence = 0.99
    check_stride = 8

    if not has_ref:
        analysis_mod = _make_fixed_point_only_analysis(
            name=analysis_name,
            obs_idx=known.obs_idx,
            blowup_idx=blowup_state_idx,
            escape_min=known.escape_min,
            escape_max=known.escape_max,
            transient_samples=transient_samples,
            outside_limit=outside_limit,
            b_max=b_max,
            fp_ids=fp_ids,
            fp_locs=fp_locs,
            fp_radii_sq=fp_radii_sq,
            fixed_point_settle_steps=fixed_point_settle_steps,
        )
    else:
        # Build PSC grid/index once per classification call
        grid_res = _choose_psc_grid_res(obs_dim)
        strides = _grid_strides(grid_res)
        ptr, cell_ids, cell_logp = _build_psc_signature_from_known(
            known=known,
            is_ref=is_ref,
            grid_res=grid_res,
            strides=strides,
            dilate_steps=1,
        )
        ht_keys, ht_start, ht_len, pairs_k, pairs_logp = _build_psc_pairs_index(
            n_attr=n_attr,
            ptr=ptr,
            cell_ids=cell_ids,
            cell_logp=cell_logp,
        )
        total_cells = float(np.prod(grid_res)) if grid_res.size > 0 else 1.0
        total_cells = max(1.0, total_cells)
        logp_null = float(np.log(1.0 / total_cells))

        analysis_mod = _make_known_hybrid_psc_analysis(
            name=analysis_name,
            n_attr=n_attr,
            obs_idx=known.obs_idx,
            blowup_idx=blowup_state_idx,
            obs_min=known.obs_min,
            obs_max=known.obs_max,
            escape_min=known.escape_min,
            escape_max=known.escape_max,
            grid_res=grid_res,
            strides=strides,
            ht_keys=ht_keys,
            ht_start=ht_start,
            ht_len=ht_len,
            pairs_k=pairs_k,
            pairs_logp=pairs_logp,
            logp_null=logp_null,
            transient_samples=transient_samples,
            min_evidence=min_evidence,
            confidence=confidence,
            check_stride=check_stride,
            outside_limit=outside_limit,
            b_max=b_max,
            fp_ids=fp_ids,
            fp_locs=fp_locs,
            fp_radii_sq=fp_radii_sq,
            fixed_point_settle_steps=fixed_point_settle_steps,
        )

    record_stride = int(max_samples + 1)
    plan = FixedStridePlan(stride=record_stride)
    record_vars_run: list[str] = []

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

    (
        state_rec_indices,
        aux_rec_indices,
        state_rec_names,
        aux_names,
    ) = sim._resolve_recording_selection(record_vars_run)
    stepper_config = sim.stepper_config()

    if use_fastpath:
        analysis_mod.validate_stepper(sim._stepper_spec)

    return {
        "analysis_name": analysis_name,
        "analysis_mod": analysis_mod,
        "use_fastpath": use_fastpath,
        "record_stride": record_stride,
        "plan": plan,
        "state_rec_indices": state_rec_indices,
        "aux_rec_indices": aux_rec_indices,
        "state_rec_names": state_rec_names,
        "aux_names": aux_names,
        "stepper_config": stepper_config,
    }


def _classify_batch_core_inner(
    sim: Sim,
    ic_arr: np.ndarray,
    params_arr: np.ndarray,
    known: KnownAttractorLibrary,
    *,
    max_samples: int,
    transient_samples: int,
    dt_use: float,
    T: float | None,
    N: int | None,
    dist_threshold: float,
    b_max: float | None,
    blowup_state_idx: np.ndarray,
    batch_size: int,
    adaptive: bool,
    fixed_point_settle_steps: int = 10,
    prepared: dict[str, object] | None = None,
) -> np.ndarray:
    """
    Inner classification logic - runs sequentially on given ICs.
    
    This is separated from _classify_batch_core to allow the outer function
    to handle chunking and parallel dispatch.
    """
    if prepared is None:
        prepared = _prepare_classifier(
            sim=sim,
            known=known,
            max_samples=max_samples,
            transient_samples=transient_samples,
            dt_use=dt_use,
            T=T,
            N=N,
            dist_threshold=dist_threshold,
            b_max=b_max,
            blowup_state_idx=blowup_state_idx,
            adaptive=adaptive,
            fixed_point_settle_steps=fixed_point_settle_steps,
        )

    analysis_name = prepared["analysis_name"]
    analysis_mod = prepared["analysis_mod"]
    use_fastpath = prepared["use_fastpath"]
    record_stride = prepared["record_stride"]
    plan = prepared["plan"]
    state_rec_indices = prepared["state_rec_indices"]
    aux_rec_indices = prepared["aux_rec_indices"]
    state_rec_names = prepared["state_rec_names"]
    aux_names = prepared["aux_names"]
    stepper_config = prepared["stepper_config"]
    
    total = int(ic_arr.shape[0])
    batch_size_use = min(batch_size, total) if total > 0 else 0
    
    labels = np.full((total,), UNRESOLVED, dtype=np.int64)
    
    for start in range(0, total, max(1, batch_size_use)):
        stop = min(total, start + max(1, batch_size_use))
        if use_fastpath:
            from dynlib.runtime.fastpath.executor import run_batch_fastpath_optimized

            results = run_batch_fastpath_optimized(
                model=sim.model,
                plan=plan,
                t0=sim.model.spec.sim.t0,
                t_end=T,
                target_steps=N,
                dt=dt_use,
                max_steps=max_samples + 1,
                transient=0.0,
                state_rec_indices=state_rec_indices,
                aux_rec_indices=aux_rec_indices,
                state_names=state_rec_names,
                aux_names=aux_names,
                params=params_arr[start:stop],
                ic=ic_arr[start:stop],
                stepper_config=stepper_config,
                parallel_mode="none",  # Sequential within chunk
                max_workers=1,
                observers=analysis_mod,
                analysis_only=True,
            )
            for j, result in enumerate(results):
                idx = start + j
                if result.status == NAN_DETECTED:
                    labels[idx] = BLOWUP
                    continue

                status = int(result.analysis_status)
                if status == 1:
                    labels[idx] = int(result.analysis_matched_id)
                elif status == 2:
                    labels[idx] = BLOWUP
                elif status == 3:
                    labels[idx] = OUTSIDE
                else:
                    labels[idx] = UNRESOLVED
            continue

        # Non-fastpath fallback
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
            view = ResultsView(result, sim.model.spec)
            
            if view.status == NAN_DETECTED:
                labels[i] = BLOWUP
                continue

            res = view.observers.get(analysis_name)
            out = None if res is None else res["out"]
            if out is None or out.size < 2:
                labels[i] = UNRESOLVED
                continue
            status = int(out[0])
            if status == 1:
                labels[i] = int(out[1])
            elif status == 2:
                labels[i] = BLOWUP
            elif status == 3:
                labels[i] = OUTSIDE
            else:
                labels[i] = UNRESOLVED
    
    return labels


def _classify_batch_core(
    sim: Sim,
    ic_arr: np.ndarray,
    params_arr: np.ndarray,
    known: KnownAttractorLibrary,
    *,
    max_samples: int,
    transient_samples: int,
    dt_use: float,
    T: float | None,
    N: int | None,
    dist_threshold: float,
    b_max: float | None,
    blowup_state_idx: np.ndarray,
    batch_size: int,
    parallel_mode: str,
    max_workers: int | None,
    adaptive: bool,
    fixed_point_settle_steps: int = 10,
) -> np.ndarray:
    """
    Core classification logic with chunk-based process parallelism support.
    
    When parallel_mode is "process" or "auto", splits the IC array into chunks
    and processes each chunk in a separate subprocess for true parallelism.
    
    Returns labels array for the given initial conditions.
    """
    total = int(ic_arr.shape[0])
    
    # Determine effective parallel mode
    use_process_parallel = parallel_mode in ("process", "auto") and total > 1000
    
    # Determine number of workers
    n_workers = _resolve_process_workers(max_workers)
    
    # For small batches or explicit "none"/"threads", run sequentially
    if not use_process_parallel or n_workers == 1 or parallel_mode == "none":
        return _classify_batch_core_inner(
            sim=sim,
            ic_arr=ic_arr,
            params_arr=params_arr,
            known=known,
            max_samples=max_samples,
            transient_samples=transient_samples,
            dt_use=dt_use,
            T=T,
            N=N,
            dist_threshold=dist_threshold,
            b_max=b_max,
            blowup_state_idx=blowup_state_idx,
            batch_size=batch_size,
            adaptive=adaptive,
            fixed_point_settle_steps=fixed_point_settle_steps,
        )
    
    # =========================================================================
    # CHUNK-BASED PROCESS PARALLELISM
    # =========================================================================
    # Split ICs into chunks and process each in a separate subprocess.
    # This avoids GIL contention and provides true parallelism.
    
    init_config = _build_process_init_config(sim, known)
    run_config = _build_process_run_config(
        max_samples=max_samples,
        transient_samples=transient_samples,
        dt_use=dt_use,
        T=T,
        N=N,
        dist_threshold=dist_threshold,
        b_max=b_max,
        blowup_state_idx=blowup_state_idx,
        batch_size=batch_size,
        adaptive=adaptive,
        fixed_point_settle_steps=fixed_point_settle_steps,
    )
    return _classify_batch_core_process(
        ic_arr,
        params_arr,
        run_config=run_config,
        n_workers=n_workers,
        init_config=init_config,
    )


def basin_known(
    sim: Sim,
    attractors: Sequence[FixedPoint | ReferenceRun],
    *,
    ic: np.ndarray | None = None,
    ic_grid: Sequence[int] | None = None,
    ic_bounds: Sequence[tuple[float, float]] | None = None,
    params: np.ndarray | None = None,
    mode: Literal["map", "ode", "auto"] = "auto",
    max_samples: int = 1024,
    transient_samples: int = 0,
    signature_samples: int = 500,
    dt_obs: float | None = None,
    observe_vars: Sequence[str | int] | None = None,
    escape_bounds: Sequence[tuple[float, float]] | None = None,
    tolerance: float = 0.1,
    tolerance_absolute: float | None = None,
    min_match_ratio: float = 0.7,
    b_max: float | None = None,
    blowup_vars: Sequence[str | int] | None = None,
    batch_size: int | None = None,
    parallel_mode: Literal["auto", "threads", "process", "none"] = "auto",
    max_workers: int | None = None,
    # Coarse-to-fine refinement options
    refine: bool = False,
    coarse_factor: int = 8,
    boundary_dilation: int = 1,
    # Fixed point fast-path options
    fixed_point_settle_steps: int = 10,
) -> BasinResult:
    """
    Classify initial conditions against known attractors using distance-based matching.
    
    This function uses a hybrid approach for efficient classification:
    - **Fixed points** (FixedPoint attractors) use a fast-path: trajectories are
      classified immediately when they stay within the attractor radius for
      consecutive steps. This is much faster than signature matching.
    - **Reference runs** (ReferenceRun attractors) use signature matching: collect
      trajectory points and check match ratio against captured reference.
    
    For basins with only fixed point attractors, set ``signature_samples=0`` to skip
    unnecessary signature capture overhead.
    
    Parameters
    ----------
    sim : Sim
        The simulation object
    attractors : sequence of FixedPoint or ReferenceRun
        List of attractors to identify. Each attractor is defined by either:
        - FixedPoint: a known fixed point with position and optional radius
        - ReferenceRun: initial conditions to run and capture the attractor trajectory
    ic : np.ndarray, optional
        Initial conditions (n_ics x n_states)
    ic_grid : sequence of int, optional
        Grid resolution for each dimension
    ic_bounds : sequence of (min, max) tuples, optional
        Bounds for each dimension when using ic_grid
    params : np.ndarray, optional
        Parameters for each IC
    mode : {'map', 'ode', 'auto'}, default 'auto'
        Simulation mode
    max_samples : int, default 1024
        Maximum number of steps to simulate for classification
    transient_samples : int, default 0
        Number of initial steps to skip before matching (used for both
        building attractors and classifying trajectories)
    signature_samples : int, default 500
        Number of steps to capture for building attractor signatures.
        Only needed for ReferenceRun attractors. For basins with only
        FixedPoint attractors, set to 0 to skip signature capture.
    dt_obs : float, optional
        Observation timestep (required for ODE mode)
    observe_vars : sequence of str or int, optional
        State variables to observe for attractor matching (default: all states)
    escape_bounds : sequence of (min, max) tuples, optional
        Bounds for escape/blowup detection. Trajectories leaving this region
        are marked as OUTSIDE. If None, computed as attractor_bounds * 1.5.
    tolerance : float, default 0.1
        Relative distance tolerance for matching (as fraction of attractor range).
        Ignored if tolerance_absolute is provided.
    tolerance_absolute : float, optional
        Absolute distance tolerance for matching in state space.
        Overrides relative tolerance if provided. Useful for fixed points.
    min_match_ratio : float, default 0.7
        Minimum fraction of points that must match for classification.
        Only used for ReferenceRun attractors.
    b_max : float, optional
        Blowup threshold. Use very high values (1e10+) or None to rely on NaN/Inf
        detection. Trajectories exceeding this are marked as BLOWUP.
        Default None means only NaN/Inf are considered blowup.
    blowup_vars : sequence of str or int, optional
        Variables to check for blowup (default: observed variables)
    batch_size : int, optional
        Batch size for parallel processing
    parallel_mode : {'auto', 'threads', 'process', 'none'}, default 'auto'
        Parallelization mode
    max_workers : int, optional
        Maximum number of parallel workers
    refine : bool, default False
        Enable coarse-to-fine grid refinement. When True, first classifies a
        coarse grid, then refines only the boundary regions at full resolution.
        This can provide 5-20x speedup for basins with large homogeneous regions.
        Only applies when ic_grid is provided.
    coarse_factor : int, default 8
        Factor to reduce grid resolution for coarse pass when refine=True.
        A value of 8 means the coarse grid is 1/8 the resolution in each dimension.
        Higher values give more speedup but may miss small features.
    boundary_dilation : int, default 1
        Number of cells to dilate boundary regions when refine=True.
        Larger values ensure boundary accuracy but increase computation.
    fixed_point_settle_steps : int, default 10
        Number of consecutive steps a trajectory must stay within a fixed point's
        radius to be classified as converged to that attractor. Lower values give
        faster classification but may cause false positives if trajectories are
        just passing through. Only affects FixedPoint attractors.
    
    Returns
    -------
    BasinResult
        Classification results with labels and registry.
        Labels: >= 0 = attractor ID, BLOWUP = NaN/Inf, OUTSIDE = escaped basin,
        UNRESOLVED = couldn't classify
    """
    _require_numba("basin_known")

    # Build known attractors library from attractor specs
    known = build_known_attractors_psc(
        sim,
        attractor_specs=attractors,
        observe_vars=observe_vars,
        escape_bounds=escape_bounds,
        mode=mode,
        dt_obs=dt_obs,
        transient_samples=transient_samples,
        signature_samples=signature_samples,
    )

    if ic is None and ic_grid is None:
        raise ValueError("Either ic or ic_grid must be provided")
    if ic is not None and ic_grid is not None:
        raise ValueError("Cannot specify both ic and ic_grid")
    if ic_grid is not None and ic_bounds is None:
        raise ValueError("ic_bounds must be provided when using ic_grid")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if transient_samples < 0:
        raise ValueError("transient_samples must be non-negative")
    if b_max is not None and b_max <= 0.0:
        raise ValueError("b_max must be positive when provided")
    if not (0.0 < tolerance < 10.0):
        raise ValueError("tolerance must be in (0, 10)")
    if not (0.0 < min_match_ratio <= 1.0):
        raise ValueError("min_match_ratio must be in (0, 1]")
    if coarse_factor < 2:
        raise ValueError("coarse_factor must be >= 2")
    if boundary_dilation < 0:
        raise ValueError("boundary_dilation must be >= 0")
    if transient_samples >= max_samples:
        raise ValueError("transient_samples must be less than max_samples")

    mode_use = _resolve_mode(mode=mode, sim=sim)
    adaptive = getattr(sim._stepper_spec.meta, "time_control", "fixed") == "adaptive"
    if mode_use == "ode" and adaptive:
        raise ValueError("basin_known requires a fixed-step stepper for ODE mode")

    state_names = list(sim.model.spec.states)
    obs_names = [state_names[int(idx)] for idx in known.obs_idx.tolist()]
    
    # Setup blowup detection
    if b_max is not None:
        blowup_use = blowup_vars
        if blowup_use is None:
            blowup_use = list(obs_names)
    else:
        blowup_use = None

    record_vars, blowup_idx = _prepare_record_vars(
        sim,
        observe_vars=obs_names,
        blowup_vars=blowup_use,
        d=int(len(obs_names)),
    )
    state_to_idx = {name: idx for idx, name in enumerate(state_names)}
    blowup_names = [record_vars[int(idx)] for idx in blowup_idx.tolist()]
    blowup_state_idx = np.array([state_to_idx[name] for name in blowup_names], dtype=np.int64)

    n_state = len(sim.model.spec.states)
    n_params = len(sim.model.spec.params)
    dtype = sim.model.dtype
    if params is None:
        params = sim.param_vector(source="session", copy=True)

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

    # Compute distance threshold
    obs_range = known.obs_max - known.obs_min
    if tolerance_absolute is not None:
        dist_threshold = float(tolerance_absolute)
    else:
        relative_threshold = tolerance * np.mean(obs_range)
        dist_threshold = max(relative_threshold, 1e-6)

    # Determine batch size
    if batch_size is None:
        batch_size_use = 4096
    else:
        batch_size_use = int(batch_size)
        if batch_size_use <= 0:
            raise ValueError("batch_size must be positive when provided")

    # Common classification parameters
    classify_kwargs = dict(
        known=known,
        max_samples=max_samples,
        transient_samples=transient_samples,
        dt_use=dt_use,
        T=T,
        N=N,
        dist_threshold=dist_threshold,
        b_max=b_max,
        blowup_state_idx=blowup_state_idx,
        batch_size=batch_size_use,
        parallel_mode=parallel_mode,
        max_workers=max_workers,
        adaptive=adaptive,
        fixed_point_settle_steps=fixed_point_settle_steps,
    )
    classify_inner_kwargs = dict(
        known=known,
        max_samples=max_samples,
        transient_samples=transient_samples,
        dt_use=dt_use,
        T=T,
        N=N,
        dist_threshold=dist_threshold,
        b_max=b_max,
        blowup_state_idx=blowup_state_idx,
        batch_size=batch_size_use,
        adaptive=adaptive,
        fixed_point_settle_steps=fixed_point_settle_steps,
    )

    # Store grid metadata
    ic_grid_meta: tuple[int, ...] | None = None
    ic_bounds_meta: tuple[tuple[float, float], ...] | None = None
    refine_used = False
    coarse_points = 0
    fine_points = 0

    if ic_grid is not None:
        ic_grid_arr = np.asarray(ic_grid, dtype=np.int64)
        if ic_grid_arr.ndim != 1:
            raise ValueError("ic_grid must be a 1D sequence")
        n_dims = len(ic_grid_arr)
        if len(ic_bounds) != n_dims:
            raise ValueError(f"ic_bounds must have {n_dims} elements to match ic_grid")
        
        fine_shape = tuple(int(x) for x in ic_grid_arr)
        ic_grid_meta = fine_shape
        ic_bounds_meta = tuple((float(bmin), float(bmax)) for bmin, bmax in ic_bounds)
        if refine:
            params_arr = np.asarray(params)
            if params_arr.ndim > 1 and params_arr.shape[0] != 1:
                raise ValueError(
                    "refine=True only supports session params; provide params as (n_params,) or (1, n_params)"
                )
        
        # Check if refinement is worthwhile
        min_coarse_size = max(4, coarse_factor)  # Need at least 4 cells per dimension
        can_refine = refine and all(n >= min_coarse_size * coarse_factor for n in fine_shape)
        
        if can_refine:
            # =========================================================
            # COARSE-TO-FINE REFINEMENT
            # =========================================================
            refine_used = True

            use_shared_pool = (
                parallel_mode in ("process", "auto")
                and _resolve_process_workers(max_workers) > 1
                and int(np.prod(fine_shape)) > 1000
            )
            prepared = None
            if not use_shared_pool:
                prepared = _prepare_classifier(
                    sim=sim,
                    known=known,
                    max_samples=max_samples,
                    transient_samples=transient_samples,
                    dt_use=dt_use,
                    T=T,
                    N=N,
                    dist_threshold=dist_threshold,
                    b_max=b_max,
                    blowup_state_idx=blowup_state_idx,
                    adaptive=adaptive,
                    fixed_point_settle_steps=fixed_point_settle_steps,
                )
            
            # Phase 1: Coarse classification
            coarse_shape = tuple(max(min_coarse_size, n // coarse_factor) for n in fine_shape)
            coarse_ics = _generate_grid_ics(coarse_shape, ic_bounds_meta, dtype)
            
            coarse_ic_arr, coarse_params_arr = _coerce_batch(
                ic=coarse_ics,
                params=params,
                n_state=n_state,
                n_params=n_params,
                dtype=dtype,
            )

            if use_shared_pool:
                n_workers = _resolve_process_workers(max_workers)
                init_config = _build_process_init_config(sim, known)
                run_config = _build_process_run_config(
                    max_samples=max_samples,
                    transient_samples=transient_samples,
                    dt_use=dt_use,
                    T=T,
                    N=N,
                    dist_threshold=dist_threshold,
                    b_max=b_max,
                    blowup_state_idx=blowup_state_idx,
                    batch_size=batch_size_use,
                    adaptive=adaptive,
                    fixed_point_settle_steps=fixed_point_settle_steps,
                )
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_init_worker,
                    initargs=(init_config,),
                ) as executor:
                    coarse_labels = _classify_batch_core_process(
                        coarse_ic_arr,
                        coarse_params_arr,
                        run_config=run_config,
                        n_workers=n_workers,
                        executor=executor,
                    )
                    coarse_points = len(coarse_labels)

                    # Phase 2: Find boundary regions
                    boundary_mask = _find_boundary_mask_nd(
                        coarse_labels, coarse_shape, dilation=boundary_dilation
                    )

                    # Phase 3: Upscale coarse labels to fine grid
                    labels = _upscale_labels(coarse_labels, coarse_shape, fine_shape)

                    # Phase 4: Refine boundary regions at full resolution
                    fine_boundary_mask = _upscale_mask(
                        boundary_mask.reshape(coarse_shape),
                        coarse_shape,
                        fine_shape,
                    ).ravel()
                    fine_indices = np.flatnonzero(fine_boundary_mask)
                    fine_points = int(fine_indices.size)

                    if fine_points > 0:
                        fine_ics = _grid_points_from_flat_indices(
                            fine_indices,
                            fine_shape,
                            ic_bounds_meta,
                            dtype,
                        )

                        fine_ic_arr, fine_params_arr = _coerce_batch(
                            ic=fine_ics,
                            params=params,
                            n_state=n_state,
                            n_params=n_params,
                            dtype=dtype,
                        )

                        # Classify fine boundary points
                        fine_labels = _classify_batch_core_process(
                            fine_ic_arr,
                            fine_params_arr,
                            run_config=run_config,
                            n_workers=n_workers,
                            executor=executor,
                        )

                        # Update labels with fine results
                        labels[fine_indices] = fine_labels
            else:
                coarse_labels = _classify_batch_core_inner(
                    sim=sim,
                    ic_arr=coarse_ic_arr,
                    params_arr=coarse_params_arr,
                    **classify_inner_kwargs,
                    prepared=prepared,
                )
                coarse_points = len(coarse_labels)

                # Phase 2: Find boundary regions
                boundary_mask = _find_boundary_mask_nd(
                    coarse_labels, coarse_shape, dilation=boundary_dilation
                )

                # Phase 3: Upscale coarse labels to fine grid
                labels = _upscale_labels(coarse_labels, coarse_shape, fine_shape)

                # Phase 4: Refine boundary regions at full resolution
                fine_boundary_mask = _upscale_mask(
                    boundary_mask.reshape(coarse_shape),
                    coarse_shape,
                    fine_shape,
                ).ravel()
                fine_indices = np.flatnonzero(fine_boundary_mask)
                fine_points = int(fine_indices.size)

                if fine_points > 0:
                    fine_ics = _grid_points_from_flat_indices(
                        fine_indices,
                        fine_shape,
                        ic_bounds_meta,
                        dtype,
                    )

                    fine_ic_arr, fine_params_arr = _coerce_batch(
                        ic=fine_ics,
                        params=params,
                        n_state=n_state,
                        n_params=n_params,
                        dtype=dtype,
                    )

                    # Classify fine boundary points
                    fine_labels = _classify_batch_core_inner(
                        sim=sim,
                        ic_arr=fine_ic_arr,
                        params_arr=fine_params_arr,
                        **classify_inner_kwargs,
                        prepared=prepared,
                    )

                    # Update labels with fine results
                    labels[fine_indices] = fine_labels
        else:
            # No refinement - classify full grid
            ic = _generate_grid_ics(fine_shape, ic_bounds_meta, dtype)
            ic_arr, params_arr = _coerce_batch(
                ic=ic,
                params=params,
                n_state=n_state,
                n_params=n_params,
                dtype=dtype,
            )
            labels = _classify_batch_core(sim, ic_arr, params_arr, **classify_kwargs)
    else:
        # Direct IC array provided - no refinement possible
        ic_arr, params_arr = _coerce_batch(
            ic=ic,
            params=params,
            n_state=n_state,
            n_params=n_params,
            dtype=dtype,
        )
        labels = _classify_batch_core(sim, ic_arr, params_arr, **classify_kwargs)

    # Build registry
    registry = [
        Attractor(id=i, fingerprint=set(), cells=set())
        for i in range(int(known.n_attr))
    ]

    # Build metadata
    meta = {
        "mode": mode_use,
        "observe_vars": tuple(obs_names),
        "attractor_names": tuple(known.names),
        "transient_samples": int(transient_samples),
        "max_samples": int(max_samples),
        "dt_obs": float(dt_use),
        "tolerance": float(tolerance),
        "min_match_ratio": float(min_match_ratio),
        "batch_size": int(batch_size_use),
    }
    if ic_grid_meta is not None:
        meta["ic_grid"] = ic_grid_meta
    if ic_bounds_meta is not None:
        meta["ic_bounds"] = ic_bounds_meta
    if refine_used:
        meta["refine"] = True
        meta["coarse_factor"] = int(coarse_factor)
        meta["coarse_points"] = int(coarse_points)
        meta["fine_points"] = int(fine_points)
        meta["total_classified"] = int(coarse_points + fine_points)

    return BasinResult(labels=labels, registry=registry, meta=meta)
