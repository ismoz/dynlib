# src/dynlib/analysis/basin_stats.py
"""
Helper utilities for computing statistics from basin of attraction results and printing them.
Change this if you ever modify BasinResult dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from dynlib.analysis.basin_codes import BLOWUP, OUTSIDE, UNRESOLVED


@dataclass(frozen=True)
class AttractorStats:
    id: int
    n_points: int
    p_points: float
    n_fingerprint: int
    n_cells: int


@dataclass(frozen=True)
class BasinStats:
    n_total: int
    n_attractors: int

    n_identified: int
    p_identified: float

    n_blowup: int
    p_blowup: float

    n_outside: int
    p_outside: float

    n_unresolved: int
    p_unresolved: float

    per_attractor: tuple[AttractorStats, ...]

    # Convenience metadata if present (not required for any computation)
    ic_grid: tuple[int, ...] | None = None
    ic_bounds: tuple[tuple[float, float], ...] | None = None


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return 100.0 * (float(n) / float(d))


def _infer_ic_grid(meta: dict[str, object]) -> tuple[int, ...] | None:
    val = meta.get("ic_grid")
    if val is None:
        return None
    if isinstance(val, tuple) and all(isinstance(x, (int, np.integer)) for x in val):
        return tuple(int(x) for x in val)
    if isinstance(val, (list, np.ndarray)) and all(isinstance(x, (int, np.integer)) for x in val):
        return tuple(int(x) for x in val)
    return None


def _infer_ic_bounds(meta: dict[str, object]) -> tuple[tuple[float, float], ...] | None:
    val = meta.get("ic_bounds")
    if val is None:
        return None
    try:
        out: list[tuple[float, float]] = []
        for pair in val:  # type: ignore[assignment]
            if pair is None or len(pair) != 2:
                return None
            out.append((float(pair[0]), float(pair[1])))
        return tuple(out)
    except Exception:
        return None


def basin_stats(result: Any, *, top_k: int | None = None) -> BasinStats:
    """
    Compute derived statistics from a BasinResult-like object.

    Requirements on `result`:
      - result.labels: np.ndarray (or array-like) of int labels
      - result.registry: list of attractor objects with .id, .fingerprint, .cells
      - result.meta: dict[str, object]

    Parameters
    ----------
    top_k:
        If provided, per_attractor is limited to the top_k attractors by n_points
        (ties broken by attractor id). Totals remain unaffected.
    """
    labels = np.asarray(result.labels)
    n_total = int(labels.size)

    uniq, cnt = np.unique(labels, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}

    n_blowup = counts.get(int(BLOWUP), 0)
    n_outside = counts.get(int(OUTSIDE), 0)
    n_unresolved = counts.get(int(UNRESOLVED), 0)
    n_identified = n_total - (n_blowup + n_outside + n_unresolved)

    registry = list(getattr(result, "registry", []))
    n_attractors = int(len(registry))

    per: list[AttractorStats] = []
    if n_total > 0 and registry:
        for A in registry:
            aid = int(A.id)
            n_points = counts.get(aid, 0)
            per.append(
                AttractorStats(
                    id=aid,
                    n_points=int(n_points),
                    p_points=_pct(int(n_points), n_total),
                    n_fingerprint=int(len(getattr(A, "fingerprint", ()) or ())),
                    n_cells=int(len(getattr(A, "cells", ()) or ())),
                )
            )

        per.sort(key=lambda r: (-r.n_points, r.id))

        if top_k is not None:
            k = int(top_k)
            if k < 0:
                raise ValueError("top_k must be non-negative when provided")
            per = per[:k]

    meta = dict(getattr(result, "meta", {}) or {})
    return BasinStats(
        n_total=n_total,
        n_attractors=n_attractors,
        n_identified=n_identified,
        p_identified=_pct(n_identified, n_total),
        n_blowup=n_blowup,
        p_blowup=_pct(n_blowup, n_total),
        n_outside=n_outside,
        p_outside=_pct(n_outside, n_total),
        n_unresolved=n_unresolved,
        p_unresolved=_pct(n_unresolved, n_total),
        per_attractor=tuple(per),
        ic_grid=_infer_ic_grid(meta),
        ic_bounds=_infer_ic_bounds(meta),
    )


def basin_summary(
    result: Any,
    *,
    top_k: int | None = None,
    include_meta: bool = True,
    meta_keys: Iterable[str] | None = None,
    sort_meta: bool = True,
    attractor_order: str = "registry",
) -> str:
    """
    Human-readable report similar to the example script.

    Parameters
    ----------
    top_k:
        Limit the printed attractor details to top_k (largest basins first).
    include_meta:
        Whether to include "Algorithm parameters" section.
    meta_keys:
        If provided, only include these meta keys.
    sort_meta:
        Sort meta keys for deterministic output.
    attractor_order:
        "registry" prints attractors in result.registry order (example-like).
        "largest" prints attractors by assigned points descending.
        If top_k is provided, "largest" is generally what you want.
    """
    st = basin_stats(result, top_k=top_k)

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("BASIN ANALYSIS RESULTS")
    lines.append("=" * 60)
    lines.append(f"Total initial conditions: {st.n_total}")
    lines.append(f"Attractors identified: {st.n_attractors}")
    lines.append(f"Points assigned to attractors: {st.n_identified} ({st.p_identified:.1f}%)")
    lines.append(f"Blowup trajectories: {st.n_blowup} ({st.p_blowup:.1f}%)")
    lines.append(f"Outside domain: {st.n_outside} ({st.p_outside:.1f}%)")
    lines.append(f"Unresolved: {st.n_unresolved} ({st.p_unresolved:.1f}%)")

    registry = list(getattr(result, "registry", []))

    if st.n_attractors > 0:
        lines.append("")
        lines.append("Attractor details:")

        if top_k is not None or attractor_order == "largest":
            # Use stats ordering (largest first), already limited by top_k if provided.
            for row in st.per_attractor:
                lines.append(
                    f"  Attractor {row.id}: {row.n_points} points, {row.n_fingerprint} cells in fingerprint"
                )
        else:
            # Registry order (example-like). Use a lookup for n_points where possible.
            lookup = {row.id: row for row in st.per_attractor}
            labels = np.asarray(result.labels)
            for A in registry:
                aid = int(A.id)
                row = lookup.get(aid)
                if row is None:
                    n_points = int(np.sum(labels == aid))
                    n_fp = int(len(getattr(A, "fingerprint", ()) or ()))
                else:
                    n_points = row.n_points
                    n_fp = row.n_fingerprint
                lines.append(
                    f"  Attractor {aid}: {n_points} points, {n_fp} cells in fingerprint"
                )

    if include_meta:
        lines.append("")
        lines.append("Algorithm parameters:")

        meta = dict(getattr(result, "meta", {}) or {})
        items: list[tuple[str, Any]]
        if meta_keys is None:
            items = [(str(k), v) for k, v in meta.items()]
            if sort_meta:
                items.sort(key=lambda kv: kv[0])
        else:
            keys = [str(k) for k in meta_keys]
            items = [(k, meta.get(k)) for k in keys]
            if sort_meta:
                items.sort(key=lambda kv: kv[0])

        for k, v in items:
            lines.append(f"  {k}: {v}")

    lines.append("")
    return "\n".join(lines)


def print_basin_summary(result: Any, **kwargs: Any) -> None:
    """Convenience wrapper."""
    print(basin_summary(result, **kwargs))
