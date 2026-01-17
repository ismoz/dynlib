# src/dynlib/analysis/post/bifurcation.py
"""Bifurcation post-processing utilities.

This module intentionally separates:
- Runtime: generating trajectories via ``dynlib.analysis.sweep.traj_sweep``
- Post-processing: extracting bifurcation scatter points from the trajectories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dynlib.analysis.sweep import TrajectoryPayload, SweepResult

import numpy as np

__all__ = ["BifurcationResult", "BifurcationExtractor"]


@dataclass
class BifurcationResult:
    param_name: str
    values: np.ndarray  # sweep grid (M,)
    mode: str
    p: np.ndarray  # scatter x-axis (P,)
    y: np.ndarray  # scatter y-axis (P,)
    meta: dict


def _extract_tail(series: np.ndarray, tail: int) -> np.ndarray:
    if series.size == 0:
        raise RuntimeError("No samples recorded; increase T/N or adjust record_interval.")
    return series[-min(tail, series.size) :]


def _extrema_from_tail(
    series: np.ndarray,
    *,
    kind: Literal["max", "min", "both"],
    max_points: int,
    min_peak_distance: int,
) -> np.ndarray:
    if series.size < 3:
        return np.array([], dtype=float)
    mid = series[1:-1]

    idx_parts: list[np.ndarray] = []
    if kind in ("max", "both"):
        rising = series[:-2] < mid
        falling = mid >= series[2:]
        peak_mask = rising & falling
        idx_parts.append(np.nonzero(peak_mask)[0] + 1)  # offset because mid skips the first element
    if kind in ("min", "both"):
        falling = series[:-2] > mid
        rising = mid <= series[2:]
        trough_mask = falling & rising
        idx_parts.append(np.nonzero(trough_mask)[0] + 1)

    if not idx_parts:
        raise ValueError("kind must be 'max', 'min', or 'both'")
    if len(idx_parts) == 1:
        idx = idx_parts[0]
    else:
        idx = np.sort(np.concatenate(idx_parts))
    if idx.size == 0:
        return np.array([], dtype=float)

    kept: list[int] = []
    last_kept = -min_peak_distance - 1
    for i in idx:
        if i - last_kept >= min_peak_distance:
            kept.append(int(i))
            last_kept = i
    if not kept:
        return np.array([], dtype=float)
    trimmed = kept[-max_points:] if max_points > 0 else kept
    return series[np.array(trimmed, dtype=int)]


def _poincare_from_series(
    section: np.ndarray,
    target: np.ndarray,
    *,
    level: float,
    direction: str,
    max_points: int,
    min_section_distance: int,
) -> np.ndarray:
    if section.size < 2:
        return np.array([], dtype=float)
    if section.shape != target.shape:
        raise ValueError("section and target series must have the same shape")

    delta0 = section[:-1] - level
    delta1 = section[1:] - level
    if direction == "positive":
        crossing = (delta0 < 0) & (delta1 >= 0)
    elif direction == "negative":
        crossing = (delta0 > 0) & (delta1 <= 0)
    elif direction == "both":
        crossing = ((delta0 < 0) & (delta1 >= 0)) | ((delta0 > 0) & (delta1 <= 0))
    else:
        raise ValueError("direction must be 'positive', 'negative', or 'both'")

    idx = np.nonzero(crossing)[0]
    if idx.size == 0:
        return np.array([], dtype=float)

    kept: list[int] = []
    last_kept = -min_section_distance - 1
    for i in idx:
        if i - last_kept >= min_section_distance:
            kept.append(int(i))
            last_kept = i
    if not kept:
        return np.array([], dtype=float)

    trimmed = kept[-max_points:] if max_points > 0 else kept
    arr = np.array(trimmed, dtype=int)

    denom = section[arr + 1] - section[arr]
    valid = denom != 0
    if not np.any(valid):
        return np.array([], dtype=float)
    arr = arr[valid]
    denom = denom[valid]

    frac = (level - section[arr]) / denom
    y0 = target[arr]
    y1 = target[arr + 1]
    return y0 + frac * (y1 - y0)


class BifurcationExtractor:
    """Post-process a trajectory sweep into bifurcation scatter points.
    
    Can be used directly (defaults to .all() mode) or call explicit methods
    like .tail(), .final(), .extrema(), or .poincare() for different extraction strategies.
    
    Examples:
        >>> from dynlib.plot import bifurcation_diagram
        >>> # Direct use (defaults to .all())
        >>> result = sweep_result.bifurcation("x")
        >>> bifurcation_diagram(result)  # Uses all points
        
        >>> # Explicit extraction mode
        >>> result = sweep_result.bifurcation("x").tail(50)
        >>> bifurcation_diagram(result)  # Uses last 50 points
    """

    def __init__(self, sweep_result: "SweepResult", var: str):
        if not isinstance(sweep_result.payload, TrajectoryPayload):
            raise TypeError("BifurcationExtractor requires a trajectory sweep result")
        self._sweep = sweep_result
        self._var = var
        self._cached_all = None  # Lazy cache for .all() result
        try:
            self._var_idx = sweep_result.record_vars.index(var)
        except ValueError:
            raise KeyError(
                f"Unknown variable {var!r}; available: {sweep_result.record_vars}"
            ) from None

    @property
    def var(self) -> str:
        return self._var

    @property
    def sweep(self) -> "SweepResult":
        return self._sweep
    
    # Duck-typing properties for direct use (defaults to .all() mode)
    @property
    def p(self) -> np.ndarray:
        """Parameter values (x-axis) - computed using .all() by default."""
        if self._cached_all is None:
            self._cached_all = self.all()
        return self._cached_all.p
    
    @property
    def y(self) -> np.ndarray:
        """Variable values (y-axis) - computed using .all() by default."""
        if self._cached_all is None:
            self._cached_all = self.all()
        return self._cached_all.y
    
    @property
    def param_name(self) -> str:
        """Parameter name."""
        return self._sweep.param_name
    
    @property
    def mode(self) -> str:
        """Extraction mode - 'all' by default."""
        if self._cached_all is None:
            self._cached_all = self.all()
        return self._cached_all.mode
    
    @property
    def meta(self) -> dict:
        """Metadata dictionary."""
        if self._cached_all is None:
            self._cached_all = self.all()
        return self._cached_all.meta
    
    @property
    def values(self) -> np.ndarray:
        """Parameter sweep grid."""
        return self._sweep.values

    def _series_iter(self):
        for p_val, arr in zip(self._sweep.values, self._sweep.data):
            yield float(p_val), np.asarray(arr[:, self._var_idx], dtype=float)

    def _array_iter(self):
        for p_val, arr in zip(self._sweep.values, self._sweep.data):
            yield float(p_val), np.asarray(arr, dtype=float)

    def all(self) -> BifurcationResult:
        """Extract all recorded points (no filtering).
        
        Returns a bifurcation result containing every recorded data point
        from all parameter values. This is useful when you want to see the
        complete trajectory data without any post-processing.
        
        Returns:
            BifurcationResult with all recorded points
            
        Example:
            >>> res = sweep.traj_sweep(sim, param="r", values=r_values, record_vars=["x"], N=500)
            >>> bif = res.bifurcation("x").all()  # Use all 500 points per parameter
        """
        p_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for p_val, series in self._series_iter():
            if series.size == 0:
                continue
            p_parts.append(np.full(series.shape, p_val, dtype=float))
            y_parts.append(np.array(series, dtype=float))

        p_out = np.concatenate(p_parts) if p_parts else np.empty((0,), dtype=float)
        y_out = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=float)

        meta = dict(self._sweep.meta)
        meta.update(var=self._var, mode="all")
        return BifurcationResult(
            param_name=self._sweep.param_name,
            values=self._sweep.values,
            mode="all",
            p=p_out,
            y=y_out,
            meta=meta,
        )

    def tail(self, n: int) -> BifurcationResult:
        if n <= 0:
            raise ValueError("n must be positive")

        p_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for p_val, series in self._series_iter():
            tail_series = _extract_tail(series, n)
            p_parts.append(np.full(tail_series.shape, p_val, dtype=float))
            y_parts.append(np.array(tail_series, dtype=float))

        p_out = np.concatenate(p_parts) if p_parts else np.empty((0,), dtype=float)
        y_out = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=float)

        meta = dict(self._sweep.meta)
        meta.update(var=self._var, mode="tail", tail=int(n))
        return BifurcationResult(
            param_name=self._sweep.param_name,
            values=self._sweep.values,
            mode="tail",
            p=p_out,
            y=y_out,
            meta=meta,
        )

    def final(self) -> BifurcationResult:
        p_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for p_val, series in self._series_iter():
            tail_series = _extract_tail(series, 1)
            p_parts.append(np.array([p_val], dtype=float))
            y_parts.append(np.array([float(tail_series[-1])], dtype=float))

        p_out = np.concatenate(p_parts) if p_parts else np.empty((0,), dtype=float)
        y_out = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=float)

        meta = dict(self._sweep.meta)
        meta.update(var=self._var, mode="final")
        return BifurcationResult(
            param_name=self._sweep.param_name,
            values=self._sweep.values,
            mode="final",
            p=p_out,
            y=y_out,
            meta=meta,
        )

    def extrema(
        self,
        *,
        tail: int | None = None,
        kind: Literal["max", "min", "both"] = "both",
        max_points: int = 50,
        min_peak_distance: int = 1,
    ) -> BifurcationResult:
        """Extract local extrema (maxima, minima, or both)."""
        if tail is not None and tail <= 0:
            raise ValueError("tail must be positive")
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        if min_peak_distance <= 0:
            raise ValueError("min_peak_distance must be positive")
        if kind not in ("max", "min", "both"):
            raise ValueError("kind must be 'max', 'min', or 'both'")

        p_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for p_val, series in self._series_iter():
            if tail is None:
                tail_series = series
            else:
                tail_series = _extract_tail(series, int(tail))
            points = _extrema_from_tail(
                tail_series,
                kind=kind,
                max_points=int(max_points),
                min_peak_distance=int(min_peak_distance),
            )
            if points.size == 0:
                continue
            p_parts.append(np.full(points.shape, p_val, dtype=float))
            y_parts.append(points)

        p_out = np.concatenate(p_parts) if p_parts else np.empty((0,), dtype=float)
        y_out = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=float)

        meta = dict(self._sweep.meta)
        meta.update(
            var=self._var,
            mode="extrema",
            kind=kind,
            max_points=int(max_points),
            min_peak_distance=int(min_peak_distance),
        )
        if tail is not None:
            meta["tail"] = int(tail)
        return BifurcationResult(
            param_name=self._sweep.param_name,
            values=self._sweep.values,
            mode="extrema",
            p=p_out,
            y=y_out,
            meta=meta,
        )

    def poincare(
        self,
        *,
        section_var: str,
        level: float = 0.0,
        direction: str = "positive",
        tail: int | None = None,
        max_points: int = 50,
        min_section_distance: int = 1,
    ) -> BifurcationResult:
        """Extract a Poincare section for the chosen variable."""
        if section_var not in self._sweep.record_vars:
            raise KeyError(
                f"Unknown section_var {section_var!r}; available: {self._sweep.record_vars}"
            )
        if tail is not None and tail <= 0:
            raise ValueError("tail must be positive")
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        if min_section_distance <= 0:
            raise ValueError("min_section_distance must be positive")

        section_idx = self._sweep.record_vars.index(section_var)
        p_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for p_val, arr in self._array_iter():
            series = arr[:, self._var_idx]
            section = arr[:, section_idx]
            if tail is not None:
                series = _extract_tail(series, int(tail))
                section = _extract_tail(section, int(tail))
            points = _poincare_from_series(
                section,
                series,
                level=float(level),
                direction=direction,
                max_points=int(max_points),
                min_section_distance=int(min_section_distance),
            )
            if points.size == 0:
                continue
            p_parts.append(np.full(points.shape, p_val, dtype=float))
            y_parts.append(points)

        p_out = np.concatenate(p_parts) if p_parts else np.empty((0,), dtype=float)
        y_out = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=float)

        meta = dict(self._sweep.meta)
        meta.update(
            var=self._var,
            mode="poincare",
            section_var=section_var,
            level=float(level),
            direction=direction,
            max_points=int(max_points),
            min_section_distance=int(min_section_distance),
        )
        if tail is not None:
            meta["tail"] = int(tail)
        return BifurcationResult(
            param_name=self._sweep.param_name,
            values=self._sweep.values,
            mode="poincare",
            p=p_out,
            y=y_out,
            meta=meta,
        )
