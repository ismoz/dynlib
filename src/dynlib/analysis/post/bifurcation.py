# src/dynlib/analysis/post/bifurcation.py
"""Bifurcation post-processing utilities.

This module intentionally separates:
- Runtime: generating trajectories via ``dynlib.analysis.post.sweep.traj``
- Post-processing: extracting bifurcation scatter points from the trajectories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dynlib.analysis.sweep import ParamSweepTrajResult

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


def _peaks_from_tail(series: np.ndarray, *, max_peaks: int, min_peak_distance: int) -> np.ndarray:
    if series.size < 3:
        return np.array([], dtype=float)
    mid = series[1:-1]
    rising = series[:-2] < mid
    falling = mid >= series[2:]
    peak_mask = rising & falling
    idx = np.nonzero(peak_mask)[0] + 1  # offset because mid skips the first element
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
    trimmed = kept[-max_peaks:] if max_peaks > 0 else kept
    return series[np.array(trimmed, dtype=int)]


class BifurcationExtractor:
    """Post-process a trajectory sweep into bifurcation scatter points.
    
    Can be used directly (defaults to .all() mode) or call explicit methods
    like .tail(), .final(), or .peaks() for different extraction strategies.
    
    Examples:
        >>> from dynlib.plot import bifurcation_diagram
        >>> # Direct use (defaults to .all())
        >>> result = sweep_result.bifurcation("x")
        >>> bifurcation_diagram(result)  # Uses all points
        
        >>> # Explicit extraction mode
        >>> result = sweep_result.bifurcation("x").tail(50)
        >>> bifurcation_diagram(result)  # Uses last 50 points
    """

    def __init__(self, sweep_result: "ParamSweepTrajResult", var: str):
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
    def sweep(self) -> "ParamSweepTrajResult":
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

    def all(self) -> BifurcationResult:
        """Extract all recorded points (no filtering).
        
        Returns a bifurcation result containing every recorded data point
        from all parameter values. This is useful when you want to see the
        complete trajectory data without any post-processing.
        
        Returns:
            BifurcationResult with all recorded points
            
        Example:
            >>> res = sweep.traj(sim, param="r", values=r_values, record_vars=["x"], N=500)
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

    def peaks(
        self,
        *,
        tail: int = 200,
        max_peaks: int = 50,
        min_peak_distance: int = 1,
    ) -> BifurcationResult:
        if tail <= 0:
            raise ValueError("tail must be positive")
        if max_peaks <= 0:
            raise ValueError("max_peaks must be positive")
        if min_peak_distance <= 0:
            raise ValueError("min_peak_distance must be positive")

        p_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for p_val, series in self._series_iter():
            tail_series = _extract_tail(series, tail)
            peaks = _peaks_from_tail(
                tail_series,
                max_peaks=int(max_peaks),
                min_peak_distance=int(min_peak_distance),
            )
            if peaks.size == 0:
                continue
            p_parts.append(np.full(peaks.shape, p_val, dtype=float))
            y_parts.append(peaks)

        p_out = np.concatenate(p_parts) if p_parts else np.empty((0,), dtype=float)
        y_out = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=float)

        meta = dict(self._sweep.meta)
        meta.update(
            var=self._var,
            mode="peaks",
            tail=int(tail),
            max_peaks=int(max_peaks),
            min_peak_distance=int(min_peak_distance),
        )
        return BifurcationResult(
            param_name=self._sweep.param_name,
            values=self._sweep.values,
            mode="peaks",
            p=p_out,
            y=y_out,
            meta=meta,
        )
