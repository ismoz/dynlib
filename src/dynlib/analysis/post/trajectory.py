"""Trajectory analysis tools for dynamical systems.

Provides statistical and temporal analysis of simulation results,
including extrema, crossings, oscillations, and summary statistics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from dynlib.runtime.results_api import ResultsView

__all__ = [
    "TrajectoryAnalyzer",
    "MultiVarAnalyzer",
]


def _validate_percentile(q: float) -> None:
    if not (0.0 <= q <= 100.0):
        raise ValueError(f"Percentile q must be in [0, 100]; got {q}")


class TrajectoryAnalyzer:
    """Analysis methods for a single trajectory variable.
    
    Provides statistical summaries, extrema detection, threshold crossings,
    and temporal analysis for a single state or auxiliary variable.
    
    Attributes:
        var: Variable name being analyzed
    """
    
    def __init__(
        self,
        results_view: "ResultsView",
        var_name: str,
        data: np.ndarray,
        time: np.ndarray
    ):
        """Initialize analyzer for a single variable.
        
        Args:
            results_view: Parent ResultsView instance
            var_name: Name of the variable
            data: 1D array of trajectory values (shape n)
            time: 1D array of time points (shape n)
        """
        self._results = results_view
        self._var = var_name
        self._data = data
        self._time = time
    
    @property
    def var(self) -> str:
        """Name of the variable being analyzed."""
        return self._var
    
    # ---- Basic statistics ----
    
    def max(self) -> float:
        """Maximum value in the trajectory."""
        return float(self._data.max())
    
    def min(self) -> float:
        """Minimum value in the trajectory."""
        return float(self._data.min())
    
    def mean(self) -> float:
        """Mean (average) value over the trajectory."""
        return float(self._data.mean())
    
    def std(self) -> float:
        """Standard deviation over the trajectory."""
        return float(self._data.std())
    
    def variance(self) -> float:
        """Variance over the trajectory."""
        return float(self._data.var())
    
    def median(self) -> float:
        """Median value of the trajectory."""
        return float(np.median(self._data))
    
    def percentile(self, q: float) -> float:
        """Compute percentile of the trajectory.
        
        Args:
            q: Percentile to compute (0-100)
            
        Returns:
            Value at the qth percentile
            
        Example:
            >>> analyzer.percentile(25)  # First quartile
            >>> analyzer.percentile(99)  # 99th percentile
        """
        _validate_percentile(q)
        return float(np.percentile(self._data, q))
    
    # ---- Extrema with timing ----
    
    def argmax(self) -> tuple[float, float]:
        """Find time and value of the maximum.
        
        Returns:
            (time, value) tuple where maximum occurs
            
        Example:
            >>> t_max, val_max = analyzer.argmax()
            >>> print(f"Peak of {val_max} at t={t_max}")
        """
        idx = int(self._data.argmax())
        return float(self._time[idx]), float(self._data[idx])
    
    def argmin(self) -> tuple[float, float]:
        """Find time and value of the minimum.
        
        Returns:
            (time, value) tuple where minimum occurs
        """
        idx = int(self._data.argmin())
        return float(self._time[idx]), float(self._data[idx])
    
    def range(self) -> float:
        """Range (max - min) of the trajectory.
        
        Returns:
            Difference between maximum and minimum values
        """
        return float(self._data.max() - self._data.min())
    
    # ---- Temporal analysis ----
    
    def initial(self) -> float:
        """First recorded value."""
        return float(self._data[0])
    
    def final(self) -> float:
        """Last recorded value."""
        return float(self._data[-1])
    
    def crossing_times(self, threshold: float, direction: str = "any") -> np.ndarray:
        """Find times when trajectory crosses a threshold.
        
        Args:
            threshold: Value to detect crossings of
            direction: "up" (below->above), "down" (above->below), or "any"
            
        Returns:
            Array of times where crossings occur (linear interpolation)
            
        Example:
            >>> # Find when x crosses 0 upward
            >>> t_cross = analyzer.crossing_times(0.0, direction="up")
        """
        vals = self._data
        above = vals >= threshold
        
        if direction == "up":
            # Find transitions from False to True
            crossings = np.where(above[1:] & ~above[:-1])[0]
        elif direction == "down":
            # Find transitions from True to False
            crossings = np.where(~above[1:] & above[:-1])[0]
        elif direction == "any":
            # Find any transition
            crossings = np.where(above[1:] != above[:-1])[0]
        else:
            raise ValueError(f"Invalid direction '{direction}'; use 'up', 'down', or 'any'")
        
        if len(crossings) == 0:
            return np.array([])
        
        # Linear interpolation to estimate exact crossing time
        times = []
        for i in crossings:
            v0, v1 = vals[i], vals[i + 1]
            t0, t1 = self._time[i], self._time[i + 1]
            # Interpolate: t = t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
            if v1 != v0:
                frac = (threshold - v0) / (v1 - v0)
                t_cross = t0 + frac * (t1 - t0)
                times.append(t_cross)
            else:
                times.append(t0)
        
        return np.array(times)
    
    def zero_crossings(self, direction: str = "any") -> np.ndarray:
        """Find times when trajectory crosses zero.
        
        Args:
            direction: "up", "down", or "any"
            
        Returns:
            Array of zero-crossing times
        """
        return self.crossing_times(0.0, direction=direction)
    
    def time_above(self, threshold: float) -> float:
        """Total time spent above a threshold.
        
        Args:
            threshold: Value to compare against
            
        Returns:
            Total duration where trajectory >= threshold
        """
        if len(self._time) < 2:
            return 0.0
        
        time_sum = 0.0
        for i in range(len(self._time) - 1):
            v0, v1 = self._data[i], self._data[i + 1]
            t0, t1 = self._time[i], self._time[i + 1]
            dt = float(t1 - t0)
            
            above0 = v0 >= threshold
            above1 = v1 >= threshold
            
            if above0 and above1:
                time_sum += dt
                continue
            if not above0 and not above1:
                continue
            
            # Linear interpolate the crossing to apportion the interval
            if v1 == v0:
                # Flat but straddling shouldn't happen; fall back to half interval
                time_sum += 0.5 * dt
                continue
            
            frac = (threshold - v0) / (v1 - v0)
            t_cross = t0 + frac * dt
            if above0:
                time_sum += float(t_cross - t0)
            else:
                time_sum += float(t1 - t_cross)
        
        return float(time_sum)
    
    def time_below(self, threshold: float) -> float:
        """Total time spent below a threshold.
        
        Args:
            threshold: Value to compare against
            
        Returns:
            Total duration where trajectory < threshold
        """
        total_time = float(self._time[-1] - self._time[0])
        return total_time - self.time_above(threshold)
    
    # ---- Summary ----
    
    def summary(self) -> dict[str, float]:
        """Compute comprehensive summary statistics.
        
        Returns:
            Dictionary with keys: min, max, mean, std, median, range, initial, final
            
        Example:
            >>> stats = analyzer.summary()
            >>> print(f"Mean: {stats['mean']}, Range: {stats['range']}")
        """
        return {
            "min": self.min(),
            "max": self.max(),
            "mean": self.mean(),
            "std": self.std(),
            "median": self.median(),
            "range": self.range(),
            "initial": self.initial(),
            "final": self.final(),
        }


class MultiVarAnalyzer:
    """Analysis methods for multiple trajectory variables.
    
    Provides the same analysis capabilities as TrajectoryAnalyzer,
    but operates on multiple variables simultaneously and returns
    results as dictionaries keyed by variable name.
    
    Attributes:
        vars: Tuple of variable names being analyzed
    """
    
    def __init__(
        self,
        results_view: "ResultsView",
        var_names: tuple[str, ...],
        data: np.ndarray,
        time: np.ndarray
    ):
        """Initialize analyzer for multiple variables.
        
        Args:
            results_view: Parent ResultsView instance
            var_names: Tuple of variable names
            data: 2D array of trajectory values (shape n × k)
            time: 1D array of time points (shape n)
        """
        self._results = results_view
        self._vars = var_names
        self._data = data
        self._time = time
        self._analyzer_cache: tuple[TrajectoryAnalyzer, ...] | None = None
    
    @property
    def vars(self) -> tuple[str, ...]:
        """Names of variables being analyzed."""
        return self._vars
    
    # ---- Basic statistics ----
    
    def max(self) -> dict[str, float]:
        """Maximum value for each variable."""
        return {var: float(self._data[:, i].max())
                for i, var in enumerate(self._vars)}
    
    def min(self) -> dict[str, float]:
        """Minimum value for each variable."""
        return {var: float(self._data[:, i].min())
                for i, var in enumerate(self._vars)}
    
    def mean(self) -> dict[str, float]:
        """Mean value for each variable."""
        return {var: float(self._data[:, i].mean())
                for i, var in enumerate(self._vars)}
    
    def std(self) -> dict[str, float]:
        """Standard deviation for each variable."""
        return {var: float(self._data[:, i].std())
                for i, var in enumerate(self._vars)}
    
    def variance(self) -> dict[str, float]:
        """Variance for each variable."""
        return {var: float(self._data[:, i].var())
                for i, var in enumerate(self._vars)}
    
    def median(self) -> dict[str, float]:
        """Median value for each variable."""
        return {var: float(np.median(self._data[:, i]))
                for i, var in enumerate(self._vars)}
    
    def percentile(self, q: float) -> dict[str, float]:
        """Compute percentile for each variable.
        
        Args:
            q: Percentile to compute (0-100)
            
        Returns:
            Dictionary mapping variable names to percentile values
        """
        _validate_percentile(q)
        return {var: float(np.percentile(self._data[:, i], q))
                for i, var in enumerate(self._vars)}
    
    # ---- Extrema with timing ----
    
    def argmax(self) -> dict[str, tuple[float, float]]:
        """Find time and value of maximum for each variable.
        
        Returns:
            Dictionary mapping variable names to (time, value) tuples
        """
        result = {}
        for i, var in enumerate(self._vars):
            col = self._data[:, i]
            idx = int(col.argmax())
            result[var] = (float(self._time[idx]), float(col[idx]))
        return result
    
    def argmin(self) -> dict[str, tuple[float, float]]:
        """Find time and value of minimum for each variable.
        
        Returns:
            Dictionary mapping variable names to (time, value) tuples
        """
        result = {}
        for i, var in enumerate(self._vars):
            col = self._data[:, i]
            idx = int(col.argmin())
            result[var] = (float(self._time[idx]), float(col[idx]))
        return result
    
    def range(self) -> dict[str, float]:
        """Range (max - min) for each variable."""
        return {var: float(self._data[:, i].max() - self._data[:, i].min())
                for i, var in enumerate(self._vars)}
    
    # ---- Temporal analysis ----
    
    def initial(self) -> dict[str, float]:
        """First recorded value for each variable."""
        return {var: float(self._data[0, i])
                for i, var in enumerate(self._vars)}
    
    def final(self) -> dict[str, float]:
        """Last recorded value for each variable."""
        return {var: float(self._data[-1, i])
                for i, var in enumerate(self._vars)}
    
    def crossing_times(self, threshold: float, direction: str = "any") -> dict[str, np.ndarray]:
        """Find crossing times for each variable.
        
        Args:
            threshold: Value to detect crossings of
            direction: "up", "down", or "any"
            
        Returns:
            Dictionary mapping variable names to arrays of crossing times
        """
        return {var: analyzer.crossing_times(threshold, direction)
                for var, analyzer in self._analyzers()}
    
    def zero_crossings(self, direction: str = "any") -> dict[str, np.ndarray]:
        """Find zero-crossing times for each variable."""
        return self.crossing_times(0.0, direction=direction)
    
    def time_above(self, threshold: float) -> dict[str, float]:
        """Total time spent above threshold for each variable."""
        return {var: analyzer.time_above(threshold)
                for var, analyzer in self._analyzers()}
    
    def time_below(self, threshold: float) -> dict[str, float]:
        """Total time spent below threshold for each variable."""
        return {var: analyzer.time_below(threshold)
                for var, analyzer in self._analyzers()}

    def _analyzers(self) -> tuple[tuple[str, TrajectoryAnalyzer], ...]:
        """Lazy cache of per-variable analyzers to avoid repeated construction."""
        if self._analyzer_cache is None:
            self._analyzer_cache = tuple(
                TrajectoryAnalyzer(self._results, var, self._data[:, i], self._time)
                for i, var in enumerate(self._vars)
            )
        return tuple(zip(self._vars, self._analyzer_cache))
    
    # ---- Summary ----
    
    def summary(self) -> dict[str, dict[str, float]]:
        """Compute comprehensive summary statistics for all variables.
        
        Returns:
            Nested dictionary: {var_name: {stat_name: value, ...}, ...}
            
        Example:
            >>> stats = analyzer.summary()
            >>> print(stats["x"]["mean"])
            >>> print(stats["y"]["max"])
        """
        return {var: {
            "min": float(self._data[:, i].min()),
            "max": float(self._data[:, i].max()),
            "mean": float(self._data[:, i].mean()),
            "std": float(self._data[:, i].std()),
            "median": float(np.median(self._data[:, i])),
            "range": float(self._data[:, i].max() - self._data[:, i].min()),
            "initial": float(self._data[0, i]),
            "final": float(self._data[-1, i]),
        } for i, var in enumerate(self._vars)}
