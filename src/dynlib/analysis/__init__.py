"""Analysis tools for dynamical systems.

This module provides:
- Parameter sweep utilities (scalar and trajectory)
- Trajectory analysis (statistics, extrema, crossings)
"""

from dynlib.analysis.sweep import (
    ParamSweepScalarResult,
    ParamSweepTrajResult,
    scalar,
    traj,
)
from dynlib.analysis.trajectory import (
    TrajectoryAnalyzer,
    MultiVarAnalyzer,
)

__all__ = [
    # Parameter sweeps
    "ParamSweepScalarResult",
    "ParamSweepTrajResult",
    "scalar",
    "traj",
    # Trajectory analysis
    "TrajectoryAnalyzer",
    "MultiVarAnalyzer",
]
