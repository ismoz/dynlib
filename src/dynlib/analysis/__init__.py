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
from dynlib.analysis.bifurcation import (
    BifurcationResult,
    BifurcationExtractor,
)

__all__ = [
    # Parameter sweeps
    "ParamSweepScalarResult",
    "ParamSweepTrajResult",
    "scalar",
    "traj",
    # Bifurcation diagrams
    "BifurcationResult",
    "BifurcationExtractor",
    # Trajectory analysis
    "TrajectoryAnalyzer",
    "MultiVarAnalyzer",
]
