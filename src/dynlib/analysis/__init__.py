"""Offline analysis namespace for dynlib."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynlib.analysis.basin import (
        BLOWUP,
        OUTSIDE,
        UNRESOLVED,
        Attractor,
        BasinResult,
        FixedPoint,
        ReferenceRun,
        KnownAttractorLibrary,
        build_known_attractors_psc,
    )
    from dynlib.analysis.basin_auto import basin_auto
    from dynlib.analysis.basin_known import basin_known
    from dynlib.analysis.basin_stats import (
        BasinStats,
        AttractorStats,
        basin_stats,
        basin_summary,
        print_basin_summary,
    )
    from dynlib.analysis.sweep import (
        SweepResult,
        TrajectoryPayload,
        scalar_sweep,
        traj_sweep,
        lyapunov_mle_sweep,
        lyapunov_spectrum_sweep,
    )
    from dynlib.analysis.post import (
        BifurcationResult,
        BifurcationExtractor,
        TrajectoryAnalyzer,
        MultiVarAnalyzer,
    )

_SWEEP_EXPORTS = {
    "SweepResult",
    "TrajectoryPayload",
    "scalar_sweep",
    "traj_sweep",
    "lyapunov_mle_sweep",
    "lyapunov_spectrum_sweep",
}

_POST_EXPORTS = {
    "BifurcationResult",
    "BifurcationExtractor",
    "TrajectoryAnalyzer",
    "MultiVarAnalyzer",
}

_BASIN_EXPORTS = {
    "BLOWUP",
    "OUTSIDE",
    "UNRESOLVED",
    "Attractor",
    "BasinResult",
    "FixedPoint",
    "ReferenceRun",
    "KnownAttractorLibrary",
    "basin_auto",
    "build_known_attractors_psc",
    "basin_known",
}

_BASIN_STATS_EXPORTS = {
    "BasinStats",
    "AttractorStats",
    "basin_stats",
    "basin_summary",
    "print_basin_summary",
}

__all__ = [
    # Sweep orchestration
    *_SWEEP_EXPORTS,
    # Post-run analysis
    *_POST_EXPORTS,
    # Basin analysis
    *_BASIN_EXPORTS,
    # Basin statistics
    *_BASIN_STATS_EXPORTS,
]


def __getattr__(name):
    if name in _SWEEP_EXPORTS:
        module = importlib.import_module("dynlib.analysis.sweep")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _POST_EXPORTS:
        module = importlib.import_module("dynlib.analysis.post")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _BASIN_EXPORTS:
        if name == "basin_auto":
            module = importlib.import_module("dynlib.analysis.basin_auto")
        elif name == "basin_known":
            module = importlib.import_module("dynlib.analysis.basin_known")
        else:
            module = importlib.import_module("dynlib.analysis.basin")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _BASIN_STATS_EXPORTS:
        module = importlib.import_module("dynlib.analysis.basin_stats")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dynlib.analysis' has no attribute '{name}'")
