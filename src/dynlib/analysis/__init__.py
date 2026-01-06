"""Unified analysis namespace for dynlib (online + post-run)."""

import importlib
from typing import TYPE_CHECKING

from dynlib.analysis.runtime import (
    AnalysisHooks,
    AnalysisModule,
    AnalysisRequirements,
    CombinedAnalysis,
    TraceSpec,
    analysis_noop_hook,
    lyapunov_mle,
)

if TYPE_CHECKING:
    from dynlib.analysis.basin import (
        BLOWUP,
        OUTSIDE,
        UNRESOLVED,
        Attractor,
        BasinResult,
        basin_auto,
    )
    from dynlib.analysis.sweep import (
        SweepResult,
        TrajectoryPayload,
        scalar,
        traj,
        lyapunov_spectrum,
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
    "scalar",
    "traj",
    "lyapunov_mle",
    "lyapunov_spectrum",
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
    "basin_auto",
}

__all__ = [
    # Runtime analysis
    "AnalysisHooks",
    "AnalysisModule",
    "AnalysisRequirements",
    "CombinedAnalysis",
    "TraceSpec",
    "analysis_noop_hook",
    "lyapunov_mle",
    # Sweep orchestration
    *_SWEEP_EXPORTS,
    # Post-run analysis
    *_POST_EXPORTS,
    # Basin analysis
    *_BASIN_EXPORTS,
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
        module = importlib.import_module("dynlib.analysis.basin")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dynlib.analysis' has no attribute '{name}'")
