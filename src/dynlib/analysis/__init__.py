"""Unified analysis namespace for dynlib (online + post-run)."""

import importlib

from dynlib.analysis.runtime import (
    AnalysisHooks,
    AnalysisModule,
    AnalysisRequirements,
    CombinedAnalysis,
    TraceSpec,
    analysis_noop_hook,
    lyapunov_mle,
)

_SWEEP_EXPORTS = {
    "ParamSweepScalarResult",
    "ParamSweepTrajResult",
    "scalar",
    "traj",
}

_POST_EXPORTS = {
    "BifurcationResult",
    "BifurcationExtractor",
    "TrajectoryAnalyzer",
    "MultiVarAnalyzer",
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
    raise AttributeError(f"module 'dynlib.analysis' has no attribute '{name}'")
