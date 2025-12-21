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

_POST_EXPORTS = {
    "ParamSweepScalarResult",
    "ParamSweepTrajResult",
    "scalar",
    "traj",
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
    # Post-run analysis
    *_POST_EXPORTS,
]


def __getattr__(name):
    if name in _POST_EXPORTS:
        module = importlib.import_module("dynlib.analysis.post")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dynlib.analysis' has no attribute '{name}'")
