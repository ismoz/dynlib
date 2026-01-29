# Analysis guide

Dynlib’s analysis subsystem translates raw simulation data into scientific insights. The guides below describe the runtime observers, sweep helpers, fixed-point/manifold tools, basins, and post-processing utilities that turn `Sim` runs into bifurcation diagrams, Lyapunov spectra, or attractor families.

## Topics

- [Basins](basin.md) — automatic or known-attractor basin calculators (`basin_auto`, `basin_known`) with grid configuration, detection thresholds, and plotting tips.
- [Fixed points](fixed-points.md) — Newton solvers exposed via `find_fixed_points` and `FullModel.fixed_points`, complete with seed handling, classification, and metadata.
- [Runtime observers](observers.md) — the observer framework, pre/post-step hooks, trace buffers, and the `lyapunov_*` factories that plug directly into `Sim.run`.
- [Lyapunov analysis](lyapunov.md) — how to use the MLE and spectrum observers to quantify chaos, request traces, and read the emitted `Results` entries.
- [Sweep utilities](sweep.md) — scalar, trajectory, and Lyapunov sweeps that vary parameters, return `SweepResult` objects, and support fast-path or parallel execution before plotting.
- [Bifurcation diagrams](bifurcation.md) — build tracers from sweep outputs, use `BifurcationExtractor` helpers, and render scatter plots with `dynlib.plot.bifurcation_diagram`.
- [Post-analysis](post-analysis.md) — the `TrajectoryAnalyzer`/`MultiVarAnalyzer` helpers that summarize runs, compute crossings/extrema, and slice sweeps for further plotting.
- [Manifold analysis](manifold.md) — 1D manifold tracing, heteroclinic/homoclinic searchers, and the metadata you feed straight into `plot.manifold` or the plotting guide’s manifold pages.
