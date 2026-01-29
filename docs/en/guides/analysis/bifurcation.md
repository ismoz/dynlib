# Bifurcation diagrams

Bifurcation diagrams are produced as a **post-processing workflow** after you sweep a parameter: the sweep helpers in `dynlib.analysis.sweep` (e.g., `traj_sweep`, `lyapunov_mle_sweep`, `lyapunov_spectrum_sweep`) record trajectories or diagnostics for each grid value, and `SweepResult.bifurcation(var)` converts those trajectories into the scatter points that form the diagram.

## Workflow

1. **Run a sweep** with `dynlib.analysis.sweep.traj_sweep` (or another sweep helper) over the parameter you're interested in. Record the variables you want to plot and keep any transients you want to discard via the `transient`, `tail`, or extractor arguments.
2. **Extract bifurcation points** with `result.bifurcation("x")`, `result.bifurcation("z").extrema(...)`, etc. The `BifurcationExtractor` provides helpers such as `extrema(...)`, `tail(...)`, `final()`, and `poincare(...)` so you can build steady-state envelopes, maxima/minima, or return-section crossings.
3. **Plot the result** with `dynlib.plot.bifurcation_diagram(extractor)` or feed the extractor’s `p`/`y` arrays into your own figure routines. The extractor carries metadata (`param_name`, `values`, `meta`) that keeps axis labels and parameter grids consistent.

The `dynlib.plot.bifurcation_diagram` helper is demonstrated in `examples/bifurcation_logistic_map.py`, and bifurcation extraction is shown alongside Lyapunov sweeps in `examples/analysis/lyapunov_sweep_mle_demo.py` and `examples/analysis/lyapunov_sweep_spectrum_demo.py`.

## Notes

- The extractor works with trajectories of different lengths—the helper normalizes the data so you can stack or iterate over runs safely.
- Many sweep helpers fill `meta` with stepper settings, so you can see whether each point used the same `dt`, `record_interval`, etc.
- When building publication figures, combine bifurcation extractors with `series.plot`, `phase.xy`, or `fig.grid` for custom layouts.
