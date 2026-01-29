# Post-analysis

Post-processing is how you turn raw simulation or sweep results into the insights you care about: summary statistics, rise/fall times, threshold crossings, and bifurcation scatter clouds. Dynlib keeps the recorded trajectories in place, so you can line up analysis helpers with the same time axis you already plotted or exported.

## From `ResultsView` to trajectory analyzers

After a run you normally call `res = sim.results()` (see the [Simulation results guide](../simulation/results.md) for the full API). `res.analyze(...)` builds the analyzer you need:

- `res.analyze("x")` returns a `TrajectoryAnalyzer` for a single variable.
- `res.analyze(["x", "y"])` or any explicit tuple returns a `MultiVarAnalyzer` for the requested columns.
- `res.analyze()` with no arguments prefers the recorded states (falls back to recorded aux variables if no states were recorded) and returns `MultiVarAnalyzer`.

Both analyzers wrap the recorded NumPy views, so all statistics and temporal helpers operate on the same grid you see in `res.t`.

```python
res = sim.results()
xa = res.analyze("x")
peak_time, peak_value = xa.argmax()
summary = res.analyze().summary()  # dict of per-variable stats
```

`TrajectoryAnalyzer` exposes:

- basic stats: `min()`, `max()`, `mean()`, `std()`, `variance()`, `median()`, `percentile(q)`, `summary()`.
- extrema timing: `argmin()`, `argmax()`, `range()`.
- temporal helpers: `initial()`, `final()`, `crossing_times(threshold, direction)`, `zero_crossings(direction)`, `time_above(threshold)`, `time_below(threshold)`.

`MultiVarAnalyzer` mirrors the same methods but returns dictionaries keyed by variable name (and lazy-caches per-variable `TrajectoryAnalyzer` instances to avoid rebuilding). Use it whenever you want side-by-side stats for multiple recorded variables.

## Parameter sweeps → bifurcation data

When you run a sweep via `dynlib.analysis.sweep.traj_sweep(...)`, the returned `SweepResult` bundles the grid, per-run payload, and any recorded stats. Call `sweep_result.bifurcation("x")` to get a `BifurcationExtractor` for that variable; this helper lazily converts the trajectories into the scatter points needed for bifurcation diagrams.

```python
from dynlib.analysis import sweep
from dynlib.plot import bifurcation_diagram

sweep_result = sweep.traj_sweep(sim, param="r", values=r_values, record_vars=["x","y"], N=2000)
extrema = sweep_result.bifurcation("x").extrema(kind="max", tail=500, max_points=80)
bifurcation_diagram(extrema)
```

`BifurcationExtractor` behaves like a thin `BifurcationResult` plus some helpers:

- `.all()` (default mode if you pass the extractor straight to a plot helper) concatenates every recorded point so you can inspect the transient and steady-state data together.
- `.tail(n)` keeps the last `n` samples per parameter (useful when the limit cycle has stabilized).
- `.final()` only keeps the final sample for each sweep value, helping you spot equilibria and slow drifts.
- `.extrema(...)` detects maxima/minima (or both) within the (optional) tail, with parameters for `max_points` and `min_peak_distance` to avoid dense clusters.
- `.poincare(section_var=..., level=..., direction=..., tail=..., max_points=..., min_section_distance=...)` builds section crossings while interpolating crossing times and the corresponding value of the target variable.

Each method returns a `BifurcationResult` dataclass with:

- `param_name`: the swept parameter (used for automatic axis labeling).
- `values`: the full sweep grid (shape `(M,)`).
- `p`, `y`: flattened arrays of parameter/value pairs suitable for scatter plots.
- `mode`: which extraction strategy produced the data.
- `meta`: copy of `SweepResult.meta` plus the analyzer settings that generated the result.

Pass the result or extractor directly to `dynlib.plot.bifurcation_diagram()` to reuse axis labels and metadata, or feed the `.p`/`.y` arrays into your own plot utilities.

## Tips for reliable post-analysis

- Verify the variable name you analyze actually appears in `res.state_names`/`res.aux_names` or `SweepResult.record_vars`; otherwise `res.analyze(...)` or `sweep_result.bifurcation(...)` raises immediately.
- When you only care about the long-term behavior, trim transients with `.tail(n)` or slice `res.segment[...]` before analyzing.
- Use `MultiVarAnalyzer` summaries to compare variables (`mean()` returns `{"x": ..., "y": ...}`) and drill into individual components with `.crossing_times(...)` if a specific variable is oscillating.
- Combine the analyzer metadata with plot helpers to annotate findings (e.g., label the time of the global maximum using `xa.argmax()` and `res.t`).
