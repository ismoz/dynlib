# Sweep analysis utilities

Dynlib keeps a dedicated `dynlib.analysis.sweep` module so you can explore how system behavior changes across a parameter grid without duplicating bootstrapping logic or manually iterating runs. Each helper reads the *current simulation session* (`sim.state_vector(source="session")` and `sim.param_vector(source="session")`) as the baseline, varies one parameter, and returns a `SweepResult` whose contents match what you recorded.

## Core result helpers

### `SweepResult`
`SweepResult` normalizes the metadata you expect (`param_name`, `values`, `kind`, `meta`) and exposes the actual data through mapping and attribute access. Scalar sweeps populate `outputs`, Lyapunov sweeps add `traces`, and trajectory sweeps attach a `TrajectoryPayload` in `payload`. The helper re-exports recorded variable names (`record_vars`), provides `t`/`t_all`, and gives you convenience helpers such as `.runs` (per-value `SweepRun` objects) or `.stack()` when all trajectories share the same length. Asking for a missing key raises immediately and lists the available fields so typos fail fast.

### `TrajectoryPayload`, `SweepRun`, and `SweepRunsView`
`TrajectoryPayload` keeps the tuple of recorded names, every run’s `t_runs`/`data` arrays, and the sweep grid `values`. It also builds an internal `_var_index`, which powers named access (`payload["x"]`, `payload.series([...])`) and the ability to stack (`payload.stack()`) or grab the default time axis (`payload.t`). When different runs log different lengths (adaptive steppers, `record_interval`, etc.), `t_all` and `data` remain aligned so you can handle each run manually.

`SweepRunsView` is a list-like wrapper over the run values; iterating it yields `SweepRun` instances. Each `SweepRun` stores `param_value`, the per-run `t`, and a variable lookup table so `run["x"]` or `run[["x","y"]]` returns the recorded trace for that parameter value.

## Sweep helpers

### `scalar_sweep`
Use `scalar_sweep` for one-number summaries per parameter value (equilibria, mean trends, min/max envelopes). You record a single `var` and pick a reduction `mode`:

- `"final"` (default): the last recorded sample
- `"mean"`: arithmetic average over the logged window
- `"max"`/`"min"`: extreme values

The helper tries a fast-path batch run (`fastpath_batch_for_sim`) for eligible configurations. If the fast path is unavailable it warns (`_warn_fastpath_fallback`) and falls back to normal `Sim.run()` calls. Result is a `SweepResult(kind="scalar")` where `outputs['y']` holds the reduced array (shape `(M,)`) and `meta` records the integration settings, stepper kind, and reduction mode.

```python
from dynlib.plot import series

res = sweep.scalar_sweep(
    sim,
    param="r",
    values=np.linspace(2.5, 4.0, 4000),
    var="x",
    mode="final",
    N=2000,
    transient=1000,
)
series.plot(x=res.values, y=res.y, xlabel="r", ylabel="x*")
```

### `traj_sweep`
`traj_sweep` records full trajectories for any combination of `record_vars` (e.g., `"x"`, `"y"`, `"z"`). Each run’s time series lives in a `TrajectoryPayload`, so you can call `res["x"]`, `res.series(["x","y"])`, `res.stack()`, or iterate `res.runs` for per-parameter plotting. The sweep supports both fast-path batch execution and a `ProcessPoolExecutor` when `values` > 1000. The `parallel_mode` argument controls how that batch run is executed (`"auto"`, `"threads"`, `"process"`, `"none"`); `max_workers` tunes the worker pool size. When number of workers resolves to one or `process` mode isn’t efficient the helper transparently downgrades to sequential execution.

`record_interval` allows you to decimate logging for memory savings, and the sweep remembers that interval in `meta`. You can also request a fixed `dt`, `t0`, `T`, or discrete iteration count `N` (useful for maps).

```python
from dynlib.plot import phase

res = sweep.traj_sweep(
    sim,
    param="A",
    values=[0.5, 1.0, 1.5],
    record_vars=["x", "y"],
    dt=0.01,
    T=20.0,
    record_interval=5,
)
for run in res.runs:
    phase.xy(x=run["x"], y=run["y"], label=f"A={run.param_value}")
```

### `lyapunov_mle_sweep`
This helper couples a parameter sweep with the maximum Lyapunov exponent (MLE) observer. For the fast-path batch execution (and the Lyapunov observers themselves) you should use a JIT-compiled sim with a fixed-step stepper and an explicit `dt`, but the helper gracefully falls back to sequential `Sim.run()` with observers attached if fast path support is unavailable. The function returns `outputs` for `mle`, `log_growth`, and `steps`, and if you provided `record_interval` it also returns `traces['mle']` (list of convergence arrays) so you can inspect how each exponent converged. `analysis_kind` lets you choose between algorithm variants.

The sweep attempts fast-path batch runs with optional `ProcessPoolExecutor` acceleration (chunks of the values list). If the fast path or fast parallel worker initialization fails it warns and falls back to sequential `Sim.run()` calls with the Lyapunov observer attached.

```python
from dynlib.plot import series

res = sweep.lyapunov_mle_sweep(
    sim,
    param="r",
    values=np.linspace(3.0, 4.0, 400),
    N=5000,
    transient=1000,
    record_interval=10,
)
series.plot(x=res.values, y=res.mle, xlabel="r", ylabel="λ_max")
```

### `lyapunov_spectrum_sweep`
Compute the first `k` Lyapunov exponents across a parameter grid. Like the MLE sweep, it is tuned for JIT + fixed `dt` fast-path execution (and accepts an optional `init_basis` for the tangent space), but it also falls back to sequential `Sim.run()` with the observer attached whenever the batch fast path or process parallelism is unavailable. The `outputs` dictionary always contains:

- `spectrum`: array of shape `(M, k)` with the normalized exponents
- `log_r`: the raw logarithmic growth values (shape `(M, k)`)
- `steps`: final number of algorithm steps per value
- `lyap0`, `lyap1`, … `lyap{k-1}`: convenient aliases for each exponent column

There are no `traces`, because the underlying observer only emits the latest spectrum. Use `parallel_mode`/`max_workers` and `record_interval` just like the MLE sweep; the helper also falls back to sequential execution when necessary.

## Practical notes

- If the fast path is closed (`fastpath_batch_for_sim` returns `None`) you’ll see a warning that the sweep is falling back to `Sim.run()`. Providing `jit=True`, using fixed-step steppers, and recording explicit `dt`/`N` values keeps the fast path healthy.
- For trajectory sweeps, `record_interval` and `max_steps` let you trade resolution for memory/cpu. Trajectories are stored exactly as produced, so you can re-use them with `dynlib.analysis.post.bifurcation.BifurcationExtractor` via `res.bifurcation("x")` to generate scatter clouds, extrema, or trimmed sample sets.
- Lyapunov helpers accept `analysis_kind` (default `1`) so you can pick the variant that best matches your system. `max_workers` defaults to machine cores (capped at 8) via `_resolve_process_workers`.
- See `examples/bifurcation_logistic_map.py` for an end-to-end script that runs `traj_sweep`, extracts `res.bifurcation("x")`, and feeds the result into `dynlib.plot.bifurcation_diagram`.
- All sweeps return `meta` metadata that includes stepper settings, timestamps, and any parallel-run configuration so you can trace how the data was generated.
