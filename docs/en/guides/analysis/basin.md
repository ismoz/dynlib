# Basin analysis

Basin-of-attraction analysis can be performed after setting up a simulation with a `Sim` object (see the [simulation guide](../simulation/index.md) for the basics of building models, steppers, and parameter assignments). Dynlib currently exposes two complementary basin calculators:

- `dynlib.analysis.basin_auto` automatically discovers attractors by watching how trajectories revisit quantized cells (PCR-BM).
- `dynlib.analysis.basin_known` matches initial conditions against fixed-point or reference-run attractors you already know, optionally via the reusable `build_known_attractors_psc` helper.

Both functions return a `BasinResult` (`labels`, `registry`, `meta`) plus the special label constants `BLOWUP`, `OUTSIDE`, and `UNRESOLVED` from `dynlib.analysis.basin`. The `labels` array preserves the order of the initial conditions you provided (grid flattening, parameter batches, or explicit `ic`), so you can reshape it back to the original layout before plotting.

## Automatic basin mapping (`basin_auto`)

`basin_auto` runs Persistent Cell-Recurrence Basin Mapping (PCR-BM). It quantizes the observation space, watches how often each trajectory revisits the same cell window, fingerprints newly discovered attractors, merges similar fingerprints (`s_merge`), and assigns the remaining trajectories via persistence (`p_in`).

Key knobs:

- **Initial conditions**: supply `ic` as a `(n_points, n_states)` array or ask `basin_auto` to build a uniform grid via `ic_grid` + `ic_bounds`.
- **Observation space**: `observe_vars` picks the variables to quantize, while `obs_min`/`obs_max` or the grid bounds define the detection region; `grid_res` controls spatial resolution.
- **Dynamics mode**: use `mode="map"` for discrete maps, `mode="ode"` for flows (or `auto` to infer); ODE mode requires a fixed-step stepper and an explicit `dt_obs` sampling interval.
- **Detection parameters**: `max_samples`, `window`, `u_th`, `recur_windows`, `post_detect_samples`, and `merge_downsample` tune the persistence scan and fingerprint merging. `transient_samples` lets you skip early transients, `b_max`/`blowup_vars` flag diverging trajectories, and `outside_limit` detects escapes from the observation region.
- **Execution controls**: `online=True` (default) streams analysis to keep memory in check; set `online=False` for offline debugging, but watch the `max_memory_bytes` guard. `parallel_mode`, `max_workers`, `batch_size`, `online_max_attr`, and `online_max_cells` trade off throughput vs. memory.

The returned `BasinResult.meta` records everything a plotting helper needs (`mode`, `observe_vars`, `grid_res`, `ic_grid`, `ic_bounds`, `dt_obs`, etc.), so you can pass the result straight to `dynlib.plot.basin_plot` (see the [basin plotting guide](../plotting/basin-plot.md)).

Example (Henon map):

```python
from dynlib import setup
from dynlib.analysis import basin_auto

sim = setup("builtin://map/henon", stepper="map", jit=True)
sim.assign(a=1.4, b=0.3)

result = basin_auto(
    sim,
    ic_grid=[200, 200],
    ic_bounds=[(-2.5, 2.5), (-2.5, 2.5)],
    grid_res=64,
    max_samples=600,
    window=64,
    u_th=0.5,
    mode="map",
)

labels = result.labels.reshape(200, 200)
```

## Targeted classification (`basin_known` + `build_known_attractors_psc`)

Use `basin_known` when you already understand the attractors in the basin (fixed points, limit cycles, strange attractors captured as reference runs). `basin_known` compares each trajectory against a `KnownAttractorLibrary` built from `FixedPoint` or `ReferenceRun` specs, so classification boils down to distance/tolerance checks plus optional escape/blowup detection.

Workflow notes:

- **Attractor specs**:
  - `FixedPoint(name, loc, radius)` fast-path: trajectories that stay within `radius` for `fixed_point_settle_steps` steps are immediately classified.
  - `ReferenceRun(name, ic, params)` captures a trajectory via `build_known_attractors_psc`, so matching is a similarity test (use `signature_samples`, `tolerance`, `min_match_ratio` to adjust strictness).
- **Grid handling**: as with `basin_auto`, pass `ic` or `ic_grid` + `ic_bounds`. With grids you can enable `refine=True` to do a coarse-to-fine pass (controls `coarse_factor`, `boundary_dilation`). The refinement benchmark in `examples/analysis/basin_refine_benchmark.py` shows how refine can speed up large grids.
- **Dynamics mode**: same `mode`/`dt_obs` rules apply (ODE mode requires fixed-step). `escape_bounds` guard against leaving the region, and `b_max`/`blowup_vars` detect blowups.
- **Parallel/execution**: `parallel_mode`, `max_workers`, and `batch_size` control concurrency. The `refine` path may spawn process pools or shared classifiers depending on your configuration.

If you prefer to reuse the attractor fingerprints, call `build_known_attractors_psc(sim, attractor_specs, ...)` once (the helper also runs `ReferenceRun` specs) and feed the returned `KnownAttractorLibrary` to downstream utilities that live outside `dynlib.analysis`.

Example (Duffing ODE fixed points):

```python
from dynlib import setup
from dynlib.analysis import basin_known, FixedPoint

sim = setup("builtin://ode/duffing", stepper="rk2", jit=True)
sim.assign(delta=0.02, alpha=-0.5, beta=0.5)

result = basin_known(
    sim,
    attractors=[
        FixedPoint(name="+1", loc=[1.0, 0.0], radius=0.3),
        FixedPoint(name="-1", loc=[-1.0, 0.0], radius=0.3),
    ],
    ic_grid=[300, 300],
    ic_bounds=[(-1.5, 1.5), (-1.5, 1.5)],
    dt_obs=0.01,
    max_samples=60000,
    signature_samples=0,
    escape_bounds=[(-2.0, 2.0), (-2.0, 2.0)],
    b_max=1e6,
)
```

## Inspecting basin results

`BasinResult.labels` carries the assignment for every initial condition, so you can reshape it and feed it to contour/`pcolormesh` helpers. The `registry` list holds `Attractor(id, fingerprint, cells)` entries and captures the discovered attractor metadata (useful for persistence debugging). `meta` includes algorithm parameters, grid metadata (`ic_grid`, `ic_bounds`), and the attractor names (for `basin_known`).

Use `dynlib.analysis.basin_stats(result)` or `basin_summary(result)` to get counts/percentages for each label and an attractor-by-attractor report. The plotting guide shows how `dynlib.plot.basin_plot(result)` combines labels, `result.meta`, and the color legend to reveal basins, escaping sets, and unresolved pockets in one figure.

### Quick summaries with `print_basin_summary`

When you need a fast console dump, `dynlib.analysis.print_basin_summary(result)` (see `examples/analysis/basin_henon_auto.py` and `examples/analysis/basin_henon_known.py`) prints attractor counts, the percentage of grid points assigned to each attractor, and the current label mix. It mirrors the structured `basin_summary` output but skips the data objects, making it ideal for iterating on grid resolutions or detection thresholds.

Because both calculators rely on Numba, there is no pure-Python `jit=False` fallback path for these helpers. Each entry point (via `dynlib.analysis.basin._require_numba`) raises `JITUnavailableError` whenever the soft dependency probe reports NumPy/Numba is missing, so install Numba and rebuild the extension before running large batches.
