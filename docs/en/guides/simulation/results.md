# Simulation Results

This guide dives into `Sim.results()` / `Sim.raw_results()` so you can understand what dynlib records, how to slice/filter/export results, and how to keep large simulations manageable.

## 1. Named access with `ResultsView`

`Sim.results()` returns a `ResultsView` that provides ergonomic access to simulation results with names derived from the model spec:

- `res.t`, `res.step`, `res.flags` give you the time axis, step indices, and status flags as NumPy views.
- `res["x"]`, `res["aux.energy"]`, or `res[["x","y"]]` return the recorded series for states and aux variables, stacking multi-variable requests into compact copies when necessary.
- `res.analyze(...)` builds a `TrajectoryAnalyzer` / `MultiVarAnalyzer` for quick statistics (max/min/crossings), and `res.observers` surfaces runtime observer outputs through the ergonomic `ObserverResult` wrapper.
- `res.segment` mirrors the main API while letting you focus on a single run (auto `run#N` names or manual tags). Each `SegmentView` slices `t`, `step`, `flags`, and even `events()` for that chunk without copying.

## 2. Raw access with: `Results`

For advanced users needing direct access to the underlying buffers, `Sim.raw_results()` hands you a `Results` dataclass that mirrors the runner buffers without copying. The key fields are the backing arrays (time `T`, states `Y`, optional aux `AUX`, `STEP`, `FLAGS`), the event log (`EVT_CODE`, `EVT_INDEX`, `EVT_LOG_DATA`), filled counts `n`/`m`, exit `status`, and snapshots of the final state/params/workspaces. Each accessor provides a view limited to the filled region so you always see contiguous records, and `Results.to_pandas()` can materialize the columns as a tidy `DataFrame` for downstream NumPy/Pandas workflows.

When you need the entire buffer, use `Sim.raw_results()`. For most users, `Sim.results()` wraps this low-level object with names, helpers, and segments.

## 3. Slicing, filtering, and exporting

- Treat `res["var"]` as your primary slicing hook; use `res[["x","y"]]` to stack multiple series and keep the natural ordering.
- For trajectory slices per segment, index `res.segment[0]` or `res.segment["run#1"]`. Each segment respects its recording window and exposes `events()` for the wrapped portion.
- When you prefer tabular exports, `Results.to_pandas()` gives `t`, `step`, `flag`, each state column, and prefixed aux columns so you can hand the frame to Pandas/NumPy directly.

### Event logging results access

Events are stored alongside the trajectory, and each event row carries a code, owning record index, and logged data blob. `ResultsView` resolves the DSL-defined names/tags so `res.event("threshold")` knows which code, fields, and tags to use. Due to NumPy's limitations on arbitrary row views, filtering (time ranges, head/tail, sorting) allocates compact arrays, but the API keeps the allocations isolated so the rest of the results stay view-only.

- Call `res.event("spike")` to get an `EventView`, then chain `.time(t0, t1)`, `.head(k)`, `.tail(k)`, or `.sort(by="t")` before grabbing individual fields with `ev["id"]` or `ev[["t","id"]]`. `ev.table()` materializes all logged columns in order.
- Use `res.event(tag="group")` for a grouped view over multiple event types; `group.select(...)` lets you union or intersect fields while `group.table(...)` can sort the combined rows.
- `res.event.summary()` gives quick counts per event type, and `res.event.names()/fields()/tags()` help you discover what is recorded.

## 4. Working with large datasets and external tools

- Control logging via `Sim.config()` and `run()` hooks: toggle `record`, jump every `record_interval` steps, or pass `record_vars`/`[]` to capture only what you need.
- Increase `cap_rec`/`cap_evt` to preallocate buffers, lower `record_interval` for downsampling, or disable state/aux logging entirely while still recording time/steps/flags.
- Use `transient`, `resume`, and snapshots (`Sim.reset()`, `Sim.create_snapshot()`) to manage staged experiments without overwhelming buffers.
- Export to NumPy/Pandas via the already exposed arrays (`res.t`, `res["x"]`, `.events()`, `.table()`) or `Results.to_pandas()` when you need a `DataFrame` with consistent column names.
- `res.segment[...]`, `res.event(...)`, and `res.observers` keep the slices you care about independent of the rest of the buffer so you can stream them into downstream analyzers without copying more than necessary.

## Summary

- Use `Sim.results()` when you want ergonomic names and helpers, and fall back to `Sim.raw_results()` when you need the raw buffer.
- Explore segments, events, and observers through `ResultsView` to extract slices, and use `Results.to_pandas()` or stacking with `res[["x","y"]]` to hand trajectories to NumPy/Pandas.
- Tune `record`, `record_interval`, `record_vars`, and buffer caps before long runs to keep memory usage stable, then replay or export the events/segments you care about.
