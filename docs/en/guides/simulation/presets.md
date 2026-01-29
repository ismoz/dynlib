# Runtime presets

This guide focuses on how to work with the preset bank while a simulation is running. The DSL side of preset definitions lives in `docs/guides/modeling/presets.md`, but once a `FullModel` is compiled the `Sim` instance keeps its own in-memory cache of those presets plus anything you add or import later.

## Preset bank in `Sim`

Every `Sim` keeps a preset bank that is populated during initialization with the inline `[presets.<name>]` tables from the model spec. Use `list_presets(pattern="*")` to inspect what is currently available; it returns alphabetically sorted names and supports `glob`-style filters (`*`, `?`, `[]`).

```python
sim = Sim(model)
print(sim.list_presets())  # ['bursting', 'regular_spiking', ...]
```

The bank is shared across runs, snapshots, and resume segments, so you can switch between presets without rebuilding the model.

## Applying a preset before a run

Call `sim.apply_preset(name)` to push a preset’s states and/or parameters into the current session. Only the keys listed in the preset are updated—anything else (time, `dt`, stepper workspace, recorded history) is left untouched—so this is a safe, incremental way to reconfigure the session prior to `run()` or between segments.

```python
sim.apply_preset("bursting")
sim.run(T=2.0, record=True)
```

If you need to change a parameter that was not part of the preset, use `sim.assign(...)` after applying the preset, or create a new preset that includes the additional key. Applying a preset also works after `reset()`/`import_snapshot()` so you can branch off a saved state with a new combination of values.

## Capturing new presets on the fly

`sim.add_preset(name, *, states=None, params=None, overwrite=False)` registers a new entry into the preset bank.

- When both `states` and `params` are omitted, the method snapshots the current session values. Otherwise, provide mappings or 1-D NumPy arrays for the variables you want to store.
- Pass `overwrite=True` to replace an existing preset, otherwise a `ValueError` is raised to avoid clobbering.

```python
sim.assign(I=15.0)
sim.run(T=1.0)
sim.add_preset("after_stim", overwrite=True)  # captures the latest states and params
```

You can also create partial presets (e.g., only storing a subset of states) by passing one of the keyword arguments.

## Importing and exporting presets

Use `sim.load_preset(name_or_pattern, path, *, on_conflict="error")` to read presets from a TOML file that follows the `dynlib-presets-v1` schema (`[__presets__].schema = "dynlib-presets-v1"`). You can match a single preset name or pass a glob pattern like `"fast_*"`. The loader validates the file, enforces numeric tables, and makes sure every referenced state/parameter exists in the active model.

- `on_conflict="error"` (default) raises if the bank already contains the preset.
- `"keep"` skips the file preset and leaves the bank untouched (warning emitted).
- `"replace"` overwrites the bank entry with the file version (warning emitted).

```python
sim.load_preset("fast_*", "presets.toml", on_conflict="replace")
sim.apply_preset("fast_spiking")
sim.run(T=5.0)
```

Conversely, `sim.save_preset(name, path, *, overwrite=False)` writes a bank entry back to disk. The helper ensures a `[__presets__]` header exists, keeps unrelated presets in the file untouched, and either appends a new `[presets.<name>]` table or updates the one that already exists (if `overwrite=True`).

```python
sim.save_preset("after_stim", "presets.toml", overwrite=True)
```

Persisting presets lets you build experiment curricula, share configurations with colleagues, or version-control the exact states/parameter sets that produced a result.

## Tips for working with runtime presets

- When switching between presets inside a single workflow, `apply_preset()` is usually enough to reconfigure the session; you only need `reset()` or snapshots if you also want to rewind time or clear the recorder.
- Combine `add_preset()` and `save_preset()` to capture a replayable state and export it for other projects.
- Remember that presets only touch the variables they list—if you need to override every parameter or keep a fixed story of time, pair them with snapshots or `assign()` calls before running.
