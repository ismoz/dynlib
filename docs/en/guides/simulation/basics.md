# Simulation Basics

This guide walks through the two core phases of running a dynlib simulation:
1. **Compile** your model into a `FullModel`.
2. **Drive** that compiled artifact with a `Sim` instance.

It also highlights the shortcut `setup()` for quickly getting a simulation up and running.

## 1. Compile and Inspect a `FullModel`

Every dynlib simulation begins with `build()` (or, when you need more control, the lower-level compiler entry points). `build()` takes the model specification (URI, path, or inline DSL) along with optional steppers, mods, and JIT flags, and returns a `FullModel`.

```python
from dynlib import build

model = build("my_model.toml", stepper="rk4", jit=True)
```

A `FullModel` includes:

- Compiled callables (`rhs`, `stepper`, `runner`, `events_pre`, `events_post`, etc.).
- Metadata (`spec`, `stepper_name`, `workspace_sig`, dtype, simulation defaults, guards).
- Helper methods, such as `export_sources()` for extracting generated Python code, and `full_model.spec` for examining states, parameters, auxiliary variables, and `[sim]` defaults.

Since the compiled `model` is a standard Python object, you can inspect or reuse it before integrating it into a runtime. For example:

- Check `model.stepper_name` to confirm the chosen integrator.
- Examine `model.spec.states`, `model.spec.aux`, and `model.spec.params` to plan what to record.
- Re-export source code for debugging using `model.export_sources("./compiled")`.

Use `FullModel` directly only when you need to inspect or reuse compiled components. Most workflows pass the `FullModel` directly to `Sim`.

## 2. Run a Simulation with `Sim`

`Sim` wraps a `FullModel` and manages resumable session state, results buffers, snapshots, and preset banks.

```python
from dynlib import Sim

sim = Sim(model)
sim.config(record_interval=5, max_steps=2000)
sim.run(T=10.0, record=True)
results = sim.results()
```

Key runtime concepts:

- `run(...)` starts the simulation. You can override any `[sim]` defaults, such as `dt`, `T`/`N`, `record`, `record_interval`, `max_steps`, and selective recording via `record_vars`.
- `Sim.config(...)` sets persistent defaults to avoid repeating settings on every `run()`.
- `Sim.assign(...)` updates states/parameters or clears history before running.
- `Sim.results()` provides a `ResultsView` for named access, while `Sim.raw_results()` gives direct array views via the low-level `Results` buffer.
- `Sim.reset()` returns to a named snapshot (the `"initial"` snapshot is created automatically on the first run) and clears recorded history.
- `Sim.create_snapshot(...)`, `list_snapshots()`, and `name_segment()` enable control over reproducible segments for multiple scenarios.
- `run(resume=True)` continues from the current state, allowing seamless stitching of simulation segments.

`Sim` respects the `[sim]` table in the DSL, so simulation parameters have sensible defaults. `run()` only changes what you specify.

## 3. Quick Setup with `setup()`

For a fast way to compile and run, use the `setup()` helper. It combines `build()` and `Sim()` in one call, applies the same defaults, and provides access to the compiled model via `sim.model`.

```python
from dynlib import setup

sim = setup("my_model.toml", stepper="rk4", jit=True)
sim.run(T=10.0)
print(sim.results().t)
```

Since `setup()` gives you a ready-to-run `Sim`, it's the quickest way to start simulations. Reserve the explicit `build()` + `Sim()` approach for cases where you need to manipulate the `FullModel` first (e.g., inspecting guards, exporting sources, or caching for reuse).

## Workflow Tips

- Build once and reuse the `FullModel` for different scenarios or data shapes.
- Keep a `Sim` instance alive for snapshots, presets, or repeated configurations; resetting and reusing is more efficient than rebuilding.
- Rely on `[sim]` defaults for most settings, using `Sim.config()` and targeted `run()` overrides as needed.
- Use `setup()` for rapid prototyping, testing, or demos.
- Enable `jit=True` only when numba is installed and simulations are long enough to justify compilation overhead. For short runs, use `jit=False` to stick with the interpreter.
