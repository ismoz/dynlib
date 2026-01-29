# Session state & printing

`Sim` keeps a live `SessionState` so you can inspect or mutate the running session without tearing down the compiled model. This guide walks through the helpers that answer "what is currently stored in the session?", how to change it safely, and how to surface the underlying DSL equations when you need to print them. For more detailed inrospection see the `export_sources()` usage.

## Inspecting the session state

- **`session_state_summary()`** returns the current time (`t`), step count, nominal `dt`, stepper name, status, and whether `resume=True` is still allowed (`can_resume`/`reason`). It also includes the stored stepper config digest so you can detect whether a future `run()` call would reuse the same configuration.
- **`can_resume()`** and `compat_check()` let you verify whether the runtime pins (spec hash, stepper, workspace signature, dtype, dynlib version) match the active model before trying to continue a session.
- The summary is a great way to build dashboards or logging right after `run()` finishes or before constructing a `Snapshot`.

### Example

```python
summary = sim.session_state_summary()
print(f"Current time {summary['t']} (step {summary['step']})")
if not summary["can_resume"]:
    print("Resume unavailable:", summary["reason"])
```

## Mutating session values mid-flight

`Sim.assign(...)` updates states and parameters by name without changing time, the workspace, or results history (unless you explicitly clear it). The method infers names from the model, offers "did you mean" hints, coerces values to the model dtype, and throws if you try to mutate unknown variables.

- Pass a mapping or keywords: `sim.assign(v=-65.0, I=12.0)` keeps the same session time but tweaks the next run's initial values.
- Use `clear_history=True` to drop the accumulated results/segments while keeping the current `SessionState`. This is useful when you want to start a new recording without resetting to an earlier snapshot.

```python
sim.assign({"v": -70.0, "I": 10.0}, clear_history=True)
sim.run(T=1.0, record=True)
```

## Exporting and printing numeric values

Several helpers let you read state/parameter vectors or dictionaries from the session, the model spec, or a named snapshot:

- **`state_vector(source='session', copy=True)`** / **`param_vector(...)`** return 1‑D NumPy arrays in the DSL declaration order. `source` can be `"session"`, `"model"`, or `"snapshot"`, and `copy=False` gives a view into the underlying storage if you need to mutate values directly.
- **`state_dict(...)`** / **`param_dict(...)`** wrap the above arrays into `name -> float` maps for quick logging or JSON serialization.
- **`state(name)`** / **`param(name)`** read a single scalar from the current session with helpful suggestions when you mistype a name.

```python
print(sim.state_dict())  # session values as a dict
print(sim.param_vector(source="model"))  # model defaults as ndarray
snapshot_states = sim.state_vector(source="snapshot", snapshot="initial")

print(sim.state("v"))  # scalar lookup
```

These helpers are useful when producing debug prints halfway through a run, rendering UI panels, or exporting checkpoint metadata alongside a snapshot or preset.

## Printing the DSL equations

`FullModel.print_equations()` reflects the original DSL specification (not the generated runner) so you can include pretty-printed equations in docs or logs.

- `tables` selects which TOML tables to render: the default `"equations"` table shows the main dynamics, but you can pass other registered tables (`"equations.inverse"`, `"equations.jacobian"`, etc.) or use `tables="all"` to print everything.
- `include_headers` toggles whether section titles appear, and `file=` lets you redirect output to any writable stream.
- `FullModel.available_equation_tables()` lists all registered keys so you can validate what you can request.

```python
model.print_equations()  # prints the main equations with headers

with open("equations.txt", "w") as out:
    model.print_equations(tables="all", include_headers=False, file=out)

print(FullModel.available_equation_tables())
```

Combining `print_equations()` with `session_state_summary()` or `state_dict()` makes it easy to produce reproducible reports that tie the numeric state to the exact equations that generated it.
