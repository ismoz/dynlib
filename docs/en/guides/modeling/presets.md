# Presets

Presets let you capture reusable sets of state and parameter values so you can quickly switch between "modes" of a model (e.g., fast vs. slow dynamics, resting vs. activated). Once defined, presets live in a Sim's in‑memory bank and can be applied, listed, saved, loaded, or created at runtime.

## Defining presets in the DSL

Inline presets are declared inside the model TOML using `[presets.<name>]` tables. Each preset may provide:

- `[presets.<name>.params]` for parameter overrides
- `[presets.<name>.states]` for state initial values

At least one of the two sections must exist, and every value must be a number (integers and floats are both accepted). A preset can omit states (parameter-only), omit params (state-only), or supply both. The declared names must match the model's `states` and `params`; invalid names are caught when the DSL is validated.

```toml
[presets.fast.params]
alpha = 2.5
beta = 0.1

[presets.fast.states]
x = 5.0
y = -1.0

[presets.rest.params]
alpha = 0.2
beta = 0.01
```

Inline presets are automatically loaded into each `Sim` instance during initialization. If a preset name appears more than once, the first definition wins and a warning is emitted.

## Working with the preset bank

Every `Sim` instance keeps a bank of presets, populated from inline definitions plus any added/loaded at runtime.

- `list_presets(pattern="*")` returns all matching names (supports `*`, `?`, `[]`) sorted alphabetically.
- `apply_preset(name)` updates only the params/states listed in the preset; time, dt, stepper workspace, step count, and recorded history remain untouched. Before applying, Dynlib validates that each key exists and casts the numeric values to the model dtype (warning if precision might be lost).

### Adding new presets on the fly

Use `add_preset(name, *, states=None, params=None, overwrite=False)` to snapshot the current session or to register custom values:

- If both `states` and `params` are `None`, the preset captures the current session's values.
- Each argument may be a mapping (`{"x": 1.0}`) or a 1‑D NumPy array (interpreted in declaration order), and may be partial (e.g., only a subset of states).
- The method raises `ValueError` if the name already exists unless `overwrite=True`, or if there is nothing to store.

### Persisting presets to disk

Dynlib can read/write presets using TOML files that follow `dynlib-presets-v1`. The file must contain:

```toml
[__presets__]
schema = "dynlib-presets-v1"

[presets.example.params]
a = 1.0
b = 2.0

[presets.example.states]
x = 0.0
```

- `load_preset(name_or_pattern, path, *, on_conflict="error")` imports presets from the file into the bank. You can pass an exact name or a glob pattern (e.g., `"fast_*"`). By default a conflict with an existing bank entry raises, but `"keep"`/`"replace"` let you skip or overwrite the bank entry (warnings highlight the action). The loader validates the schema header, enforces numeric tables, and ensures all referenced names exist in the active model.
- `save_preset(name, path, *, overwrite=False)` appends or writes a preset from the bank to disk. It creates or updates the `[__presets__]` header, leaves existing unrelated presets intact, and respects `overwrite` for name collisions inside the file.

Together these helpers make it easy to build curriculums of parameter/state sets, share them across projects, or export the state of a numerical experiment for later reuse.

## Example

```toml
[model]
type = "ode"

[states]
v = -65.0
u = -13.0

[params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0
I = 10.0
v_th = 30.0

[equations]
expr = """
dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
du = a * (b * v - u)
"""

[events.reset]
cond = "v >= v_th"
phase = "post"
action = """
v = c
u = u + d
"""

# PRESETS:
[presets.regular_spiking.params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0

[presets.intrinsic_bursting.params]
a = 0.02
b = 0.2
c = -55.0
d = 4.0

[presets.bursting.params]
a = 0.02
b = 0.2
c = -50
d = 2

[presets.fast_spiking.params]
a = 0.1
b = 0.2
c = -65
d = 2

[presets.low_threshold.params]
a = 0.02
b = 0.25
c = -65
d = 2

[presets.resonator.params]
a = 0.1
b = 0.26
c = -65
d = 2
```

In the simulation file you can choose one of the existing presets. TOML presets are added to the presets bank automatically.

See simulation for runtime usage of presets.