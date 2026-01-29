# Simulation Configuration

This guide explains how to tune runtime defaults, adjust the model state/parameter bank, and control recording and caps when working with `Sim`.

## Persisting defaults with `Sim.config()`

`Sim.config()` lets you set *persistent* defaults that `run()` will use whenever a particular argument is unspecified. It covers the most common simulation knobs plus any stepper-specific `Config` fields.

```python
sim.config(
    dt=0.01,
    max_steps=5000,
    record=True,
    record_interval=10,
    cap_rec=2048,
    cap_evt=4,
    tol=1e-6,          # forwarded to the active stepper config
)
```

Highlights:

- `dt` sets the nominal time step (or label spacing for discrete models) and is stored on the simulation state as `_nominal_dt`. It must be positive.
- `max_steps` becomes the default safety limit (continuous) or target iteration count (discrete) whenever you omit `N`/`T`.
- `record` and `record_interval` define the default logging behavior and are inherited by future `run()` calls unless overridden.
- `cap_rec`/`cap_evt` control the initial sizes of the trajectory/event buffers. They grow automatically if needed, but larger initial caps can reduce reallocations.
- Any extra keyword arguments are forwarded to `Sim.stepper_config()` so you can configure `tol`, `max_iter`, or other stepper knobs globally.

Remember that explicit arguments to `run()` always override these defaults, so mix `config()` with `run(...)` overrides for reproducible scripts.

## Tweaking states and parameters with `Sim.assign()`

`Sim.assign()` updates the session’s current state and parameter vectors by name, without recompiling anything.

```python
sim.assign({"v": -65.0, "I": 5.0})
sim.assign(I=8.0, clear_history=True)
```

Key behaviors:

- Accepts a mapping and/or keyword arguments; keywords override map entries.
- Resolves names first against `states`, then `params`; unknown names raise a clear `ValueError` with “did you mean …?” suggestions.
- Casts inputs to the model dtype, emitting a warning if precision would be lost.
- `clear_history=True` wipes the accumulated `Results`, segments, and pending tags without altering time, workspace, snapshots, or stepper config.
- Changes take effect immediately for the next `run()` unless you pass explicit `ic`/`params` arguments to `run()`.

Use `assign()` when you want to reuse a `Sim` with new conditions, tweak parameters between experiments, or prime the system before resuming.

## Recording options

`Sim.run()` offers fine control over logging alongside the defaults you already set via `config()` or `[sim]` table.

- `record` (bool): turn recording on/off. When `False`, only the global time axis is updated; no state/aux buffers grow.
- `record_interval` (int): capture every N-th step (default `1`). Useful for downsampling or capturing fast simulations cheaply.
- `record_vars`: selective recording list. Acceptable entries:
  - `None` (default) : All available state variables.
  - Unprefixed names refer to states.
  - `"aux.<name>"` explicitly targets auxiliary variables, but posting `aux` names without the prefix is also accepted and disambiguated.
  - An empty list (`[]`) disables state/aux recording while still logging timestamps, steps, and flags.

Selective recording keeps the same `Results` buffer layout but only fills the requested subsets, which saves memory/time for large state vectors.

You can also adjust recording capacity before a run with `cap_rec` and `cap_evt`, or let dynlib grow the buffers automatically while `record_interval`/`record_vars` decide what gets captured.

## Managing simulation horizons

- `dt` records the nominal step size. Use `config(dt=…)` for a persistent default, or override per-run.
- `T` (continuous) or `N` (discrete) define how far the runner goes; if both are omitted, `max_steps` takes over (the default still respects `[sim].max_steps`).
- When working with maps (`kind="map"`), `N` determines iterations and `T` is derived; for ODEs, `T` is the end time and `N` is inferred.
- `max_steps` is enforced as a safety guard on continuous models and acts as the default iteration count when you omit `N` on discrete ones. Raise it when the horizon grows or shrink it to avoid runaway loops. If a runner reaches the `max_steps`, then a warning is raised to avoid silent unexpected behaviors.
- `transient` can skip recording for an initial warm-up period (either time or iterations) without influencing the stored `Results`. Nothing is recorded during the `transient` period. Beware that time is started from `t0` after the transient period, so this might be counter-intuitive for some users.
- `resume=True` allows restarting from the last `SessionState`; note that `ic`, `params`, `t0`, and `dt` cannot be overridden in resume mode.

Pair these options with `Sim.reset()`/`Sim.create_snapshot()` when orchestrating multi-stage experiments, so you retain control over segments and recorded history even as you tweak horizons.

## Summary

- Use `Sim.config()` to declare long-lived defaults for `dt`, `max_steps`, recording, capacities, and stepper tuning.
- Call `Sim.assign()` anytime you need to update state/parameter values without rebuilding the model.
- Leverage `run(record_vars=…)`, `record_interval`, and the buffer caps to trade off fidelity vs. memory.
- Adjust `T`, `N`, and `max_steps` depending on whether you simulate maps or continuous systems, and rely on `transient`/`resume` to control staging.
- Combine configuration options with snapshots, segments, and `setup()` when you need both reproducibility and speed.
