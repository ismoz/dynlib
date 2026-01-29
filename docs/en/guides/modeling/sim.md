# Simulation ([sim])

The `[sim]` table specifies default runtime knobs for any model. Most fields feed into the `Sim` facade (`Sim.run`) and the runner that executes integrations or maps. Think of this table as the “recommended defaults” for users and tools that load your model.

## Known keys

- `t0` – Initial time. Defaults to `0.0`. It seeds both the runtime clock and the derived `Nominal dt / T` arithmetic.
- `t_end` – End time for continuous (ODE-like) models. Defaults to `1.0`. The `Sim.run` façade uses it when you do not override `T`.
- `dt` – Nominal time step (or discrete map spacing). Defaults to `1e-2`. The runner caches this as the “nominal dt” used for both stepping and as the fallback when you omit `dt` in `Sim.run`.
- `stepper` – Name of the default stepper (e.g., `"rk4"` for ODEs, `"map"` for maps). The compiler chooses a sane default based on the model kind but you can override it here to pin a specific integrator.
- `record` – Boolean controlling the default recording behavior. Defaults to `true`. When `Sim.run` is called without `record`, this value decides whether state/aux samples are accumulated.
- `atol`, `rtol` – Adaptive-stepper tolerances (default `1e-8`/`1e-5`). They only apply when the configured stepper exposes adaptive control via its `Config` dataclass.
- `max_steps` – Maximum number of steps before the runner stops (default `1_000_000`). For discrete models it also serves as the default iteration count (`N`) when you do not supply `N` or `T`.
- `stop` – Early exit condition evaluated every step, usually in the `post` phase. You can write a simple string `stop = "x > threshold"` or a table:
  ```toml
  [sim.stop]
  cond = "max_energy > threshold"
  phase = "post"  # only "post" is currently supported
  ```
  When the condition is true the runner raises `EARLY_EXIT` and `Results.status` reflects that fact.
- **Extra keys** – Any other entries (anything besides the keys listed above) are forwarded into `SimDefaults._stepper_defaults` and automatically mapped onto the active stepper’s `Config` fields. This lets you pass `stepper-specific` defaults such as `tol`, `max_iter`, or any enum-typed option without duplicating the stepper name in code.

## Interaction with runs

1. **Run-time overrides win** – Every public `Sim.run` argument (`t0`, `T`/`N`, `dt`, `max_steps`, `record`, etc.) overrides the values in `[sim]`. This includes stepper kwargs passed via `**stepper_kwargs`, which take precedence over `[sim]` extras.
2. **Precedence chain** – Stepper config defaults come from: stepper class default < `[sim]` extra fields < `Sim.run(... stepper_kwargs...)`. This merging happens automatically through `ConfigMixin.default_config`, so you only need to declare the keys once.
3. **Discrete vs. continuous** – `Sim.run` interprets `[sim].t_end`/`dt` differently depending on the model kind (`map` vs `ode`). For maps, `[sim].max_steps` becomes the default iteration count when `N`/`T` are omitted; for ODEs, `t_end` is the default integration horizon.
4. **Stop condition evaluation** – When `[sim].stop` is present, the compiler wires it into the appropriate runner (pre/post according to `phase`). The condition shares the usual expression context (states, params, aux, functions, built-in macros) and is checked every committed step.
5. **Recording defaults** – `[sim].record` only sets the default; `Sim.run(record=False)` still disables logging for that specific invocation. If you use selective recording (`record_vars`), the `[sim]` table remains unchanged because selection is runtime-specific.

## Examples

```toml
[sim]
t0 = 0.0
t_end = 5.0
dt = 0.01
stepper = "rk4"
record = true
atol = 1e-9
rtol = 1e-6
max_steps = 500_000
tol = 1e-8        # extra field forwarded to the stepper config
stop = "energy > 100"  # triggers early exit
```

If the stepper exposes a `Config` with a field named `tol`, that value overrides its default but can still be overridden later via `Sim.run(t, dt, tol=explicit_value)`.

## Best practices

1. **Only add stepper-specific keys you need** so the list of extras stays focused.
2. **Document your early-exit condition** when you use `[sim].stop`—readers should know when and why the simulation aborts.
3. **Keep defaults consistent with physical time units** (seconds, iterations) so scripts using your model do not need to repeatedly override `t0`, `dt`, or `t_end`.
4. **Use `[sim]` mostly for safe defaults**; rely on `Sim.run` arguments when reproducing experiments or steering explorers.
