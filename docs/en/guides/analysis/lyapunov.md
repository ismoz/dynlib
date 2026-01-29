# Lyapunov analysis

Dynlib exposes dedicated analysis helpers for computing **maximum Lyapunov exponents (MLE)** and full **Lyapunov spectra**. These helpers are implemented as observer factories, so the bulk of the runtime wiring is already documented in the [observers guide](observers.md), but this page focuses on using `lyapunov_mle_observer` and `lyapunov_spectrum_observer` as standalone analysis tools (with or without additional observers).

## When to use Lyapunov diagnostics

Lyapunov exponents quantify how infinitesimal perturbations grow or contract along trajectories:

- **Positive MLE** signals sensitive dependence on initial conditions and chaotic regimes.
- **Zero or near-zero MLE** often indicates periodic or quasiperiodic motion.
- **Negative MLE** implies stable fixed points or sinks.

The spectrum generalizes the MLE by tracking every exponent, so it is useful whenever you want a fuller picture of stability (e.g., flows with multiple contracting/expanding directions or low-dimensional maps).

## Observer factories

Both factories live in `dynlib.runtime.observers` and share a consistent signature:

```python
from dynlib.runtime.observers import lyapunov_mle_observer, lyapunov_spectrum_observer

factory = lyapunov_mle_observer(model=sim.model, record_interval=record_every)
```

Common arguments:

- `model`: the active `Model` (can be passed explicitly or injected when using the factory directly).
- `record_interval` / `trace_plan`: determines how often convergence traces are stored; omit to only track the final outputs.
- `mode`: `"flow"`, `"map"`, or `"auto"` to control whether time or iteration counts appear in denominators (auto derives from `model.spec.kind`).
- `prefer_variational_combined`: reuse the stepper’s combined variational integrator when available; bypass this to fall back to the tangent-only path.
- `analysis_kind`: integer tag that flows through metadata builders and caches (set it to distinguish multiple analyses in the same run).
- `k` (spectrum only): how many leading exponents to compute; the first exponent is the MLE.

These factories declare their requirements (`need_jvp`, variational stepping, trace alignment) so the runner enforces compatible steppers just as described in the [observers guide](observers.md).

## Common workflow

1. **Register the observers** when calling `Sim.run(...)`.
2. **Request trace sampling** (set `record_interval` or a `FixedTracePlan`) if you want convergence data.
3. **Inspect `sim.results().observers[...]`** after the run for per-observer outputs/traces.

### Example snippet (logistic map)

The logistic map demo (`examples/analysis/lyapunov_logistic_map_demo.py`) ties everything together:

```python
sim.assign(x=0.4, r=4.0)
sim.run(
    N=5000,
    dt=1.0,
    record_interval=1,
    observers=[
        lyapunov_mle_observer(model=sim.model, record_interval=1),
        lyapunov_spectrum_observer(model=sim.model, k=1, record_interval=1),
    ],
)
result = sim.results()
```

Afterwards, the `ResultView` exposes ergonomic helpers such as `result.observers["lyapunov_mle"].mle` for the converged exponent and `result.observers["lyapunov_spectrum"]["lyap0"]` for the trace values.

## Reading outputs

Each Lyapunov observer writes:
- **Output slots** (`output_names`) for final statistics such as `mle`, `log_growth`, or `lyap0`.
- **Traces** (`trace_names`) when a trace plan is active.

Use the `ResultView.observers` dictionary to iterate over observers, and rely on `trace_steps`, `trace_time`, and direct indexing (`result["mle"]`) to align diagnostics with the recorded trajectory.

## Tips

- Combine `lyapunov_mle_observer` with other diagnostics (e.g., convergence tracers or event logs) as long as they share the same `TracePlan`.
- The `lyapunov_spectrum_observer` stores its exponents in alternating trace buffers, so filter out the zero entries when reading the last non-zero exponent. See the logistic map demo for a working extraction example.
- If you only need the final value (no trace), set `record=False` on `Sim.run` and skip the trace plan; the observers still deposit output buffers and metadata.
```
