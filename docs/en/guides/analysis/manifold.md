# Manifold analysis

Dynlib currently supports **1D manifold tracing** for both discrete maps and ODEs, plus **search/trace utilities for heteroclinic and homoclinic orbits** of ODE models. Once you extract the manifolds (stable/unstable branches or connecting orbits) you can feed the results directly into `dynlib.plot.manifold` or the plotting guide in [docs/guides/plotting/manifold-plot.md](../plotting/manifold-plot.md).

## 1D manifold tracing

Two helpers live in `dynlib.analysis.manifold`:

- `trace_manifold_1d_map(...)` for autonomous maps whose stable or unstable subspace is 1D. Stable branches require the model to expose an analytic inverse map (`model.inv_rhs`), while unstable branches only need the forward map (`model.rhs`).
- `trace_manifold_1d_ode(...)` for ODE systems; it always uses an internal RK4 integrator (independent of the Sim stepper) and traces forward (unstable) or backward (stable) in time from an equilibrium point.

### Map manifolds (`trace_manifold_1d_map`)

Key arguments and workflow:

- `sim`, `fp`: supply a `Sim` whose compiled model is a map plus the equilibrium you want to expand (dict or array is accepted).
- `kind="stable"` or `"unstable"` picks the branch. Stable mode additionally requires `model.inv_rhs`.
- `params` override extra parameters, and `bounds` defines an `(n_state, 2)` observation box so tracing stops once the branch leaves it.
- `clip_margin` adds a fractional buffer when integrating; `seed_delta` perturbs the fixed point along the chosen eigenvector.
- `steps`, `hmax`, `max_points_per_segment`, and `max_segments` control how far the sampler walks and how many segments it stores.
- `eig_rank`, `strict_1d`, `eig_unit_tol`, and `eig_imag_tol` tune the eigenvalue selection so you can force a particular root or relax the strict-1D assumption.
- `jac="auto" | "fd" | "analytic"` picks the Jacobian strategy; `fd_eps` sets the finite-difference step.
- `fp_check_tol` optionally verifies that `fp` is still a fixed point at the provided parameters.

If Numba/JIT is available and the model was compiled with `jit=True`, the helper employs fast batch evaluation or preallocated fastpaths; otherwise it falls back to safe Python loops with a warning.

### ODE manifolds (`trace_manifold_1d_ode`)

Key knobs:

- `sim`, `fp`, `params`, and `bounds` work as above. The `bounds` box is respected during integration, and you can set `clip_margin` to buffer it while `strict_1d` ensures the selected eigenvector really spans a 1D manifold.
- `dt`, `max_time`, and `max_points` cap the internal RK4 integration. `resample_h` (if non-`None`) re-samples each branch to roughly equal arc-length spacing for cleaner plotting.
- `seed_delta` seeds the branch along the normalized eigenvector (both positive and negative directions are traced unless the branch leaves early).
- Jacobian handling mirrors the map helper (`jac`, `fd_eps`, `eig_real_tol`, `eig_imag_tol`). Use `eig_rank` when multiple stable/unstable eigenvalues exist.
- `fp_check_tol` lets you refuse to trace if `fp` is no longer a steady state (e.g., due to parameter overrides).

Like the map helper, the ODE tracer prefers a JIT-compiled model but runs even without Numba (with a warning about fallback).

### `ManifoldTraceResult`

Both tracing utilities return a `ManifoldTraceResult` with these attributes:

- `kind`: `"stable"` or `"unstable"`.
- `fixed_point`: the equilibrium that seeded the branches.
- `branches`: a tuple `(positive_side, negative_side)` where each side is a list of point sequences (`np.ndarray` of shape `(n_points, n_state)`).
- `branch_pos` / `branch_neg`: convenient views of the tuple above.
- `eigenvalue`, `eigenvector`, `eig_index`, `step_mul`: spectral information used during the trace.
- `meta`: dictionary recording the configuration that produced the result (bounds, params, dt, clip margins, etc.).

These results are directly consumable by `dynlib.plot.manifold` and expose `branches`, so you can overlay them with heteroclinic traces, time series, or other decorations (see [the plotting guide](../plotting/manifold-plot.md)).

Concrete examples:

- `examples/analysis/manifold_henon.py` traces the Henon map stable/unstable manifolds and renders them with `plot.manifold`.
- `examples/analysis/manifold_ode_saddle.py` walks through an analytic saddle, showing how to seed both sides, check the traced curve against closed-form expressions, and plot the result.

## Heteroclinic and homoclinic finder/tracer

For ODE models you can search for or trace connecting orbits without manually tweaking shooting segments. The workflow typically is:

1. Call a **finder** to locate a parameter value (and equilibrium pair) that yields a connection.
2. Use a **tracer** at the confirmed parameter to record the orbit.
3. Plot the resulting trace alongside the source/target manifolds using `dynlib.plot.manifold`.

### Heteroclinic utilities

- `heteroclinic_finder(...)` searches for a parameter `param` in `[param_min, param_max]` whose unstable manifold from `source_eq_guess` lands near the stable manifold of `target_eq_guess`. The simplified API accepts the `preset` (`"fast"`, `"default"`, `"precise"`), a `window` to constrain search, and convergence tolerances such as `gap_tol` (miss distance) and `x_tol` (parameter refinement). The return value, `HeteroclinicFinderResult`, contains:
  - `success`: whether a valid orbit was found.
  - `param_found`: the parameter value that minimized the miss distance.
  - `miss`: diagnostic struct (`HeteroclinicMissResult2D` or `HeteroclinicMissResultND`) with crossing points, gap metrics, and solver status.
  - `info`: auxiliary diagnostics (preset name, number of scans, etc.).
- After the finder succeeds, `heteroclinic_tracer(...)` records the actual connection at `param_value`. You must specify both equilibria (`source_eq`, `target_eq`) and an unstable direction sign (`sign_u`). The tracer exposes:
  - `HeteroclinicTraceResult` with fields `t`, `X`, `meta`, `branches`, and a boolean `success` property.
  - `hit_radius`: controls how close the unstable segment must get to the target before stopping (default `1e-2`).
  - The same `preset`/`window`/`t_max`/`r_blow` shortcuts plus the full `HeteroclinicBranchConfig` if you need finer control.

#### Configuration dataclasses

The finder/tracer pairs also accept structured dataclasses from `dynlib.analysis.manifold` when you need deterministic control. `heteroclinic_finder` can take a `cfg` (`HeteroclinicFinderConfig2D` or `HeteroclinicFinderConfigND`) while `heteroclinic_tracer` accepts `cfg_u`, and both can be overridden with the simplified `preset`, `trace_cfg`, and keyword arguments described above.

- `HeteroclinicRK45Config` tunes the internal RK45 integrator (`dt0`, `min_step`, `dt_max`, `atol`, `rtol`, `safety`, `max_steps`), which is stored on each branch config.
- `HeteroclinicBranchConfig` bundles the settings for a single manifold trace: equilibrium refinement (`eq_tol`, `eq_max_iter`, optional `eq_track_max_dist`), the leave radius (`eps_mode`, `eps`, `eps_min`, `eps_max`, `r_leave`, `t_leave_target`), integration caps (`t_max`, `s_max`, `r_blow`), optional sections/windowed exit conditions (`window_min`, `window_max`, `t_min_event`, `require_leave_before_event`), spectral tolerances (`eig_real_tol`, `eig_imag_tol`, `strict_1d`), Jacobian handling (`jac`, `fd_eps`), and the `rk` field that points to a `HeteroclinicRK45Config`.
- `HeteroclinicFinderConfig2D`/`HeteroclinicFinderConfigND` pair two branch configs (`trace_u`, `trace_s`) with search behavior (`scan_n`, `max_bisect`, `x_tol`, `gap_tol`, `gap_fac`, `branch_mode`, `sign_u`, `sign_s`, `r_sec`, `r_sec_mult`, `r_sec_min_mult`, plus `tau_min` for ND) and optionally `eq_tol`/`eq_max_iter` overrides. This full config is passed through `cfg` and bypasses the simplified kwargs when present.
- `HeteroclinicPreset` packages a branch config, RK settings, and scan parameters so you can request `"fast"`, `"default"`, or `"precise"` (or create a custom preset by instantiating the dataclass yourself) instead of manually setting every field.

See `examples/analysis/heteroclinic_finder_tracer.py` for a complete heteroclinic hunt + plotting routine.

### Homoclinic utilities

- `homoclinic_finder(...)` searches for a parameter such that the unstable and stable manifolds of the same saddle equilibrium reconnect. It accepts the same `preset` names and simplified overrides (`window`, `scan_n`, `max_bisect`, `gap_tol`, `x_tol`, `t_max`, `r_blow`, `r_sec`, `t_min_event`) or a full `HomoclinicFinderConfig`. The returned `HomoclinicFinderResult` mirrors the heteroclinic finder (with a `HomoclinicMissResult` describing the closest hit).
- `homoclinic_tracer(...)` follows a single sign-defined unstable branch until it lands back on the saddle; it returns a `HomoclinicTraceResult` whose `branches` attribute can be sent to `plot.manifold`. The tracer uses `HomoclinicBranchConfig`, allowing you to tweak RK45 tolerances, leave/return radii, and event detection guards.

#### Configuration dataclasses

The finder/tracer pair exposes structured configuration dataclasses for advanced tuning. `homoclinic_finder` accepts a `cfg: HomoclinicFinderConfig` while `homoclinic_tracer` can take `cfg_u: HomoclinicBranchConfig`; supplying these objects bypasses the simplified `preset`, `trace_cfg`, and keyword arguments.

- `HomoclinicRK45Config` owns the RK45 parameters (`dt0`, `min_step`, `dt_max`, `atol`, `rtol`, `safety`, `max_steps`).
- `HomoclinicBranchConfig` layers equilibrium refinement (`eq_tol`, `eq_max_iter`, `eq_track_max_dist`), leave-event control (`eps_mode`, `eps`, `eps_min`, `eps_max`, `r_leave`, `t_leave_target`, `r_sec`, `t_min_event`, `require_leave_before_event`), integration caps (`t_max`, `s_max`, `r_blow`), optional window constraints, spectral thresholds (`eig_real_tol`, `eig_imag_tol`, `strict_1d`), Jacobian handling (`jac`, `fd_eps`), and an `rk` field referencing a `HomoclinicRK45Config`.
- `HomoclinicFinderConfig` wraps a `trace` branch config with search-specific knobs (`scan_n`, `max_bisect`, `gap_tol`, `x_tol`, `branch_mode`, `sign_u`) so you can control both the tracing and parameter bisection behavior.
- `HomoclinicPreset` bundles a branch config, RK settings, and scan tolerances so `"fast"`, `"default"`, or `"precise"` can be passed directly as the `preset` argument; you can also instantiate your own preset if those defaults are not aggressive enough.

Preset Summary:

| Name | Description |
| --- | --- |
| `fast` | Quick scan with looser tolerances for exploration. |
| `default` | Balance of speed and robustness for standard use cases. |
| `precise` | Tight tolerances, smaller steps, and longer integrations for demanding orbits. |

Both finders/tracers log `diag` metadata in their result `meta` objects so you can inspect ODE step counts, RK adjustments, and event triggers if a run fails or only barely succeeds.

## Next steps

- Use `trace_manifold_1d_map` or `trace_manifold_1d_ode` once you know the target equilibrium and want to visualize its stable/unstable branches. Combine their `ManifoldTraceResult` with reference plots in `docs/guides/plotting/manifold-plot.md`.
- Run the heteroclinic/homoclinic finder scripts from `examples/analysis/{heteroclinic_,homoclinic_}finder_tracer.py` to hunt for parameter values, then trace the orbit at the discovered parameter to create publication-ready visuals.
