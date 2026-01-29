# Fixed points and equilibria

Fixed points (for maps) or equilibria (for ODE models) are the state vectors where the right-hand-side of the model vanishes. Dynlib exposes two entry points for exploring them:

- the standalone `find_fixed_points(...)` helper inside `dynlib.analysis.fixed_points`, which operates on any callable RHS/jacobian pair, and
- the `FullModel.fixed_points(...)` convenience method that bridges the solver with a compiled model (parameter defaults, runtime workspace, timestep handling, and an optional analytic Jacobian).

Both routes drive the same Newton solver with configurable convergence criteria, deduplication, and stability diagnostics, but `FullModel.fixed_points(...)` is the preferred API for most users because it automatically wires up parameters, seeds, and Jacobians from the compiled model. They work both for maps and ODE models. To keep the API simple, the term fixed-point is used for the helpers.

## Quick Start

To find fixed points, you need:
- A model (for `FullModel.fixed_points`) or RHS function (for `find_fixed_points`).
- Initial guesses (seeds) near the expected fixed points.
- Optional parameters and configuration.

For a compiled model, use:

```python
from dynlib.analysis import FixedPointConfig

# Create or load your model
model = ...  # e.g., from dynlib import compile_model; model = compile_model('your_model.toml')

# Configure the solver
cfg = FixedPointConfig(tol=1e-10, classify=True)

# Find fixed points
result = model.fixed_points(
    params={"param_name": value},  # Optional: parameter values
    seeds=[[x1, y1], [x2, y2]],    # Initial guesses
    cfg=cfg
)

# Check results
print("Fixed points:", result.points)
print("Stability:", result.stability)
```

This returns a `FixedPointResult` with the solutions and diagnostics.

## `find_fixed_points(...)`

This helper solves `f(x, params) = 0` starting from one or more user-supplied seeds.

- **Function signature:** `f(x, params)` takes NumPy arrays and must return a vector with the same shape as `x`. If you provide `jac(x, params)`, its output is validated against `(n_state, n_state)` before solving.
- **Seed/pattern handling:** `seeds` accepts a single vector or a batch of shape `(n_seeds, n_state)`. The helper raises if the shape does not match. `params` must be a 1-D array; pass `None` to keep the default zero-length vector.
- **Newton settings:** `cfg` (see below) tunes tolerance, maximum iterations, finite-difference epsilon, and whether to perform eigenvalue-based classification. The solver returns as soon as the residual norm falls below `cfg.tol`.
- **Deduplication:** after solving every seed, the helper reports unique solutions. If `cfg.unique_tol` is positive, it merges nearby points (within the tolerance) and keeps the copy with the smallest residual. Setting `unique_tol` to `None` or a non-positive value disables deduplication.
- **Classification:** when classification is enabled via `cfg.classify`, the solver computes eigenvalues for each unique root (for maps it adds the identity to the Jacobian before eigendecomposition), then labels each point as `stable`, `unstable`, `neutral`, or `saddle` based on `cfg.stability_tol`.
- **Meta information:** every run records per-seed residuals, iteration counts, convergence flags, and mapping from seeds to unique points. `FixedPointResult.meta` exposes this data alongside the evaluated parameter vector and the `FixedPointConfig` used.

## `FixedPointConfig`

| Field | Description |
| --- | --- |
| `method` | Solver method name (`"newton"` only for now). |
| `tol` | Residual tolerance for convergence (default `1e-10`). |
| `max_iter` | Maximum Newton iterations per seed (default `50`). |
| `unique_tol` | Distance for merging duplicates (default `1e-6`). Set to `None`/≤0 to keep every converged seed. |
| `jac` | Jacobian mode: `"auto"` (use analytic if provided, otherwise finite diff), `"fd"`, or `"provided"` (requires a `jac` callable). |
| `fd_eps` | Step size for finite-difference Jacobians (default `1e-6`). |
| `classify` | Enable eigenvalue computation and stability labels (default `True`). |
| `kind` | Either `"ode"` or `"map"`—`find_fixed_points` enforces this value and uses it for classification. |
| `stability_tol` | Margin around the unit circle/imaginary axis where eigenvalues are considered neutral (default `1e-6`). |

## Interpreting `FixedPointResult`

The `FixedPointResult` object contains all the information about the found fixed points. Here's what each field means:

- `points`: A list of NumPy arrays, each representing a unique fixed point (state vector where the model RHS is zero). These are deduplicated based on `unique_tol`.
- `residuals`: The Euclidean norm of the RHS function at each point (should be very small, e.g., < 1e-10, for converged solutions).
- `jacobians`: The Jacobian matrices evaluated at each point (used for stability analysis). For maps, this includes the identity added (J + I).
- `eigvals`: Complex eigenvalues of the Jacobian (only if `classify=True`). Used to determine stability.
- `stability`: A list of strings labeling each point as 'stable', 'unstable', 'neutral', or 'saddle'. For ODEs, based on eigenvalue real parts; for maps, based on eigenvalue magnitudes relative to 1.
- `meta`: A dictionary with detailed diagnostics:
  - `seed_points`: The original seeds provided.
  - `seed_residuals`: Residuals after solving each seed.
  - `seed_converged`: Boolean flags indicating if each seed converged.
  - `seed_iterations`: Number of Newton iterations per seed.
  - `seed_to_unique`: Mapping from seed indices to unique point indices.
  - `unique_seed_indices`: Indices of seeds that produced unique points.
  - `params`: The parameter vector used.
  - `cfg`: The `FixedPointConfig` used.

If a seed doesn't converge (e.g., due to bad initial guess or model issues), check `meta['seed_converged']` and adjust seeds or tolerances.

## `FullModel.fixed_points(...)`

Calling this method on a compiled `FullModel` routes the solver through the model runtime so you get consistent defaults and Jacobians:

- `params`/`seeds` accept either sequences (applied directly) or mappings from names to overrides (which update the default parameter or state vectors).
- `method`, `tol`, `max_iter`, `unique_tol`, and `classify` mirror `FixedPointConfig` fields; pass a `cfg` object if you need multiple overrides at once.
- `jac` lets you request finite differences (`"fd"`), force analytic mode (`"analytic"`), or leave it `"auto"` (the default). When `jac='analytic'` the model must expose a Jacobian.
- `t` controls the evaluation time for non-autonomous problems; it defaults to `spec.sim.t0` so steady-state solutions match the simulation start.
- Internally the method builds a runtime workspace, evaluates the RHS (and Jacobian if available), subtracts the identity for maps, and then delegates to `find_fixed_points(...)`. It also updates `cfg.kind` from `spec.kind` before calling the helper so ODE vs. map behavior stays correct.

## Example

Here's a complete example using the logistic map model. Assume you have a compiled model for the logistic map equation: `x_{n+1} = r * x_n * (1 - x_n)`.

```python
from dynlib.analysis import FixedPointConfig

# Assuming 'model' is a compiled FullModel for the logistic map
# For r=3.8, expected fixed points are around 0.0 and 0.7368

top_cfg = FixedPointConfig(unique_tol=1e-8, classify=True)
result = model.fixed_points(
    params={"r": 3.8},  # Set parameter r
    seeds=[[-0.5, 0.2], [0.4, 0.4]],  # Initial guesses near expected points
    cfg=top_cfg,
)

# Inspect results
print("Number of fixed points found:", len(result.points))
print("Fixed points:")
for i, point in enumerate(result.points):
    print(f"  Point {i}: {point}")
print("Stability labels:", result.stability)
print("Residuals (should be near 0):", result.residuals)

# Example output (approximate):
# Number of fixed points found: 2
# Fixed points:
#   Point 0: [0.0]
#   Point 1: [0.73684211]
# Stability labels: ['unstable', 'stable']
# Residuals (should be near 0): [0.0, 1.11022302e-16]
```

This example shows how seeds converge to unique fixed points, with stability determined by eigenvalues. For maps, stability depends on whether the eigenvalue magnitude is less than 1. Use `result.meta` to trace convergence details per seed.

## Tips and Common Issues

- **Choosing seeds**: Start with educated guesses based on model behavior or prior simulations. For example, plot the vector field or run short trajectories to estimate where fixed points might be.
- **Convergence problems**: If seeds don't converge, try increasing `max_iter`, lowering `tol`, or using better initial guesses. Check `result.meta['seed_converged']` for failures.
- **Deduplication**: Set `unique_tol` to merge nearby points; set to `None` to keep all converged solutions.
- **Stability analysis**: Enable `classify=True` to get stability labels. For ODEs, stable points have all eigenvalues with negative real parts; for maps, magnitudes < 1.
- **Performance**: Providing an analytic Jacobian speeds up convergence and improves accuracy over finite differences.
- **Errors**: Ensure seeds match the state vector shape `(n_state,)`. For models, verify parameters are correctly named.

For more examples, see the analysis examples in the repository.
