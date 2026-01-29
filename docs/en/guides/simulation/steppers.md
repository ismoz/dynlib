# Steppers

`dynlib` uses steppers to advance one simulation step. They are mostly integrators but they are not called integrators or solvers because a map type stepper is used to advance discrete maps. Also ODE integrators / solvers might not be suitable for future dynamical system types. Choose the stepper that matches your problem class (ODE vs map), time-control strategy (fixed vs adaptive), and numerical scheme (explicit, implicit, splitting). You can override the model default via `build(..., stepper="rk4")` or, for the fast path in application code, `setup(..., stepper="rk4")`. Every compiled `Model` or `Sim` surface exposes `model.stepper_name` so you can confirm which integrator was selected after compilation.

## Choosing a Stepper

There are three axes you should always check before running a simulation:

- **Kind** (`Kind = "ode" | "map"`) describes the mathematical nature of the stepper. ODE steppers expect an RHS `f(t, y)` and multiply by `dt`, while the `map` stepper treats the compiled callable as a discrete update that already returns the next state (dt is only a label).
- **Time control** (`TimeCtrl = "fixed" | "adaptive"`) determines whether the integrator advances with a constant `dt` or internally retries and resizes the step. Adaptive steppers (RK45, BDF2a, TR-BDF2a) expose `atol/rtol` tolerant controls, while fixed-step steppers rely on the driver `Sim.run()` arguments.
- **Scheme** (`Scheme = "explicit" | "implicit" | "splitting"`) is the algebraic structure of the method. Explicit steppers have no nonlinear solves, whereas implicit steppers (SDIRK2, BDF2, BDF2a, TR-BDF2a) call Newton iterations and usually support optional analytic Jacobians. Splitting schemes will show up in the future.

The combination of these axes plus the `family`/`order` metadata in each `StepperMeta` gives you a concise view of what is physically happening. If you need Jacobians, dense output, or variational stepping for Lyapunov analysis, look at the `StepperCaps` block (in the docs we expose `dense_output`, `jacobian`, `jit_capable`, `requires_scipy`, and `variational_stepping` flags).

## Available Steppers

| Name      | Kind | Time control | Scheme | Order | Key notes |
| `map`     | map | fixed    | explicit | 1  | Discrete iterates (`F(t, y)` returns next state). `dt` only labels time. |
| `euler`   | ode | fixed    | explicit | 1  | Forward Euler, minimal workspace, variational-stepping capable. |
| `rk2`     | ode | fixed    | explicit | 2  | Explicit midpoint (RK2) with simple 2-stage update and variational support. |
| `rk4`     | ode | fixed    | explicit | 4  | Classic Runge–Kutta 4th order, alias `rk4_classic`, `classical_rk4`. |
| `rk45`    | ode | adaptive | explicit | 5¹ | Dormand–Prince RK45 with embedded order 4 error estimate. |
| `ab2`     | ode | fixed    | explicit | 2  | Adams–Bashforth 2 multistep with Heun startup, maintains derivative history. |
| `ab3`     | ode | fixed    | explicit | 3  | Adams–Bashforth 3 with a two-step startup that hands control over to the multistep loop. |
| `sdirk2`  | ode | fixed    | implicit | 2  | Alexander SDIRK2 (γ = (2−√2)/2), stiffly accurate but requires Jacobians. |
| `bdf2`    | ode | fixed    | implicit | 2  | Implicit BDF2 with Newton solver; optionally accepts external Jacobians. |
| `bdf2a`   | ode | adaptive | implicit | 2  | Variable-step BDF2 with error estimation. |
| `tr-bdf2a`| ode | adaptive | implicit | 2  | TR-BDF2 adaptive integrator (L-stable, BE partner) with the same config knobs as `bdf2a`. |

¹ Embedded order: 4 (error estimate). Adaptive steppers override `dt` internally but still report `dt_next` for the runner.

Each canonical stepper name is registered once; aliases such as `forward_euler`, `rk4_classic`, `trbdf2a`, and `sdirk2_jit` are automatically mapped to the same spec. Use the canonical name to avoid surprises when sharing configs or presets.

For maps you don't have to explicitly define `stepper=map`. For ODE models `rk4` is the default stepper.

## Stepper Registry & Discovery

The stepper registry is both user-facing and developer-facing. It is populated automatically when the stepper modules import, but you can also register custom specs with `dynlib.register()` if you need a specialized method.

```python
from dynlib import list_steppers, select_steppers, get_stepper

print(list_steppers(kind="ode"))
infos = select_steppers(scheme="implicit", stiff=True, jit_capable=True)
print([info.name for info in infos])
spec = get_stepper("rk45")
print(spec.meta.order, spec.meta.aliases)
```

`list_steppers()` returns sorted canonical names and accepts the same keyword filters as `select_steppers()` (kind, scheme, stiff, jit_capable, etc.). `select_steppers()` yields `StepperInfo` instances (aliased to `StepperMeta`), and you can also pass `name_pattern` or a custom `predicate` for fine-grained discovery (e.g., look for options that support variational stepping or dense-output).

The CLI mirrors the Python API: `dynlib steppers list` prints the same canonical names, and available flags mirror the keyword filters you saw above so you can narrow the output (for example by kind or scheme).

### Stepper metadata fields

- `name`: canonical stepper name (aliases resolve to this spec).
- `kind`: `ode` vs `map`.
- `time_control`: `fixed` or `adaptive`.
- `scheme`: `explicit`, `implicit`, or `splitting`.
- `geometry`: reserved set for geometry-aware methods (currently empty for built-in steppers).
- `family`: classification such as `runge-kutta`, `adams-bashforth`, `bdf`, `dirk`, `tr-bdf2`, or `iter`.
- `order`, `embedded_order`: describe the primary and embedded accuracy.
- `stiff`: indicates whether the method is intended for stiff problems.
- `aliases`: other names that map to the canonical spec.
- `caps`: see below.

### Stepper capability flags (`StepperCaps`)

- `dense_output`: supports continuous interpolation / dense output (currently False for built-ins).
- `jacobian`: `"none" | "internal" | "optional" | "required"` describes how the stepper consumes external Jacobians.
- `jit_capable`: true for all built-in steppers; false if the method relies on foreign dependencies.
- `requires_scipy`: true if SciPy is needed.
- `variational_stepping`: indicates support for `emit_step_with_variational()` (used in Lyapunov analysis).

The registry helpers (`list_steppers`, `select_steppers`) accept these metadata fields as filters. For example, `select_steppers(kind="ode", variational_stepping=True)` returns only ODE steppers that also implement the variational interface.

## Stepper Workspace

The stepper workspace is a private, stepper-specific scratch area that lives alongside the runtime metadata. Each stepper defines a NamedTuple that describes the arrays it needs during a single step. The workspace is allocated once per simulation via the stepper’s `make_workspace(n_state, dtype)` hook and reused for every step.

### Key characteristics

- **Ownership**: private to each stepper spec and passed as `stepper_ws` to the ABI.
- **Lifetime**: persistent until the `Sim` is destroyed or the workspace memory is explicitly freed.
- **Content**: NumPy arrays, e.g., stage buffers, derivative histories, Jacobian scratch, Newton guesses.
- **Allocation**: driven by the stepper’s `workspace_type()` and `make_workspace()`.
- **Type**: a `NamedTuple` so fields are accessed by name instead of by index.

### Example workspace layouts

#### Euler workspace
```python
class Workspace(NamedTuple):
    dy: np.ndarray
    kv: np.ndarray
```

#### RK4 workspace
```python
class Workspace(NamedTuple):
    y_stage: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    k3: np.ndarray
    k4: np.ndarray
    v_stage: np.ndarray
    kv1: np.ndarray
    kv2: np.ndarray
    kv3: np.ndarray
    kv4: np.ndarray
```

#### RK45 workspace (adaptive)
```python
class Workspace(NamedTuple):
    y_stage: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    k3: np.ndarray
    k4: np.ndarray
    k5: np.ndarray
    k6: np.ndarray
    k7: np.ndarray
```

### Workspace allocation

Steppers implement `make_workspace(n_state, dtype, model_spec=None)` to instantiate the NamedTuple and zero-initialize the arrays. The `workspace_type()` hook informs tooling (e.g., the compiler) how to pack the workspace into serialized snapshots, and `variational_workspace()` is available when the stepper supports Lyapunov/variational analysis.

### Why every stepper needs RHS arrays

Whatever the scheme, the stepper needs to store one or more evaluations of `f(t, y)` so it can combine them into a proposal. The workspace keeps these RHS buffers (`dy` for Euler, `k1..k4` for RK4, `f_prev/f_curr` for AB2/AB3, Newton residuals for implicit steppers). Every method writes the derivative vectors into its own slots, then assembles `y_prop`, `t_prop`, `dt_next`, and `err_est`.

## Runtime Workspace

The runtime workspace handles lag buffers and other DSL machinery state, separate from stepper scratch space.

### Structure

```python
RuntimeWorkspace = namedtuple(
    "RuntimeWorkspace",
    ["lag_ring", "lag_head", "lag_info"],
)
```

- `lag_ring`: circular buffer storing historical state snapshots.
- `lag_head`: current head indices used for each lagged state.
- `lag_info`: metadata describing the layout of each lag (state index, depth, offset).

Lag buffers are accessed with runtime helpers inside generated code, e.g.:

```python
prev_x = lag_value(runtime_ws, state_idx=0, lag_steps=1)
```

## Workspace design benefits

- **Separation of concerns**: stepper scratch stays isolated from runtime metadata.
- **Type safety**: NamedTuple workspaces carry descriptive field names instead of numeric offsets.
- **Flexibility**: each stepper shapes its workspace to exactly the buffers it needs.
- **Safety**: runtime workspaces reserve their own head slots, so lag buffers never interfere with stepper scratch.

## Stepper ABI

The compiled stepper callable follows a fixed ABI so the runner and results infrastructure can stay generic. The signature is:

```python
status = stepper(
    t: float,
    dt: float,
    y_curr: float[:],
    rhs,
    params: float[:] | int[:],
    runtime_ws,
    stepper_ws,
    stepper_config: float64[:],
    y_prop: float[:],
    t_prop: float[:],
    dt_next: float[:],
    err_est: float[:],
) -> int32
```

- `rhs`: compiled RHS function the stepper repeatedly invokes.
- `runtime_ws`: shared runtime workspace for lag buffers and metadata.
- `stepper_ws`: the Active stepper workspace described above.
- `stepper_config`: config array packed from the stepper’s dataclass (use `pack_config()`).
- `y_prop`, `t_prop`, `dt_next`, `err_est`: output buffers that the runner consumes after each call.

Adaptive steppers rewrite `dt_next` and `err_est` before the runner accepts the step, while fixed-step steppers usually write the current `dt + t` and zero error.

## Workspace Serialization

Stepper workspaces are serializable so you can snapshot a running simulation and restore it later. Use the helper functions that capture the workspace arrays when taking a snapshot:

```python
snapshot = snapshot_workspace(stepper_ws)
restore_workspace(stepper_ws, snapshot)
```

Snapshots capture both the runtime workspace and the stepper workspace so you can pause, rewind, or branch simulations with exact stepper state.

## Variational Stepping Overview

Some steppers support a combined state + tangent integration path that is useful for Lyapunov, sensitivity, or variational analyses. Look for `StepperSpec` implementations that set `StepperCaps(variational_stepping=True)`; they expose a `variational_workspace()` hook and `emit_step_with_variational()` method. Those facilities keep extra tangent buffers next to the state workspace, let you invoke the Jacobian-vector product (`jvp_fn`) in lockstep with the state RHS, and keep the tangent state synchronized with `y_prop`. When you need Lyapunov exponents or other variational diagnostics, choose one of these steppers (Euler, RK2, RK4, RK45, AB2, AB3, etc.) and call the variational stepping helpers rather than manually chaining the state and tangent integrators.

## Extending Steppers (Developer Guide)

For contributors who need a custom integrator or map, the `dynlib` stepper stack is intentionally modular. A new stepper spec simply implements the `StepperSpec` protocol, registers itself, and optionally exposes runtime configuration via `ConfigMixin`.

### 1. Define the metadata and capabilities

- Create a `StepperMeta` that describes the new method: canonical `name`, `kind`, `time_control`, `scheme`, `family`, `order`, `stiff`, and any `aliases`.
- Provide a `StepperCaps` instance that advertises optional behaviors (`dense_output`, `jacobian`, `jit_capable`, `requires_scipy`, `variational_stepping`). These flags are used by `select_steppers()`/`list_steppers()` and CLI filters, so set them truthfully for tooling discoverability.

### 2. Implement the spec (use `ConfigMixin` when you need runtime knobs)

```
class MyStepperSpec(ConfigMixin):
    @dataclass
    class Config:
        tol: float = 1e-6
        max_iter: int = 20
        __enums__ = {"method": {"foo": 0, "bar": 1}}

    def __init__(self, meta: StepperMeta | None = None):
        self.meta = meta or StepperMeta(
            name="my_stepper",
            kind="ode",
            time_control="fixed",
            scheme="explicit",
            family="custom",
            order=2,
            stiff=False,
            caps=StepperCaps(jit_capable=True, variational_stepping=True),
        )

    def workspace_type(self) -> type | None:
        return MyStepperWorkspace

    def make_workspace(...):
        ...

    def emit(...):
        ...
```

- `ConfigMixin` automatically implements `config_spec()`, `default_config()`, `pack_config()` and `config_enum_maps()` based on the nested `Config` dataclass, so you only need to declare the fields you care about.
- Inside `emit`, read `stepper_config` (packed float array) to apply runtime overrides, defaulting to `default_config(model_spec)` for values left unspecified by the user.

### 3. Workspace & combinatorial helpers

- Describe the workspace layout as a `NamedTuple` with NumPy arrays for stage buffers, histories, Jacobian scratch, or Newton residuals. `workspace_type()` exposes this layout, while `make_workspace(n_state, dtype)` zero-initializes the buffers.
- If the stepper supports Lyapunov/variational analysis, implement `variational_workspace()` to describe the analysis scratch (see `EulerSpec.variational_workspace()` for a pattern).

### 4. Register the spec

Register the stepper with the global registry so it becomes discoverable to `build()`, `setup()`, and `dynlib steppers list`.

```python
from dynlib import register

register(MyStepperSpec())
```

Call `register()` once (typically at module import time) using the canonical spec instance. Canonical names and aliases are deduplicated by the registry helpers.

### 5. Testing and CLI visibility

- Prepare tests like `test_<stepper_name>_basic.py` or add the stepper into the parameter list of existing tests like `test_ode_stepper_contract.py`.
- Update the CLI or presets if you want the stepper to appear in `dynlib steppers list` examples.

Following this flow keeps user-visible metadata, config, ABI, and workspace wiring aligned with the built-in steppers, making your custom integrator pluggable into the existing tooling.
