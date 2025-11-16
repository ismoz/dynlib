# Stepper Workspace

The stepper workspace is a private, stepper-specific data structure that provides scratch space for numerical computations. Workspaces are owned by their respective components and cleanly separate stepper scratch from runtime state, which uses dedicated runtime workspace offsets and head slots for lag buffers.

## Stepper Workspace

Each stepper defines its own workspace as a `NamedTuple` containing NumPy arrays for temporary storage during integration steps. The workspace is allocated once per simulation and reused across steps.

### Key Characteristics

- **Ownership**: Private to each stepper instance
- **Lifetime**: Persistent across steps within a simulation run
- **Content**: NumPy arrays for stages, derivatives, histories, Jacobians, etc.
- **Allocation**: Determined by stepper's `make_workspace()` method
- **Type**: NamedTuple with descriptive field names

### Example Workspace Definitions

#### Euler Stepper
```python
class Workspace(NamedTuple):
    dy: np.ndarray  # RHS evaluation buffer (single evaluation)
```

#### RK4 Stepper
```python
class Workspace(NamedTuple):
    y_stage: np.ndarray  # Intermediate state for stages
    k1: np.ndarray      # Stage 1 derivative (RHS evaluation)
    k2: np.ndarray      # Stage 2 derivative (RHS evaluation)
    k3: np.ndarray      # Stage 3 derivative (RHS evaluation)
    k4: np.ndarray      # Stage 4 derivative (RHS evaluation)
```

#### RK45 Stepper (Adaptive)
```python
class Workspace(NamedTuple):
    y_stage: np.ndarray  # Intermediate state
    k1: np.ndarray      # Stage 1 (RHS evaluation)
    k2: np.ndarray      # Stage 2 (RHS evaluation)
    k3: np.ndarray      # Stage 3 (RHS evaluation)
    k4: np.ndarray      # Stage 4 (RHS evaluation)
    k5: np.ndarray      # Stage 5 (RHS evaluation)
    k6: np.ndarray      # Stage 6 (RHS evaluation)
    k7: np.ndarray      # Stage 7 (RHS evaluation)
```

### Workspace Allocation

Workspaces are allocated by the stepper's `make_workspace()` method:

```python
def make_workspace(
    self,
    n_state: int,
    dtype: np.dtype,
    model_spec=None,
) -> Workspace:
    # Allocate arrays based on n_state and dtype
    return self.Workspace(
        # ... arrays initialized to zeros ...
    )
```

### Why All Steppers Need RHS Arrays

All ODE steppers need arrays to store right-hand side (RHS) function evaluations. The RHS function `f(t, y)` computes the derivatives. Different integration methods require different numbers of RHS evaluations:

- **Euler**: 1 evaluation (`dy`) - simplest explicit method
- **RK4**: 4 evaluations (`k1`-`k4`) - 4th-order Runge-Kutta  
- **RK45**: 7 evaluations (`k1`-`k7`) - Dormand-Prince method with error estimation

The arrays store the derivative vectors `f(t, y)` at different points, which are then combined according to the method's formula.

## Runtime Workspace

The runtime workspace handles lag buffers and other DSL machinery state, separate from stepper scratch space.

### Structure

```python
RuntimeWorkspace = namedtuple(
    "RuntimeWorkspace",
    ["lag_ring", "lag_head", "lag_info"],
)
```

- **`lag_ring`**: Circular buffer for historical state values
- **`lag_head`**: Current head indices for each lagged state
- **`lag_info`**: Metadata for lag buffer layout (state_idx, depth, offset)

### Lag Buffer Access

Lag buffers are accessed via runtime workspace helpers in generated code:

```python
# Generated code accesses lags through runtime_ws
def rhs(t, y, dy, params, runtime_ws):
    # Access lagged values
    prev_x = lag_value(runtime_ws, state_idx=0, lag_steps=1)
    dy[0] = -prev_x  # Example: delayed feedback
```

## Workspace Design Benefits

The workspace architecture provides clean separation between stepper scratch space and runtime state:

- **Cleaner separation**: Stepper vs runtime responsibilities
- **Type safety**: NamedTuple fields instead of indexed arrays
- **Flexibility**: Each stepper defines its own workspace layout
- **Safety**: Dedicated runtime workspace offsets and head slots for lag buffers


## Stepper ABI

The stepper function signature now includes workspaces:

```python
status = stepper(
    t: float, dt: float,
    y_curr: float[:], rhs,
    params: float[:] | int[:],
    runtime_ws,        # Runtime workspace (lags)
    stepper_ws,        # Stepper workspace (scratch)
    stepper_config: float64[:],
    y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:]
) -> int32
```

## Workspace Serialization

Workspaces support snapshot/restore for simulation resume:

```python
# Capture workspace state
snapshot = snapshot_workspace(stepper_ws)

# Restore later
restore_workspace(stepper_ws, snapshot)
```

This enables resuming simulations with exact stepper state.