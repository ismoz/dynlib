# Lag System Design & Implementation

## Overview

The lag system provides access to historical state values in dynlib models using the notation:
- `lag_<name>()` - Access state `<name>` from one step ago
- `lag_<name>(k)` - Access state `<name>` from k steps ago

**Key Features:**
- ✅ On-demand activation (only lagged states consume memory)
- ✅ O(1) circular buffer access (Numba-compatible)
- ✅ Counted after successful committed steps (immune to buffer growth, early breaks, resume)
- ✅ Works with both ODE and map models
- ✅ Dedicated runtime workspace (stepper ABI extended with runtime_ws parameter)

---

## DSL Syntax

### Supported Usage

```toml
[model]
type = "map"

[states]
x = 0.1

[params]
r = 3.5
alpha = 0.3

[equations.rhs]
# Mix current and lagged states
x = "r * (alpha * x + (1 - alpha) * lag_x(1)) * (1 - x)"

# Use zero-arg shorthand for one step back
x = "r * (alpha * x + (1 - alpha) * lag_x()) * (1 - x)"
```

### Lag Depths

```toml
[aux]
# Use multiple lag depths - max is automatically detected
delayed_diff = "x - lag_x(5)"

[equations.rhs]
x = "v + 0.1 * lag_x(2)"  # Max lag for x = 5 (from aux)
v = "-x - lag_v(3)"       # Max lag for v = 3
```

### Restrictions

**Only for state variables:**
```toml
[states]
x = 0.1

[params]
a = 2.0

[equations.rhs]
x = "lag_x(1)"   # ✅ Valid - x is a state
x = "lag_a(1)"   # ❌ ERROR - a is a parameter, not a state
```

**Lag argument must be integer literal:**
```toml
x = "lag_x(2)"       # ✅ Valid
x = "lag_x(k)"       # ❌ ERROR - k is not a literal
x = "lag_x(2 + 1)"   # ❌ ERROR - expression not allowed
```

**Sanity limit:**
```toml
x = "lag_x(1000)"    # ❌ ERROR - exceeds sanity limit (1000)
```

---

## Lagging Auxiliary Variables

**Auxiliary variables CANNOT be lagged directly.** Instead, use lagged states in expressions:

### ❌ **NOT Supported:**
```toml
[aux]
energy = "0.5 * v^2 + 0.5 * k * x^2"

[equations.rhs]
v = "-x - 0.1 * lag_energy(1)"  # ERROR: energy is aux, not state
```

### ✅ **Correct Approach:**
```toml
[aux]
energy = "0.5 * v^2 + 0.5 * k * x^2"  # Current energy (optional)

[equations.rhs]
# Compute lagged energy from lagged states
v = "-x - 0.1 * (0.5 * lag_v(1)^2 + 0.5 * k * lag_x(1)^2)"
```

**Rationale:** Auxiliaries are ephemeral derived quantities. Lagging `energy` is mathematically equivalent to computing `energy` from lagged states.

### Alternative: Promote Aux to State

If a derived quantity is frequently lagged:

```toml
[states]
x = 0.1
v = 0.0
energy = 0.005  # Promoted from aux

[params]
k = 2.0

[equations.rhs]
x = "v"
v = "-k * x - 0.1 * lag_energy(1)"  # Clean access
energy = "0.5 * v^2 + 0.5 * k * x^2"  # Track as ODE
```

**Trade-off:** Increases system dimension by 1.

---

## Storage Architecture

### RuntimeWorkspace Structure

Lag buffers are stored in a dedicated `RuntimeWorkspace` NamedTuple:

```python
RuntimeWorkspace = namedtuple(
    "RuntimeWorkspace",
    ["lag_ring", "lag_head", "lag_info"],
)
```

**Components:**
- `lag_ring`: Contiguous array storing all circular buffers (dtype matches model)
- `lag_head`: Array of current head indices for each lagged state (int32)
- `lag_info`: Metadata array with shape (n_lagged_states, 3) containing (state_idx, depth, offset)

### Circular Buffer Layout

Each lagged state gets a contiguous segment in `lag_ring`:

```
lag_ring layout (contiguous):
┌──────────────────┬──────────────────┬─────────────┐
│ lag buffer for x │ lag buffer for y │ (unused)    │
│  (depth 5)       │  (depth 3)       │             │
└──────────────────┴──────────────────┴─────────────┘
   offset=0           offset=5          end
```

**Allocation:**
- Each lagged state with depth `k` gets `k` consecutive elements in `lag_ring`
- Total `lag_ring` size = sum of all lag depths
- `lag_head` has one entry per lagged state
- `lag_info[j] = (state_idx, depth, offset)` for lagged state `j`

---

## Circular Buffer Mechanics

### Access Pattern

For `lag_x(k)` where state `x` has:
- `depth = 5` (max lag)
- `offset = 0` (starts at lag_ring[0])
- `head_index = 0` (head at lag_head[0])

**Lowered expression:**
```python
runtime_ws.lag_ring[offset + ((runtime_ws.lag_head[head_index] - k) % depth)]
#  runtime_ws.lag_ring[0 + ((runtime_ws.lag_head[0] - k) % 5)]
```

### Initialization (at t=t0)

```python
# Fill with IC for each lagged state
for j, (state_idx, depth, offset) in enumerate(lag_info):
    value = y_curr[state_idx]
    runtime_ws.lag_ring[offset : offset + depth] = value
    runtime_ws.lag_head[j] = depth - 1  # head at last position
```

**Why depth-1?** So that after first step commit, head wraps to 0.

### Update After Committed Step

**CRITICAL:** Updates happen ONLY after successful step commits, not after rejected steps or buffer growths.

```python
# In runner.py, after commit:
for j in range(n_lagged_states):
    state_idx, depth, offset = lag_info[j]
    head = int(lag_head[j]) + 1
    if head >= depth:
        head = 0
    lag_head[j] = head
    lag_ring[offset + head] = y_curr[state_idx]
```

**Example trace for x with depth=3:**

```
Step 0 (IC=0.1):
lag_ring = [0.1, 0.1, 0.1], head=2

Step 1 (y_curr=0.2):
head = (2+1) % 3 = 0
lag_ring[0] = 0.2 → lag_ring = [0.2, 0.1, 0.1], head=0

Step 2 (y_curr=0.3):
head = (0+1) % 3 = 1
lag_ring[1] = 0.3 → lag_ring = [0.2, 0.3, 0.1], head=1

Step 3 (y_curr=0.4):
head = (1+1) % 3 = 2
lag_ring[2] = 0.4 → lag_ring = [0.2, 0.3, 0.4], head=2

Access lag_x(1) at step 3:
lag_ring[0 + ((2 - 1) % 3)] = lag_ring[1] = 0.3 ✓ (step 2 value)

Access lag_x(2) at step 3:
lag_ring[0 + ((2 - 2) % 3)] = lag_ring[0] = 0.2 ✓ (step 1 value)
```

---

## Safety & Correctness

### Buffer Growth (GROW_REC, GROW_EVT)

**Lag buffers are NOT reallocated** during recording/event buffer growth:
- Runtime workspace sizes are determined by lag depths (model-specific, not trajectory-dependent)
- Wrapper doubles `rec`/`ev` buffers, but leaves runtime workspace unchanged
- Lag state preserved across re-entry

### Early Breaks (STEPFAIL, NAN_DETECTED, USER_BREAK)

- Runner commits state **before** break
- Lag buffers contain values up to last successful commit
- Resume uses `workspace_seed` to restore exact lag state

### Resume & Snapshots

- `RuntimeWorkspace` supports `snapshot_workspace()` and `restore_workspace()`
- Lag buffers automatically included in workspace snapshots
- No special handling needed

**Correctness guarantee:** Lags are counted after committed steps only.

---

## Implementation Status

### ✅ Completed

1. **Detection & Validation** (`dsl/astcheck.py`)
   - `collect_lag_requests()` scans all expressions
   - Validates lag depths, state existence, integer literals

2. **Metadata** (`dsl/spec.py`)
   - `ModelSpec.lag_map`: {state_name -> (depth, offset, head_index)}
   - Tracks which states need lagging and their maximum depths

3. **Runtime Workspace** (`runtime/workspace.py`)
   - `RuntimeWorkspace` NamedTuple with `lag_ring`, `lag_head`, `lag_info`
   - `make_runtime_workspace()` allocates lag buffers
   - `initialize_lag_runtime_workspace()` seeds with initial conditions
   - `snapshot_workspace()`/`restore_workspace()` support

4. **Allocation** (`compiler/build.py`)
   - `_compute_lag_state_info()` converts lag_map to runtime metadata
   - Runtime workspace allocated with proper lag buffer sizes

5. **Expression Lowering** (`compiler/codegen/rewrite.py`)
   - `_make_lag_access()` generates runtime workspace access
   - `runtime_ws.lag_ring[offset + ((runtime_ws.lag_head[idx] - k) % depth)]`

6. **Function Signatures** (`compiler/codegen/emitter.py`)
   - RHS: `def rhs(t, y_vec, dy_out, params, runtime_ws)`
   - Events: `def events_pre(t, y_vec, params, evt_log_scratch, runtime_ws)`

7. **Runner Updates** (`compiler/codegen/runner.py`, `runner_discrete.py`)
   - Lag buffer updates after successful step commits
   - Both continuous and discrete runners supported

8. **Stepper ABI** (`steppers/base.py`, implementations)
   - All steppers use new workspace-based ABI
   - `StepperSpec.workspace_type()` declares workspace structure
   - `StepperSpec.make_workspace()` allocates stepper-specific workspace

### ✅ Tested

- Unit tests for lag detection and validation
- Integration tests for lag buffer mechanics
- Resume/snapshot functionality with lag buffers
- Both ODE and map models with lagging

---

## Example: Logistic Map with Delay

```toml
[model]
type = "map"
name = "Delayed Logistic Map"

[states]
x = 0.1

[params]
r = 3.8
alpha = 0.7  # Mix of current and delayed feedback

[equations.rhs]
# Delay-coupled logistic map
x = "r * (alpha * x + (1 - alpha) * lag_x()) * (1 - x)"

[sim]
t0 = 0.0
dt = 1.0
stepper = "map"
```

**Execution trace:**
```
n=0: x=0.1, lag_x(1)=0.1 (IC)
n=1: x = 3.8*(0.7*0.1 + 0.3*0.1)*(1-0.1) = 0.342
     lag_x(1)=0.1
n=2: x = 3.8*(0.7*0.342 + 0.3*0.1)*(1-0.342) = 0.627
     lag_x(1)=0.342
n=3: x = 3.8*(0.7*0.627 + 0.3*0.342)*(1-0.627) = 0.788
     lag_x(1)=0.627
...
```

---

## Performance

### Memory Overhead

Per lagged state with depth `k`:
- Storage: `k * sizeof(dtype)` bytes in `RuntimeWorkspace.lag_ring`
- Head indices: 1 int32 per lagged state in `RuntimeWorkspace.lag_head`
- Metadata: 3 int32 per lagged state in `RuntimeWorkspace.lag_info`
- Example: 3 states, depth 10, float64 → 3 * 10 * 8 = 240 bytes + overhead

### Computational Cost

- **Per step:** O(n_lagged_states) writes to circular buffer
- **Lag access:** O(1) modulo + array index
  - Numba optimizes `(x - k) % depth` to bitwise AND if depth is power-of-2

### Optimization: Power-of-2 Depths

Round up lag depths to next power-of-2 for faster modulo:

```python
depth_requested = 7
depth_allocated = 8  # 2^3

# Access becomes:
index = (head - k) & 7  # bitwise AND (faster than %)
```

**Trade-off:** Up to 2x memory for ~3x speedup. **Not yet implemented.**

---

## Future Enhancements

1. **Fractional Lags** (interpolation)
   ```toml
   x = "lag_x(1.5)"  # Interpolate between lag_x(1) and lag_x(2)
   ```
   Requires dense output or Hermite interpolation.

2. **Time-Based Lags** (delay differential equations)
   ```toml
   x = "lag_x(t - tau)"  # Lag by time delay tau, not steps
   ```
   Requires time-indexed history with interpolation.

3. **Multi-Step Method Integration**
   Share lag buffers with Adams-Bashforth f-history in stepper workspace.

4. **Diagnostic API**
   ```python
   res.lag_buffer_usage()  # Memory statistics
   res.lag_history(state="x", k=5)  # Retrieve full lag buffer
   ```

---

## References

- **Runtime Workspace:** `runtime/workspace.py`
- **DSL Spec:** `dsl/spec.py`
- **Expression Lowering:** `compiler/codegen/rewrite.py`
- **Runner ABI:** `runtime/runner_api.py`
- **Stepper Base:** `steppers/base.py`

---

**Status:** Fully implemented with workspace-based architecture.
