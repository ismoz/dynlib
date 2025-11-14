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
- ✅ No ABI changes (reuses existing `ss` and `iw0` banks with partitioning)

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

### Circular Buffers in `ss` Bank

Each lagged state gets a contiguous circular buffer:

```
ss layout (lane-packed):
┌──────────────────┬──────────────────┬────────────┐
│ lag buffer for x │ lag buffer for y │ stepper ss │
│  (depth 5 lanes) │  (depth 3 lanes) │   (rest)   │
└──────────────────┴──────────────────┴────────────┘
   ss[0..4]            ss[5..7]           ss[8..]
```

**Allocation:**
- Each lagged state with depth `k` gets `k` lanes in `ss`
- Stepper's own `ss` allocation follows lag buffers
- Total `ss_size = lag_lanes + stepper_ss_lanes`

### Head Indices in `iw0` Bank

```
iw0 layout (int32 elements):
┌────────┬────────┬──────────────┐
│ head_x │ head_y │ stepper iw0  │
└────────┴────────┴──────────────┘
  iw0[0]   iw0[1]    iw0[2..]
  ↑                   ↑
  lag reserved        stepper offset
```

**Partitioning:**
- `iw0[0..iw0_lag_reserved-1]`: Lag circular buffer heads (RESERVED)
- `iw0[iw0_lag_reserved..]`: Available for stepper use

**Stepper Impact:**
Steppers that use `iw0` must offset their accesses:

```python
# In stepper code:
LAG_RESERVED = 2  # embedded compile-time constant

my_counter = iw0[LAG_RESERVED + 0]  # stepper's first index
my_ring_head = iw0[LAG_RESERVED + 1]  # stepper's second index
```

---

## Circular Buffer Mechanics

### Access Pattern

For `lag_x(k)` where state `x` has:
- `depth = 5` (max lag)
- `ss_offset = 0` (starts at ss[0])
- `iw0_index = 0` (head at iw0[0])

**Lowered expression:**
```python
ss[ss_offset + ((iw0[iw0_index] - k) % depth)]
#  ss[0 + ((iw0[0] - k) % 5)]
```

### Initialization (at t=t0)

```python
# Fill with IC for each lagged state
for state_idx, depth, ss_offset, iw0_index in lag_state_info:
    ic_value = y_curr[state_idx]
    
    for i in range(depth):
        ss[ss_offset + i] = ic_value  # [IC, IC, IC, IC, IC]
    
    iw0[iw0_index] = depth - 1  # head at last position
```

**Why depth-1?** So that after first step commit, head wraps to 0.

### Update After Committed Step

**CRITICAL:** Updates happen ONLY after successful step commits, not after rejected steps or buffer growths.

```python
# In runner.py, after commit:
for k in range(n_state):
    y_prev[k] = y_curr[k]
    y_curr[k] = y_prop[k]

# NEW: Update lag buffers (if use_history=True)
for lag_idx in range(num_lagged_states):
    state_idx = lag_state_indices[lag_idx]
    head = iw0[lag_idx]
    depth = lag_depths[lag_idx]
    ss_offset = lag_offsets[lag_idx]
    
    # Advance head (circular wraparound)
    new_head = (head + 1) % depth
    iw0[lag_idx] = new_head
    
    # Write new value at head position
    ss[ss_offset + new_head] = y_curr[state_idx]
```

**Example trace for x with depth=3:**

```
Step 0 (IC=0.1):
ss = [0.1, 0.1, 0.1], head=2

Step 1 (y_curr=0.2):
head = (2+1) % 3 = 0
ss[0] = 0.2 → ss = [0.2, 0.1, 0.1], head=0

Step 2 (y_curr=0.3):
head = (0+1) % 3 = 1
ss[1] = 0.3 → ss = [0.2, 0.3, 0.1], head=1

Step 3 (y_curr=0.4):
head = (1+1) % 3 = 2
ss[2] = 0.4 → ss = [0.2, 0.3, 0.4], head=2

Access lag_x(1) at step 3:
ss[0 + ((2 - 1) % 3)] = ss[1] = 0.3 ✓ (step 2 value)

Access lag_x(2) at step 3:
ss[0 + ((2 - 2) % 3)] = ss[0] = 0.2 ✓ (step 1 value)
```

---

## Safety & Correctness

### Buffer Growth (GROW_REC, GROW_EVT)

**Lag buffers are NOT reallocated** during recording/event buffer growth:
- `ss` and `iw0` sizes are determined by lag depths (model-specific, not trajectory-dependent)
- Wrapper doubles `rec`/`ev` buffers, but leaves `ss`/`iw0` unchanged
- Lag state preserved across re-entry

### Early Breaks (STEPFAIL, NAN_DETECTED, USER_BREAK)

- Runner commits state **before** break
- Lag buffers contain values up to last successful commit
- Resume uses `workspace_seed` to restore exact lag state

### Resume & Snapshots

- `wrapper.py` already captures/restores `ss` and `iw0` via `_apply_workspace_seed()`
- Lag buffers automatically included in workspace snapshots
- No special handling needed

**Correctness guarantee:** Lags are counted after committed steps only.

---

## Implementation Status

### ✅ Completed

1. **Detection & Validation** (`dsl/astcheck.py`)
   - `collect_lag_requests()` scans all expressions
   - Validates lag depths, state existence, integer literals

2. **Metadata** (`dsl/spec.py`, `steppers/base.py`)
   - `ModelSpec.lag_map`: {state_name -> (depth, ss_offset, iw0_index)}
   - `StructSpec.iw0_lag_reserved`: Prefix slots for lag heads

3. **Allocation** (`compiler/build.py`)
   - Augments stepper's `StructSpec` with lag requirements
   - `ss_size += lag_lanes`, `iw0_size += num_lagged_states`
   - Converts `lag_map` to `lag_state_info` for runtime

4. **Expression Lowering** (`compiler/codegen/rewrite.py`)
   - `_NameLowerer` handles `lag_<name>(k)` calls (argument optional, defaults to 1)
   - Generates circular buffer access: `ss[offset + ((iw0[idx] - k) % depth)]`

5. **Function Signatures** (`compiler/codegen/emitter.py`)
   - RHS: `def rhs(t, y_vec, dy_out, params, ss, iw0)`
   - Events: `def events_pre(t, y_vec, params, evt_log_scratch, ss, iw0)`

6. **Initialization** (`runtime/wrapper.py`)
   - `_initialize_lag_buffers_by_index()` fills with IC at t=t0
   - Only on first run (resume uses `workspace_seed`)

### ❌ Remaining Work

1. **Runner Update Hooks** (`compiler/codegen/runner.py`)
   - **TODO:** Add lag buffer update after step commit
   - **TODO:** Embed lag metadata as compile-time constants in generated runner
   - **TODO:** Handle both continuous (`runner.py`) and discrete (`runner_discrete.py`) runners

2. **Stepper ABI Updates**
   - **TODO:** Update all stepper `emit()` functions to pass `ss`/`iw0` to RHS calls
   - Currently: `rhs(t, y_stage, k, params)`
   - Required: `rhs(t, y_stage, k, params, ss, iw0)`

3. **Documentation**
   - **TODO:** Update `docs/stepper_banks.md` with iw0 partitioning rules
   - **TODO:** Add examples to main README

4. **Testing**
   - **TODO:** Unit tests for `collect_lag_requests()`
   - **TODO:** Integration test: simple logistic map with lag
   - **TODO:** Test buffer initialization, wraparound, resume

---

## Example: Logistic Map with Delay

```toml
[model]
type = "map"
label = "Delayed Logistic Map"

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
- Storage: `k * sizeof(dtype)` bytes
- Example: 3 states, depth 10, float64 → 3 * 10 * 8 = 240 bytes

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
   Share `ss` bank between lag buffers and Adams-Bashforth f-history:
   ```python
   ss_layout = [lag_x_buffer, lag_y_buffer, ab_f_history]
   ```

4. **Diagnostic API**
   ```python
   res.lag_buffer_usage()  # Memory statistics
   res.lag_history(state="x", k=5)  # Retrieve full lag buffer
   ```

---

## References

- **Stepper Banks Design:** `docs/stepper_banks.md`
- **DSL Spec:** `dsl/spec.py`
- **Expression Lowering:** `compiler/codegen/rewrite.py`
- **Runner ABI:** `runtime/runner_api.py`

---

**Status:** Partially implemented (detection, allocation, lowering complete; runner hooks pending).
