# Float Banks

## `sp` — Scratch (Ephemeral Within an Attempt)

- **Who writes/reads:** Stepper.
- **Lifetime:** Within a single attempt (discardable between attempts and steps).
- **Use for:** Temporary stage states (e.g., `y_stage`), short copies, tiny transposes, quick stash when reusing another bank.

### Patterns:
- **Euler/RK:** `y_stage` or "stash original `k1` before reuse".
- **RK45:** 1 lane is enough for `y_stage`.

---

## `ss` — Stepper State (Persistent Across Attempts/Steps)

- **Who writes/reads:** Stepper; runner may read for tiny maintenance hooks only (history/dense-output), never reinterpret.
- **Lifetime:** Across attempts and across accepted steps.
- **Use for:** Data you must carry forward:
  - FSAL caches (e.g., store `k_last` to reuse as next `k1`),
  - Dense-output coefficients,
  - Multi-step history (e.g., rings of `f(y)`),
  - Any per-stepper state needed by its algorithm (without touching global state).

### Patterns:
- **AB/AM:** Keep past `f` vectors.
- **DDE/hist:** Heads/indices plus float history (with `iw0` for heads).
- **RK45 (if not lane-packing `sw*`):** Spill `k5`, `k6` or store DO coeffs.

---

## `sw0`, `sw1`, `sw2`, `sw3` — Stage Work (Ephemeral Within an Attempt)

- **Who writes/reads:** Stepper.
- **Lifetime:** Within a single attempt (no contract to persist after return).
- **Use for:** Stage derivatives, intermediate vectors that must coexist during combination.

### Lane Packing:
You may use multiple lanes per bank (e.g., `sw0[:n]`, `sw0[n:2*n]`).

### Patterns:
- **RK4:** `k1..k4` as 4 lanes across `sw0..sw3` (1 lane each).
- **RK45:** Neat fit with lanes:
  - `sw0`: `k1`, `k2`
  - `sw1`: `k3`, `k4`
  - `sw2`: `k5`, `k6`
  - `sw3`: `k7` (or dense-out cache), leaving `sp` for `y_stage`.

### Rule of Thumb:
- `sw*` = "throwaway" inside the attempt.
- `ss` = "keep it" across attempts/steps.
- `sp` = "transient scratch".

---

# Integer / Byte Banks

## `iw0:int32` — Indices & Heads

### Ownership & Partitioning

**When lag system is active** (`use_history=True` due to lagged variables in DSL):

- **`iw0[0..iw0_lag_reserved-1]`**: Reserved for lag circular buffer heads
  - One slot per lagged state
  - Managed automatically by runner after step commit
  - **DO NOT MODIFY** in stepper code

- **`iw0[iw0_lag_reserved..]`**: Available for stepper-specific use
  - Ring heads, step counters, retry flags, etc.
  - Stepper must offset all accesses by `iw0_lag_reserved`

**When no lags are present** (`use_history=False` or no lag notation in model):
- `iw0_lag_reserved = 0`
- Entire `iw0` is available to stepper
- No offset needed

### Stepper Implementation Pattern

```python
# In stepper emit():
def my_stepper(..., iw0, ...):
    # Access stepper-owned indices with offset
    LAG_RESERVED = iw0_lag_reserved  # compile-time constant from metadata
    
    my_counter = iw0[LAG_RESERVED + 0]
    my_ring_head = iw0[LAG_RESERVED + 1]
    
    # ... stepper logic ...
    
    iw0[LAG_RESERVED + 0] = my_counter + 1
    iw0[LAG_RESERVED + 1] = (my_ring_head + 1) % history_depth
```

**Important:** The `iw0_lag_reserved` constant is embedded in generated stepper 
source code, ensuring zero runtime overhead.

### Who writes/reads

- **Lag system (runner):** Writes to `iw0[0..iw0_lag_reserved-1]` only
- **Stepper:** Reads/writes `iw0[iw0_lag_reserved..]` only
- **Never overlap:** Strict partitioning enforced at compile time

### Lifetime

Persistent across steps.

### Use for

- **Lag heads (automatic):** Circular buffer positions for lagged states
- **Stepper indices:** Ring heads, counters, small integer state (e.g., multi-step position, history window indices, retry counters if needed)

---

## `bw0:uint8` — Bit Flags / Small Masks

- **Who writes/reads:** Runner/stepper.
- **Lifetime:** Persistent.
- **Use for:** Compact flags (e.g., which dense-output coeffs valid), feature toggles, tiny state machines.

---

# Proposal/Outs (Always Output-Only)

- **`y_prop[n_state]`:** Proposed next state (stepper writes only).
- **`t_prop[1]`:** Proposed next time (write only).
- **`dt_next[1]`:** Suggestion for next `dt` (write only; runner may clamp).
- **`err_est[1]`:** Scalar error metric (write only; used by runner for policy/diagnostics).

---

# Ownership & Mutation Rules (Tight)

- **Stepper:**
  - May read `y_curr`, `params`, `banks`.
  - May mutate `sp`, `ss`, `sw*`, `iw0`, `bw0`.
  - Must write only `y_prop`, `t_prop[0]`, `dt_next[0]`, `err_est[0]`.
- **Runner:**
  - Never interprets stepper layouts.
  - May do tiny generic maintenance guarded by flags in `StructSpec` (e.g., rotate history rings using `iw0`).
- **Events:**
  - Never touch banks; only states/params.

---

# Lane-Count Rule (Applies to Float Banks)

- **Sizes in `StructSpec` are lane counts:**
  - `0` ⇒ unused (0-length array),
  - `1` ⇒ `n_state`,
  - `k` ⇒ `k * n_state`.
- **Only contiguous, stride-1 slicing by lanes is allowed** (e.g., `a[:n]`, `a[n:2*n]`).

---

# Typical Layouts (Examples)

### Euler (Fixed):
- `sw0=1` (`k1`), `sp=1` (`y_stage`), `ss=0`.

### RK4 (Fixed):
- `sw0..sw3=1` (`k1..k4`), `sp=1`, `ss=0`.

### RK45 (Adaptive, Clean Lane Packing):
- `sw0=2` (`k1`, `k2`), `sw1=2` (`k3`, `k4`), `sw2=2` (`k5`, `k6`), `sw3=1` (`k7`), `sp=1` (`y_stage`), `ss=0`.
  - If you prefer FSAL/dense-output storage across steps, set `ss>0` and use it for that state.

### AB/AM Multistep:
- Use `ss = m` lanes for `f` history, `iw0` for ring head; `sw*` for current `f`.

### Dense Output:
- Store last-step DO coeffs in `ss` lanes as declared by the stepper.