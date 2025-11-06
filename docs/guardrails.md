# DYNLIB v2 Guardrails — ABI, DSL, Steppers

> Canonical, compact references for implementation checks. No back-compat, no Optional, ABI frozen.

---

## 1) ABI — Wrapper/Runner/Stepper

### Wrapper → Runner (single call; runner owns loop)
```
runner(
  # scalars
  t0: float, t_end: float, dt_init: float,
  max_steps: int, n_state: int, record_every_step: int,
  # state/params
  y_curr: float[:], y_prev: float[:], params: float[:] | int[:],
  # struct banks (views)
  sp: float[:], ss: float[:],
  sw0: float[:], sw1: float[:], sw2: float[:], sw3: float[:],
  iw0: int32[:], bw0: uint8[:],
  # proposals/outs (len-1 arrays where applicable)
  y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:],
  # recording
  T: float64[:], Y: float[:, :], STEP: int64[:], FLAGS: int32[:],
  # event log (present; cap may be 1 if disabled)
  EVT_TIME: float64[:], EVT_CODE: int32[:], EVT_INDEX: int32[:],
  # cursors & caps
  i_start: int64, step_start: int64, cap_rec: int64, cap_evt: int64,
  # control/outs (len-1)
  user_break_flag: int32[:], status_out: int32[:], hint_out: int32[:],
  i_out: int64[:], step_out: int64[:], t_out: float64[:],
  # function symbols (jittable callables)
  stepper, rhs, events_pre, events_post
) -> int32
```
**Exit statuses** (int32): `DONE=9`, `GROW_REC=10`, `GROW_EVT=11`, `USER_BREAK=12`, `STEPFAIL=2`, `NAN_DETECTED=3`. Internal codes (no exit): `OK=0`, `REJECT=1`.

**Ownership**: runner performs capacity checks, pre/post events, commit, record, and returns only with exit codes above.

### Runner → Stepper (per attempt)
```
status = stepper(
  t: float, dt: float,
  y_curr: float[:], rhs,
  params: float[:] | int[:],
  sp: float[:], ss: float[:],
  sw0: float[:], sw1: float[:], sw2: float[:], sw3: float[:],
  iw0: int32[:], bw0: uint8[:],
  y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:]
) -> int32
```
**Rules**: reads `t, dt, y_curr, params, sp, ss`; writes `y_prop, t_prop[0], dt_next[0], err_est[0]` and may mutate `ss`. **Never** touches record/log buffers or cursors. Adaptive steppers may loop internally until accept/fail.

### RHS
```
rhs(t: float, y_vec: float[:], dy_out: float[:], params: float[:] | int[:]) -> None
```
- Recompute **aux from `y_vec`** each call. Pure numeric; no allocations/Python objects.

### Events
- `events_pre` runs on **committed** state before stepping; `events_post` after commit.
- May **mutate only states/params**; may request logging/recording per DSL.

### Dtypes & Buffers
- Model primary dtype from `[model].dtype` (default `float64`).
- `T`, `EVT_TIME`: `float64`; `Y`: model dtype; `STEP:int64`, `FLAGS:int32`.
- Work banks `sp, ss, sw*`: model dtype; `iw0:int32`, `bw0:uint8`.
- `t_prop` is model dtype; committed `t` written to `T` as `float64`.

### Growth/Resume
- On `GROW_REC`/`GROW_EVT`, wrapper reallocates (geometric), copies slices, and re-enters runner with updated caps/cursors. Runner must resume seamlessly.

---

## 2) DSL — TOML Schema (no legacy keys)

### Required
```toml
[model]
type = "ode"          # "ode" | "map"
label = "..."         # optional
dtype = "float64"     # optional (default float64)

[states]               # order = file order
x = 1.0

[params]
a = 2.0
```

### Equations (choose A and/or B; no duplicate targets)
**A. Per-state RHS**
```toml
[equations.rhs]
x = "-a*x"
```
**B. Block form**
```toml
[equations]
expr = """
dx = -a*x
"""
```

### Aux (derived, read-only; acyclic)
```toml
[aux]
E = "0.5*a*x^2"
```

### Functions (pure numeric helpers)
```toml
[functions.sat]
args = ["u","c"]
expr = "u/(1+abs(u)^c)"
```

### Events (named subtables; text order execution)
```toml
[events.reset]
phase  = "post"              # "pre" | "post" | "both"
cond   = "x >= 1"
action.x = "0"               # keyed form
log    = ["t", "x", "aux:E", "param:a"]
```
**Block action alternative**
```toml
[events.bump]
phase = "pre"
cond  = "t >= 10"
action = '''
x = x + 0.5
'''
```
**Legality**: only states/params assignable; aux/buffers/internals forbidden.

### Simulation Defaults
```toml
[sim]
t0 = 0.0
t_end = 100.0
dt = 0.01
stepper = "rk4"
record = true
```

### Mods (build-time overlays)
**Verbs**
```toml
[mod.set.states]
x = 2.0

[mod.set.params]
a = 3.0

[mod.add.events.reset]
phase = "post"
cond  = "x >= 1"
action.x = "0"

[mod.replace.events.reset]
phase = "post"
cond  = "x >= 1"
action = '''x = 0'''

[mod.remove.events]
names = ["reset"]
```
**Grouping**
```toml
[mod]
name = "drive_low"
group = "drive"
exclusive = true
priority = 0
```
**Application order**: remove → replace → add → set (per selected mod list order). Then validate → codegen/JIT.

### Paths/Registry
- Config: Linux `${XDG_CONFIG_HOME:-~/.config}/dynlib/config.toml`; macOS `~/Library/Application Support/dynlib/config.toml`; Windows `%APPDATA%\dynlib\config.toml`.
- Env override: `DYNLIB_CONFIG=/abs/custom.toml`.
- `[paths]` tag roots; optional `DYN_MODEL_PATH` one-shot env with **prepended** roots.
- URIs: `inline:...`, absolute, relative, `TAG://file`, fragments `#mod=NAME`.

### Validation Highlights
- Unique names; acyclic aux/functions; no duplicate equation targets.
- Events mutate only states/params; dtype rules enforced.

---

## 3) Steppers — Spec, StructSpec, Registry

### StepperMeta / StepperInfo
```
name: str
kind: "ode" | "map"
time_control: "fixed" | "adaptive"
scheme: "explicit" | "implicit" | "splitting"
geometry: FrozenSet[str]
family: str
order: int
embedded_order: int | None
dense_output: bool
stiff_ok: bool
aliases: tuple[str, ...]
```

### StepperSpec contract
- Accept `meta: StepperMeta` in `__init__`.
- Provide `emit(...)` for codegen.
- Provide `struct_spec() -> StructSpec`.

### StructSpec (sole extension point)
```
StructSpec(
  sp_size, ss_size,
  sw0_size, sw1_size, sw2_size, sw3_size,
  iw0_size, bw0_size,
  use_history=False, use_f_history=False,
  dense_output=False, needs_jacobian=False,
  embedded_order=None, stiff_ok=False
)
```
**Add storage by sizes only.** Never add runner/stepper args. Runner reads flags for tiny maintenance hooks (e.g., history ring updates).

### Registry Rules
- Unique `name`; `aliases` map to same spec.
- `kind` must match model kind at build time.
- `time_control` tells runner fixed vs adaptive behavior.

## 4) Stepper banks — Guardrails (usage, sizing, and validation)

The following guardrails consolidate and formalize the guidance from the stepper banks manual into explicit rules the build-time validator and stepper authors must follow. These are intended to make incorrect bank usage fail fast at build-time (or raise a clear warning) rather than producing subtle runtime bugs.

### Contract (short)
- sp, ss, sw0..sw3 are float banks sized in "lanes"; lane count × n_state = items.
- iw0 (`int32`) and bw0 (`uint8`) are small persistent integer/flag banks.
- Stepper may read `y_curr`, `params`, `sp`, `ss`, and may mutate `sp`, `ss`, `sw*`, `iw0`, `bw0`.
- Stepper MUST write only `y_prop[:]`, `t_prop[0]`, `dt_next[0]`, `err_est[0]` as outputs. It MUST NOT touch record/log buffers, cursors or runner-owned structures.

### Lane-count and layout rules (enforced)
- Sizes in `StructSpec` are lane counts: `0` => unused, `1` => `n_state`, `k` => `k * n_state`.
- Lanes must be contiguous and stride-1. Allowed indexing inside stepper code is by whole lanes (for example `sw0[:n_state]`, `sw0[n_state:2*n_state]`). Non-contiguous or strided lane access is forbidden and must be rejected at build-time.
- A stepper declaring lane counts must not assume partial lanes (no fractional lanes). If the algorithm needs a partial lane it must be reworked into full-lane storage or use `sp`/`ss` appropriately.

### Bank semantics & permitted mutations (enforced)
- sp: ephemeral scratch within a single attempt. Must not be relied on across attempts; build-time check: steppers that declare persistence in `sp` should trigger a warning.
- sw0..sw3: ephemeral stage work for an attempt. Must not be used to persist data across attempts/steps. Any scheme that expects stage banks to persist between calls must instead use `ss`.
- ss: persistent stepper state across attempts/steps. Allowed uses: FSAL caches, dense-output coefficients, multi-step history (together with `iw0` indices). Mutations here are allowed.
- iw0 and bw0: persistent integer heads/flags. Their declared sizes must be small integers >=0. `iw0` must be `int32`, `bw0` must be `uint8` (enforced by validator).

### API/ownership hard rules (must be validated)
- Stepper code must never read or write runner-owned record/log buffers, `T`, `Y`, `STEP`, `FLAGS`, event logs, or cursor/cap scalars. Violations are build-time errors.
- Stepper must not add or depend on extra function arguments beyond the declared ABI (no hidden args, no extra closures that capture runtime objects). This is part of the JIT/hot-path policy.

### StructSpec sizing and compatibility checks (build-time)
Validator must enforce at model/stepper registration time:
- All `StructSpec` sizes are non-negative integers.
- `sw*`, `sp`, `ss` lane counts multiplied by `n_state` must be representable within available memory; extremely large lane counts should produce a warning and require explicit confirmation (to avoid accidental OOM).
- If `dense_output=True` is declared, `ss_size` must be sufficient to store the declared dense-output coefficients; otherwise raise an error.
- If `use_history=True` or `use_f_history=True` is declared, then `ss_size` and `iw0_size` must be large enough for the requested history length; otherwise raise an error.

### Static code checks for steppers (recommended, implement in codegen/validation)
At codegen/build-time perform static checks on emitted stepper code/metadata:
- Verify the stepper writes only the allowed output arrays/positions (`y_prop`, `t_prop[0]`, `dt_next[0]`, `err_est[0]`). If writes to other outputs or to record buffers are detected, abort with explanatory error.
- Ensure any accesses to `sp`, `sw*`, `ss` obey whole-lane slicing patterns. Reject uses of slicing that imply strided, non-contiguous, or negative indexing on lanes.
- Detect persistence patterns: if local `sw*` lanes are used across attempts (e.g., used to pass values between calls), emit a warning or error instructing to move that storage to `ss`.
- Ensure `iw0` and `bw0` are only used for small integer indices/flags. If code attempts to store large floats or non-integer types into these banks, raise an error.

### Error vs Warning policy
- Errors (block build/runtime): illegal writes (to record/log/cursors), illegal bank dtype usage (writing floats to `iw0`), non-integer/negative StructSpec sizes, non-contiguous lane access, missing storage when `dense_output/use_history` is declared.
- Warnings (informational, but recommend fix): overly large bank sizes that look accidental, use of `sp` as persistent storage across attempts, use of many lanes where fewer suffice (suggest optimization), excessive dependence on `iw0` for complex state machines (suggest `ss`).

### Helpful messages for stepper authors (short checklist)
- Does my stepper only write `y_prop`, `t_prop[0]`, `dt_next[0]`, `err_est[0]`? If not, fix it.
- Are all my banks sized as whole lanes? Use `sp`, `ss` for stuff that must persist.
- Is any data that must survive an attempt stored in `ss` (not `sw*`/`sp`)? Move it if needed.
- Are index/flag uses in `iw0`/`bw0` tightly bounded and integer/bitwise in nature? Otherwise use `ss`.

### Examples (quick)
- RK4: `sw0..sw3=1`, `sp=1`, `ss=0` — OK.
- RK45 (lane-packed): `sw0=2`, `sw1=2`, `sw2=2`, `sw3=1`, `sp=1`, `ss=0` — validator must ensure lane counts × `n_state` matches buffer lengths.
- AB/AM multistep: `ss = m` lanes used for f-history and `iw0` used for ring head — ensure `iw0_size>=1` and `ss_size>=m`.

### Suggested implementation notes for the validator (pseudocode)
- On stepper registration:
  - Check `StructSpec` sizes >= 0 and are ints.
  - Expand lane sizes to element counts: lane_count * n_state and verify against backend buffer allocation logic.
  - Static-scan emitted stepper code (or check metadata) for illegal writes (record buffers, other runner-owned arrays).
  - Confirm slicing patterns are whole-lane only; reject strided/partial lane use.

### Follow-ups
- Implement these checks in `compiler/` (codegen/emit validation) and in `compiler/jit/*` where function bodies are decorated. Start with blocking illegal writes and lane-layout checks; add heuristics for warnings later.

---

## 5) JIT Policy

**Optional JIT (runtime toggle, user‑facing)**
- Users can enable/disable JIT via a single flag (e.g., `jit=true|false`) read at **build time**.
- The **same Python function bodies** are used in both modes; the compiler layer conditionally applies `@njit` (or not). **No alternate code paths.**
- JIT decoration happens **only** in `compiler/jit/*`; runtime/hot loops contain no Python objects either way.
- ABI and behavior are identical with or without JIT.


**Hot-path constraints (always):**
- No allocations or Python objects in hot loops.
- No dynamic shapes/fancy indexing; arrays are preallocated and passed in.


## Notes
- These guardrails are deliberately strict: failing early at build-time avoids subtle runtime behavior and makes steppers portable and JIT-friendly.
- If a stepper legitimately needs an uncommon pattern, extend `StructSpec` with a narrowly scoped opt-in flag and document the precise invariants required. Such extensions must be reviewed and added to the registry rules above.