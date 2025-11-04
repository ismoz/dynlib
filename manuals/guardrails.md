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
- Model primary dtype from `[model].dtype` (default `float64`). ODEs must be float; maps may be float or int.
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
record = true
log    = ["x", "aux:E", "param:a"]
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
- Events mutate only states/params; dtype rules enforced (ODEs must be float).

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

### JIT Policy
**Two distinct rules (don’t mix them):**

1) **Optional JIT (runtime toggle, user‑facing)**
- Users can enable/disable JIT via a single flag (e.g., `jit=true|false`) read at **build time**.
- The **same Python function bodies** are used in both modes; the compiler layer conditionally applies `@njit` (or not). **No alternate code paths.**
- JIT decoration happens **only** in `compiler/jit/*`; runtime/hot loops contain no Python objects either way.
- ABI and behavior are identical with or without JIT.


**Hot-path constraints (always):**
- No allocations or Python objects in hot loops.
- No dynamic shapes/fancy indexing; arrays are preallocated and passed in.
