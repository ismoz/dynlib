# DYNLIB v2.26 Guardrails — ABI, DSL, Steppers, Workspaces

> Canonical, compact references for implementation checks. No back-compat, no Optional, ABI frozen.

---

## 1) ABI — Wrapper/Runner/Stepper

### Wrapper → Runner (single call; runner owns loop)
```
runner(
  # scalars
  t0: float, t_end: float, dt_init: float,
  max_steps: int, n_state: int, record_interval: int,
  # state/params
  y_curr: float[:], y_prev: float[:], params: float[:] | int[:],
  # workspaces
  runtime_ws, stepper_ws,
  # proposals/outs (len-1 arrays where applicable)
  y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:],
  # recording
  T: float64[:], Y: float[:, :], STEP: int64[:], FLAGS: int32[:],
  # event log (present; cap may be 1 if disabled)
  EVT_CODE: int32[:], EVT_INDEX: int32[:], EVT_LOG_DATA: float[:, :],  # EVT_INDEX stores owning record index (or -1)
  evt_log_scratch: float[:],
  # cursors & caps
  i_start: int64, step_start: int64, cap_rec: int64, cap_evt: int64,
  # control/outs (len-1)
  user_break_flag: int32[:], status_out: int32[:], hint_out: int32[:],
  i_out: int64[:], step_out: int64[:], t_out: float64[:],
  # function symbols (jittable callables)
  stepper, rhs, events_pre, events_post
) -> int32
```
**Exit statuses** (int32): `DONE=9`, `GROW_REC=10`, `GROW_EVT=11`, `USER_BREAK=12`, `STEPFAIL=2`, `NAN_DETECTED=3`. Internal code (no exit): `OK=0` (step accepted, runner continues).

**Stepper contract**:
- **Fixed-step** (euler, rk4): single attempt per step; return `OK` (accepted) or `NAN_DETECTED`/`STEPFAIL` (terminal error).
- **Adaptive** (rk45): internal accept/reject loop; runner only sees `OK` (accepted) or `NAN_DETECTED`/`STEPFAIL` (terminal error).

**Ownership**: runner performs capacity checks, pre/post events, commit, record, and returns only with exit codes above.

### Runner → Stepper (per attempt)
```
status = stepper(
  t: float, dt: float,
  y_curr: float[:], rhs,
  params: float[:] | int[:],
  stepper_ws, stepper_config,
  y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:]
) -> int32
```
**Rules**: reads `t, dt, y_curr, params, stepper_ws`; writes `y_prop, t_prop[0], dt_next[0], err_est[0]` and may mutate `stepper_ws`. **Never** touches record/log buffers, cursors, or runtime_ws. Adaptive steppers may loop internally until accept/fail.

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
- `T`: `float64`; `Y`/`EVT_LOG_DATA`: model dtype; `STEP:int64`, `FLAGS:int32`; `EVT_CODE`/`EVT_INDEX:int32`.
- `t_prop` is model dtype; committed `t` written to `T` as `float64`.

### Growth/Resume
- On `GROW_REC`/`GROW_EVT`, wrapper reallocates (geometric), copies slices, and re-enters runner with updated caps/cursors and workspaces. Runner must resume seamlessly.

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
phase  = "post"              # optional; defaults to "post"
cond   = "x >= 1"
action.x = "0"               # keyed form
log    = ["t", "x", "aux:E", "param:a"]
```
**Block action alternative**
```toml
[events.bump]
phase = "pre"                # override default
cond  = "t >= 10"
action = '''
x = x + 0.5
'''
```
**Legality**: only states/params assignable; aux/buffers/internals forbidden. Events without an
explicit `phase` run after the step ("post").

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

## 3) Steppers — Spec, Workspace, Registry

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
stiff: bool
aliases: tuple[str, ...]
```

### StepperSpec contract
- Accept `meta: StepperMeta` in `__init__`.
- Provide `emit(...)` for codegen.
- Provide `workspace_type() -> type` returning a NamedTuple type for stepper workspace.
- Provide `make_workspace(n_state: int, dtype) -> workspace_instance` to allocate workspace.

### Registry Rules
- Unique `name`; `aliases` map to same spec.
- `kind` must match model kind at build time.
- `time_control` tells runner fixed vs adaptive behavior.

## 4) Stepper Workspaces — Guardrails (usage, ownership, and validation)

The following guardrails formalize the separation of stepper and runtime workspaces introduced in v2.26.0. Workspaces cleanly separate responsibilities: stepper workspaces are private to each stepper for scratch and state, while runtime workspaces handle lag buffers and DSL machinery.

### Workspace Separation
- **Stepper Workspace**: Private NamedTuple-of-NumPy-views containing stepper-specific scratch arrays (e.g., stages, histories, Jacobians). Owned and managed by the stepper; allocated via `StepperSpec.make_workspace()`.
- **Runtime Workspace**: Private NamedTuple containing lag buffers (lag_ring, lag_head, lag_info) for historical state access. Owned by the runner and DSL machinery; not accessible to steppers.

### Contract (short)
- Stepper may read `y_curr`, `params`, `stepper_ws`, and may mutate `stepper_ws`.
- Stepper MUST write only `y_prop[:]`, `t_prop[0]`, `dt_next[0]`, `err_est[0]` as outputs. It MUST NOT touch record/log buffers, cursors, runtime_ws, or any runner-owned structures.
- Runtime workspace is managed by the runner for lag system; steppers never access it directly.

### Ownership and Permitted Mutations (enforced)
- Stepper workspace: Fully owned by stepper; may be mutated freely within a step/attempt. Persistence across steps/attempts is allowed (e.g., for FSAL caches, histories).
- Runtime workspace: Read-only to steppers; managed by runner for lag buffers. Stepper code must never access or mutate runtime workspace.

### API/Ownership Hard Rules (must be validated)
- Stepper code must never read or write runner-owned record/log buffers, `T`, `Y`, `STEP`, `FLAGS`, event logs, cursors/cap scalars, or runtime_ws. Violations are build-time errors.
- Stepper must not add or depend on extra function arguments beyond the declared ABI (no hidden args, no extra closures that capture runtime objects). This is part of the JIT/hot-path policy.
- Workspace types must be NamedTuples with NumPy array fields; no Python objects or dynamic structures.

### Workspace Allocation and Compatibility Checks (build-time)
Validator must enforce at model/stepper registration time:
- `StepperSpec.workspace_type()` returns a valid NamedTuple type with array fields.
- `StepperSpec.make_workspace(n_state, dtype)` allocates arrays of correct shapes and dtypes matching model dtype.
- Workspace sizes must be representable within available memory; extremely large workspaces should produce a warning and require explicit confirmation (to avoid accidental OOM).
- For steppers with dense output or history, workspace must include sufficient storage; otherwise raise an error.

### Static Code Checks for Steppers (recommended, implement in codegen/validation)
At codegen/build-time perform static checks on emitted stepper code/metadata:
- Verify the stepper writes only the allowed output arrays/positions (`y_prop`, `t_prop[0]`, `dt_next[0]`, `err_est[0]`). If writes to other outputs, record buffers, or runtime_ws are detected, abort with explanatory error.
- Ensure stepper only accesses `stepper_ws` fields; reject any access to runtime_ws or runner-owned structures.
- Detect illegal mutations: stepper must not mutate inputs except `stepper_ws`.

### Error vs Warning Policy
- Errors (block build/runtime): illegal writes (to record/log/cursors, runtime_ws), access to forbidden structures, invalid workspace types/shapes.
- Warnings (informational, but recommend fix): overly large workspace sizes that look accidental, unused workspace fields (suggest optimization).

### Helpful Messages for Stepper Authors (short checklist)
- Does my stepper only write `y_prop`, `t_prop[0]`, `dt_next[0]`, `err_est[0]`? If not, fix it.
- Does my stepper only access `stepper_ws` for storage? Never touch runtime_ws or runner buffers.
- Is my workspace a NamedTuple with NumPy arrays? No Python objects.
- Is any data that must persist across steps stored in stepper_ws? Ensure allocation is sufficient.

### Examples (quick)
- Euler: Minimal workspace with no arrays (empty NamedTuple).
- RK4: Workspace with stage arrays (e.g., k1, k2, k3, k4 as arrays of shape (n_state,)).
- RK45: Workspace with stages plus error estimation arrays.
- AB2: Workspace with history ring for previous derivatives.

### Suggested Implementation Notes for the Validator (pseudocode)
- On stepper registration:
  - Validate `workspace_type()` returns NamedTuple with array fields.
  - Check `make_workspace()` allocates correct shapes/dtypes.
  - Static-scan emitted stepper code for illegal accesses (runtime_ws, runner buffers).
  - Confirm stepper only mutates `stepper_ws`.

### Follow-ups
- Implement these checks in `compiler/` (codegen/emit validation) and in `compiler/jit/*` where function bodies are decorated. Start with blocking illegal accesses and workspace validation; add heuristics for warnings later.

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
- Workspaces provide clean separation: stepper workspaces for algorithm-specific storage, runtime workspaces for DSL features like lagging.
- If a stepper legitimately needs an uncommon workspace pattern, extend the NamedTuple with narrowly scoped fields and document the precise invariants required. Such extensions must be reviewed and added to the registry rules above.
