# Core Architecture

### Encapsulation: Wrapper ⊃ Runner ⊃ Stepper
- **Wrapper**: Calls the runner once; handles interrupts (growth/failure/user break), resumes, and final result packing.
- **Runner (jitted)**: Owns the main loop, capacity checks, event execution (pre/post/both), accepts/commits steps, and recording.
- **Stepper (jitted)**: Proposes next step (including internal reject/try logic), `dt` control, and uses its own preallocated workspace. Never commits global state.

### Frozen Call Contracts (ABI)

#### Runner ← Wrapper
Arguments are fixed: scalars + ndarrays only (no `None`/Optionals/objects).

- **Scalars**: `t0`, `t_end`, `dt_init`, `max_steps`, `n_state`, `record_every_step`.
- **State/params**: `y_curr[n_state]`, `y_prev[n_state]`, `params[P]`.
- **Struct banks (workspace)**
  - Float64/Float32 (model primary dtype): `sw0[:]`, `sw1[:]`, `sw2[:]`, `sw3[:]`
  - Int32: `iw0[:]`
  - UInt8: `bw0[:]`
  - *(0 lanes = unused; 1 lane = n_state; All are views carved from contiguous dtype-specific buffers allocated at init.)*
- **Proposal/outs**: `y_prop[n_state]`, `t_prop[1]`, `dt_next[1]`, `err_est[1]`.
- **Recording**: `T[CAP]` (float64 time), `Y[n_state, CAP]` (model dtype), `STEP[CAP]` (int64), `FLAGS[CAP]` (int32).
- **Optional event log**: `EVT_TIME[EVTCAP]`, `EVT_CODE[EVTCAP]`, `EVT_INDEX[EVTCAP]` *(present with minimal size if logging disabled)*.
- **Cursors & caps**: `i_start`, `step_start`, `cap_rec`, `cap_evt`.
- **Control/outs (1-elem arrays)**: `user_break_flag`, `status_out`, `hint_out`, `i_out`, `step_out`, `t_out`.
- **Function symbols**: `stepper`, `rhs`, `events_pre`, `events_post` *(all jittable callables; no Python objects in hot path)*.

#### Runner Status Codes (int32):
- `OK=0` (internal)
- `REJECT=1`
- `STEPFAIL=2`
- `NAN_DETECTED=3`
- `DONE=9`
- `GROW_REC=10`
- `GROW_EVT=11`
- `USER_BREAK=12`

*Runner exits to wrapper only with: DONE/GROW_*/STEPFAIL/NAN/USER_BREAK.*

#### Stepper ← Runner (same for every stepper)
```python
status = stepper(
  t, dt, y_curr, rhs, params,
  sp, ss, sw0, sw1, sw2, sw3, iw0, bw0,
  y_prop, t_prop, dt_next, err_est
) -> int32
```
- **Read**: `t`, `dt`, `y_curr`, `params`, read-only constants `sp`, persistent `ss`.
- **Write**: `y_prop`, `t_prop`, `dt_next`, `err_est`, and internal `ss` as needed.
- **Do not touch**: `y_curr`, record buffers, runner cursors.
- **Behavior**: Adaptive methods loop internally until accept or fail; fixed-step returns `OK` once.

#### RHS (used by all steppers)
```python
rhs(t, y_vec, dy_out, params)
```
- Compiled from DSL; recomputes aux from the `y_vec` it receives, so multi-stage/adaptive evaluations are correct.

# DSL (TOML) Schema

### Required
```toml
[model]           # type="ode" | "map", label?, dtype? (default float64)
[states]          # names with initial values (order = file order)
[params]          # scalars/arrays (cast to model dtype)
```

#### Equations (choose A or B, or mix without duplicate targets)
- **A: per-state rhs**
```toml
[equations.rhs]
x = "expr"
y = "expr"
```
- **B: block form**
```toml
[equations]
expr = """
dx = ...
dy = ...
"""
```
*(‘dx’ maps to state ‘x’; same for others.)*

#### Auxiliaries (derived, read-only)
```toml
[aux]
E = "expr"
```
- Aux can depend on `t`, states, params, and earlier aux (acyclic).
- RHS inlines/derives aux from its `y_vec`.
- Events read aux computed from committed state.

#### Functions (pure numeric helpers)
```toml
[functions.name]
args = ["u","c","s"]
expr = "expr"
```
- Acyclic, stateless, numeric only; usable in rhs/aux/events.

#### Events (named subtables; execution = text order)
```toml
[events.ev_name]
phase  = "pre" | "post" | "both"
cond   = "expr"
action = "expr"
tags   = ["..."]     # optional
record = true/false  # optional (record event time)
log    = ["x", "y", "aux:E", "param:a"]   
```
- Allowed assignments: states, params. Forbidden assignments: aux, buffers, stepper internals.

### Simulation defaults
```toml
[sim]
t0=..., t_end=..., dt=..., stepper="euler"|"rk4"|..., record=true
```

### Meta
```toml
[meta]
title = "Neuron Model with Current" # metadata only for high level API
```

# Mods (build-time overlays)

### Purpose: modify a base model before codegen/JIT. Deterministic, no runtime cost.
Where they apply
- **states** (initial values)
- **params**
- **aux** (definitions)
- **functions**
- **events**
- *(optionally) sim defaults*

### Mod tables & verbs
- **Set / upsert** (create or update)
  - `[mod.set.states]`, `[mod.set.params]`
  - `[mod.set.aux]` (same syntax as [aux])
  - `[mod.set.functions.NAME]` with args + body
- **Add** (for named entities that are ordered or have bodies)
  - `[mod.add.events.EVENT_NAME]` → full event body
  - `[mod.add.functions.NAME]` → full function body
  - `[mod.add.aux.NAME]` → expr string
- **Replace** (by name; overwrites the whole entity)
  - `[mod.replace.events.EVENT_NAME]`
  - `[mod.replace.functions.NAME]`
  - `[mod.replace.aux.NAME]`
- **Remove** (by name list)
  - `[mod.remove.states]` names = ["v","u"]
  - `[mod.remove.params]` names = ["I"]
  - `[mod.remove.aux]` names = ["E"]
  - `[mod.remove.functions]names = ["sat"]
  - `[mod.remove.events]` names = ["reset","kick"]

### Event body forms (both allowed)
- **Keyed actions**:
```toml
  [mod.add.events.reset]
  phase = "post"
  cond  = "v >= 30"
  action.v = "-65"
  action.u = "u + 2"
  record = true
  log    = ["v","u","aux:E"]
```
- **Block action**:
```toml
  action = '''
  v = -65
  u = u + 2
  '''
```

### Grouping & exclusivity
Inside each mod:
```toml
[mod]
name = "drive_low"
group = "drive"      # bucket
exclusive = true     # at most ONE mod from this group may be active
priority = 0         # optional tie-break (higher wins) if multiple provided
```

### Build rules:
- If multiple active mods share a group and any has `exclusive=true`: error (unless only one is active).
- Otherwise, apply in the given list order (last-wins for conflicting keys).

### Application order (deterministic)
1. Parse base TOML → ModelSpec.
2. For each selected mod in order:
  - remove.* → delete
  - replace.* → overwrite entity by name
  - add.* → append entity (events keep text order of application)
  - set.* → upsert scalar tables (states/params), overwrite by key for aux/functions
3. Validate (names, cycles, dtype rules, events/actions legality).
4. Freeze spec → codegen/JIT.

### Validation highlights
- Unknown targets in remove.* → error (fail loud).
- set.states/set.params may create new keys; if you want stricter behavior, require add.* for creation and make set.* “update-only”.
- No assigning to aux inside events (events may read aux).
- Dtype rules still enforced after mods.
- Event names are unique; replace.events.X requires X to exist; add.events.X requires it not to exist.

### Selection API (examples)
- Inline string: `build(model="path.toml", mods=["inline: ..."])`
- File + named mod: `build(model="model.toml", mods=["mods.toml#mod=drive_low"])`
- Multiple mods: `mods=[MOD_GRP_1, MOD_GRP_2]` → exclusivity check triggers error for same group if both exclusive.

### Interaction with performance/JIT
- Mods are build-time only: they mutate the ModelSpec before codegen.
- No changes to runner/stepper ABI, buffers, or JIT behavior.
- Event record and log buffers are sized based on the final event set after mods.

# Model Paths / Registry (build-time resolution)

### Config locations
- Linux: `${XDG_CONFIG_HOME:-~/.config}/dynlib/config.toml`
- macOS: `~/Library/Application Support/dynlib/config.toml`
- Windows: `%APPDATA%\dynlib\config.toml`
- Override with env: `DYNLIB_CONFIG=/absolute/path/to/custom.toml`

### Config schema
```toml
[paths]
# Each key is a TAG; value is a list of directory roots (strings)
proj = ["./models", "./course/models"]
user = ["~/shared/models"]
labA = ["Z:/labA/models", "//server/share/models"]
```

### One-shot env override (optional)
- `DYN_MODEL_PATH` can supply extra tags/roots at runtime.
  - POSIX: `proj=/p1,/p2:user=~/u1`
  - Windows: `proj=C:\p1,C:\p2;user=%USERPROFILE%\u1`
- Merge policy: env entries are prepended to the tag’s root list (so they win on first match).

### Source URI forms accepted by `build(...)`
- Inline TOML string: `"inline: ..." → parse directly.
- Absolute path: `"/abs/path/model.toml"` → open directly.
- Relative path: `"models/ho.toml"` → resolve from CWD.
- Tagged URI: `"proj://ho.toml"` → search in `[paths].proj` roots (first match wins).
- Fragment selectors: `"file.toml#mod=drive_low"` (or multiple fragments later if needed).
- Optional: allow extensionless hints (`proj://ho`) to match `ho.toml` (document if you enable it).

### Resolution order (deterministic)
1. If `src` starts with `inline:` → use inline string.
2. Else if absolute → use it.
3. Else if `TAG://relpath`:
  - Expand `~` and env vars in each root.
  - Join `root/relpath`; if the path has no extension and extensionless resolution is enabled, try appending `.toml`.
  - First existing file wins.
4. Else treat as relative path from CWD (same extension rules).
5. On failure: raise a clear `ModelLoadError` listing all tried candidates.

### Security & portability
- Normalize to absolute paths after resolution.
- Reject traversal outside the declared root when resolving a tag (e.g., `TAG://../../secret` → error).
- Store content hash (not path) in caches to avoid wrong-code reuse after file moves.

### What can be resolved via paths
- Base model files.
- Mods files (same URI rules), including "inline: ..." mods.
- Later: allow `proj://bundle/` directories if you add package-style model bundles.

### Caching & JIT
- Compute a spec hash from:
  - file contents of model + selected mods (after path resolution),
  - chosen stepper, dtype, and StructSpec,
  - dynlib + numba version pins.
- Use that hash to name compiled artifacts (stable across machines if content matches).

### Error handling (fail loud, student-friendly)
- Unknown tag → list known tags from config.
- No file found → show the search roots and the exact candidate paths checked.
- Ambiguous extensionless match (if enabled) → error listing candidates; require explicit filename.

### Nice extras (optional, but harmless)
- `build("proj://ho.toml", mods=["user://extras/drive.toml#mod=high"])`
- `build_many(["proj://a.toml","proj://b.toml"], ...)` (batch scans use the same resolver)
- CLI `dynlib build proj://ho.toml` uses the same resolution function.

# Execution Semantics (per accepted step)
1. Pre phase: compute aux from committed (t, y_curr) → run events_pre (mutate y_curr/params), recompute aux if needed.
2. Stepper: calls rhs(t_s, y_stage, k_s, params) per stage; returns OK (accepted) or failure codes.
3. Commit: on OK, y_prev ← y_curr, y_curr ← y_prop, t ← t_prop.
4. Post phase: compute aux from committed state; run events_post.
5. Record: write T, Y, STEP, FLAGS; capacity check → GROW_REC.
6. Loop until t ≥ t_end or max_steps.

### Dtypes
- One primary dtype per model (from `[model].dtype`, default float64).
- ODEs must use float dtype. Maps may use float or int dtype.
- Recording: T=float64, Y=model dtype, STEP=int64, FLAGS=int32.
- Float banks use model dtype; iw0=int32; bw0=uint8.
- Expression validation enforces dtype rules (e.g., no sin in int models unless you explicitly add casts).

### StructSpec (per stepper, resolved at build time)
- Declares sizes for banks: SP, SS, SW0..SW3 (model dtype), IW0 (int32), BW0 (uint8).
- Declares feature flags (e.g., use_history, dense_output, needs_jacobian, embedded_order, stiff_ok).
- Builder allocates contiguous buffers per dtype and slices views for banks; signature stays fixed.

### Maintenance Hooks (post-commit, tiny & generic)
- If use_history: push (t,y) and/or f(y) into history ring (iw0 holds heads).
- If use_f_history: update multi-step RHS cache.
- If dense_output: store last step’s stage/coefs in ss/sw* (stepper decides layout).

### Recording & Growth
- Amortized O(1) geometric growth on GROW_REC (and GROW_EVT if enabled).
- Wrapper doubles capacity, copies filled slices, updates cursors, resumes the same runner (no reinit).

### Error/Interrupt Discipline
- Stepper returns: OK, REJECT (retry internally), STEPFAIL, NAN_DETECTED.
- Runner escalates only generic signals to wrapper: DONE, GROW_*, STEPFAIL, NAN_DETECTED, USER_BREAK.
- Commit/record/events happen only after OK (no side-effects on rejected attempts).

### Numba Policy
- Numba-first: runner/stepper/rhs/events compiled with @njit.
- Optional JIT: same function bodies without decorators if user disables JIT—no alternate code paths.
- Avoid Python objects, Optional, dynamic shapes, fancy indexing, or allocations in the hot path.

### Performance Notes
- Expect 10–100× vs pure Python loops for typical ODE workloads; overhead of function-argument calls is negligible compared to RHS math.
- Bottleneck will shift to memory bandwidth when recording every step; consider (later) stride recording or dense-output sampling if needed.

### Validation Checklist
- Unique names; no collisions among states/params/aux/functions.
- equations.rhs + equations.expr don’t assign the same state twice.
- Aux/function graphs are acyclic.
- Events only assign states/params; expressions reference known symbols.
- Dtype rules enforced (no floats in int models unless explicit and supported).
- For ODEs with int dtype → error.

### Minimal “do/don’t”
- **Do**: keep ABI frozen; declare all buffers in StructSpec; use feature flags for extras.
- **Don’t**: add Optional args, Python objects, or per-step allocations; mutate global state inside steppers; run events on uncommitted state.

This is the whole playbook. If you stick to these contracts, you can add RK45, multistep, DDE history, dense output, Jacobians, and new DSL features without touching the runner signature or re-entangling the pipeline.

# Suggested File Tree

.
├── CHANGELOG.md
├── README.md
├── pyproject.toml
├── manuals/
│   ├── ABI.md                 # runner ABI, buffers, status codes
│   ├── DSL.md                 # TOML schema, mods verbs, events
│   └── Steppers.md            # StepperSpec, StructSpec contracts
├── examples/
│   ├── models/                # sample TOML models + mods for docs/tests
│   └── plot_demo.py
├── src/
│   └── dynlib/
│       ├── __init__.py        # minimal public API re-exports
│       │
│       ├── runtime/           # hot path: wrapper/runner, buffers, sim/results
│       │   ├── __init__.py
│       │   ├── wrapper.py
│       │   ├── runner_api.py  # frozen call signatures + status codes
│       │   ├── buffers.py     # T/Y/STEP/FLAGS + evt buffers, growth helpers
│       │   ├── results.py     # Results dataclass + accessors
│       │   ├── model.py       # Model(spec, compiled_runner, stepper_name)
│       │   ├── sim.py         # thin Sim facade around Model + Wrapper
│       │   └── types.py       # Literal aliases (Kind/TimeCtrl/Scheme)
│       │
│       ├── steppers/          # jittable, numba-first implementations
│       │   ├── __init__.py
│       │   ├── base.py        # StepperMeta/Info, StepperSpec, StructSpec
│       │   ├── registry.py    # name -> StepperSpec
│       │   ├── euler.py
│       │   ├── rk4.py
│       │   └── rk45.py        # (later) adaptive; shares ABI
│       │
│       ├── dsl/               # TOML → ModelSpec (front-end)
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── astcheck.py
│       │   ├── parser.py
│       │   └── spec.py        # ModelSpec, SimDefaults, hashing
│       │
│       ├── compiler/          # resolve → mods → validate → codegen → JIT
│       │   ├── __init__.py
│       │   ├── build.py       # orchestrator: build(model_uri, mods, stepper)->Model
│       │   ├── paths.py       # TAG:// resolver, config loader
│       │   ├── mods.py        # apply set/add/replace/remove deterministically
│       │   ├── codegen/
│       │   │   ├── __init__.py
│       │   │   ├── emitter.py
│       │   │   ├── rewrite.py
│       │   │   └── runner.py  # generate_runner_source(...)
│       │   └── jit/
│       │       ├── __init__.py
│       │       ├── cache.py   # content/ABI hashing, cache paths
│       │       └── compile.py # numba compile/load from cache
│       │
│       ├── builtins.py        # EVENT_HELPERS, FUNCTION_DEFAULTS (pure numeric)
│       ├── errors.py          # ModelLoadError, ValidationError, MergeConflictError, …
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── arrays.py      # dtype casts, contiguous checks, view carving
│       │   ├── growth.py      # geometric growth sizing
│       │   └── hashing.py     # stable content hashing
│       │
│       └── plot/              # existing plotting utilities (unchanged)
│           ├── __init__.py
│           ├── _export.py
│           ├── _facet.py
│           ├── _fig.py
│           ├── _primitives.py
│           └── _theme.py
└── tests/
    ├── unit/
    │   ├── dsl/
    │   ├── compiler/
    │   ├── steppers/
    │   └── runtime/
    ├── integration/
    │   ├── ode_basic/
    │   └── events_logging/
    └── data/
        └── models/            # small TOML fixtures for tests




