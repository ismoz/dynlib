# DYNLIB v2 — Numba-Safe Implementation Plan

## Slice 0 — ABI Freeze + Numba Probe (no product JIT yet)

### Goal
Lock names/shapes and prove they compile under `@njit` right now. Product code stays JIT-free; tests do the probing.

### 0.1 Files (compile-free runtime)
- **`src/dynlib/runtime/runner_api.py`**
  - `IntEnum`/consts: `OK=0`, `REJECT=1`, `STEPFAIL=2`, `NAN_DETECTED=3`, `DONE=9`, `GROW_REC=10`, `GROW_EVT=11`, `USER_BREAK=12`.
  - Single source of truth for the runner signature (docstring includes exact shapes/dtypes).
  - Tiny `RunnerABI` dataclass (discoverability only; never used in hot path).
- **`src/dynlib/runtime/types.py`**
  - `Kind = Literal["ode","map"]`, `TimeCtrl = Literal["fixed","adaptive"]`, `Scheme = Literal["explicit","implicit","splitting"]`.
- **`src/dynlib/steppers/base.py`**
  - `StepperMeta`, alias `StepperInfo = StepperMeta`.
  - `StructSpec(sp_size, ss_size, sw0_size, sw1_size, sw2_size, sw3_size, iw0_size, bw0_size, use_history=False, use_f_history=False, dense_output=False, needs_jacobian=False, embedded_order=None, stiff_ok=False)`.
  - Abstract class `StepperSpec`: `def __init__(self, meta: StepperMeta); def struct_spec(self) -> StructSpec; def emit(self, rhs_fn, struct: StructSpec): ...`
- **`src/dynlib/steppers/registry.py`**
  - Minimal name→spec map; `register(spec)`, `get_stepper(name)`.
- **`src/dynlib/utils/arrays.py`**
  - Guards (not in hot loop): `require_c_contig`, `require_dtype`, `require_len1`, `carve_view(a, start, length)`.
- **`src/dynlib/__init__.py`**
  - Re-export frozen constants/types for stable imports.

### 0.2 Numba Probe (tests only)
- **`tests/test_numba_probe.py`**
  - JIT dummy `_rhs`, `_events_noop`, `_stepper`, `_runner` with your exact frozen signature (no unions, no `Optional`).
  - Allocate arrays with the model `dtype = float64` (for ODEs), use len-1 outs, pass callables into the jitted `_stepper` and `_runner`.
  - Assert `_runner(...) == DONE` and that `y_curr` committed and `t_out[0]` advanced.
- **Rules enforced in the probe:**
  - Concrete types only in jitted signatures (e.g., a single `params: float64[:]` for ODEs).
  - `t_prop`, `dt_next`, `err_est` are `model_dtype[1]`.
  - Function pointer args (`rhs`, `events_pre`, `events_post`, `stepper`) are all jitted.

### 0.3 CI matrix (fast)
- **Job A:** normal run (Numba enabled) → runs the probe.
- **Job B:** `NUMBA_DISABLE_JIT=1` → ensures everything (besides the probe) executes without needing alternate code paths.
- Add a lint step (`ruff`/`mypy`) but no `Optional` in any hot-path types.

### Exit criteria for Slice 0
- Imports succeed everywhere with the frozen names.
- Probe compiles & passes on CI A.
- CI B passes (probe skipped/xfail guarded by environment).

## Slice 1 — Buffers, Results, Wrapper (still JIT-free product code)

### Goal
Lock growth semantics and the wrapper↔runner re-entry contract. No codegen yet; no real runner yet.

### 1.1 Files
- **`src/dynlib/runtime/buffers.py`**
  - Contiguous pools per dtype; carve banks: `sp`, `ss`, `sw0..sw3` (model dtype), `iw0:int32`, `bw0:uint8`.
  - Recording: `T:float64`, `Y:model_dtype`, `STEP:int64`, `FLAGS:int32`; optional EVT buffers `EVT_TIME:float64`, `EVT_CODE:int32`, `EVT_INDEX:int32`.
  - Geometric growth helpers returning `(new_arr, new_cap)`; copy only filled slices.
- **`src/dynlib/runtime/results.py`**
  - `Results` dataclass exposing views, not copies: `T[:n]`, `Y[:, :n]`, etc.
  - Tiny helpers for plotting/pandas (out of hot path).
- **`src/dynlib/runtime/wrapper.py`**
  - Orchestrator that:
    - allocates banks & buffers using `StructSpec`,
    - calls the compiled runner once,
    - handles `GROW_*`, `NAN_DETECTED`, `STEPFAIL`, `USER_BREAK`,
    - reallocates and re-enters the same runner with updated caps/cursors.
  - **Recording discipline (frozen):**
    - Record at `t0` if `record=true`.
    - `record_every_step=N` → record on `step % N == 0` after post-events.
  - **Prev semantics (frozen):** after commit of step `k`: `y_prev ← previous committed y_curr`; on the very first commit, `y_prev := y_curr`.

### 1.2 Tests
- Buffer growth copies only filled region; cursors preserved.
- Wrapper re-entry adjusts caps and resumes (use a fake runner that returns `GROW_REC`/`GROW_EVT` immediately to exercise the loop).

### Exit criteria for Slice 1
- Deterministic growth discipline; wrapper calls runner only once per attempt and re-enters on growth.

## Slice 2 — DSL v2 (schema, parser, validation) + Mods

### Goal
Parse exactly the v2 TOML (no legacy), build a `ModelSpec`, and compute a content hash. Still no runnable model.

### 2.1 Files
- **`src/dynlib/dsl/schema.py`** — structural checks; forbid duplicates across `[equations.rhs]` and block expr.
- **`src/dynlib/dsl/parser.py`** — parse `[model]`, `[states]`, `[params]`, `[equations]`, `[aux]`, `[functions]`, `[events.*]`, `[sim]`.
- **`src/dynlib/dsl/astcheck.py`** — acyclicity for aux/functions; event legality: only states/params assignable; dtype rules (ODEs must be float).
- **`src/dynlib/compiler/mods.py`** — verbs `remove → replace → add → set` with groups/priority/exclusive, deterministic order.
- **`src/dynlib/dsl/spec.py`** — `ModelSpec`, `SimDefaults`, `compute_spec_hash`.

### 2.2 Tests
- Golden TOMLs for each table, grouped mods, fragments `#mod=...`, `TAG://` resolution (stub).
- Acyclic detection; duplicate target rejection; dtype enforcement.

### Exit criteria for Slice 2
- `ModelSpec` builds & validates with the exact fields you froze in the Guardrails.

## Slice 3 — Codegen (rhs/events) + JIT cache + Build pipeline (ODE only)

### Goal
Generate pure-numeric callables from the DSL, JIT them (optional JIT toggle lives only here), and produce a `Model`.

### 3.1 Files
- **`src/dynlib/compiler/codegen/rewrite.py`**
  - Sanitize names; inline `functions.*`; rewrite expressions to numeric Python; aux recomputed from `y_vec` in every `rhs` call.
- **`src/dynlib/compiler/codegen/emitter.py`**
  - Emit:
    - `rhs(t, y_vec, dy_out, params)` (pure numeric; no allocs).
    - `events_pre(t, y, params)`, `events_post(t, y, params)` (mutate only states/params; no buffer access).
- **`src/dynlib/compiler/jit/cache.py` / `compile.py`**
  - Optional JIT toggle (`jit=True|False`) applied inside compile layer.
  - Cache key: `(spec hash + stepper name + structspec + model dtype + version pins)`.
- **`src/dynlib/compiler/build.py`**
  - Resolve URI + mods; produce `Model(spec, compiled_runner, stepper_name)`.
- **`src/dynlib/runtime/model.py` & `src/dynlib/runtime/sim.py`**
  - Thin façade; `Sim(model).run(...)` delegates to wrapper.

### 3.2 Tests
- Tiny models (decay) generate `rhs`/`events` that execute under the probe stepper/runner from Slice 0.
- JIT toggle parity: with/without JIT → identical results for `rhs`/`events` unit tests.

### Exit criteria for Slice 3
- You can evaluate `rhs`/`events` in isolation (no real stepper yet), both jitted and non-jitted.

## Slice 4 — First real Stepper (Euler) + Generic Runner (JIT)

### Goal
Close the loop end-to-end with a fixed-step method and the generic runner.

### 4.1 Files
- **`src/dynlib/steppers/euler.py`**
  - `StructSpec` mostly size-1 banks.
  - `emit(...)` returns a jittable stepper:
    ```python
    rhs(t, y_curr, sw0, params)
    y_prop = y_curr + dt * sw0
    t_prop[0] = t + dt
    dt_next[0] = dt
    err_est[0] = 0
    return OK
    ```
- **`src/dynlib/compiler/codegen/runner.py`**
  - One generic `@njit` runner with the frozen signature:
    1. pre-aux → `events_pre`
    2. stepper loop (accept/fail)
    3. commit: `y_prev`, `y_curr`, `t`
    4. post-aux → `events_post`
    5. record; capacity checks; loop end; return status

### 4.2 Tests
- `dx/dt = -a*x`: matches analytic solution within tolerance.
- Event mutation (reset at threshold), event logging, and recording discipline verified.
- Growth paths triggered by small caps; wrapper re-enters and completes.

### Exit criteria for Slice 4
- End-to-end simulation works with Euler; results identical with JIT on/off.

## Slice 5 — More Steppers (rk4 fixed; rk45 adaptive)

### Goal
Add steppers without touching wrapper/results/ABI.
- **`steppers/rk4.py`**: fixed.
- **`steppers/rk45.py`**: adaptive, internal accept/reject loop; uses `err_est` and sets `dt_next[0]`.

### Tests
- Order checks, adaptive step count varies but trajectory matches tolerance; no ABI changes.

## Slice 6 — Paths/Registry & CLI niceties (optional)
- Config/ENV per your spec (`DYNLIB_CONFIG`, `DYN_MODEL_PATH` prepend semantics).
- `TAG://` resolver; error messages hardened.

## Cross-cutting “Never drift from Numba” rules (frozen now)
- No `Optional`/`Union` in any jitted signature, ever.
- Hot path = only scalars/ndarrays of fixed dtypes; no Python objects/allocations.
- All banks/outs are C-contiguous; len-1 outs have shape `(1,)`.
- ODE model `dtype` is float; `T`/`EVT_TIME` are always `float64`.
- Events may mutate only states/params; they never touch buffers or cursors.