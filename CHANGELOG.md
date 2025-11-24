## Changelog

---

## [2.30.6] – 2025-11-25
### Added
- Added a new `analysis` module with parameter sweep tools for running simulations with different parameter 
  values and collecting results. It is an early sketch.

---

## [2.30.5] – 2025-11-24
### Added
- Added support for `range()` function in DSL expressions. Arguments are automatically cast to integers for 
  proper Numba compatibility.
- Added validation to check that all identifiers used in expressions are properly declared as states, parameters, 
  auxiliary variables, functions, or constants, or are supported builtins and macros. This prevents typos and 
  undefined references.

### Tests
- Updated test models to remove manual `int()` casts around `range()` calls, as the function now handles type 
  conversion internally.

---

## [2.30.4] – 2025-11-24
### Added
- Added user-defined `[constants]` table support to inline model-specific numeric literals across DSL
  expressions, with collision guards against states/params/aux and reserved identifiers.

---

## [2.30.3] – 2025-11-24
### Added
- DSL builtin constants `pi` and `e` are now inlined as numeric literals across equations, aux/functions,
  events, and initial value expressions. These identifiers are reserved and cast to the model dtype at
  codegen time to avoid runtime lookups or dtype mismatches.
- Added cross-section identifier guard in `src/dynlib/dsl/astcheck.py` and wired it into build_spec so models 
  now error when a name is reused across states/params/aux/functions (e.g., param V vs aux V).

### Tests
- Added semantic validation coverage in `tests/unit/test_semantic_validation.py` to ensure these conflicts 
  raise a clear `ModelLoadError`.

---

## [2.30.2] – 2025-11-23
### Changed
- Runners now refresh auxiliary variable values before recording initial conditions to ensure aux data is 
  available at the start for recording.
- Results API now properly handles variable names from recorded data instead of listing all available states. 

---

## [2.30.1] – 2025-11-23
### Changed
- Warm-up now matches the new runner ABI and quadruplet callables so JIT compilation is triggered during 
  build again.
- `_warmup_jit_runner` now allocates AUX/selective-recording buffers, computes event log width, and passes 
  `state_rec_indices`, `aux_rec_indices`, and counts to the runner along with the AUX buffer.
- `_all_compiled` checks `update_aux` too, preventing skips when only the aux updater lacks signatures.

---

## [2.30.0] – 2025-11-23
### Added
- Added `update_aux` callable to compute auxiliary variables from current state values. This function is 
  called after each committed step to ensure aux variables are available for recording and event conditions. 
  Inside RHS the aux values are still replaced by expressions to perform fast stepper multi-stage calculations.
- Added selective variable recording via new `record_vars` parameter in `Sim.run()`. Users can now specify 
  exactly which variables to record:
  - `record_vars=None` (default): Record all states (backward compatible)
  - `record_vars=["x", "y"]`: Record specific states
  - `record_vars=["aux.energy"]`: Record specific aux variables with explicit prefix
  - `record_vars=["energy"]`: Record aux variables with auto-detection (no prefix needed)
  - `record_vars=["x", "energy", "aux.power"]`: Mix states and aux variables
  - `record_vars=[]`: Record nothing (only time, step, flags; equivalent to `record=False`)
- Added `aux_values` array to `RuntimeWorkspace` for storing computed auxiliary variable values during 
  simulation.
- Enhanced `Results` class with `AUX` array for recorded auxiliary variables, plus `state_names` and `aux_names` 
  metadata.
- Added `get_var()` and `__getitem__()` methods to `Results` for accessing recorded variables by name with auto-
  detection.
- Added `to_pandas()` support for auxiliary variables in selective recording.

### Changed
- Updated runner functions to call `update_aux` after each committed step to maintain aux variable values.
- Modified JIT compilation to handle quadruplet (rhs, events_pre, events_post, update_aux) instead of triplet.
- Enhanced `ResultsView` to support accessing auxiliary variables alongside states.
- Updated `Sim` class to handle selective recording with proper buffer allocation and metadata tracking.
- Modified `run_with_wrapper` to pass selective recording parameters to runners.

### Tests
- Added 4 new tests in `tests/unit/test_selective_recording.py` to verify auto-detection behavior:
  - `test_aux_auto_detection_without_prefix`: Aux variables work without prefix
  - `test_mixed_auto_detect_and_explicit_prefix`: Mixing both syntaxes
  - `test_state_priority_over_aux_same_name`: States take priority in detection
  - `test_unknown_variable_helpful_error`: Error messages list available variables
- Updated existing tests to match improved error messages.
- Added comprehensive test coverage for selective recording functionality including buffer growth, resume 
  behavior, and error handling.

### Known Issues
- JIT warm-up is broken again. Possibly related to the new quadruplet approach.

---

## [2.29.6] – 2025-11-23
### Added
- Added support for adding and removing parameters via mods. Now `add.params` can be used to add new 
  parameters and `remove.params` to remove existing ones.
- Added validation to prevent unsupported targets in mod operations. Now clear error messages are shown 
  when trying to use invalid targets like `add.states` or `remove.states`.
- Added comprehensive documentation for mods in `docs/mods.md`, explaining all verb operations, supported 
  targets, error handling, and best practices.

### Tests
- Added tests for parameter modification via mods, including adding/removing parameters and validation of 
  unsupported targets.

---

## [2.29.5] – 2025-11-23
### Added
- Added support for `sum()` and `prod()` generator comprehensions in DSL expressions. These allow summing or 
  multiplying over ranges with optional conditions, like `sum(i*i for i in range(10))` or 
  `prod((i+1) for i in range(1, 5) if i % 2 == 0)`. They are compiled into efficient for-loops and work in 
  equations, aux variables, events, and functions.
- Updated documentation in `docs/dsl_macros_and_functions.md` with a new "Generator Comprehensions" section 
  explaining the syntax and examples.

### Tests
- Added comprehensive tests in `tests/unit/test_sum_generator_lowering.py` to verify the functionality works 
  correctly with both JIT and non-JIT compilation.

---

## [2.29.4] – 2025-11-21
### Added
- Added validation to prevent auxiliary variables from using reserved names like `t` to avoid conflicts 
  with runtime symbols. `_AUX_RESERVED_NAMES` list can be expanded to restrict aux names in the future.

### Changed
- Renamed auxiliary variable in Ikeda map model from `t` to `theta` for clarity and to avoid reserved 
  name conflict.

### Tests
- Added a test for reserved auxiliary name validation.

---

## [2.29.3] – 2025-11-20
### Added
- Added documentation for DSL model file template in `docs/dsl_model_template.md`.
- Added several builtin models for ODE and MAP types:
  - ODE models: `exp_if` (Exponential Integrate-and-Fire), `fitzhugh_nagumo`, `hodgkin_huxley`, `leaky_if` 
    (Leaky Integrate-and-Fire), `quadratic_if`, `resonate_if`.
  - MAP models: `henon`, `ikeda`, `logistic`, `lozi`, `sine`, `standard`.

---

## [2.29.2] – 2025-11-20
### Added
- Added `choose_default_stepper()` function to automatically select appropriate steppers based on model type
  if DSL model spec and user does not provide one. The hard-coded defaults are `map` -> `map` and `ode` -> `rk4`.
- Added theme usage examples in collatz.py, detect_transition.py, and logistic_map.py to demonstrate theme 
  presets.

### Changed
- Improved plotting primitives with enhanced docstrings and better parameter handling.
- Standardized arguments of plotting functions.

---

## [2.29.1] – 2025-11-20
### Added
- Added plotting theme system using `ThemeSpec` dataclass with inheritance support for better theme management 
  and customization.
- Added `themes_demo.py` example to demonstrate all available theme presets with sample plots.

### Changed
- Enhanced savefig function to properly handle constrained layout without clipping.
- Improved style resolution with clear priority hierarchy separating visual patterns from rendering properties.

---

## [2.29.0] – 2025-11-20
### Added
- Added `dynlib` CLI entry point with `model validate`, `steppers list`, and `cache` management 
  subcommands for model validation, registry inspection, and JIT cache cleanup. The CLI entry point is
  placed into `src/dynlib/cli.py`.

### Tests
- Added unit tests covering the new CLI flows (model validation success/failure, stepper filters, cache 
  listing/clearing).

---

## [2.28.9] – 2025-11-20
### Added
- Added validation script for adaptive ODE steppers tolerance sweep.
- Added RK2 (explicit midpoint) stepper for fixed-step ODE simulations.
- Added SDIRK2 (Alexander) stepper, a JIT-compatible implicit method for stiff ODEs.

### Tests
- Added basic accuracy and contract tests for RK2 and SDIRK2 steppers.

---

## [2.28.8] – 2025-11-20
### Added
- Added automatic initial step size selection for adaptive ODE steppers using Hairer/Shampine-style WRMS 
  norm heuristics. Now dt arg for adaptive ode steppers is just a suggestion and max step bound. Stepper 
  tol, atol, meta.order values are accessed automatically during heuristics. Placed the algorithm into 
  `initial_step.py` module. Heuristics are disabled if `resume=True`. 

### Changed
- Updated simulation wrapper to choose initial dt based on stepper type and configuration.
- Modified Sim class to support WRMS config for adaptive steppers.

### Tests
- Added tests for new initial step size selection heuristic feature.

---

## [2.28.7] – 2025-11-20
### Changed
- Improved `bdf2a` stepper startup by using Richardson extrapolation for better accuracy on the first step. 
- Optimized `tr-bdf2a` stepper by switching to modified Newton method with frozen Jacobian for faster 
  convergence. Full explicit Backwards Euler stage can be further optimized by estimating the error instead 
  of full implicit solve operation. However, for the sake of robustness, I skipped this optimization.

### Fixed
- Fixed wrong error estimation coefficients for `bdf2a` and `tr-bdf2a` steppers. Using error**0.5 was causing 
  parity between numba and python results. Used math.sqrt(error) for exactly matching results.

### Tests
- Added accuracy and contract tests for `bdf2a` and `tr-bdf2a` steppers.

---

## [2.28.6] – 2025-11-19
### Added
- Added `bdf2a` stepper, an adaptive BDF2 method with variable step size for stiff ODEs.
- Added `tr-bdf2a` stepper, an adaptive TR-BDF2 method combining trapezoidal rule and BDF2 for better 
  stability. It is not optimized, I might optimize it in the next version if it is feasible.

### Changed
- Improved RK45 stepper performance by moving k1 computation outside the adaptive retry loop.
- Renamed `StepperMeta` `stiff_ok` key to `stiff`.

---

## [2.28.5] – 2025-11-19
### Added
- Added `select_steppers()` function to filter steppers by metadata fields like kind, scheme, jit_capable, 
  etc.
- Added `list_steppers()` function to get a list of stepper names matching filter criteria.
- Added `validation/` folder with `ode_steppers_dt_sweep.py` script for benchmarking ODE stepper accuracy 
  across different time steps.

### Tests
- Fixed stepper name in `test_stepper_config.py` test to use a stepper with model config key.

---

## [2.28.4] – 2025-11-19
### Added
- Added `state()` and `param()` methods to `Sim` class for accessing individual state and parameter 
  values by name.
- Added `stepper` property to `Sim` class to access the stepper specification.
- Added a simple exponential decay model (`expdecay.toml`) as a builtin example.

### Changed
- Moved guards configuration earlier in `build()` to ensure guards are ready before JIT compilation.
- Updated `get_guards()` to install guard consumers automatically.

### Fixed
- `jit=True` and `disk_cache=False` combination was raising nopython errors for NaN/Inf guards (like 
  allfinite1d). Added `register_guards_consumer` function in `guards.py` to allow guards to update 
  existing stepper namespaces. In the future a proper numba inlineable functions registry would be 
  better but this fix works right now.
- Added endpoint clipping in the runner to handle cases where dt would overshoot `t_end`, ensuring 
  accurate final time steps.
- Registered guards consumers in RK45 and BDF2 steppers for proper NaN/Inf detection updates.
- Reordered history rotation in BDF2 stepper to prevent aliasing issues between current and proposed 
  states.

---

## [2.28.3] – 2025-11-19
### Added
- Added `bdf2a_scipy` stepper which is adaptive BDF2 solver based on scipy root solvers.

### Tests
- Added accuracy and contract tests for `bdf2a_scipy`.

### Fixed
- `bdf2` was underperforming according to the `accuracy_demo.py`. Added scale-aware residual bookkeeping 
  and a correction-based convergence check to the Newton loop so the solver can no longer declare success 
  while the update is still O(dt) in size; both BDF1 and BDF2 branches now track the largest state 
  magnitude and require the max residual and the scaled correction to fall below newton_tol before exiting, 
  which restores the intended second-order accuracy. This fixed its very low accuracy.

---

## [2.28.2] – 2025-11-18
### Added
- Added Van der Pol oscillator model and its example with jit-enabled `bdf2` stepper.
- Added a warning when `max_steps` is hit by the runners.
- Added `accuracy_demo.py` to examples for comparing errors of ode steppers against known models.

### Changed
- Renamed `bdf2_jit` -> `bdf2` and `bdf2` -> `bdf2_scipy` because jittable custom bdf implementation is 
  way faster than minpack-based scipy solver and its accuracy is similar. I will treat jittable BDF2 
  implementation as the main BDF stepper.

### Tests
- Added accuracy and contract tests for `bdf2_scipy`.

---

## [2.28.1] – 2025-11-18
### Added
- Added support for extra stepper configuration defaults in the [sim] section of model files. Previously 
  only hardcoded rtol/atol values were handled as [sim] stepper config values. Now unknown keys in [sim] 
  are stored as stepper defaults and used when building stepper configurations. The precedence is still 
  as in v2.28.0. [sim] stepper config values are not used if user overrides the stepper. Not all configs 
  should be listed. The stepper defaults are applied for the ones not provided.

### Changed
- Updated `SimDefaults` class to handle extra stepper configuration keys dynamically.

### Tests
- Added tests for stepper config handling, including runtime overrides, model defaults, and extra sim keys.

---

## [2.28.0] – 2025-11-18
### Added
- Introduced `ConfigMixin` base class in `config_base.py` for automatic stepper configuration handling. 
  It makes stepper declarations more concise.
- Added `config_utils.py` file that provides common tools for stepper config values handling. 
- Added new `bdf2` stepper that uses `scipy.optimize.root` for solving implicit equations.
- Added `requires_scipy` flag to stepper capabilities to indicate scipy dependency.
- Added support for string enum values in stepper configurations via `config_enum_maps()`.

### Changed
- Refactored all stepper implementations to use `ConfigMixin` for config management.
- Updated `_build_stepper_config` to automatically convert string config values to integer enum values 
  using `convert_config_enums` (provided in `config_utils.py`).
- Updated stepper config default value precedence to: Stepper Config Defaults < Model [sim] table values 
  < User Inputs. This is handled by the `default_config` method of the `ConfigMixin` class. So this base 
  class must be applied to the steppers for this precedence to apply.

---

## [2.27.2] – 2025-11-18
### Added
- Added `jit_capable` flag to `StepperCaps` to specify if a stepper supports JIT compilation.
- Introduced `softdeps.py` module for centralized detection of optional dependencies like numba and 
  scipy.
- Added `stepper_checks.py` module to validate stepper capabilities and dependencies before building 
  models.
- New `StepperJitCapabilityError` exception raised when requesting JIT for incompatible steppers.

### Changed
- Updated `build()` function to perform stepper capability checks, ensuring compatibility with 
  requested options.
- Refactored dependency detection in `guards.py`, `jit/compile.py`, `runner.py`, and 
  `runner_discrete.py` to use the centralized `softdeps` system.

### Tests
- Added `test_stepper_jit_capability.py` to test JIT capability validation and error handling.

---

## [2.27.1] – 2025-11-17
### Changed
- Improved BDF2_JIT stepper by adding checks for NaN/Inf values during calculations to exit early in 
  case of invalid data. Also improved Jacobian calculations.

### Tests
- Added BDF2_JIT stepper contract and accuracy tests.

---

## [2.27.0] – 2025-11-17
### Added
- Jit compatible BDF2 (Backward Differentiation Formula 2nd Order) stepper `bdf2_jit` is added. It 
  utilizes a simple Newton method plus custom numeric Jacobian. Therefore, it is not as reliable as 
  other solvers utilizing minpack or similar dedicated root finders. However, it is numba compatible, 
  so it can be used for fast simulations or analysis of stiff models to some extend.

## Changed
- Changed `JacobianPolicy` values as `none`, `internal`, `optional`, `required`.
  - `none`     : No Jacobian during calculations.
  - `internal` : Uses a numerical Jacobian approximation. Users can't pass externally.
  - `optional` : Users can provide external Jacobian. Fallback is `internal`.
  - `required` : USers should pass an external Jacobian.

### Fixed
- Simulations were continuing after transient or subsequent runs with `resume=True` even though a 
  `STEPFAIL` was raised by the steppers previously. Ensured the transient warm-up respects runner 
  failures by validating each warm-result before touching session state or shifting time, so `Sim.run` 
  now aborts immediately on a `STEPFAIL`/`NAN_DETECTED` instead of rolling forward with stale values. 
  Wrapped the recorded run in the same guard, so no rebasing, session-state updates, or segment 
  stitching happens unless the wrapper reports `DONE`, keeping resume histories consistent.
- Added `_ensure_runner_done` to give a clear RuntimeError that includes the failing phase and status 
  name, centralizing the exit check.

### Tests
- Added regression tests `tests/unit/test_sim_failure.py` that monkeypatch `_execute_run` to simulate 
  `STEPFAIL` returns, covering both transient and main segments, and verifying that state/records remain 
  untouched and `run()` raises.
- Updated `tests/unit/test_nan_inf_guards.py` to import pytest, expect `Sim.run()` to raise a RuntimeError 
  mentioning `NAN_DETECTED`, and assert the session state remains untouched after the aborted run. This 
  aligns the test with the stricter failure handling added to `Sim.run`.

---

## [2.26.5] – 2025-11-16
### Added
- Added new `StepperCaps` dataclass to hold stepper-specific features that can be added or removed 
  without changing the rest of the stepper `StepperMeta` declarations.

### Changed
- Moved `dense_output` flag from `StepperMeta` to `StepperCaps` for better organization.
- Updated all stepper implementations (Euler, RK4, RK45, AB2, AB3, Map) to use the new caps structure.

---

## [2.26.4]
### Tests
- Updated all tests according to the new workspaces design.
- All tests pass and examples work at this point.

### Fixed
- Running the entire unit suite started failing with `ModuleNotFoundError: dynlib_stepper_<digest>` 
  because numba’s cache metadata points to generated module names that were no longer importable 
  once earlier tests cleaned up their in-memory modules. Added a cache importer meta-path finder 
  plus sys.modules registration (`cache_importer.py`) so cached steppers/runners/triplets can 
  always be re-imported from the dynlib cache root, which restores `pytest tests/unit` stability 
  while keeping individual test 
  runs unchanged.

---

## [2.26.3] – 2025-11-16
### Changed
- Removed NaN/Inf checks from `AB2` and `AB3` steppers, since they are fixed-step solvers.
- Removed workbanks related docstrings from steppers.

---

## [2.26.2] – 2025-11-16
### Changed
- Updated `snapshot_demo.py` and `uri_demo.py` examples. All examples work at this point.

### Tests
- Updated `test_snapshot_persistence.py` test.

---

## [2.26.1] – 2025-11-16
### Changed
- Updated the docs throughout the package. Removed remnants of the old workbanks docs.
- Removed stepper_banks.md file and introduced stepper_workspace.md file.
- Updated ISSUES.md and TODO.md files.

### Fixed
- Cobweb plotter was still using the old workbanks API. Now it also uses runtime workspace.

---

## [2.26.0] – 2025-11-15
### Added
- Introduced separate stepper and runtime workspaces to cleanly separate responsibilities:
  - **Stepper workspace**: Private to each stepper, implemented as a NamedTuple-of-NumPy-views 
    containing stepper-specific scratch arrays (e.g., stages, histories, Jacobians).
  - **Runtime workspace**: Private to the runner and DSL machinery, implemented as a NamedTuple 
    containing lag buffers (lag_ring, lag_head, lag_info) for historical state access.
- Added `StepperSpec.workspace_type()` and `StepperSpec.make_workspace()` methods for steppers 
  to declare and allocate their workspace.
- Added `RuntimeWorkspace` NamedTuple in `src/dynlib/runtime/workspaces.py` for lag buffer 
  management.
- Added `make_runtime_workspace()` helper in `src/dynlib/runtime/workspaces.py` to allocate 
  runtime workspace from lag metadata.

### Changed
- Removed `WorkBanks` and `StructSpec` from public API: Eliminated the shared banking scheme 
  (sp, ss, sw*, iw0, bw0) that mixed stepper scratch with runtime state. Workspaces are now owned 
  by their respective components.
- Updated stepper ABI: Simplified stepper signature to `stepper(t, dt, y_curr, rhs, params, 
  runtime_ws, stepper_ws, stepper_config, y_prop, t_prop, dt_next, err_est) -> int32`, removing bank 
  arguments and passing workspaces directly.
- Updated runner ABI: Runner now accepts `runtime_ws` and `stepper_ws` instead of bank arrays. 
  Runner handles lag updates using runtime workspace instead of global `_LAG_STATE_INFO` and ss/iw0 
  banks. Removed `_LAG_STATE_INFO`.
- Refactored lagging system: Lag buffers moved to dedicated runtime workspace with circular 
  buffer access. Removed partitioning of ss/iw0 banks for lags.
- Migrated all steppers: Updated Euler, RK4, RK45, AB2, AB3, and Map steppers to use new 
  workspace pattern. Each stepper defines its own NamedTuple workspace type and factory.
- Updated code generation: RHS/event lowering now accesses lags via runtime workspace helpers 
  instead of ss/iw0. Generated code remains Numba-friendly.
- Enhanced workspace serialization: Added `snapshot_workspace()` and `restore_workspace()` 
  helpers for workspace persistence during resume.

### Fixed
- Eliminated cross-talk between lagging and stepper workspaces, preventing lag corruption in complex 
  models.
- Improved workspace reallocation safety: workspaces never grow during runs, only RecordingPools / 
  EventPools do.

### Tests
- Added comprehensive tests for new workspace allocation and migration.
- Updated all stepper tests to use new ABI.
- Added tests for lag system with runtime workspace.
- Verified resume/snapshot functionality with separated workspaces.

### Known Issues
- Many parts need refactoring for the new workspace approach. Most tests and examples will fail at 
  this point.

---

## [2.25.1] – 2025-11-15
### Added
- Added AB3 (Adams-Bashforth 3rd order) stepper for ODE simulations.
- Added basic and contract tests for AB3.

---

## [2.25.0] – 2025-11-15
### Added
- Added AB2 (Adams-Bashforth 2nd order) stepper for ODE simulations.
- Added basic tests for AB2 stepper accuracy.
- Added contract tests for ODE steppers to ensure JIT on/off parity and proper registration.
- Added separate test files for RK4 and RK45 steppers using the new API.

### Changed
- Updated RK45 stepper default tolerances (atol=1e-6, rtol=1e-3, max_factor=5.0) for better balance.

### Tests
- Reorganized ODE stepper tests. Now each stepper will have two test files:
    1) `test_<stepper_name>_basic.py`
      - Accuracy vs analytic decay
      - Order test (dt vs dt/2)
    2) `test_ode_stepper_contract.py` (single file for all steppers)
      - JIT on/off parity
      - Buffer growth invariance (your cap_rec tests)
      - Transient warm-up behavior
      - Registry / alias correctness
      - Maybe a transient or growth test
- Refactored test files to use the `setup()` helper instead of manual model building.
- Deleted duplicate test model files (`decay_rk4.toml`, `decay_rk45.toml`).
- Removed old combined RK4/RK45 integration test file.

---

## [2.24.1] – 2025-11-14
### Added
- Added scalar DSL macros usable in aux, equations, and event actions: `sign(x)`, `heaviside(x)`, 
  `step(x)`, `relu(x)`, `clip(x, a, b)`, and `approx(x, y, tol)`. They lower to comparisons and 
  builtins only, keeping generated code Numba-friendly.

### Tests
- Added regression coverage for the new macros in `tests/unit/test_scalar_macros.py`.

---

## [2.24.0] – 2025-11-14
### Added
- Added DSL event macros for common transition detection: 
    - `cross_up(state, threshold)`, 
    - `cross_down(state, threshold)`, 
    - `cross_either(state, threshold)`, 
    - `changed(state)`, 
    - `in_interval(value, lower, upper)`, 
    - `enters_interval(state, lower, upper)`, 
    - `leaves_interval(state, lower, upper)`, 
    - `increasing(state)`, `decreasing(state)`.

### Changed
- Updated `detect_transition.py` example to use the new `cross_up` macro instead of manual lag 
  condition.

### Tests
- Added comprehensive tests for event macros in `test_event_macros.py`.
- Updated lag detection to recognize macro usage in expressions.

---

## [2.23.6] – 2025-11-14
### Fixed
- Event log buffer reallocation was causing data loss for post events. Reordered event handling in 
  runners (both `runner.py` and `runner_discrete.py`) to check post-events on proposed state before 
  committing. This way reallocation during post-events now occur before commit, and this prevents 
  uncaught event logs.

### Changed
- DSL event tables now default to `phase = "post"` when the key is omitted, simplifying the common
  case where only post-step events are needed.

### Tests
- Added unit coverage ensuring the parser backfills the default event phase.

---

## [2.23.5] – 2025-11-14
### Added
- Added support for numeric expressions in states and parameters. You can now use strings like 
  "8/3" or "1/2" that get evaluated to numbers.
- Added a new example `detect_transition.py` showing how to use the lag system to detect when 
  a state variable crosses from negative to positive.

### Changed
- Improved TOML parsing error messages with better context, line numbers, and helpful hints for 
  common mistakes like division in values.
- Updated plotting functions to accept single numbers in `vlines` parameter, not just tuples.

### Tests
- Added tests for numeric expressions in model states and parameters.
- Added comprehensive tests for improved TOML error messages.

### Known Issues
- Event log buffer reallocation causes data loss.

---

## [2.23.4] – 2025-11-14
### Added
- Added `uses_lag` and `equations_use_lag` flags to model classes to track lag feature usage.
- Added `detect_equation_lag_usage` function to check if model equations depend on lag functions. 
  It tries to uncover all dependencies because aux and functions used in equations may also rely 
  on lagged values. 

### Changed
- Updated cobweb plotting to prevent usage with models that have lag in equations, as it cannot 
  evaluate them properly. Future analysis / plot tools that use model equations should not forget 
  that lag mechanism will not work without a proper Sim object. They should perform a similar check.

---

## [2.23.3] – 2025-11-14
### Fixed
- `_LAG_STATE_INFO` value was shared globally between different python (non-jitted) runners. This 
  was causing corrupted lag info between different runners. Fixed lag state info to be per-runner 
  instance instead of global, preventing interference between models with different lag configurations.

### Tests
- Added new tests covering lag info corruption issue to `test_lag_system.py`.

---

## [2.23.2] – 2025-11-14
### Changed
- Removed support for the `prev_<name>` DSL shorthand. Now `lag_<name>()` is used as a shorthand for 
  one-step lag. `lag_<name>(k)` usage stays the same.

---

## [2.23.1] – 2025-11-14
### Added
- Added `ss_lag_reserved` field to `StructSpec` for lag buffer allocation in stepper state. If
  a stepper needs to use the ss bank, it should use starting from this index. iw0 bank already
  has `iw0_lag_reserved` from the previous version.

### Changed
- Updated stepper banks documentation with lag system partitioning rules for `ss` banks.
- Updated build process to include lag reservations in struct specification.

---

## [2.23.0] – 2025-11-14
### Added
- Added lag system to access historical state values in models using `lag_<name>(k)` for k steps
  back or `prev_<name>` for one step back. This enables delay differential equations and lagged 
  feedback in both ODE and map models.
- Added automatic detection and validation of lag usage in model expressions, with sanity limits 
  on lag depths.
- Added circular buffer storage for lagged states using existing `ss` and `iw0` stepper banks with 
  partitioning to avoid ABI changes.
- Added lag buffer initialization with initial conditions and updates after committed steps only.
- Added comprehensive documentation for the lag system in `docs/lag_system.md`.
- Added `lag_state_info` metadata to `FullModel` and `Model` classes for runtime lag buffer 
  management.

### Changed
- Updated all stepper implementations (`euler`, `rk4`, `rk45`, `map`) to pass `ss` and `iw0` 
  parameters to RHS and event functions for lag support.
- Updated runner functions (`runner.py`, `runner_discrete.py`) to maintain lag buffers after step 
  commits and embed lag metadata as compile-time constants.
- Updated code generation (`emitter.py`, `rewrite.py`) to handle lag notation in expressions and 
  generate circular buffer access code.
- Updated model building (`build.py`) to augment stepper struct specs with lag requirements and 
  convert lag maps to runtime indices.
- Updated DSL parsing and validation (`astcheck.py`, `spec.py`) to collect lag requests and build 
  lag metadata.
- Updated simulation wrapper (`wrapper.py`) to initialize lag buffers on first run.
- Updated stepper banks documentation (`stepper_banks.md`) with `iw0` partitioning rules for lag 
  heads.
- Updated plotting primitives (`_primitives.py`) to handle lag buffers in cobweb plots.

### Tests
- Added comprehensive unit tests for lag system functionality in `tests/unit/test_lag_system.py`, 
  including buffer tracking, resume behavior, and correctness validation.
- Updated existing tests to accommodate new function signatures with `ss` and `iw0` parameters.

### Known Issues
- Some lag feature related issues will be resolved in the following updates.

---

## [2.22.0] – 2025-11-13
### Added
- Added support for builtin models using the "builtin://" URI scheme. This lets users access bundled 
  models without setting up paths manually.
- Added Izhikevich neuron model as a builtin model with presets for different spiking patterns like 
  regular spiking, bursting, and fast spiking.
- Updated path resolution to automatically include the builtin models directory in the search paths.

### Changed
- Updated izhikevich.py example to use the builtin Izhikevich model and show how to apply presets 
  during simulation runs.

### Tests
- Updated path resolution tests to check that the builtin tag is registered and models can be found.

---

## [2.21.6] – 2025-11-13
### Added
- Added `add_preset()` method to `Sim` class. It lets you create new presets from the current 
  session state or by providing specific values for states and parameters.
- Added support for partial presets. You can now define presets with only some states or 
  parameters instead of requiring all of them. Parameters are no longer mandatory.

### Changed
- Updated preset validation to allow empty states or params sections, as long as at least one is 
  defined.
- Modified `apply_preset()` to update only the provided values, leaving other states and parameters
  unchanged.
- Enhanced `load_preset()` and `save_preset()` to handle partial presets and provide better error 
  messages.
- Updated presets demo in examples to show how to use the new preset features.

### Tests
- Added new tests for partial presets to the `test_presets.py` and updated existing ones.

---

## [2.21.5] – 2025-11-13
### Changed
- Updated cobweb plotting function so that it works with new v2 sim or model objects.
- Updated logistic_map.py example to use themes, grid layouts, and cobweb plots.

---

## [2.21.4] – 2025-11-13
### Added
- Added `state_vector()`, `param_vector()`, `state_dict()`, and `param_dict()` methods to `Sim` 
  class. They let getting state and parameter values as arrays or dictionaries from the current 
  session, model defaults, or saved snapshots.

### Changed
- Updated `izhikevich.py` to show how to access state/parameter values from snapshots.

---

## [2.21.3] – 2025-11-13
### Added
- Added `Sim.config()` method to set default simulation settings like dt, max_steps, record options,
  and capacities. Stepper specific parameters can also be set with this method. They are forwarded
  to `stepper_config()`.
- Added `facet.py` example showing how to create faceted plots with multiple subplots.
- Added support for colored bands in plotting functions, allowing bands to have custom colors.

### Changed
- Renamed plotting parameters from `events` to `vlines` for better clarity in time series plots.
- Updated `izhikevich.py` example to use `sim.config(dt=0.01)` and added enhanced plotting features 
  like ylim, colored bands, and vertical lines.
- Moved plot related examples into the `examples/plot` folder.

---

## [2.21.2] – 2025-11-13
### Added
- Introduced `Sim.assign()` method in `src/dynlib/runtime/sim.py` for assigning state and parameter
  values dynamically during a simulation session.

### Changed
- Enhanced `_select_seed()` in `Sim` to use current session state values as defaults for 
  `resume=False` runs, allowing explicit overrides with `ic` and `params` arguments.
- Updated `examples/izhikevich.py` to demonstrate dynamic parameter assignment using `Sim.assign()`.

### Tests
- Added `tests/unit/test_sim_assign.py` to validate the functionality of `Sim.assign()`.

---

## [2.21.1] – 2025-11-13
### Changed
- Removed `tomli` package fallbacks and updated Python requirement as >= 3.11 instead of 3.10.

---

## [2.21.0] – 2025-11-12
### Added
- Reintroduced `guards.py` in `src/dynlib/compiler/` to provide universal finiteness checks for 
  steppers. This guard is applied universally to all steppers inside the runners and adaptive 
  steppers also use these checks internally inside their step size calculation loops.
- Added `allfinite1d` and `allfinite_scalar` functions for finiteness checks.
- Integrated `guards` with `runner`, `runner_discrete`, and `rk45` stepper for NaN/Inf detection.
- Added `test_nan_inf_guards.py` to validate the functionality of `guards`.

### Changed
- Updated `RK45Spec` to use `guards` for internal loops.
- Enhanced `build_callables` in `src/dynlib/compiler/build.py` to configure finiteness guards 
  based on JIT settings.

---

## [2.20.2] – 2025-11-12
### Changed
- Removed `guards.py` because it was poorly designed and implemented; was causing a lot of numba
  compatibility and caching issues.

### Known Issues
- All tests pass now but there is no NaN/Inf checks anywhere at this point.

---

## [2.20.1] – 2025-11-12
### Fixed
- `build()` warm-up function `_warmup_jit_runner` was generating the wrong runner when dtype is not 
  float64. Updated `_warmup_jit_runner` in `src/dynlib/compiler/build.py` to ensure stepper control 
  values are always float64, regardless of model dtype. This fixed wrong runner caching issue during 
  warm-up.
- Enhanced comments in `src/dynlib/compiler/build.py` to clarify the use of Python floats for stepper 
  configurations.
- Fixed exception handling in `_jit_compile_with_disk_cache` in `src/dynlib/compiler/jit/compile.py` 
  to catch `DiskCacheUnavailable` correctly.

### Known Issues
- `rk45.py` tests are failing. 

---

## [2.20.0] – 2025-11-12
### Added
- Introduced `Segment` dataclass in `src/dynlib/runtime/sim.py` to represent simulation segments.
- Added `SegmentsView` and `SegmentView` classes in `src/dynlib/runtime/results_api.py` for 
  accessing recorded simulation segments.
- Implemented `Sim.name_segment` and `Sim.name_last_segment` methods for renaming simulation 
  segments.

### Changed
- Enhanced `Sim.run` in `src/dynlib/runtime/sim.py` to support tagging and recording simulation 
  segments.
- Updated `Sim.results` to include segment metadata in `ResultsView`.

### Tests
- Added new tests to `tests/steppers/common/test_sim_session.py` to validate segment functionality.

---

## [2.19.4] – 2025-11-12
### Added
- Added `examples/collatz.py` to demonstrate map simulation with integer dtype and ternary if 
  usage.

### Changed
- [params] table was mandatory in DSL model declarations. Made it optional.

### Fixed
- Fixed a bug in `src/dynlib/compiler/codegen/runner_discrete.py` where the `header` variable was 
  causing caching issues. Replaced `textwrap.dedent` with `inspect.cleandoc`during header creation.
  Did the same with `runner.py` for symmetry.
- When using `dtype=int64` (or other integer dtypes) for models, the stepper control arrays (dt_next, 
  t_prop, err_est) were incorrectly being created with the model's dtype instead of float64. This
  was causing non-monotone time series. Now they are always float64 alongside tracked time. 

### Tests
- Added `test_int_dtype.py` for testing integer data type usage with maps.

---

## [2.19.3] – 2025-11-11
### Added
- Introduced `guards.py` in `src/dynlib/runtime/` to provide universal finiteness checks for 
  steppers. This guard is applied universally to all steppers inside the runners and adaptive
  steppers also use these checks internally inside their step size calculation loops.
- Added `allfinite1d` function and `configure_allfinite_guard` to toggle between Python and 
  JIT implementations.

### Changed
- Refactored `runner.py` and `runner_discrete.py` to use `guards.allfinite1d` for finiteness 
  checks.
- Updated `EulerSpec`, `RK4Spec`, `RK45Spec`, and `MapSpec` to remove redundant finiteness 
  checks and rely on `guards.allfinite1d`.
- Enhanced `build_callables` in `src/dynlib/compiler/build.py` to configure finiteness guards 
  based on JIT settings.

---

## [2.19.2] – 2025-11-11
### Changed
- Gathered all ode-solver steppers under `src/dynlib/steppers/ode` folder.

---

## [2.19.1] – 2025-11-11
### Added
- Added `examples/logistic_map.py` to demonstrate the logistic map simulation using the new 
  discrete runner.
- Implemented `RunnerDiskCache` in `src/dynlib/compiler/codegen/_runner_cache.py` for managing 
  disk-backed runner caching.

### Changed
- Refactored `runner` and `runner_discrete` to use `RunnerDiskCache` for caching.
- Enhanced `src/dynlib/compiler/codegen/runner.py` and `runner_discrete.py` to improve modularity
  and maintainability.

### Fixed
- `runner_discrete` jit option was not implemented properly and it was accepting a wrong argument. 
  Refactored `runner_discrete` so that now it completely mirrors `runner`. The caching was also 
  not working properly. The new common `RunnerDiskCache` solved that issue.

---

## [2.19.0] – 2025-11-11
### Changed
- Split runners into two: `runner_discrete` for discrete-time models; `runner` (old `runner` 
  untouched) for continuous-time models.
- Sim.run: replaced legacy `t_end` with `T` (continuous end time) and added `N` for discrete
  iteration counts. `Sim` now distinguishes discrete (maps) vs continuous systems and enforces
  the correct parameters, improving transient and resume behavior.
- Wrapper: `run_with_wrapper` accepts `discrete`/`target_steps` and passes a general `horizon`
  argument to the runner so it can work in iteration (N) or time (T) modes.
- Build/codegen: added and exported a discrete runner (`runner_discrete`) and wired build to
  select the discrete runner for `map`-kind steppers; disk-cache configuration was added for
  the discrete runner as well.
- Steppers: exported the discrete `map` stepper so it is available for selection/registration.
- Docs/examples: updated usages to call `sim.run(T=...)` where appropriate (replacing
  `t_end`).

### Tests
- Reorganized whole tests folder.
- Added `test_discrete_runner.py` test for new `runner_discrete` and maps.

---

## [2.18.0] – 2025-11-11
### Added
- Added source code export functionality for compiled models. All compiled models now store the 
  generated Python source code for RHS, events, and stepper functions.
- Added `export_model_sources(model, output_dir)` function in `src/dynlib/compiler/build.py` to 
  export all compiled sources to a directory for inspection and debugging.
- Added source code fields to `FullModel` and `Model` classes:
  - `rhs_source`: Generated RHS (right-hand side) function source code
  - `events_pre_source`: Pre-step event handler source code
  - `events_post_source`: Post-step event handler source code
  - `stepper_source`: Numerical integration stepper source code
- Added `examples/export_sources_demo.py` demonstrating basic source export.
- Added comprehensive documentation in `docs/export_sources.md`.

### Changed
- Modified `CompiledPieces` dataclass to store source code from compilation.
- Updated `build_callables()` to preserve source code through the compilation pipeline.
- Enhanced `_StepperCacheEntry` to cache stepper source code for reuse.
- Source code is now available regardless of `disk_cache` setting (always stored in model object).

---

## [2.17.1] – 2025-11-11
### Added
- Added `setup()` helper to `src/dynlib/__init__.py`. It combines `build()` + `Sim()` calls. It is 
  more convenient for end users.

---

## [2.17.0] – 2025-11-11
### Added
- Added presets feature for quick storage of state/param values. Presets can be defined inside model
  file or in external toml files.
- Added `_read_presets` function in `src/dynlib/dsl/parser.py` to parse `[presets.<name>]` blocks 
  from TOML files.
- Introduced `PresetSpec` dataclass in `src/dynlib/dsl/spec.py` to represent presets in the model 
  DSL.
- Added `Sim` presets API in `src/dynlib/runtime/sim.py`:
  - `list_presets(pattern)`: Lists preset names matching a glob pattern.
  - `apply_preset(name)`: Applies a preset to the current session.
  - `load_preset(name_or_pattern, path, on_conflict)`: Loads presets from a TOML file.
  - `save_preset(name, path, include_states, overwrite)`: Saves a preset to a TOML file.
- Added `examples/presets_demo.py` to demonstrate the presets feature.

### Changed
- Enhanced `build_spec` in `src/dynlib/dsl/spec.py` to validate and include presets in the model 
  specification.
- Updated `validate_tables` in `src/dynlib/dsl/schema.py` to validate the `[presets]` table.
- Enhanced `Sim` initialization in `src/dynlib/runtime/sim.py` to auto-load inline presets from the 
  model specification.

### Tests
- Added `tests/unit/test_presets.py` to cover inline and file-based presets, including validation, 
  loading, saving, and error handling.

---

## [2.16.2] – 2025-11-10
### Added
- Persisted runtime `stepper_config` data in `SessionState`, snapshots, and snapshot metadata so
  resumes continue with the exact tolerances last used (plus field-name lists for inspection).
- Added `Sim.stepper_config(**kwargs)` helper and extended `session_state_summary()` diagnostics with
  stepper-config previews/digests.

---

## [2.16.1] – 2025-11-10
### Added
- Introduced snapshot export/import functionality in `Sim`:
  - `export_snapshot()`: Exports session state to disk as a strict snapshot file.
  - `import_snapshot()`: Imports session state from a snapshot file, replacing the current session.
  - `inspect_snapshot()`: Returns parsed metadata from a snapshot file without modifying simulation 
    state.
- Added `examples/snapshot_demo.py` to demonstrate snapshot export/import and inspection.

### Changed
- Updated `Sim` class in `src/dynlib/runtime/sim.py`:
  - Added internal helpers for snapshot handling, including `_snapshot_pick_state`, 
    `_snapshot_build_meta`, `_snapshot_write_npz`, `_snapshot_read_npz`, and `_snapshot_restore`.
  - Enhanced `Sim.run()` to support snapshot-based workflows.

### Tests
- Added `tests/integration/test_snapshot_persistence.py` to validate snapshot export/import 
  functionality.

---

## [2.16.0] – 2025-11-10
### Added
- `Sim` now tracks an internal `SessionState` so `run(resume=True)` continues from the exact last
  integrator conditions (time, state, params, dt, workspace).
- Snapshot API: `create_snapshot()`, `reset()`, and `list_snapshots()` capture/restore
  SessionState, with an auto `initial` snapshot created before the first run.
- New helpers `session_state_summary()`, `can_resume()`, and `compat_check()` surface resume
  diagnostics.
- Seam-aware result stitching drops duplicate seam samples, offsets STEP/EVT indices, and asserts
  monotone time; events referencing a dropped seam record are dropped (documented policy).

### Changed
- `Results` contract now includes `final_params_view`, `t_final`, `final_dt`, `step_count_final`, 
  and `final_stepper_ws`. `run_with_wrapper` captures these values along with snapshots of the 
  stepper workspace and accepts a `workspace_seed` for resume.
- Runner redefines `EVT_INDEX` as the owning record index (or `-1` when no record exists);
  downstream docs/tests updated accordingly.

### Tests
- Added integration coverage for resume stitching, record-off then resume, and snapshot reset in
  `tests/integration/test_sim_session.py`.

---

## [2.15.3] – 2025-11-09
### Added
- `Sim.run()` accepts a `transient` warm-up duration that advances the model before recording while
  keeping events functional and resetting the public time axis to `t0`.
- `Results` now exposes the final committed state via `final_state_view` for scenarios that need to
  reuse the converged state (e.g., transient warm-up, chained simulations).
### Changed
- `run_with_wrapper` captures the final committed state and stores it on the returned `Results`.

---

## [2.15.2] – 2025-11-09
### Changed
- Renamed `run()` args: 
  - `y0` -> `ic` 
  - `record_every_step` -> `record_interval`
- Renamed `build()` args:
  - `stepper_name` -> `stepper`
  - `model_dtype` -> `dtype`

---

## [2.15.1] – 2025-11-09
### Changed
- Forgot to add `**stepper_kwargs` in the previous version. Refactored `Sim.run()` in `sim.py` to 
  accept `**stepper_kwargs` for runtime overrides instead of explicit stepper parameters.
- Updated `_build_stepper_config()` in `Sim` to construct stepper configuration arrays from
  `**stepper_kwargs`.

---

## [2.15.0] – 2025-11-09
### Added
- Introduced runtime stepper configuration system. Now steppers can declare their internal config 
  values. During build process a read-only struct buffer is filled to pass these values into the 
  steppers. `Sim.run()` can pass these values at runtime. If not provifed, then model [sim] defaults 
  are used. If it is also not present, then internal defaults of the stepper are used. Here are the 
  details:
  - `config_spec()`: Returns dataclass type or None for stepper configuration.
  - `default_config(model_spec)`: Creates default configuration with model-specific overrides.
  - `pack_config(config)`: Packs configuration into a float64 array.
- Added `stepper_config` parameter to `runner` ABI in `src/dynlib/runtime/runner_api.py`.
- Enhanced `Sim.run()` in `src/dynlib/runtime/sim.py` to accept `**stepper_kwargs` for runtime 
  overrides.
- Added `RK45Config` dataclass in `src/dynlib/steppers/rk45.py` with runtime parameters:
  - `atol`, `rtol` (tolerances)
  - `safety`, `min_factor`, `max_factor` (step control)
  - `max_tries`, `min_step` (failure thresholds)
- Added `_build_stepper_config()` in `Sim` to construct stepper configuration arrays.

### Changed
- Updated `run_with_wrapper` in `src/dynlib/runtime/wrapper.py` to pass `stepper_config` to the 
  runner.
- Enhanced `_warmup_jit_runner` in `src/dynlib/compiler/build.py` to initialize `stepper_config` 
  during warmup.
- Refactored `RK45Spec` in `src/dynlib/steppers/rk45.py` to use `RK45Config` for runtime 
  configuration.
- Updated `runner` in `src/dynlib/compiler/codegen/runner.py` to accept `stepper_config` as a 
  parameter.
- Modified `EulerSpec` and `RK4Spec` to return `None` for `config_spec()` and handle empty 
  configurations gracefully.

### Tests
- Added tests for runtime stepper configuration in `tests/unit/test_stepper_config.py`.

---

## [2.14.2] – 2025-11-08
### Added
- Previously only runners were cached. Added disk caching support for stepper and triplet functions
  in `runner.py`. This way all jittable parts are cached. This improved build times but compilation 
  cost is unavoidable. Use numba for long simulations or simulations / analyses that call `run()` 
  repeatedly. Use disk cache only for fixed models.

### Changed
- Enhanced `build_callables` in `src/dynlib/compiler/build.py` to include disk caching for RHS and 
  event functions.
- Improved `emit_rhs_and_events` in `src/dynlib/compiler/codegen/emitter.py` to return source code 
  for RHS and events.
- Refactored `jit_compile` in `src/dynlib/compiler/jit/compile.py` to support disk caching.

---

## [2.14.1] – 2025-11-08
### Added
- Introduced `Timer` utility in `src/dynlib/utils/timer.py` for measuring execution time.
- Added `izhikevich_benchmark.py` to observe build and run times with and without JIT and disk
  caching.

### Changed
- Enhanced `build` function in `src/dynlib/compiler/build.py` to support warm-up for JIT runners.
  So `build()` also causes numba compilation instead of lazy compilation after `run()` call.
- Improved error handling in `src/dynlib/compiler/codegen/runner.py` for disk cache unavailability.

### Fixed
- Updated `resolve_cache_root` in `src/dynlib/compiler/paths.py` to handle unwritable cache 
  directories gracefully. Cache root resolution now probes writability and falls back to a temp 
  directory when the platform default cannot be written to (such as sandboxed environments), 
  ensuring disk caching actually speeds up repeated builds instead of silently falling back to 
  in-memory JIT. 

---

## [2.14.0] – 2025-11-07
### Added
- Introduced opt-in disk-backed runner caching via `build(..., disk_cache=...)`, including
  configurable cache roots, deterministic digesting, and automatic regeneration on corruption.
  - `cache_root=True`: persistent on-disk cache.
  - `cache_root=False`: in-memory (per-process) JIT only; no files written.
  - Config defines the cache root. If not available platform defaults are used:
      Linux: ${XDG_CACHE_HOME:-~/.cache}/dynlib
      macOS: ~/Library/Caches/dynlib
      Windows: %LOCALAPPDATA%\dynlib\Cache

### Tests
- Added `tests/unit/test_runner_diskcache.py` to cover materialization, reuse, recovery, and
  fallback scenarios for the new cache layer.

---

## [2.13.2] – 2025-11-07
### Fixed
- The runner was dropping records when buffer growth was triggered. Enhanced `runner` function
  in `src/dynlib/compiler/codegen/runner.py`:
  - Added logic to handle pending steps before growth.
  - Improved recording mechanism for steps during re-entries.

---

## [2.13.1] – 2025-11-07
### Fixed
- Buffer reallocation was resetting the wrapper time value. Refactored `run_with_wrapper` in 
  `src/dynlib/runtime/wrapper.py` to track committed time and step size for re-entries, 
  ensuring that reallocation does not corrupt recording.

### Added
- Added `Sim` to `__all__` in `src/dynlib/__init__.py` for better accessibility.
- Added `izhikevich.py` example in `examples/` demonstrating neuron spiking behavior.

### Tests
- `test_euler_growth_matches_reference` in `tests/integration/test_euler_basic.py` now ensures 
  buffer growth does not alter recorded trajectories.

---

## [2.13.0] – 2025-11-07
### Added
- Introduced `Sim.results` and `Sim.raw_results` methods for accessing simulation results. The 
  first one returns new `ResultsView` object while the latter returns old low-level `Results` object. 
- Added `EventAccessor` and `EventGroupView` in `src/dynlib/runtime/results_api.py` for grouped 
  event access.

### Changed
- Updated `Sim.run` in `src/dynlib/runtime/sim.py` to return `None` and store results internally.

### Fixed
- Updated tests for the new `ResultsView` API.
- Added `test_sim_results_api.py` test for testing the new `ResultsView` API.

---

## [2.12.5] – 2025-11-07
### Changed
- Removed unused `src/dynlib/utils/arrays.py` and `utils` folder because user inputs are always
  copied with `np.array()` and this file is not useful right now.

---

## [2.12.4] – 2025-11-07
### Changed
- Updated `validate_stepper_function` in `src/dynlib/compiler/codegen/validate.py` to include
 `StructSpec` validation.
- Enhanced `report_validation_issues` to handle warnings and errors more effectively.
- Modified `build` function in `src/dynlib/compiler/build.py` to pass `StructSpec` to 
  `validate_stepper_function`.

### Added
- Introduced `test_stepper_guardrails.py` in `tests/unit/` to validate `StructSpec` sizes, 
  persistence flags, and bank assignments.
- Added new validation rules for `iw0` and `bw0` banks to reject float assignments.
- Added warnings for ephemeral banks being read before write.

---

## [2.12.3] – 2025-11-07
### Added
- Introduced `StepperKindMismatchError` to handle mismatched stepper and model kinds.

### Changed
- Updated `build` function to validate stepper kind against model kind and raise 
  `StepperKindMismatchError` if incompatible.

---

## [2.12.2] – 2025-11-07
### Changed
- Removed `priority` field from `ModSpec` in `src/dynlib/compiler/mods.py`.
- Updated exclusivity handling in `apply_mods_v2` to enforce stricter group rules.
- Improved error messages for exclusivity conflicts in `src/dynlib/compiler/mods.py`.

### Tests
- Removed priority fields from `tests/unit/test_mods.py` .
- Updated `test_mods_group_exclusive_conflict_raises` to validate stricter exclusivity rules.

---

## [2.12.1] – 2025-11-07
### Added
- TODO.md, ISSUES.md files.

### Changed
- Mods won't change [Sim] defaults in models. This is documented.

---

## [2.12.0] – 2025-11-06
### Added
- Event tagging feature for compile-time metadata and filtering:
  - Added `tags: Tuple[str, ...]` field to `EventSpec` dataclass in `src/dynlib/dsl/spec.py`
  - Tags are immutable, order-stable (sorted), and deduplicated tuples
  - Compile-time only; no ABI or runtime impact
  - Added `tag_index: Dict[str, Tuple[str, ...]]` to `ModelSpec` for fast reverse lookup
  - Tag index maps each tag to tuple of event names that have that tag
- DSL support for event tags:
  - Parser accepts `tags = ["tag1", "tag2", ...]` under each event in TOML files
  - Empty or absent tags field defaults to empty tuple (no tags)
  - Tags are normalized: duplicates removed, alphabetically sorted
- Tag validation in `src/dynlib/dsl/astcheck.py`:
  - Added `validate_event_tags` function with pattern `[A-Za-z_][A-Za-z0-9_-]*`
  - Tags must start with letter or underscore, contain only alphanumerics, underscores, hyphens
  - Empty tags rejected; non-string tags caught by parser
  - Duplicates normalized away (not an error)

### Changed
- Spec hash includes tags for deterministic cache invalidation.
- Updated `_json_canon` in `src/dynlib/dsl/spec.py` to serialize tags in EventSpec and tag_index
  in ModelSpec.
- Added helper `_build_tag_index` to construct reverse index during spec building.

### Tests
- Added comprehensive test suite in `tests/unit/test_event_tags.py`:
  - Tag parsing, normalization, and validation
  - Tag index construction and event lookup
  - Format validation (valid slugs, invalid special chars, empty tags)
  - Spec hash stability and changes with tags
  - Integration with TOML file loading
- Added test data file `tests/data/models/tagged_events.toml` demonstrating tag usage.

---

## [2.11.3] – 2025-11-06
### Changed
- Exclusive groups now raise a ModelLoadError when more than one exclusive mod is supplied,
  so conflicts no longer slip through unnoticed.
 - `src/dynlib/compiler/mods.py`:  updated exclusivity docs and enforce a conflict check 
   that raises with the conflicting mod names instead of silently selecting a winner.

### Tests
- Replaced the previous “pick a winner” assertion with a conflict-raises check in 
  `tests/unit/test_mods.py`.

---

## [2.11.2] – 2025-11-06
### Changed
- Updated `_apply_remove` in `src/dynlib/compiler/mods.py`:
  - Added validation to raise errors for non-existent `events`, `aux`, and `functions` during 
    removal.
  - Improved error messages for better debugging.
- Enhanced `_apply_replace` in `src/dynlib/compiler/mods.py`:
  - Added validation to ensure replaced `events`, `aux`, and `functions` exist.
  - Improved error handling for invalid replacements.

### Tests
- Added new test cases in `tests/unit/test_mods.py`:
  - `test_mods_remove_nonexistent_event_raises`: Verifies error is raised for non-existent event 
    removal.
  - `test_mods_replace_nonexistent_raises`: Ensures replacement of non-existent entities raises 
    errors.
- Updated `tests/unit/test_mods_aux_functions.py`:
  - `test_remove_aux_nonexistent_raises`: Validates error handling for non-existent aux removal.
  - `test_remove_functions_nonexistent_raises`: Ensures error is raised for non-existent function 
    removal.

---

## [2.11.1] – 2025-11-06
### Changed
- Updated `Results` class in `src/dynlib/runtime/results.py`:
  - Added `status` field to store runner exit status.
  - Added `ok` property to check if the runner exited cleanly.
  - Updated docstrings to reflect the new `status` field.
- Enhanced `run_with_wrapper` function in `src/dynlib/runtime/wrapper.py`:
  - Added `status` field to `Results` object returned by the function.
  - Improved early termination handling with warnings for specific statuses.
  - Refactored status handling logic for clarity.

### Tests
- New assertions for `status` and `ok` properties in `tests/unit/test_wrapper_reentry.py`.

---

## [2.11.0] – 2025-11-06
### Changed
- Removed `REJECT` status code from the stepper/runner contract:
  - Clarified architectural contracts: fixed-step steppers do single attempts; adaptive steppers 
    handle internal accept/reject loops
  - Runner never sees rejection codes; adaptive steppers (like RK45) handle retries internally 
    and only return terminal codes
  - Updated status enum in `src/dynlib/runtime/runner_api.py`: removed `REJECT=1`
  - Updated all exports in `src/dynlib/__init__.py` to remove `REJECT`
  - Simplified runner comments in `src/dynlib/compiler/codegen/runner.py`
  - Updated wrapper imports in `src/dynlib/runtime/wrapper.py`
  - Stepper contract now clear: return `OK` (step accepted) or terminal codes (`NAN_DETECTED`, 
    `STEPFAIL`)
  
### Added
- NaN/Inf detection in fixed-step steppers:
  - Added finiteness checks to Euler stepper (`src/dynlib/steppers/euler.py`)
  - Added finiteness checks to RK4 stepper (`src/dynlib/steppers/rk4.py`)
  - Both now return `NAN_DETECTED` if proposal contains non-finite values
  - Maintains consistency with adaptive stepper (RK45) which already had such checks

---

## [2.10.2] – 2025-11-06
### Changed
- Dropped the legacy `EVT_TIME` buffer entirely; logged times live in `EVT_LOG_DATA`.

---

## [2.10.1] – 2025-11-06
### Changed
- Removed `record` key from events in favor of unified `log` mechanism:
  - Events no longer support `record=True/False`
  - Use `log=["t"]` to capture event occurrence times
  - Use `log=["t", "x", ...]` to capture both time and other values
  - The `"t"` signal is treated like any other loggable value in `EVT_LOG_DATA`
  - Eliminates non-orthogonal design where `record` and `log` overlapped
  - Event buffers only grow when events actually have `log` items
  - Migration: Replace `record=true` with `log=["t"]`, or add `"t"` to existing log arrays

### Fixed
- Event buffer allocation is now more efficient:
  - No buffer space wasted on events without logging
  - Events only increment buffer counter when they have actual log data

### Changed
- Updated event function signature in `src/dynlib/compiler/codegen/emitter.py`:
  - Old: `events_phase(...) -> (event_code, has_record, log_width)`
  - New: `events_phase(...) -> (event_code, log_width)`
- Updated runner in `src/dynlib/compiler/codegen/runner.py`:
  - Removed `has_record` conditional logic
  - Simplified event recording: only fires when `log_width > 0`
  - All log values (including `"t"`) go to `EVT_LOG_DATA`
- Updated `EventSpec` dataclass in `src/dynlib/dsl/spec.py`:
  - Removed `record: bool` field
- Updated parsers to reject deprecated `record` key with helpful error:
  - `src/dynlib/dsl/parser.py`: Raises error directing users to use `log=["t"]`
  - `src/dynlib/compiler/mods.py`: Same validation in mod files
- Updated `src/dynlib/runtime/runner_api.py` documentation

### Tests
- Updated all test models and test cases to use `log=["t"]` instead of `record=True`.

---

## [2.10.0] – 2025-11-06
### Fixed
- Event logging now properly separates `record` and `log` functionality:
  - Previously, events only logged if BOTH `record=True` AND `log` was non-empty, silently 
    ignoring events with `record=True` but empty `log=[]`
  - `EVT_INDEX` was stuck at zero and log signal values were never materialized
  - New behavior:
    - `record=True` → logs event occurrence times to `EVT_TIME` and `EVT_CODE`
    - `log=[signals]` → logs signal values to new `EVT_LOG_DATA` buffer (independent of `record`)
    - Both features are now orthogonal and can be used independently or together
  - `EVT_INDEX` now stores the log width (number of signals logged per event)
  - Events with `log` but no `record` set `EVT_TIME=-1.0` as a sentinel

### Added
- New `EVT_LOG_DATA` buffer in `EventPools` to store logged signal values:
  - Shape: `(cap_evt, max_log_width)` where `max_log_width` is computed from all events
  - Values are model dtype (same as state variables)
  - Accessible via `Results.EVT_LOG_DATA_view` property
- Helper function `_parse_log_signal()` in `src/dynlib/compiler/codegen/emitter.py`:
  - Supports formats: `"x"` (state), `"param:a"`, `"aux:E"`, `"t"` (time)
  - Validates that referenced symbols exist in the model spec
- Event log scratch buffer `evt_log_scratch` passed to runner for temporary log value storage

### Changed
- Event function signature in `src/dynlib/compiler/codegen/emitter.py`:
  - Old: `events_phase(t, y_vec, params) -> event_code`
  - New: `events_phase(t, y_vec, params, evt_log_scratch) -> (event_code, has_record, log_width)`
  - Events now write log values into `evt_log_scratch` before returning
- Runner signature in `src/dynlib/compiler/codegen/runner.py`:
  - Added `EVT_LOG_DATA` and `evt_log_scratch` parameters
  - Runner now copies log data from scratch buffer to `EVT_LOG_DATA[m, :]` when `log_width > 0`
  - Records event time/code only when `has_record=True`
- Updated `allocate_pools()` in `src/dynlib/runtime/buffers.py`:
  - Added `max_log_width` parameter
  - Allocates `EVT_LOG_DATA` with shape `(cap_evt, max(1, max_log_width))`
- Updated `grow_evt_arrays()` in `src/dynlib/runtime/buffers.py`:
  - Added `dtype` parameter for allocating `EVT_LOG_DATA` with correct dtype
  - Copies existing log data during growth
- Updated `Sim.run()` in `src/dynlib/runtime/sim.py`:
  - Calculates `max_log_width` from event specs before calling wrapper
- Added `EVT_LOG_DATA` to forbidden writes in `src/dynlib/compiler/codegen/validate.py`

### Tests
- Updated `test_event_logging_basic()` in `tests/integration/test_event_logging.py`:
  - Verifies `EVT_INDEX` contains log width (not zero)
  - Checks that `EVT_LOG_DATA` contains logged signal values
  - Validates logged values are within expected ranges
- Fixed `test_codegen_triplet.py` to use new event signature with scratch buffer
- Fixed `test_buffers_growth.py` to pass `max_log_width` and `dtype` parameters

---

## [2.9.0] – 2025-11-06
### Added
- Implemented complete mod support for DSL aux and functions:
  - Added `add.aux` and `add.functions` verbs in `src/dynlib/compiler/mods.py`
  - Added `replace.aux` and `replace.functions` verbs with existence validation
  - Added `remove.aux` and `remove.functions` verbs (silent for non-existent items)
  - Added `set.aux` and `set.functions` verbs with upsert semantics
  - All verbs follow the same deterministic application order: remove → replace → add → set

### Changed
- Enhanced `_apply_remove`, `_apply_replace`, `_apply_add`, and `_apply_set` in 
  `src/dynlib/compiler/mods.py`:
  - Extended all verb handlers to support aux and functions alongside existing events support
  - Added `_normalize_function` helper for consistent function definition validation
  - String expression validation for aux values
  - Function args and expr validation matching parser requirements

### Tests
- Updated tests and removed redundant ones.
- Added 19 comprehensive tests in `tests/unit/test_mods_aux_functions.py`:
  - `test_add_aux` - Adding new auxiliary variables
  - `test_add_aux_duplicate_raises` - Duplicate detection
  - `test_replace_aux` - Replacing existing aux expressions
  - `test_replace_aux_nonexistent_raises` - Error on missing aux
  - `test_remove_aux` - Removing aux variables
  - `test_remove_aux_nonexistent_silent` - Silent handling of non-existent removals
  - `test_set_aux_upsert` - Upsert semantics (create or update)
  - `test_add_functions` - Adding new functions
  - `test_add_functions_duplicate_raises` - Duplicate detection
  - `test_replace_functions` - Replacing existing functions
  - `test_replace_functions_nonexistent_raises` - Error on missing function
  - `test_remove_functions` - Removing functions
  - `test_remove_functions_nonexistent_silent` - Silent handling
  - `test_set_functions_upsert` - Upsert semantics
  - `test_verb_order_remove_then_add` - Verb ordering validation
  - `test_multiple_mods_aux_and_functions` - Sequential application
  - `test_invalid_aux_value_type` - Type validation for aux
  - `test_invalid_function_args` - Args validation
  - `test_invalid_function_expr` - Expr validation

---

## [2.8.0] – 2025-11-06
### Added
- DSL block equations are now parsed (previously they were omitted).
- Introduced `StructSpec` validation in `src/dynlib/steppers/base.py`:
  - Ensures all sizes are non-negative integers.
  - Validates compatibility with declared dense-output coefficients and history lengths.
- Added `validate_name_collisions` in `src/dynlib/dsl/schema.py`:
  - Detects duplicate equation targets across RHS and block forms.

### Changed
- Enhanced `emit_rhs_and_events` in `src/dynlib/compiler/codegen/emitter.py`:
  - Improved error messages for invalid LHS in block equations.
- Updated `build_spec` in `src/dynlib/dsl/spec.py`:
  - Added stricter validation for unknown states and missing equals in equations.

### Tests
- Added unit tests in `tests/unit/test_equations_block_form.py`:
  - Verified detection of duplicate targets and invalid LHS in block equations.
  - Tested auxiliary variable usage and user-defined functions in block equations.
- Added integration tests in `tests/integration/test_block_equations_sim.py`:
  - Verified simulation correctness with mixed RHS and block equations.
  - Tested conservation laws and stepper compatibility.

---

## [2.7.1] – 2025-11-06
### Changed
- Updated `_edges_for_aux_and_functions` in `src/dynlib/dsl/astcheck.py`:
  - Function dependencies now include references to auxiliary variables.
- Enhanced `build_spec` in `src/dynlib/dsl/spec.py`:
  - Added validation steps for acyclic expressions, event legality, and function signatures.

### Tests
- Added integration tests in `tests/integration/test_semantic_validation.py`:
  - Verified detection of cyclic dependencies in auxiliary variables and functions.
  - Tested event legality and function argument validation.

---

## [2.7.0] – 2025-11-05
### Changed
- Centralized JIT compilation logic in `src/dynlib/compiler/jit/compile.py`:
  - Introduced `jit_compile` function for consistent error handling.
  - Updated `maybe_jit_triplet` to use `jit_compile`.
- Enhanced `runner` function in `src/dynlib/compiler/codegen/runner.py`:
  - Added event logging for pre/post events.
  - Improved event buffer growth handling.
- Updated `emit_rhs_and_events` in `src/dynlib/compiler/codegen/emitter.py`:
  - Added event codes for logging-enabled events.
  - Ensured default return value for no events fired.
- Modified `run_with_wrapper` in `src/dynlib/runtime/wrapper.py`:
  - Preserved event cursor during buffer growth.

### Tests
- Added integration tests in `tests/integration/test_event_logging.py`:
  - Verified event logging functionality.
  - Tested multiple event firings and state captures.
- Updated `tests/data/models/decay_with_event.toml`:
  - Added `log` field to reset event.

---

## [2.6.1] – 2025-11-05
### Changed
- Preceding newlines are removed from the inline model declarations. This way `inline:`
  statement can be placed above `[model]` statements.

### Known Issues
- A FIX_PLAN.md file is created to implement planned but missing features. These features
  will be added.

---

## [2.6.0] – 2025-11-05
### Added
- Implemented comprehensive path resolution system in `src/dynlib/compiler/paths.py`:
  - Platform-specific config file locations (Linux/XDG, macOS, Windows)
  - `DYNLIB_CONFIG` environment variable for custom config paths
  - `DYN_MODEL_PATH` environment variable with prepend semantics for runtime tag additions
  - `TAG://` URI scheme for model resolution from configured directories
  - `inline:` URI scheme for embedding model definitions directly
  - Support for absolute and relative paths with automatic `.toml` extension resolution
  - Fragment selectors (`#mod=NAME`) for selecting embedded mods from files
  - Path traversal prevention outside declared roots
  - Clear error messages listing all searched candidates on failure
- Enhanced `build()` function in `src/dynlib/compiler/build.py`:
  - Accepts both URI strings and ModelSpec objects
  - `mods` parameter for applying multiple mod files via URIs
  - `config` parameter for custom PathConfig
  - Automatically resolves stepper from model's sim defaults if not specified
- Added `load_model_from_uri()` function for loading models with mod application
- New exception classes in `src/dynlib/errors.py`:
  - `ModelNotFoundError`: lists all searched paths
  - `ConfigError`: configuration file or environment errors
  - `PathTraversalError`: security violation detection
  - `AmbiguousModelError`: multiple files match extensionless reference
- Updated `src/dynlib/compiler/__init__.py` to export new public API

### Changed
- `build()` signature enhanced with optional `config` parameter
- `build()` now accepts `Union[ModelSpec, str]` for model parameter

### Tests
- Added 37 unit tests in `tests/unit/test_paths.py`:
  - Config loading from all platforms
  - Environment variable handling
  - TAG:// resolution with multiple roots and first-match-wins
  - inline:, absolute, relative path handling
  - Fragment extraction
  - Path traversal security checks
  - Error message quality verification
- Added 16 integration tests in `tests/integration/test_uri_loading.py`:
  - End-to-end model building from all URI schemes
  - Embedded mod selection with fragments
  - External mod file application
  - Multiple mods in sequence
  - Error handling with helpful diagnostics
  - Backward compatibility with direct ModelSpec usage

### Documentation
- URI schemes supported:
  - `inline: [model]\ntype='ode'\n...` - Direct TOML content
  - `/abs/path/model.toml` - Absolute file path
  - `relative/model.toml` - Relative to current working directory
  - `TAG://model.toml` - Resolve using configured tag roots
  - Any of above with `#mod=NAME` - Select embedded mod
- Config file format:
  ```toml
  [paths]
  proj = ["/home/user/models", "/opt/shared/models"]
  user = "/home/user/personal/models"
  ```
- Config File Paths:
  - Linux: `${XDG_CONFIG_HOME:-~/.config}/dynlib/config.toml`
  - macOS: `~/Library/Application Support/dynlib/config.toml`
  - Windows: `%APPDATA%\dynlib\config.toml`
- Environment variables:
  - `DYNLIB_CONFIG=/custom/path/config.toml` - Override config location
  - `DYN_MODEL_PATH=proj=/extra/path,/another:new=/path` - Add paths at runtime

---

## [2.5.0] – 2025-11-05
### Added
- Implemented `RK45Spec` in `src/dynlib/steppers/rk45.py` for Dormand-Prince adaptive stepper 
  with embedded 4th/5th order error estimation.

### Changed
- Enhanced buffer allocation in `src/dynlib/runtime/buffers.py`:
  - Improved `allocate_pools` to handle workspace banks (sp, ss, sw0-sw3) with size 0 convention.
  - Size 0 now means "allocate n_state elements", size >= 1 uses spec size as-is.
  - Ensures all RK methods get sufficient workspace without ABI changes.
- Updated `RK4Spec` for the new banks allocation rule.

### Tests
- Added comprehensive integration tests in `tests/integration/test_rk4_rk45.py`:
  - `test_rk4_accuracy`: Verifies RK4 achieves < 1e-5 relative error for exponential decay.
  - `test_rk4_order_convergence`: Confirms 4th-order convergence (16x error reduction when 
    halving dt).
  - `test_rk45_adaptive_accuracy`: Validates RK45 adaptive stepping accuracy.
  - `test_rk45_step_adaptation`: Verifies RK45 outperforms Euler significantly.
  - `test_rk4_jit_parity` and `test_rk45_jit_parity`: Ensure identical results with JIT on/off.
  - `test_stepper_registration`: Confirms all steppers and aliases are properly registered.
- Added test models `tests/data/models/decay_rk4.toml` and `decay_rk45.toml`.

---

## [2.4.1] – 2025-11-05
### Added
- Added `get_runner` in `src/dynlib/compiler/codegen/runner.py` for obtaining a generic runner
  function.
- Created `runner` function in `src/dynlib/compiler/codegen/runner.py` for fixed-step execution 
  with events and recording.
- Introduced `EulerSpec` in `src/dynlib/steppers/euler.py` for explicit Euler stepper 
  implementation.

### Changed
- Updated `build` in `src/dynlib/compiler/build.py` to use `get_runner` instead of generating 
  runner source code from string.
- Enhanced `emit` in `src/dynlib/steppers/euler.py` to return a callable Python function instead 
  of source code.
- Improved `StepperSpec` in `src/dynlib/steppers/base.py` to include detailed docstrings for 
  `emit` method.

---

## [2.4.0] – 2025-11-04
### Added
- Introduced `Sim.run` method in `src/dynlib/runtime/sim.py` for executing simulations with 
  compiled models.
- Added `run_with_wrapper` in `src/dynlib/runtime/wrapper.py` for orchestrating simulation runs.
- Created `EulerSpec` in `src/dynlib/steppers/euler.py` for explicit Euler stepper implementation.

### Changed
- Updated `allocate_pools` in `src/dynlib/runtime/buffers.py` to ensure `sw0` size is at least 
  `n_state` for steppers requiring workspace.
- Enhanced `Model` dataclass in `src/dynlib/runtime/model.py` with additional attributes for 
  compiled stepper and runner callables.

### Tests
- Added integration tests in `tests/integration/test_euler_basic.py` for Euler simulations, event 
  handling, and buffer growth.
- Updated unit tests in `tests/unit/test_wrapper_reentry.py` to validate re-entry logic for buffer 
  growth.

---

## [2.3.1] – 2025-11-04
### Fixed
- Removed redundant validation functions `validate_dtype_rules` and `validate_equation_targets` 
  from `src/dynlib/dsl/astcheck.py`.
- Fixed unused import `from email.mime import message` in `src/dynlib/errors.py`.

### Tests
- Updated tests in `tests/unit/test_ast_check.py` to reflect changes in validation logic.

---

## [2.3.0] – 2025-11-04
### Added
- Introduced `build_callables` in `src/dynlib/compiler/build.py` for generating RHS and event 
  callables.
- Added `emit_rhs_and_events` in `src/dynlib/compiler/codegen/emitter.py` for code generation.
- Implemented `JITCache` in `src/dynlib/compiler/jit/cache.py` for caching compiled callables.
  - Added `maybe_jit_triplet` in `src/dynlib/compiler/jit/compile.py` for JIT compilation toggle.
- Created `Model` dataclass in `src/dynlib/runtime/model.py` for simulation models.
- Added `Sim` class in `src/dynlib/runtime/sim.py` as a placeholder for simulation runners.

### Changed
- Fixed column-major layout in `tests/unit/test_numba_probe.py` for recording buffers.
- Enhanced `_normalize_event` in `src/dynlib/compiler/mods.py` to handle nested TOML dicts for 
  actions.
- Improved `_read_events` in `src/dynlib/dsl/parser.py` to support nested TOML keys for event 
  actions.

### Tests
- Added unit tests for `build_callables` in `tests/unit/test_codegen_triplet.py`.

---

## [2.2.0] – 2025-11-04
### Added
- Implemented `parse_model_v2` in `src/dynlib/dsl/parser.py` for parsing DSL TOML into normalized 
  models.
- Added validation functions in `src/dynlib/dsl/schema.py` for model headers, tables, and name 
  collisions.
- Introduced `SimDefaults`, `EventSpec`, and `ModelSpec` dataclasses in 
  `src/dynlib/dsl/spec.py`.
- Added `build_spec` and `compute_spec_hash` in `src/dynlib/dsl/spec.py` for model specification 
  and hashing.
- Created `src/dynlib/errors.py` for custom exceptions like `ModelLoadError`.
- Added `allocate_pools`, `grow_rec_arrays`, and `grow_evt_arrays` in `src/dynlib/runtime/buffers.py` 
  for memory management.
- Introduced `Results` dataclass in `src/dynlib/runtime/results.py` for simulation outputs.
- Added `run_with_wrapper` in `src/dynlib/runtime/wrapper.py` for orchestrating simulation runs.

### Tests
- Added unit tests for `parse_model_v2` in `tests/unit/test_ast_check.py`.
- Added tests for buffer growth in `tests/unit/test_buffers_growth.py`.
- Added schema and parser tests in `tests/unit/test_dsl_schema_parser.py`.
- Added tests for `build_spec` and `compute_spec_hash` in `tests/unit/test_dsl_spec.py`.
- Added tests for `apply_mods_v2` in `tests/unit/test_mods.py`.
- Added re-entry tests for `run_with_wrapper` in `tests/unit/test_wrapper_reentry.py`.

---

## [2.1.0] – 2025-11-04
### Added
- Introduced `src/dynlib/__init__.py` to re-export constants, types, steppers, and utilities 
  for stable imports.
- Added `src/dynlib/runtime/runner_api.py` defining stable exit/status codes and the frozen 
  runner ABI.
- Added `src/dynlib/runtime/types.py` for type literals like `Kind`, `TimeCtrl`, and `Scheme`.
- Added `src/dynlib/steppers/base.py` with metadata and struct specifications for steppers.
- Added `src/dynlib/steppers/registry.py` to manage stepper registration and retrieval.
- Added `src/dynlib/utils/arrays.py` with utility functions for array validation and slicing.

### Tests
- Added `tests/unit/test_numba_probe.py` to validate future JIT-compiled runner's numba
  compliance.

---

## [2.0.0] – 2025-11-03
### Changed
- v1 hit a dead end and v0 was a failure. Starting a new design from scratch. Only plot 
  tools are preserved. They should be modified in the future for the new model format.
