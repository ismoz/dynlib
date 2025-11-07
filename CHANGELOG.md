## Changelog

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
  - Added `model_dtype` parameter for allocating `EVT_LOG_DATA` with correct dtype
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
- Fixed `test_buffers_growth.py` to pass `max_log_width` and `model_dtype` parameters

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
