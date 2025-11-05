## Changelog

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