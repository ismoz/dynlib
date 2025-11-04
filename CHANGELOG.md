## Changelog

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