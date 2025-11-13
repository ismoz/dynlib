## Changelog

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
