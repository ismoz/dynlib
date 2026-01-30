## Changelog

---

## [0.37.2] – 2026-01-30
### Added
- BSD 3-Clause Licence.
- Turkish README file.

### Changed
- Minor documentation changes.

---

## [0.37.1] – 2026-01-30
### Added
- Added Turkish documentation and improved existing English documentation.

---

## [0.37.0] – 2026-01-29
### Added
- `mkdocs` compatible documentation in docs folder with multiple language support. Only English docs are available
  in this version.
- Added auto-generation scripts `gen_model_docs.py` and `mkdocs_helpers.py` for mkdocs. `tools/gen_model_docs.py`
  is a single wrapper; so it should be run to generate docs locally. 

### Changed
- Made `jit=False` and `disk_cache=False` defaults in `setup()` and `build()`.
- Removed unnecessary [meta] tables from built-in models.

---

## [0.36.13] – 2026-01-27
### Changed
- Renamed `label` key in [model] DSL table with `name` key. Replaced all occurrences throughout the package.
- Removed temporary development docs. 

---

## [0.36.12] – 2026-01-27
### Added
- Added `model.print_equations()` for printing DSL equations (no generated code) and a demo example:
  `print_equations_demo.py`. It accepts `tables` argument so that users can print various equations like 
  `equations.inverse` and `equations.jacobian`. This method is extendible via `register_equation_table()`
  method inside `build.py` file. This method is only available for quickly checking equations. To view other
  tables like aux values and events use `export_sources()` method.

---

## [0.36.11] – 2026-01-27
### Added
- Added `homoclinic_finder` analysis tool for searching and finding a homoclinic orbit for a given equilibrium.
  A single parameter of an ODE system is changed between an interval during the search.
- Added `homoclinic_tracer` analysis tool for finding a homoclinic orbit of an equilibrium for fixed parameter values. 
  Its result can be plotted using `plot.manifold` plotter utility.
- Added example `homoclinic_finder_tracer.py` for finding a parameter value of an ODE model that might have a
  homoclinic orbit and the tracer is used to visualize the homoclinic orbit for the found parameter.

---

## [0.36.10] – 2026-01-26
### Added
- Added `heteroclinic_finder` analysis tool for searching and finding a heteroclinic orbit between two equilibria.
  A single parameter of an ODE system is changed between an interval during the search.
- Added `heteroclinic_tracer` analysis tool for finding a heteroclinic orbit between two equilibria for fixed
  parameter values. Its result can be plotted using `plot.manifold` plotter utility.
- Added example `heteroclinic_finder_tracer.py` for finding a parameter value of an ODE model that might have a
  heteroclinic orbit between its two equilibria. The tracer is used to visualize the heteroclinic orbit for the found 
  parameter.

---

## [0.36.9] – 2026-01-25
### Added
- Added analysis tool `trace_manifold_1d_ode()` for tracing stable/unstable 1D manifolds of ODE models. The results
  can be plotted using the same `plot.manifold()` utility introduced previously.
- Added an example `manifold_ode_saddle.py` for demonstrating ODE manifold tracing. 

---

## [0.36.8] – 2026-01-25
### Added
- Added analysis tool `trace_manifold_1d_map()` for tracing stable/unstable 1D manifolds of nD maps.
- Added `plot.manifold()` plot utility for plotting manifold analysis results.
- Added an example `manifold_henon.py` for demonstrating manifold tracing.
- Added another built-in Henon map model in `henon2.toml` which is the version used in Kathleen Alligood's book.

---

## [0.36.7] – 2026-01-18
### Added
- Introduced a fixed points / equilibria calculator in `analysis/fixed_points.py` file. It uses Newton solver with
  model type awareness (ODE vs map). Seeds are required as initial guesses. `fixed_points()` method is attached to 
  the `FullModel` class for user convenience (fixed points of a model can be calculated easily for that model with
  its current parameters).

### Tests
- Introduced `test_fixed_points.py` for testing the fixed point calculation feature.

---

## [0.36.6] – 2026-01-17
### Fixed
- `basin_auto` plot results were rotated. `basin_auto.py` now uses `np.meshgrid(..., indexing="ij")` with explicit 
  order="C" raveling for consistent grid flattening.

---

## [0.36.5] – 2026-01-17
### Changed
- Moved all of the runtime analysis utilities from `./analysis/runtime` folder to `./runtime/observers` folder.
- Renamed all runtime analyses as `observers`. `Sim.run()` now accepts `observers` argument instead of `analysis`.
  Added _observer suffix to the names of these utilities to prevent import name clashes. These changes should provide 
  a clear separation between runtime and offline standalone analysis utilities.
- Added _sweep suffix to the names of sweep utilities to prevent import name clashes.
- Updated all analysis related examples and tests.

---

## [0.36.4] – 2026-01-12
### Added
- Inverse RHS equations support for map DSL definitions. [equations.inverse] (with `expr` keyword) or 
  [equations.inverse.rhs] can be used to define inverse map equations. Numba compatibility and caching is provided. 

### Tests
- Added `test_inverse_equations.py` to test new inverse map behavior and updated old DSL related tests.

---

## [0.36.3] – 2026-01-08
### Added
- Added basin of attraction calculation analysis tool `basin_known`. It can calculate basin of attraction of known
  attractors numerically.
- Added various basin of attraction calculation examples.
- Added `basin_plot` utility for plottin basin of attraction results.
- Added various utilities for investigating basin analysis results in `basin_stats.py`.

### Changed
- For sweep and basin analyses introduced chunk-based process parallelism with efficient worker initialization: 
  a chunk of the analysis is assigned to a worker; each worker process initializes its `Sim` object once and reuses 
  it across all chunks assigned to that worker, avoiding redundant JIT compilation and allocation overheads.
- Renamed `analysis/basin.py` -> `analysis/basin_auto.py`.

### Tests
- Fixed tests that contain `exprs` keyword.

---

## [0.36.2] – 2026-01-06
### Added
- Added basin of attraction calculation analysis tool `basin_auto` in `analysis/basin.py`. It can search for 
  attractors and determine their basins using recurrence-based basin estimation method (Datseris & Wagemakers 2022), 
  with an additional persistence-based early assignment and coarsened-grid fingerprint merging. Internally it is 
  called PCR-BM (Persistent Cell-Recurrence Basin Mapping).
- Added Henon map basin of attraction calculation demo example: `basin_henon_demo.py`.
- Added ODE-based limit cycle basin of attraction calculation example: `basin_limit_cycle.py`.
- Added basin of attraction plotting utility `plot_basin` in `plot/basin.py`.
- Added ETO (Energy Template Oscillator) with Circular L Curve built-in model `eto-circular.toml`.
- Added Duffing oscillator built-in model: `duffing.toml`.

### Changed
- Replaced underscore (_) from the built-in model names with hypen (-). For example: `exp_if.toml` -> `exp-if.toml`.
- Analyses can now trigger early exit from hooks by setting `runtime_ws.stop_flag`, even when the DSL stop table is 
  absent, as long as the analysis declares a `stop_phase_mask` (e.g., post‑step).

### Fixed
- Some DSL tables were using `expr` keyword for equation definitions and some were using `exprs`. Unified all 
  equation expression keywords as `expr`. Typos were not causing any errors but steppers would not advance. Now 
  any typo raises an error with a hint.

### Tests
- Added tests for analysis triggered early exit in `test_stop_early_exit.py`.

---

## [0.36.1] – 2026-01-04
### Added
- Added early exit feature: simulations can now stop early when a specified condition is met using the `stop` 
  field in the [sim] section.
- Added support for stop conditions with built-in DSL macros like `cross_up()`, `in_interval()`, `increasing()`,
  `decreasing()`, and others.
- Added `exited_early` property to results to check if simulation stopped due to a stop condition.
- Added example `early_exit_demo.py`.
- Extended runtime workspace with `stop_flag` and `stop_phase_mask` arrays for stop condition evaluation. Since 
  only phase=`post` is supported `stop_phase_mask` is not necessary but it is there for future phase=`pre` and 
  phase=`post` extension.

### Changed
- Updated all runners (base, analysis, fastpath variants for both continuous and discrete systems) to evaluate 
  stop conditions after each step.
- Modified `update_aux()` function to also handle stop flag updates when stop conditions are enabled.
- Updated results API to consider `EARLY_EXIT` as a successful completion status alongside `DONE`.
- Enhanced model specification to include `StopSpec` for parsing and validating stop conditions.
- Updated AST checker to validate stop condition expressions and collect lag requests for stop conditions.
- Modified mods system to support setting `sim.stop` via TOML modifications.
- Updated emitter to compile stop condition expressions into the `update_aux` function.
- Attached exiting `EARLY_EXIT` status code (7) to the new early termination mechanism.

### Tests
- Added test file `test_stop_early_exit.py`.

---

## [0.36.0] – 2026-01-02
### Added
- Added `RunnerVariant` enum to consolidate runner architecture: `BASE`, `ANALYSIS`, `FASTPATH`, and 
  `FASTPATH_ANALYSIS`.
- Added FASTPATH runner templates for both continuous (ODE) and discrete (map) systems. Fastpath runners skip
  event processing and buffer growth checks for fixed-step execution.
- Added `get_runner()` unified API in `runner_variants.py` to replace separate `get_runner_variant()` and 
  `get_runner_variant_discrete()` paths. Now supports explicit variant selection with hooks baked in as globals.
- Added `EARLY_EXIT` status code (7) in `runner_api.py` for future use with basin of attraction analysis.
- Renamed `runtime/fastpath/runner.py` to `runtime/fastpath/executor.py` for clarity.
- Added `runner_cache.py` disk cache helper module for runner variants with variant-aware and template-version 
  aware cache keys.

### Changed
- Consolidated runner templates into single source of truth in `runner_variants.py`, reducing code duplication.
- Analysis hooks are now always injected as global symbols (`ANALYSIS_PRE`, `ANALYSIS_POST`) rather than function 
  arguments. This enables simpler, more JIT-friendly code generation.
- Unified wrapper and fastpath execution to use `get_runner()` with explicit `RunnerVariant` selection based on 
  analysis presence and execution mode (continuous vs discrete).
- Updated `run_with_wrapper()` to always require `model_hash` and `stepper_name` for runner selection.
- Removed legacy dual-ABI runner path (analysis_kind + dispatch arguments). All variants now use hook globals.
- Disk cache keys now include variant type and template version for invalidation support.
- Fastpath now uses `RunnerVariant.FASTPATH_ANALYSIS` instead of legacy `ANALYSIS` path, enabling 
  future capability specialization.

### Removed
- Removed legacy `compiler/codegen/runner.py` (ODE runner template).
- Removed legacy `compiler/codegen/runner_discrete.py` (discrete runner template).
- Removed `analysis_dispatch_pre` and `analysis_dispatch_post` from runner ABI. Hooks are now baked in.

### Tests
- Updated `test_analysis_runtime.py` to use new unified runner API and explicit `model_hash`/`stepper_name`.
- Updated `test_fastpath_runner.py` to import from `executor` module.
- Updated `test_wrapper_reentry.py` to provide required `model_hash` and `stepper_name` arguments.

---

## [0.35.9] – 2025-12-29
### Fixed
- The `ab2` and `ab3` state history usage was wrong. Shifted tangent‑only Lyapunov propagation into `pre_step` 
  so `ab2`/`ab3` see (y_n, v_n) when forming g_n = J(y_n)v_n. `post_step` now only normalizes/accumulates (flow) 
  or computes Jv for map mode.

---

## [0.35.8] – 2025-12-29
### Changed
- Changed version numbers to v0 and updated all git tags accordingly. This will be an indicator that the package
  is still alpha not a stable release.

---

## [0.35.7] – 2025-12-29
### Added
- Variational stepping support to the steppers: `rk2`, `ab2`, and `ab3`.

### Changed
- Multi-step variational stepping is now supported for Lyapunov analyses. However, combined calculations (advance 
  the state and the tangents together in a single stepper call) is not possible for them. All calculations are
  tangent-only (advance only the tangents using a variational step routine while the state is advanced by the 
  normal runner path). Also Lyapunov Spectrum calculations always prefer tangent-only calculations.

---

## [0.35.6] – 2025-12-28
### Added
- `sweep.lyapunov_spectrum()` utility for plotting Lyapunov spectrum change of a system for a range of parameters.
  Just like `sweep.lyapunov_mle()` it supports fast-path runner and parallelization.
- Added `sweep.lyapunov_spectrum()` example `lyapunov_sweep_spectrum_demo.py`.

### Changed
- Removed `SweepAnalysis.peaks()` method and introduced `SweepAnalysis.extrema()`. `peaks()` was only finding 
  maxima. New extrema() method can find maxima, minima and both (default) via its `kind` argument.
- Renamed `lyapunov_sweep_demo.py` -> `lyapunov_sweep_mle_demo.py`.

### Fixed
- Fast-path runner was treating transient duration as part of the `T` value. This issue was fixed for `N` before 
  but forgotten for `T` values. Now `T` values also ignore transient duration and recording starts at `t0` after 
  the `transient`.

### Tests
- Fixed `test_fastpath_runner.py`. It was assuming wrong fast-path transient behavior.

---

## [0.35.5] – 2025-12-27
### Added
- Added `JITUnavailableError` for clear failures when `jit=True` but numba is missing.
- Added variational workspace helpers for Euler and RK4 steppers so Lyapunov analyses can reuse stepper buffers.
- Added labeled horizontal-line support to plots.

### Changed
- Lyapunov MLE/spectrum now use stepper-provided variational workspaces and can prefer combined state+tangent steps.
  Previously state+tangent steps were hardcoded in `lyapunov.py`.
- JIT-related code paths now raise instead of warning when numba is unavailable (guards, runners, steppers).
- Plot line label placement is more configurable (position, pad, rotation, color) and theme defaults are updated.
- Examples are updated to use the new vline/hline/vband helpers.

### Tests
- Renamed variational stepping tests to `tests/unit/test_variational_stepping.py`.

---

## [0.35.4] – 2025-12-27
### Added
- Added `builtin://ode/lorenz` model definition.
- Added support for horizontal lines (`hlines`) and horizontal bands (`hbands`) in plot utilities.
- Added variational stepping support to `Euler` stepper. Both Euler and RK4 now advertise 
  `variational_stepping=True` capability.
- Added `lyapunov_lorenz_demo.py` demonstrating Max Lyapunov and Lyapunov spectrum analyses of the Lorenz system.

### Changed
- Renamed `bands` argument to `vbands` in plot functions to distinguish from the new `hbands`.
- Lyapunov analyses (`lyapunov_mle`, `lyapunov_spectrum`) now strictly require the stepper to support variational 
  stepping. If the stepper does not support it, a `ValueError` is raised (no longer falls back to Euler).
- `CombinedAnalysis` now enforces stricter rules:
    - Rejects combination if more than one analysis requires runner-level variational stepping.
    - Rejects combination if any analysis mutates state.
- `AnalysisResult` now provides `trace_steps`, `trace_time`, and `record_interval` attributes.
- Transient warm-up phase now explicitly disables analysis hooks.

### Tests
- Updated `test_rk4_variational.py` to reflect strict stepper requirements and new `CombinedAnalysis` rules. 
  Added `test_transient_warmup_skips_analysis_hooks`.

---

## [0.35.3] – 2025-12-24
### Added
- Variational stepping support (simultaneous numerical integration of both the original dynamical system and 
  its variational / tangent equations using the same numerical method) for RK4 stepper via 
  `emit_step_with_variational()` and `emit_tangent_step()` methods.

### Changed
- Lyapunov exponent analysis (`lyapunov_mle` and `lyapunov_spectrum`) now automatically uses variational 
  stepping when the chosen stepper supports it (currently RK4), otherwise falls back to Euler integration.
- When computing Lyapunov exponents, the analysis tracks how small perturbations grow over time by integrating 
  "tangent vectors" (directional derivatives). For accurate results, both the main system and tangent vectors 
  should use the same numerical method. This update enables RK4 to integrate tangent vectors with full 
  4th-order accuracy instead of 1st-order Euler, significantly improving reliability for continuous systems.

### Fixed
- Lyapunov exponent calculations now use the same numerical method (e.g., RK4) for both the system state and 
  the tangent vectors, ensuring mathematical consistency and improved accuracy. Previously, tangent vectors 
  were always integrated using Euler method regardless of the chosen stepper.

### Tests
- Added comprehensive test suite (`test_rk4_variational.py`) covering RK4 variational mode selection, runtime 
  verification, Euler fallback behavior for steppers without variational support, and numerical accuracy 
  validation with known linear systems.

### Known Issues
- Only RK4 stepper is supported for variational stepping. Other fixed step steppers should be modified similarly.

---

## [0.35.2] – 2025-12-24
### Fixed
- Combined runtime analyses now share trace capacity correctly. Each module writes into its own slice with a
  shared step-level cursor, preventing double-increment of `trace_count` and eliminating spurious
  `TRACE_OVERFLOW` when multiple analyses with different trace widths run together. Both Python and generated
  JIT hooks use the synchronized counter and import `numpy` explicitly in the generated namespace.

---

## [0.35.1] – 2025-12-24
### Changed
- Refactored analysis hook dispatch to eliminate `NumbaExperimentalFeatureWarning`. Analysis hooks (`pre_step`,
  `post_step`) are now injected as global symbols (`ANALYSIS_PRE`, `ANALYSIS_POST`) into generated runner source
  code via `exec()` with a custom namespace, making call targets statically resolvable by Numba instead of using 
  first-class function arguments.
- Runner variants are cached (64 entries max) keyed by (model_hash, stepper_name, analysis_signature_hash, 
  runner_type, jit), ensuring the same runner/analysis combination compiles only once per parameter sweep. Cache
  mechanism uses LRU (Least Recently Used) eviction.
- Replaced `literal_unroll` over callable containers with explicit sequential call codegen for combined hooks, 
  avoiding callable containers entirely.
- When no analysis is active, base runner templates are used without any `analysis_kind` branching overhead.

### Added
- `signature(dtype) -> tuple` method to `AnalysisModule` protocol for stable cache key generation.
- New module `dynlib.compiler.codegen.runner_variants` with `get_runner_variant()`, 
  `get_runner_variant_discrete()`, `analysis_signature_hash()`, and `clear_variant_cache()` API.

### Fixed
- Resolved `NumbaExperimentalFeatureWarning` caused by first-class function types in analysis hook dispatch.

---

## [0.35.0] – 2025-12-24
### Added
- `lyapunov_spectrum()` analysis utility for performing Benettin QR / Shimada–Nagashima (reorthonormalization) 
  style Lyapunov spectrum analysis.
- Introduced `mode` option to `lyapunov_mle` and `lyapunov_spectrum` analyses which can be `auto`, `flow`, or `map`.
  `auto` (default) mode can detect type of Lyapunov exponent calculation from the DSL model type.

### Known Issues
- Analyses hooks cause `NumbaExperimentalFeatureWarning` because they are passed as first-class functions to the 
  jitted runner.
- Analyses occasionally cause `TRACE_OVERFLOW` which should not be possible.
- `flow` mod analyses use Euler integration regardless of the `Sim` stepper.

---

## [0.34.10] – 2025-12-24
### Changed
- Removed the legacy `dynlib.runtime.model.Model` dataclass. `Sim` now expects the `FullModel` returned by `build()`, 
  eliminating the duplicate runtime model type. Maintaining both `Model` and `FullModel` class was hard. For example 
  `export_sources()` shortcut was attached to the old `Model` class instead of the newer `FullModel` class.
- Updated `export_sources.md` doc file and `export_sources_demo.py` example.

### Tests
- Replaced `Model` class usage with `FullModel` class.

---

## [0.34.9] – 2025-12-24
### Tests
- Added comprehensive test coverage for external Jacobian mode in implicit steppers (`bdf2`, `bdf2a`, `tr-bdf2a`, 
  `sdirk2`) via new test file `test_jacobian_external_mode.py`. Tests verify accuracy against analytic solutions,
  consistency between internal and external modes, and correct behavior on both simple (1D exponential decay) and
  complex (2D Van der Pol oscillator) systems.

---

## [0.34.8] – 2025-12-23
### Changed
- Modified steppers relying on Jacobian matrices (`bdf2`, `bdf2a`, `tr-bdf2a`, `sdirk`) so that they can utilize
  DSL-generated Jacobian functions in their calculations alongside their previous finite difference numerical
  approximations. These two modes can be selected using the stepper config value `jacobian_mode` which can be
  `external` (use DSL Jacobian) or `internal` (use numerical approximation). Default is `internal` because I am
  worried about users providing a faulty Jacobian matrix.
- Set JacobianPolicy of `bdf2`, `bdf2a`, `tr-bdf2a`, `sdirk` steppers to `optional` meaning that they can work
  both with external and internal Jacobians.
- For steppers to utilize DSL Jacobian functions, added `jacobian_fn` and `jvp_fn` callable args to the `emit()` 
  function of all steppers. They are optional so steppers not using Jacobians are not changed.
- Ensured that `aux` values are calculated exactly once in Jacobian matrix entries to prevent repeated 
  recalculations and slowing down the steppers unnecessarily. This feature is called as aux hoisting.
- Removed `bdf2_scipy` and `bdf2a_scipy` because they were performing very poorly and there was no reason to keep
  them around. However, stepper features like `jit_capable` are kept for future use.
- Updated `export_sources.md` file encouraging users to utilize `Model.export_sources()` instead of the standalone
  function.

### Added
- Example for comparing run times of steppers with `external` and `internal` Jacobian modes. `external` mode is
  slightly faster for all steppers.

### Tests
- Added `test_aux_hoist_jacobian.py` to test correct dependency resolution during `aux` variable hoisting in DSL 
  Jacobian functions.
- Removed `bdf2_scipy` and `bdf2a_scipy` related tests.

---

## [0.34.7] – 2025-12-23
### Changed
- Each parameter result had its own result data class. Unified sweep results with a single template: `SweepResult` 
  + `TrajectoryPayload`. Removed individual data classes like `ParamSweepTrajResult` and the `ParamSweepMLEResult`
  introduced in the previous version. Future sweep analyses should use this template.

---

## [0.34.6] – 2025-12-22
### Added
- Lyapunov MLE parameter sweep functionality via `sweep.lyapunov_mle()` for computing maximum Lyapunov exponents 
  across parameter ranges.
  - Returns `ParamSweepMLEResult` with converged MLE values, log growth, step counts, and optional convergence 
    traces.
  - Supports parallel execution via `parallel_mode` parameter ("auto", "threads", "none").
  - Includes `stack_traces()` method for uniform-length trace analysis.
  - Exported in `dynlib.analysis` namespace alongside `scalar`, `traj` sweep functions.
- Example script `examples/analysis/lyapunov_sweep_demo.py` demonstrating bifurcation diagram and MLE sweep 
  visualization for the logistic map.

### Changed
- `_LyapunovModule.resolve_hooks()` now caches compiled JIT hooks per dtype to avoid redundant JIT compilation 
  on every run, improving performance for repeated analysis calls.

---

## [0.34.5] – 2025-12-22
### Added
- `ResultsView.analysis` now returns `AnalysisResult` wrappers that dynamically expose analysis-specific outputs 
  and traces via named access.
  - Generic design: field names are auto-discovered from each analysis module's `output_names` and `trace_names` 
    metadata - no hardcoding per analysis type.
  - Attribute access for values: e.g., for Lyapunov MLE, `lyap.log_growth`, `lyap.steps` (from `output_names`), 
    and `lyap.mle` (from `trace_names`). Different analyses will have different field names.
  - Trace attributes return final scalar values: `lyap.mle` returns the converged value (last element).
  - Bracket access for full arrays: `lyap["mle"]` returns complete trace array for plotting/analysis.
  - Mapping interface is provided as low-level generic API: `lyap["out"]`, `lyap["trace"]`, `lyap["stride"]`.
  - Discovery support: `lyap.output_names`, `lyap.trace_names`, `list(lyap)`, `dir(lyap)` for introspection and 
    tab-completion.
  - Scales from 1D (single MLE value) to nD (Lyapunov spectrum, multiple metrics) automatically.

### Changed
- Updated `lyapunov_logistic_map_demo.py` to demonstrate new API: `lyap.mle` instead of manual 
  `lyap["out"][0] / lyap["out"][1]` computation.

### Tests
- Extended `test_analysis_runtime.py` for the new `AnalysisResult` feature.

---

## [0.34.4] – 2025-12-22
### Changed
- Moved `src/dynlib/analysis/post/sweep.py` -> `src/dynlib/analysis/post/sweep.py` to gather all sweeps into one 
  place. Changed sweep imports accordingly.

### Fixed
- `lyapunov_mle()` now validates inputs consistently: a model or explicit `jvp` (with `n_state`) is required, and
  `record_interval` is optional (defaults to stride 1). This prevents silent misuse when the factory is called
  without a model context while keeping `Sim`-injected factory usage unchanged.

---

## [0.34.3] – 2025-12-22
### Changed
- Simplified runtime analysis API with consistent factory pattern:
  - `lyapunov_mle()` now uses simple conditional logic: if `model` parameter is provided, returns `AnalysisModule` 
    directly; otherwise returns a factory function.
  - Factory accepts `model` parameter that `Sim.run()` injects automatically via signature introspection.
  - All parameters have sensible defaults: `jvp` and `n_state` extracted from model when not provided.
  - Usage patterns:
    - `analysis=lyapunov_mle()` — factory mode, Sim injects model (most common).
    - `analysis=lyapunov_mle(record_interval=2)` — factory with custom params, Sim injects model.
    - `analysis=lyapunov_mle(model=sim.model)` — direct mode, returns AnalysisModule immediately.
  - This simple if/else pattern generalizes cleanly to future analysis modules with different signatures.
- Updated logistic map Lyapunov example to demonstrate simplified factory API.

---

## [0.34.2] – 2025-12-22
### Fixed
- Runtime `CombinedAnalysis` now composes child hooks with precomputed offsets and numba-friendly closures, 
  making combined analyses eligible for fast-path/JIT execution while preserving the Python path when `jit=False`.
- Analysis JIT compilation remains opt-in; pure-Python runners continue to dispatch uncompiled hooks when JIT is 
  not requested or numba is unavailable.
- With this and previous changes, all of the issues blocking numba compatibility (`jit=True`) of runtime analysis 
  modules are resolved.

---

## [0.34.1] – 2025-12-21
### Added
- DSL models can now declare Jacobians directly in TOML via `[equations.jacobian].exprs` with deterministic 
  state order semantics (state decleration order is used for determining the order of the matrix).
- Jacobian declarations generate JVP (Jacobian Vector Product) operators (and optional dense fill) that are 
  JIT/disk-cache aware and wired into models.
- Lyapunov runtime analysis now consumes JVPs, making Jacobian-dependent analyses work without Python 
  callbacks.

### Changed
- State declaration order is now part of the spec hash to avoid cache ambiguity and is preserved through mods
  /build.
- Capability checks now distinguish JVP and dense-Jacobian requirements; `Sim` validates analysis requirements 
  early.
- Logistic map Lyapunov example updated to use DSL Jacobian instead of a Python function.

### Known Issues
- Models without a declared Jacobian still fall back to Python-only analyses; declarative TOML Jacobians are 
  now compiled to JVPs but dense numeric Jacobians are not generated automatically.
- Steppers using numerical Jacobian approximations can benefit from builtin Jacobian functions.

---

## [0.34.0] – 2025-12-21
### Added
- Runtime analysis system for computing diagnostics during simulation execution:
  - New `dynlib.analysis.runtime` module with `AnalysisModule`, `AnalysisHooks`, and `TraceSpec` for building 
    analysis pipelines.
  - `lyapunov_mle()` function computes maximum Lyapunov exponents for detecting chaos in discrete and continuous 
    systems.
  - Analysis modules can run alongside integration via `Sim.run(analysis=...)` parameter.
  - Support for both Python and JIT-compiled analysis hooks for performance.
  - Trace buffers allow sampling analysis results at configurable intervals during runs.
  - Fast-path integration now supports analysis modules when JIT hooks are provided.
- New Lyapunov example (`lyapunov_logistic_map_demo.py`) demonstrating chaos detection in the logistic map.

### Changed
- Reorganized analysis tools into two submodules for clarity:
  - `dynlib.analysis.post`: Post-run analysis (sweeps, bifurcations, trajectory statistics) - moved from 
    `dynlib.analysis`.
  - `dynlib.analysis.runtime`: During-run analysis (Lyapunov, online diagnostics).
- Enhanced `Results` and `ResultsView` with `.analysis` property for accessing runtime analysis outputs.
- Analysis module imports updated: use `from dynlib.analysis.post import sweep, traj` instead of 
  `from dynlib.analysis import .sweep, traj`
- Runner ABI version incremented to 3 to accommodate analysis buffer arguments
- All runners (continuous/discrete) now accept analysis workspace, output, trace buffers, and dispatch hooks.
- Fast-path capability checking now validates analysis module requirements (fixed-step, Jacobian, event 
  compatibility).

### Tests
- Added comprehensive tests for runtime analysis infrastructure (`test_analysis_runtime.py`).

### Known Issues
- Numba option will not work for runtime analysis modules. There are lots of problems for numba compatibility.
- Jacobian equation definitions are very crude and creates numba compatibility issues. TOML DSL definition would 
  be better.

---

## [0.33.1] – 2025-12-18
### Changed
- Fixed some minor plot related bugs. 
- `export.savefig()` now can infer save format from the file extension of path. Otherwise `fmts` arg should 
  be used. If `fmts` and a path with file extension are used together, then an error is raised.

### Tests
- Added unit tests for `savefig()` behavior.

---

## [0.33.0] – 2025-12-18
### Added
- Introduced comprehensive bifurcation analysis tools for exploring parameter-dependent dynamics:
  - `BifurcationExtractor` class provides post-processing of trajectory sweeps into bifurcation scatter data
  - Multiple extraction modes: `.all()` (all points), `.tail(n)` (attractor cloud), `.final()` (convergent 
    states), `.peaks()` (local maxima)
  - `BifurcationResult` dataclass holds bifurcation data with metadata for plotting
  - `bifurcation_diagram()` plotting function with scatter-optimized defaults
  - Convenient API: `sweep_result.bifurcation("x").tail(50)` extracts bifurcation data from parameter sweeps
  - New module `dynlib.analysis.bifurcation` for bifurcation post-processing utilities
  - New module `dynlib.plot.bifurcation` for bifurcation diagram plotting
- Added three comprehensive bifurcation examples:
  - `bifurcation_logistic_map.py`: Basic bifurcation diagram demonstration
  - `bifurcation_logistic_map_annotated.py`: Advanced analysis with annotations and zoomed cascade views
  - `bifurcation_logistic_map_comparison.py`: Comparison of different extraction modes (final/tail/peaks)
- `sweep.traj()` now accepts `parallel_mode` and `max_workers` parameters for controlling fast-path batch 
  execution parallelism.

### Changed
- Plot module has been reorganized for better API clarity and consistency:
  - Moved `cobweb()` from `dynlib.plot.analysis` to `dynlib.plot.cobweb()`
  - Moved `hist()` from `dynlib.plot.analysis` to `dynlib.plot.utils`
  - Added top-level convenience function `return_map()` as alias for `phase.return_map()`
  - Removed `dynlib.plot.analysis` module (replaced by domain-specific modules)
  - Updated examples to use new plot API imports

### Fixed
- Fixed critical bug where `N` parameter was incorrectly reduced by transient steps, causing fewer points to 
  be recorded than expected. Now `N` correctly specifies the number of recorded steps *after* transient warm
  up completes.
- Vector field `scale` parameter now properly applies only when `normalize=True` is set, fixing unintended 
  scaling behavior.

### Tests
- Added comprehensive test coverage for bifurcation diagram functionality
- Added regression test `test_fastpath_transient_with_N()` to prevent transient/N interaction bugs

---

## [0.32.1] – 2025-12-16
### Changed
- Applied minor bug fixes to the new fast-path runner feature.
- Centralized fast-path capability gating in `src/dynlib/analysis/sweep.py` by adding `_assess_fastpath_support`, 
  reusing the same support for both batch and per-run fallbacks so we don’t re-evaluate or fabricate support 
  objects.
- `_run_batch_fast` now returns `(result, support)` without emitting its own warnings; batch callers warn once via
  `_warn_fastpath_fallback`, including the capability reason when available.
- Fallback runs reuse the same `FastpathSupport` and suppress per-run warnings, ensuring only a single, informative 
  warning per sweep when fast-path is unavailable.

---

## [0.32.0] – 2025-12-16
### Added
- Introduced a fast-path analysis runner (`dynlib.runtime.fastpath`) optimized for parameter sweeps and 
  batch simulations:
  - **Recording plans**: `FixedStridePlan` and `TailWindowPlan` pre-allocate buffers to avoid dynamic growth
  - **Capability gating**: Automatic validation ensures fast-path constraints are met before execution
  - **Stateless execution**: Runs don't mutate `Sim` session state, enabling safe parallel execution
  - **Three-level API hierarchy**:
    - Low-level: `run_single_fastpath()`, `run_batch_fastpath()` for raw model arrays
    - High-level standalone: `fastpath_for_sim()`, `fastpath_batch_for_sim()` for `Sim` objects
    - User-facing method: `Sim.fastpath()` for ergonomic one-off executions
  - **Parallel execution modes**:
    - `parallel_mode="auto"`: Automatically selects optimal strategy (default)
    - `parallel_mode="threads"`: ThreadPoolExecutor with GIL-free execution for JIT builds (~Nx speedup)
    - `parallel_mode="none"`: Sequential execution for debugging
    - `max_workers` parameter controls thread pool size
  - **Transient warm-up**: Optional `transient` parameter discards initial samples before recording
  - **Selective recording**: Compatible with `record_vars` for memory-efficient variable selection
- Added `RecordingPlan` base class with `FixedStridePlan(stride)` for regular sampling and 
  `TailWindowPlan(stride, window)` for keeping only last N samples.
- Added `FastpathSupport` and `assess_capability()` for runtime validation of fast-path eligibility.

### Changed
- Parameter sweeps (`analysis.sweep.scalar()`, `analysis.sweep.traj()`) now attempt fast-path backend when 
  eligible before falling back to `Sim.run()`, providing automatic optimization without user intervention.
- Fast-path batch execution achieves near-linear speedup with JIT compilation (e.g., 8x on 8 cores) by 
  leveraging Numba's GIL-free execution in threaded environments.
- Sweep operations now emit a one-time warning when falling back to `Sim.run()`, helping users identify 
  performance optimization opportunities without flooding output. Pure Python + threads: ~1.1-1.3x speedup 
  (limited by GIL).

### Known Issues
- Fast-path requires fixed-step steppers (Euler, RK4, Map, etc.); adaptive steppers fall back to `Sim.run()`.
- Event logging is unsupported (apply-only events work fine).
- Lagged systems (`lag_x(k)`) currently disabled pending ring-buffer management improvements.
- Recording interval must be positive and known in advance.

---

## [0.31.5] – 2025-12-15
### Added
- Vector field plots can now create animations across parameter sweeps via `vectorfield_animate()` with 
  configurable frame rates, repeat settings, and custom update functions for parameters and fixed states.
- Vector field plots support multi-panel parameter sweeps via `vectorfield_sweep()` with shared or independent 
  colormaps, custom titles, and flexible grid layouts.
- Added examples `vectorfield_sweep_demo.py`, `vectorfield_animate_demo.py`, and `vectorfield_animation.py` to 
  demonstrate the new features.

### Changed
- When `normalize=True` in vector field plots, quiver arrows now use data-unit scaling to show unit vectors at 
  their true size instead of being auto-rescaled.

### Tests
- Added unit tests for vector field animations covering parameter sweeps, custom functions, frame control, and 
  animation properties.
- Added unit tests for vector field sweep functionality including shared speed normalization, custom sweep 
  definitions, and facet titles.

---

## [0.31.4] – 2025-12-15
### Added
- Vector field plots can color arrows and streamlines by speed magnitude via `speed_color=True`, with optional
  `speed_cmap`/`speed_norm` forwarded to Matplotlib.

---

## [0.31.3] – 2025-12-15
### Added
- Vector field plotting now supports a streamlines mode via `mode="stream"` with kwargs forwarded to 
  `matplotlib.streamplot()`.

---

## [0.31.2] – 2025-12-15
### Added
- Vectorfield plot demo `vectorfield_highdim_demo.py` for demonstrating vector fields of 2D slices of higher
  dimensional systems.

### Changed
- Renamed `series.multi()` `series` argument to `y` for API consistency.

---

## [0.31.1] – 2025-12-15
### Added
- Interactive vector field plots respond to clicks by simulating trajectories from that point using the compiled 
  model’s stepper, while nullclines can be toggled without recomputation (key `N`) and trajectories can be cleared 
  via keyboard shortcuts (key `C`).
- Vector field plotting now accepts a `stepper` override when compiling from URIs/ModelSpecs and exposes `T`/`dt` 
  arguments of the stepper.

### Tests
- Added coverage for stepper overrides in vector field plotting and interactive trajectory handling.

---

## [0.31.0] – 2025-12-15
### Added
- Added vector field plotting functions for ODE models into `plot/vectorfield.py`. It creates quiver plots and 
  can calculate nullclines numerically. 
- Added demo example for vector field plotting.

### Tests
- Added unit tests for vector field evaluation and plotting in `tests/unit/test_vectorfield_eval.py`.

---

## [0.30.11] – 2025-12-10
### Added
- Added `TrajectoryAnalyzer`/`MultiVarAnalyzer` utilities under `dynlib.analysis` and exported them for easy 
  imports.
- Added `ResultsView.analyze()` helper that returns trajectory analyzers for recorded states/aux variables 
  with stats, extrema, crossing detection, and time-above/below helpers. `summary()` call returns all trajectory 
  analysis results.
- Added `examples/analysis/demo_trajectory_analysis.py` showing analyzer usage on a damped oscillator run.

### Tests
- Added unit tests covering analyzer selection (states vs aux), percentile validation, crossing/time-above 
  calculations, and analyzer caching for multi-var cases.

---

## [0.30.10] – 2025-12-03
### Added
- Added iterable `.runs` property to `ParamSweepTrajResult` for intuitive access to individual sweep runs:
  - `SweepRun` dataclass provides `.param_value`, `.t`, and `["var"]` access for each run
  - `SweepRunsView` provides list-like interface supporting indexing, iteration, and length
  - Consistent API: `run["x"]` works like `res["x"]` but for individual runs
  - Example: `for run in res.runs: plot(run.t, run["x"])`

### Changed
- Removed redundant `.t_all` shortcut properties from `ParamSweepTrajResult` in favor of unified `.runs` interface. 
  Now `res.t` is equivalent to `res.runs[0].t` as a shortcut.

---

## [0.30.9] – 2025-12-02
### Added
- Added `dt_max` parameter to adaptive ODE steppers (RK45, BDF2, BDF2A_scipy, TR-BDF2A) to limit maximum step size 
  during step size calculation.
- Added `inf` and `nan` imports to generated stepper modules for proper numba caching.

### Changed
- Enhanced initial step size selection to respect `dt_max` configuration from stepper configs.
- Listed `prod()` DSL expression as builtin (it was implemented but not defined).

---

## [0.30.8] – 2025-12-02
### Added
- Added `phase.multi()` function to plot multiple 2D phase trajectories on the same axes, useful for showing how 
  trajectories change with different parameters in phase space.
- Added `parameter_sweep.py` example showing how to use the sweep functions with a simple exponential decay model.

### Changed
- Enhanced `sweep.scalar()` and `sweep.traj()` functions with better documentation, support for initial time 
  offsets, and improved result classes with named variable access.

---

## [0.30.7] – 2025-11-25
### Added
- Enhanced the `analysis.sweep()` utility with data stacking for consistent run lengths, named access to 
  variables, and time axis convenience properties:
  - `res.t` will yield time values for the first sim. 
  - `res.t_runs` and `res.t_all` will allow access to whole time value ndarray.
  - `res[x]` will yield parameter sweep results for the `x` variable as (K,N) ndarray.
- Added support for numpy ndarrays in `plot.multi` function, allowing direct plotting of 1D and 2D arrays with 
  automatic naming.

---

## [0.30.6] – 2025-11-25
### Added
- Added a new `analysis` module with parameter sweep tools for running simulations with different parameter 
  values and collecting results. It is an early sketch.

---

## [0.30.5] – 2025-11-24
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

## [0.30.4] – 2025-11-24
### Added
- Added user-defined `[constants]` table support to inline model-specific numeric literals across DSL
  expressions, with collision guards against states/params/aux and reserved identifiers.

---

## [0.30.3] – 2025-11-24
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

## [0.30.2] – 2025-11-23
### Changed
- Runners now refresh auxiliary variable values before recording initial conditions to ensure aux data is 
  available at the start for recording.
- Results API now properly handles variable names from recorded data instead of listing all available states. 

---

## [0.30.1] – 2025-11-23
### Changed
- Warm-up now matches the new runner ABI and quadruplet callables so JIT compilation is triggered during 
  build again.
- `_warmup_jit_runner` now allocates AUX/selective-recording buffers, computes event log width, and passes 
  `state_rec_indices`, `aux_rec_indices`, and counts to the runner along with the AUX buffer.
- `_all_compiled` checks `update_aux` too, preventing skips when only the aux updater lacks signatures.

---

## [0.30.0] – 2025-11-23
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

## [0.29.6] – 2025-11-23
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

## [0.29.5] – 2025-11-23
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

## [0.29.4] – 2025-11-21
### Added
- Added validation to prevent auxiliary variables from using reserved names like `t` to avoid conflicts 
  with runtime symbols. `_AUX_RESERVED_NAMES` list can be expanded to restrict aux names in the future.

### Changed
- Renamed auxiliary variable in Ikeda map model from `t` to `theta` for clarity and to avoid reserved 
  name conflict.

### Tests
- Added a test for reserved auxiliary name validation.

---

## [0.29.3] – 2025-11-20
### Added
- Added documentation for DSL model file template in `docs/dsl_model_template.md`.
- Added several builtin models for ODE and MAP types:
  - ODE models: `exp_if` (Exponential Integrate-and-Fire), `fitzhugh_nagumo`, `hodgkin_huxley`, `leaky_if` 
    (Leaky Integrate-and-Fire), `quadratic_if`, `resonate_if`.
  - MAP models: `henon`, `ikeda`, `logistic`, `lozi`, `sine`, `standard`.

---

## [0.29.2] – 2025-11-20
### Added
- Added `choose_default_stepper()` function to automatically select appropriate steppers based on model type
  if DSL model spec and user does not provide one. The hard-coded defaults are `map` -> `map` and `ode` -> `rk4`.
- Added theme usage examples in collatz.py, detect_transition.py, and logistic_map.py to demonstrate theme 
  presets.

### Changed
- Improved plotting primitives with enhanced docstrings and better parameter handling.
- Standardized arguments of plotting functions.

---

## [0.29.1] – 2025-11-20
### Added
- Added plotting theme system using `ThemeSpec` dataclass with inheritance support for better theme management 
  and customization.
- Added `themes_demo.py` example to demonstrate all available theme presets with sample plots.

### Changed
- Enhanced savefig function to properly handle constrained layout without clipping.
- Improved style resolution with clear priority hierarchy separating visual patterns from rendering properties.

---

## [0.29.0] – 2025-11-20
### Added
- Added `dynlib` CLI entry point with `model validate`, `steppers list`, and `cache` management 
  subcommands for model validation, registry inspection, and JIT cache cleanup. The CLI entry point is
  placed into `src/dynlib/cli.py`.

### Tests
- Added unit tests covering the new CLI flows (model validation success/failure, stepper filters, cache 
  listing/clearing).

---

## [0.28.9] – 2025-11-20
### Added
- Added validation script for adaptive ODE steppers tolerance sweep.
- Added RK2 (explicit midpoint) stepper for fixed-step ODE simulations.
- Added SDIRK2 (Alexander) stepper, a JIT-compatible implicit method for stiff ODEs.

### Tests
- Added basic accuracy and contract tests for RK2 and SDIRK2 steppers.

---

## [0.28.8] – 2025-11-20
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

## [0.28.7] – 2025-11-20
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

## [0.28.6] – 2025-11-19
### Added
- Added `bdf2a` stepper, an adaptive BDF2 method with variable step size for stiff ODEs.
- Added `tr-bdf2a` stepper, an adaptive TR-BDF2 method combining trapezoidal rule and BDF2 for better 
  stability. It is not optimized, I might optimize it in the next version if it is feasible.

### Changed
- Improved RK45 stepper performance by moving k1 computation outside the adaptive retry loop.
- Renamed `StepperMeta` `stiff_ok` key to `stiff`.

---

## [0.28.5] – 2025-11-19
### Added
- Added `select_steppers()` function to filter steppers by metadata fields like kind, scheme, jit_capable, 
  etc.
- Added `list_steppers()` function to get a list of stepper names matching filter criteria.
- Added `validation/` folder with `ode_steppers_dt_sweep.py` script for benchmarking ODE stepper accuracy 
  across different time steps.

### Tests
- Fixed stepper name in `test_stepper_config.py` test to use a stepper with model config key.

---

## [0.28.4] – 2025-11-19
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

## [0.28.3] – 2025-11-19
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

## [0.28.2] – 2025-11-18
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

## [0.28.1] – 2025-11-18
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

## [0.28.0] – 2025-11-18
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

## [0.27.2] – 2025-11-18
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

## [0.27.1] – 2025-11-17
### Changed
- Improved BDF2_JIT stepper by adding checks for NaN/Inf values during calculations to exit early in 
  case of invalid data. Also improved Jacobian calculations.

### Tests
- Added BDF2_JIT stepper contract and accuracy tests.

---

## [0.27.0] – 2025-11-17
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

## [0.26.5] – 2025-11-16
### Added
- Added new `StepperCaps` dataclass to hold stepper-specific features that can be added or removed 
  without changing the rest of the stepper `StepperMeta` declarations.

### Changed
- Moved `dense_output` flag from `StepperMeta` to `StepperCaps` for better organization.
- Updated all stepper implementations (Euler, RK4, RK45, AB2, AB3, Map) to use the new caps structure.

---

## [0.26.4]
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

## [0.26.3] – 2025-11-16
### Changed
- Removed NaN/Inf checks from `AB2` and `AB3` steppers, since they are fixed-step solvers.
- Removed workbanks related docstrings from steppers.

---

## [0.26.2] – 2025-11-16
### Changed
- Updated `snapshot_demo.py` and `uri_demo.py` examples. All examples work at this point.

### Tests
- Updated `test_snapshot_persistence.py` test.

---

## [0.26.1] – 2025-11-16
### Changed
- Updated the docs throughout the package. Removed remnants of the old workbanks docs.
- Removed stepper_banks.md file and introduced stepper_workspace.md file.
- Updated ISSUES.md and TODO.md files.

### Fixed
- Cobweb plotter was still using the old workbanks API. Now it also uses runtime workspace.

---

## [0.26.0] – 2025-11-15
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

## [0.25.1] – 2025-11-15
### Added
- Added AB3 (Adams-Bashforth 3rd order) stepper for ODE simulations.
- Added basic and contract tests for AB3.

---

## [0.25.0] – 2025-11-15
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

## [0.24.1] – 2025-11-14
### Added
- Added scalar DSL macros usable in aux, equations, and event actions: `sign(x)`, `heaviside(x)`, 
  `step(x)`, `relu(x)`, `clip(x, a, b)`, and `approx(x, y, tol)`. They lower to comparisons and 
  builtins only, keeping generated code Numba-friendly.

### Tests
- Added regression coverage for the new macros in `tests/unit/test_scalar_macros.py`.

---

## [0.24.0] – 2025-11-14
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

## [0.23.6] – 2025-11-14
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

## [0.23.5] – 2025-11-14
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

## [0.23.4] – 2025-11-14
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

## [0.23.3] – 2025-11-14
### Fixed
- `_LAG_STATE_INFO` value was shared globally between different python (non-jitted) runners. This 
  was causing corrupted lag info between different runners. Fixed lag state info to be per-runner 
  instance instead of global, preventing interference between models with different lag configurations.

### Tests
- Added new tests covering lag info corruption issue to `test_lag_system.py`.

---

## [0.23.2] – 2025-11-14
### Changed
- Removed support for the `prev_<name>` DSL shorthand. Now `lag_<name>()` is used as a shorthand for 
  one-step lag. `lag_<name>(k)` usage stays the same.

---

## [0.23.1] – 2025-11-14
### Added
- Added `ss_lag_reserved` field to `StructSpec` for lag buffer allocation in stepper state. If
  a stepper needs to use the ss bank, it should use starting from this index. iw0 bank already
  has `iw0_lag_reserved` from the previous version.

### Changed
- Updated stepper banks documentation with lag system partitioning rules for `ss` banks.
- Updated build process to include lag reservations in struct specification.

---

## [0.23.0] – 2025-11-14
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

## [0.22.0] – 2025-11-13
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

## [0.21.6] – 2025-11-13
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

## [0.21.5] – 2025-11-13
### Changed
- Updated cobweb plotting function so that it works with new v2 sim or model objects.
- Updated logistic_map.py example to use themes, grid layouts, and cobweb plots.

---

## [0.21.4] – 2025-11-13
### Added
- Added `state_vector()`, `param_vector()`, `state_dict()`, and `param_dict()` methods to `Sim` 
  class. They let getting state and parameter values as arrays or dictionaries from the current 
  session, model defaults, or saved snapshots.

### Changed
- Updated `izhikevich.py` to show how to access state/parameter values from snapshots.

---

## [0.21.3] – 2025-11-13
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

## [0.21.2] – 2025-11-13
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

## [0.21.1] – 2025-11-13
### Changed
- Removed `tomli` package fallbacks and updated Python requirement as >= 3.11 instead of 3.10.

---

## [0.21.0] – 2025-11-12
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

## [0.20.2] – 2025-11-12
### Changed
- Removed `guards.py` because it was poorly designed and implemented; was causing a lot of numba
  compatibility and caching issues.

### Known Issues
- All tests pass now but there is no NaN/Inf checks anywhere at this point.

---

## [0.20.1] – 2025-11-12
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

## [0.20.0] – 2025-11-12
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

## [0.19.4] – 2025-11-12
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

## [0.19.3] – 2025-11-11
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

## [0.19.2] – 2025-11-11
### Changed
- Gathered all ode-solver steppers under `src/dynlib/steppers/ode` folder.

---

## [0.19.1] – 2025-11-11
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

## [0.19.0] – 2025-11-11
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

## [0.18.0] – 2025-11-11
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

## [0.17.1] – 2025-11-11
### Added
- Added `setup()` helper to `src/dynlib/__init__.py`. It combines `build()` + `Sim()` calls. It is 
  more convenient for end users.

---

## [0.17.0] – 2025-11-11
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

## [0.16.2] – 2025-11-10
### Added
- Persisted runtime `stepper_config` data in `SessionState`, snapshots, and snapshot metadata so
  resumes continue with the exact tolerances last used (plus field-name lists for inspection).
- Added `Sim.stepper_config(**kwargs)` helper and extended `session_state_summary()` diagnostics with
  stepper-config previews/digests.

---

## [0.16.1] – 2025-11-10
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

## [0.16.0] – 2025-11-10
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

## [0.15.3] – 2025-11-09
### Added
- `Sim.run()` accepts a `transient` warm-up duration that advances the model before recording while
  keeping events functional and resetting the public time axis to `t0`.
- `Results` now exposes the final committed state via `final_state_view` for scenarios that need to
  reuse the converged state (e.g., transient warm-up, chained simulations).
### Changed
- `run_with_wrapper` captures the final committed state and stores it on the returned `Results`.

---

## [0.15.2] – 2025-11-09
### Changed
- Renamed `run()` args: 
  - `y0` -> `ic` 
  - `record_every_step` -> `record_interval`
- Renamed `build()` args:
  - `stepper_name` -> `stepper`
  - `model_dtype` -> `dtype`

---

## [0.15.1] – 2025-11-09
### Changed
- Forgot to add `**stepper_kwargs` in the previous version. Refactored `Sim.run()` in `sim.py` to 
  accept `**stepper_kwargs` for runtime overrides instead of explicit stepper parameters.
- Updated `_build_stepper_config()` in `Sim` to construct stepper configuration arrays from
  `**stepper_kwargs`.

---

## [0.15.0] – 2025-11-09
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

## [0.14.2] – 2025-11-08
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

## [0.14.1] – 2025-11-08
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

## [0.14.0] – 2025-11-07
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

## [0.13.2] – 2025-11-07
### Fixed
- The runner was dropping records when buffer growth was triggered. Enhanced `runner` function
  in `src/dynlib/compiler/codegen/runner.py`:
  - Added logic to handle pending steps before growth.
  - Improved recording mechanism for steps during re-entries.

---

## [0.13.1] – 2025-11-07
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

## [0.13.0] – 2025-11-07
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

## [0.12.5] – 2025-11-07
### Changed
- Removed unused `src/dynlib/utils/arrays.py` and `utils` folder because user inputs are always
  copied with `np.array()` and this file is not useful right now.

---

## [0.12.4] – 2025-11-07
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

## [0.12.3] – 2025-11-07
### Added
- Introduced `StepperKindMismatchError` to handle mismatched stepper and model kinds.

### Changed
- Updated `build` function to validate stepper kind against model kind and raise 
  `StepperKindMismatchError` if incompatible.

---

## [0.12.2] – 2025-11-07
### Changed
- Removed `priority` field from `ModSpec` in `src/dynlib/compiler/mods.py`.
- Updated exclusivity handling in `apply_mods_v2` to enforce stricter group rules.
- Improved error messages for exclusivity conflicts in `src/dynlib/compiler/mods.py`.

### Tests
- Removed priority fields from `tests/unit/test_mods.py` .
- Updated `test_mods_group_exclusive_conflict_raises` to validate stricter exclusivity rules.

---

## [0.12.1] – 2025-11-07
### Added
- TODO.md, ISSUES.md files.

### Changed
- Mods won't change [Sim] defaults in models. This is documented.

---

## [0.12.0] – 2025-11-06
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

## [0.11.3] – 2025-11-06
### Changed
- Exclusive groups now raise a ModelLoadError when more than one exclusive mod is supplied,
  so conflicts no longer slip through unnoticed.
 - `src/dynlib/compiler/mods.py`:  updated exclusivity docs and enforce a conflict check 
   that raises with the conflicting mod names instead of silently selecting a winner.

### Tests
- Replaced the previous “pick a winner” assertion with a conflict-raises check in 
  `tests/unit/test_mods.py`.

---

## [0.11.2] – 2025-11-06
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

## [0.11.1] – 2025-11-06
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

## [0.11.0] – 2025-11-06
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

## [0.10.2] – 2025-11-06
### Changed
- Dropped the legacy `EVT_TIME` buffer entirely; logged times live in `EVT_LOG_DATA`.

---

## [0.10.1] – 2025-11-06
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

## [0.10.0] – 2025-11-06
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

## [0.9.0] – 2025-11-06
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

## [0.8.0] – 2025-11-06
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

## [0.7.1] – 2025-11-06
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

## [0.7.0] – 2025-11-05
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

## [0.6.1] – 2025-11-05
### Changed
- Preceding newlines are removed from the inline model declarations. This way `inline:`
  statement can be placed above `[model]` statements.

### Known Issues
- A FIX_PLAN.md file is created to implement planned but missing features. These features
  will be added.

---

## [0.6.0] – 2025-11-05
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

## [0.5.0] – 2025-11-05
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

## [0.4.1] – 2025-11-05
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

## [0.4.0] – 2025-11-04
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

## [0.3.1] – 2025-11-04
### Fixed
- Removed redundant validation functions `validate_dtype_rules` and `validate_equation_targets` 
  from `src/dynlib/dsl/astcheck.py`.
- Fixed unused import `from email.mime import message` in `src/dynlib/errors.py`.

### Tests
- Updated tests in `tests/unit/test_ast_check.py` to reflect changes in validation logic.

---

## [0.3.0] – 2025-11-04
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

## [0.2.0] – 2025-11-04
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

## [0.1.0] – 2025-11-04
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

## [0.0.0] – 2025-11-03
### Changed
- Initial commit.
