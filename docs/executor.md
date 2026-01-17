### Architecture Reference

Consolidated runner templates into single source of truth in `runner_variants.py`, reducing code duplication. Now `runner_variants.py` is the only source for all of the runners. `build()` returns a base / generic runner. `Sim.run()` selects the base runner if observers are not provided. If observers are attached then `runtime/fastpath/executor.py` returns a specialized runner according to those observers. Each executor runner is cached separately.

**`runner_variants.py`** (compiler/codegen/)
- Templates and compilation for all runner variants (BASE, ANALYSIS, FASTPATH, FASTPATH_ANALYSIS)
- Unified `get_runner(variant, ...)` API to select and compile runners based on capabilities needed
- Manages LRU cache of compiled runners and disk cache tokens
- Responsible for: generating Python source, injecting observer hooks as globals, JIT compilation
- Used by: `wrapper.py` (normal execution) and `executor.py` (fastpath execution)

**`executor.py`** (runtime/fastpath/)
- High-level orchestration for fixed-step fastpath execution
- Handles buffer allocation, workspace setup, transient warm-up, result building
- Calls runners via `get_runner(RunnerVariant.FASTPATH, ...)` or `get_runner(RunnerVariant.FASTPATH_ANALYSIS, ...)`
- Supports single and batch execution with optional parallelization
- Responsible for: orchestration, result marshaling, observer finalization, user-facing API (`fastpath_for_sim`)
- Does NOT implement trajectory logic (that's in runner templates)
