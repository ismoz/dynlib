# Runner Types

Understanding how the runner layer is structured makes it easier to reason about performance trade‑offs, observer support, and how the fast-path API diverges from the ordinary wrapper loop.

## Ordinary runners

`run_with_wrapper` (see `src/dynlib/runtime/wrapper.py`) orchestrates the default execution path. It:

- converts `Sim` knobs (t0/tend, adaptive flags, discrete horizons, stop phases, selective recording choices, observers, etc.) into buffers and scalar inputs that match the frozen runner ABI.
- allocates recording/event arrays and workspaces once, then calls a `RunnerVariant.BASE` or `RunnerVariant.ANALYSIS` runner through `runner_variants.get_runner`.
- handles runner signals such as `GROW_REC`, `GROW_EVT`, `DONE`, `EARLY_EXIT`, `USER_BREAK`, or `NAN_DETECTED`, growing buffers or pausing/resuming as needed so the compiled kernel never has to reallocate memory.
- copies final state/dt, traces, and observer metadata into a `Results` object before returning. The wrapper therefore keeps the hot loop lean and lets the compiled runner focus on numerical stepping.

Because this path tracks events, variable recording lengths, stop phases, and growth codes, we call it the **ordinary runner**. It is the most flexible runner and the one used by `Sim.run` unless `fastpath` execution is explicitly requested. Both the continuous (ODE) and discrete (map) steppers share the ordinary runner templates (`RunnerVariant.BASE` or `RunnerVariant.ANALYSIS` with `discrete` flag), so the wrapper can handle time-based horizons and iteration-budget horizons with the same ABI.

## Fast-path runners

`runtime/fastpath/executor.py` drives the specialized, fixed-step path. The executor allocates everything upfront (workspaces, selective recording buffers, stop flags, variational hooks) and then calls `runner_variants.get_runner` with `RunnerVariant.FASTPATH` or `RunnerVariant.FASTPATH_ANALYSIS`.

Fast-path runners are stripped down:

- no event-log growth or sticky buffer resizing—everything is sized based on the chosen `RecordingPlan` (via `RecordingPlan.capacity`) before the first invocation.
- no `GROW_*` statuses; the runner assumes the buffers it received are large enough, which keeps the loop tight.
- no event/interruption loop inside the runner template itself; the executor is responsible for any preparatory transient warm-up, final trimming, and metadata construction.

This makes fast-path runners ideal for repeated batch runs (`run_batch_fastpath`), throughput benchmarks, or any scenario where you can guarantee fixed-step sizes and memory caps. The executor still marshals observer traces, variational hooks, and runtime workspaces, but it does so outside the numerically hot loop.

## Analysis runners

Whenever observers are attached, both the ordinary and fast-path runners switch to an analysis variant:

- `RunnerVariant.ANALYSIS` (wrapper path) and `RunnerVariant.FASTPATH_ANALYSIS` (fast-path path) inject the observer hooks as globals (`ANALYSIS_PRE` and `ANALYSIS_POST`) so the wrapper or executor does not need to pass function handles through the runner ABI.
- `runner_variants.compile_analysis_hooks` resolves `ObserverModule.resolve_hooks`, pre-compiles both hooks (or uses a no-op fallback), and stores them in the `runner` cache key, which includes `analysis_signature_hash`.
- The analysis runners also wire up `analysis_ws`, `analysis_out`, and trace buffers, plus they honor `analysis.trace.record_interval()` and metadata such as `analysis_kind`, so users can capture side-channel data alongside the main simulation steps.
- Variational observers can supply a `runner_variational_step` callback that the runner calls instead of the default stepper, allowing hooks to adjust proposals without breaking JIT compatibility.

The analysis runners inherit all of the base runner features (recording, stopping, growth) or fast-path simplifications, depending on the variant.

-## Architecture Reference

`runner_variants.py` is the consolidated source of truth for every runner template. `Sim.run()` uses the base runner when no observers are attached, while `runtime/fastpath/executor.py` asks for analysis-aware fast-path runners when observers are present. Each executor/ordinary runner variant is separately cached so that fast-path batches and wrapper calls get their own compiled kernels.


### `runner_variants.py`

- Defines templates and compilation logic for all runner variants (`BASE`, `ANALYSIS`, `FASTPATH`, `FASTPATH_ANALYSIS`), covering both continuous and discrete models.
- Exposes `get_runner(variant, ...)`, which builds a cache key, injects observer hooks as globals, optionally JIT-compiles with Numba, and stores both LRU (in-memory) and on-disk runner caches.
- Responsible for generating Python source, bootstrapping analysis hooks, and ensuring the same runner templates back both `wrapper.py` and `executor.py`.


### `executor.py` (fast-path)

- Orchestrates fixed-step fast-path execution, allocating buffers, workspaces, optional transient warm-up, and result marshalling.
- Chooses runners with `get_runner(RunnerVariant.FASTPATH, ...)` or `RunnerVariant.FASTPATH_ANALYSIS` depending on observer presence.
- Supports single and batch execution (with optional parallelization) while leaving trajectory logic inside the shared runner templates.
- Finalizes observer traces/metadata so the caller receives the same analysis payloads as the ordinary wrapper path.


## Runner generation via runner_variants.py

`src/dynlib/compiler/codegen/runner_variants.py` is the single source of truth for every runner template.
- `RunnerVariant` enumerates the four supported flavors: `BASE`, `ANALYSIS`, `FASTPATH`, `FASTPATH_ANALYSIS`.
- `_RUNNER_TEMPLATE_MAP` pairs each variant and the continuous/discrete flag with the right template string and function name.
- `get_runner` builds a cache key composed of `(model_hash, stepper_name, analysis_sig, variant, runner_kind, dtype, cache_token, jit flag, template version)` and looks in either `_variant_cache_continuous` or `_variant_cache_discrete`.
- If the runner is missing, it synthesizes the source, optionally JIT-compiles it with Numba, injects the `ANALYSIS_PRE`/`ANALYSIS_POST` hook globals, and stores the callable in both the local LRU and the on-disk `runner_cache`.
- `analysis_signature_hash` reduces each observer set to a stable 16-character hash so runners remain cachable even when observers resolve to dynamically generated hooks.

Both the wrapper and the fast-path executor call `get_runner`, so adding a new runner variant (e.g., a variant that always writes to a special log or that fuses extra diagnostics) means adding a new template string, registering it in `_RUNNER_TEMPLATE_MAP`, and calling it from the appropriate execution path.

## Fast-path executor responsibilities

`executor.py` does more than call a runner:

- It implements `_RunContext`, `_WorkspaceBundle`, and `_call_runner`, which allocate the right buffers, manage cursor resets, and unify the call site between `run_single_fastpath`, `run_batch_fastpath`, and the optimized batch helpers.
- It selects the fast-path runner variant based on whether observers are provided and passes `analysis=None` or the actual observer module to `get_runner`, mirroring the wrapper’s logic but in a more constrained setting.
- When observers are present, the executor still collects trace buffers, metadata (`build_observer_metadata`), and optional variational step callbacks so the runner can emit analysis output.
- Batch helpers optionally use a thread pool to run the same runner across multiple IC/parameter sets, relying on the fact that the compiled runner is jit‑safe and GIL-free.

The separation means the fast-path executor and ordinary wrapper reuse the same templates and caching infrastructure but diverge in how much bookkeeping happens around the inner loop.
