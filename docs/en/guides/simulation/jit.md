# JIT Compilation in Simulations

dynlib currently relies on **Numba** as the sole just-in-time backend. Because `build()` and `setup()` now default to `jit=False` and `disk_cache=False`, turning on JIT is a deliberate choice: pass `jit=True` when you need the compiled kernels. If Numba is not installed, that flag raises `JITUnavailableError`, so install numba first. The runtime design assumes the compiled kernel is GIL-free, therefore no other JIT engines (PyPy, Cython, LLVM wrappers, etc.) are supported.

## Turning on JIT

- A plain `build(model)` or `setup(model)` runs the simulation entirely in Python.
- Pass `jit=True` to compel dynlib to compile the RHS, events, auxiliary updater, stepper, runner, and guards with Numba.
- While `jit=False` is safe for quick experiments or short batches, `jit=True` pays off for long-running simulations where the upfront compilation cost is amortized.
- Pair `jit=True` with `disk_cache=True` to persist compiled artifacts across processes; leaving `disk_cache=False` keeps everything in-memory and avoids writing to the cache root.

```python
from dynlib import setup

sim = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=True)
sim.run(T=20.0)
```

You can also call `build()` directly when you need to inspect the `FullModel`. Its compiled attributes (`model.rhs`, `model.stepper`, `model.runner`, etc.) expose the usual Numba `signatures` once JIT-ed.

## What gets compiled

The JIT path covers the pieces that run inside the numerically hot loop:

- **Triplet functions**: `rhs`, `events_pre`, `events_post`, and `update_aux` are compiled together so they stay performant when invoked from the stepper/runner.
- **Stepper**: `stepper.emit()` produces the integration kernel, which is also JIT-compiled when the stepper is marked as JIT-capable.
- **Runner**: The runner template (ordinary, fast-path, or analysis variant) is compiled with the same `jit` flag so the entire wrapper → runner stack runs without crossing back to Python.
- **Guards**: When JIT is enabled, guard helpers that validate states/params are compiled once and reused to keep the `nopython` contract intact.

## Caching JIT artifacts

Caching eliminates redundant compilation when you rerun the same model or rebuild it in another process.


- `disk_cache=True` : Persists compiled triplets, steppers, and runners under 
    - `~/.cache/dynlib/jit/...` (Linux), 
    - `~/Library/Caches/dynlib/jit/...` (macOS), 
    - `%LOCALAPPDATA%/dynlib/Cache/jit/...` (Windows). 

- `disk_cache=False` : Keeps everything in memory; useful when you cannot write to the cache root or when you specifically want a clean compilation per run. 

- Source visibility : Generated source code stays available on the `model` object (`model.rhs_source`, `model.stepper_source`, …) regardless of `disk_cache`.

### How the cache key is constructed

- `runner_variants.get_runner` and the triplet/stepper cache builders derive cache keys from deterministic inputs: the model hash, stepper name, dtype, guard struct signature, runner variant, analysis signature, `cache_token`, JIT flag, and template version.
- The triplet/stepper caches store compiled modules under `cache_root/jit/triplets|steppers/.../<digest>`, so two runs with the same hash reuse the compiled artifact immediately.
- Runners inject `analysis` hooks and thus only cache non-analysis variants (`njit(cache=True)` flags are omitted for analysis-aware runners because the hooks are resolved at runtime). Variants are still cached in-process and on disk via `runner_cache`, and `cache_token` (based on the configured cache context) makes sure caches are invalidated when workspace layout or dtype changes.

### Configuring the cache root

You can override the cache root via configuration files or environment variables:

1. `DYNLIB_CONFIG` points to a TOML file that may contain `cache_root = "/custom/root"` or a `[cache]` table with `root = "/custom/root"`.
2. `load_config()` also honors `DYN_MODEL_PATH`, so you can combine `cache_root` overrides with custom TAG roots.
3. If the configured root is unwritable, dynlib falls back to `/tmp/dynlib-cache` (and warns once). If that too fails, the runtime warns and keeps the JIT cache entirely in memory (no files written).

### Cache resilience

- The on-disk cache rebuilds itself when it detects corruption: it deletes the bad module, re-renders the generated source, and retries compilation.
- `CacheLock` guards prevent races when multiple processes try to populate the same digest simultaneously.
- The caches also support a manual invalidation path by touching the `cache_token` (the runner builder picks up the current struct signature and dtype), so ABI-affecting changes automatically create new entries.
- Changes in runner or stepper source code are not tracked or hashed. Any change in the source code will corrupt the cache or you will get old behavior due to cached artifacts. In such cases delete the cache. CLI can be used to delete the cache (for example: `dynlib cache clear --all`).

## Checking compilation status

If you want to know whether a callable was actually JIT-compiled:

```python
from dynlib import build

def is_jitted(fn):
    return hasattr(fn, "signatures") and bool(fn.signatures)

model = build("model.toml", stepper="euler", jit=True)
print("RHS jitted", is_jitted(model.rhs))
print("Runner jitted", is_jitted(model.runner))
```

`model.export_sources(...)` still works even when `disk_cache=True`; the exported directory contains every compiled component (`rhs.py`, `stepper.py`, `runner.py`, etc.), which makes it easy to inspect what dynlib is compiling behind the scenes.

## Best practices

- Reserve `jit=True` for simulations that execute long enough to amortize the upfront compilation cost.
- Keep `disk_cache=True` for development cycles where you rebuild the same models repeatedly; turn it off when you need a clean slate (e.g., CI or tests that ensure a fresh compilation).
- If a run fails with a Numba error, inspect the traceback, fix any unsupported Python constructs, and rerun – dynlib does not silently fall back to Python when `jit=True`.
- When you add new observer hooks or change the dtype/stepper, the cache key changes automatically, so you do not need to manually delete cache directories.

With these knobs you can achieve near-native performance while still keeping traceability and reproducibility via persistent caches.
