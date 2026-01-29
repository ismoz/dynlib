# Exporting compiled sources

When you need to peek inside what dynlib is generating, `FullModel.export_sources()` (and the sibling helper, `dynlib.compiler.build.export_model_sources`) write every available callable into a directory so you can open it in your editor, run linting, or keep a record of a simulation build.

Exporting sources is useful whenever you:

- want to understand how the DSL equations become Python functions (rhs, steppers, events, etc.)
- need to audit what changed between steppers, events, or solver options
- are preparing artifacts for regression testing, demos, or sharing with teammates

## Step-by-step export workflow

```python
from dynlib import build

model = build("decay.toml", stepper="euler", jit=True)
files = model.export_sources("compiled_sources")
print(files["rhs"])
```

1. Call `build(...)` or `setup(...)` to get a `FullModel`. If you already have a `Sim`, use `sim.model` to reach the compiled artifact.
2. Pass a writable `output_dir`. The directory is created automatically (`mkdir -p` semantics).
3. The return value is a dictionary mapping component names to the `Path` of the written file.
4. Every time you rebuild with different options you can export again into a new directory to compare source diffs.

If you prefer a free-standing helper, import `export_model_sources` from `dynlib.compiler.build` and pass the `FullModel` instance.

## What gets written

The export writes a `.py` file for each compiled component that carries source text on the model object:

- `rhs.py`, `events_pre.py`, `events_post.py`, `update_aux.py`
- `stepper.py` and (when available) `jvp.py`, `jacobian.py`, `inv_rhs.py`

In addition, `model_info.txt` summarizes the spec: spec hash, kind, stepper name, dtype, listed states/parameters, RHS equations, and a short preview of any events (phase, guard, and the first ~50 characters of the action block). Having this metadata alongside the code makes it easier to correlate a compiled snapshot to the DSL input.

Because `FullModel` retains the sources regardless of `disk_cache` or the stepper you selected, the export works even when the compiler reused cached artifacts or when you swap between `euler`, `rk4`, or custom steppers. The files are written with UTF-8 encoding so you can open them in any standard editor.

## Tips

- If you need a record of the generated runner/stepper for debugging, export after compiling and before running long simulations.
- Treat each export directory as a snapshot: keep it around to track regressions or to document how a particular option (e.g., `jit=True` vs `jit=False`) affects the generated code.
- The helper returns `Path` objects so you can immediately read the contents (e.g., `files["stepper"].read_text()`).
