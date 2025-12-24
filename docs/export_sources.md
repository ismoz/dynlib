# Exporting and Inspecting Compiled Models

## Overview

When you compile a model with `dynlib.build()`, the system generates Python source code for the RHS function, event handlers, and stepper. Previously, this source code was only accessible through the disk cache (if enabled), making inspection cumbersome.

Now you can easily **export and inspect** the generated source code to verify compilation correctness.

## Quick Start

**Recommended approach:** Use `export_sources()` on the model object:

```python
from dynlib import build

# Compile your model
model = build("my_model.toml", stepper="euler", jit=True)

# Export all generated source files (preferred method)
files = model.export_sources("./compiled_sources")

# Inspect what was exported
for component, filepath in files.items():
    print(f"{component}: {filepath}")
```

**Alternative: Using `setup()` to create a Sim:**

```python
from dynlib import setup

# Setup simulation (combines build + Sim creation)
sim = setup("my_model.toml", stepper="euler", jit=True)

# Export sources through the model attribute
files = sim.model.export_sources("./compiled_sources")
```

<details>
<summary>Alternative: Using the standalone function</summary>

You can also import and use the standalone function (exports additional `model_info.txt`):

```python
from dynlib import build
from dynlib.compiler.build import export_model_sources

model = build("my_model.toml", stepper="euler", jit=True)
files = export_model_sources(model, "./compiled_sources")
# Also includes files["info"] with model metadata
```

However, the `model.export_sources()` method is more convenient and easier to discover.
</details>

## What Gets Exported?

The `export_sources()` method exports all available compiled source code:

1. **`rhs.py`** - The right-hand side function (derivatives for ODE, next state for maps)
2. **`events_pre.py`** - Pre-step event handler
3. **`events_post.py`** - Post-step event handler
4. **`update_aux.py`** - Auxiliary variable updater
5. **`stepper.py`** - The numerical integration stepper function (if available)
6. **`jvp.py`** - Jacobian-vector product function (if available)
7. **`jacobian.py`** - Dense Jacobian filler function (if available)

Note: The standalone `export_model_sources()` function additionally exports:
- **`model_info.txt`** - Summary of the model specification

## Source Code Availability

Source code is **always available** regardless of the `disk_cache` setting:

- `disk_cache=True`: Sources are stored in cache AND in the model object
- `disk_cache=False`: Sources are only in the model object (not persisted)

You can check availability:

```python
model = build("model.toml", stepper="rk4")

print(f"RHS source available: {model.rhs_source is not None}")
print(f"Stepper source available: {model.stepper_source is not None}")
```

## Use Cases

### 1. Verify Compilation Correctness

Export the sources and manually inspect them to ensure the DSL was compiled correctly:

```python
from dynlib import build

model = build("complex_model.toml", stepper="rk45")
files = model.export_sources("./verify")

# Read and check the RHS
rhs_code = files["rhs"].read_text()
print(rhs_code)
```

### 2. Debug Numerical Issues

If your simulation produces unexpected results, inspect the generated code:

```python
from dynlib import build

model = build("problematic_model.toml", stepper="euler")
model.export_sources("./debug_output")

# Check the stepper implementation
# Check if auxiliary variables are computed correctly
# Verify event conditions
```

### 3. Learn How DSL Translates to Code

Understand how dynlib translates your model:

```python
# Your DSL model
model_dsl = """
[model]
type = "ode"

[states]
x = 1.0
y = 0.0

[params]
omega = 1.0

[equations.rhs]
x = "y"
y = "-omega * x"
"""

model = build(model_dsl, stepper="rk4")
files = model.export_sources("./learn")

# See exactly how "y = -omega * x" becomes Python code
```

### 4. Archive Compiled Models

Save the exact compiled version for reproducibility:

```python
from datetime import datetime
from pathlib import Path

model = build("production_model.toml", stepper="rk45", disk_cache=True)

# Archive with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
archive_dir = Path(f"./model_archives/{timestamp}")

files = model.export_sources(archive_dir)
print(f"Model archived to: {archive_dir}")
```

## Checking JIT Compilation Status

To verify that functions were actually JIT-compiled (not just pure Python):

```python
def is_jitted(fn):
    """Check if function is JIT compiled"""
    return hasattr(fn, 'signatures') and bool(fn.signatures)

model = build("model.toml", stepper="euler", jit=True)

print(f"RHS JIT compiled: {is_jitted(model.rhs)}")
print(f"Stepper JIT compiled: {is_jitted(model.stepper)}")
print(f"Runner JIT compiled: {is_jitted(model.runner)}")
```

## Example Output

Running `export_model_sources()` on a simple decay model produces:

```
exported_model/
├── rhs.py                # Generated RHS function
├── events_pre.py         # Pre-step events
├── events_post.py        # Post-step events
└── update_aux.py         # Auxiliary variable updater

Note: `stepper.py`, `jvp.py`, and `jacobian.py` may also be present if sources are available.
The `model_info.txt` file is only created by the standalone `export_model_sources()` function.
```

**`rhs.py` content:**
```python
import math

def rhs(t, y_vec, dy_out, params):
    dy_out[0] = -params[0] * y_vec[0]
```

**`model_info.txt` content:**
```
Model Information
============================================================
Spec Hash: 1b860ede6c41...
Kind: ode
Stepper: euler
Dtype: float64

States: x
Parameters: a

Equations (RHS):
  x = -a * x
```

## Disk Cache vs. Export

| Feature | Disk Cache | Export Sources |
|---------|-----------|----------------|
| Location | `~/.cache/dynlib/jit/...` | User-specified directory |
| Format | Python modules + metadata | Plain `.py` files |
| Organization | By digest hash | Named by component |
| Easy to find | No (hash-based paths) | Yes (explicit directory) |
| Persistence | Permanent (until cleared) | User-controlled |
| Purpose | Performance optimization | Inspection & debugging |

## Complete Example

See [`examples/inspect_compilation.py`](../examples/inspect_compilation.py) for a comprehensive example that:

- Builds and compiles a model
- Checks JIT compilation status
- Exports all sources
- Displays compilation details
- Shows source code previews

Run it:

```bash
python examples/inspect_compilation.py
```

## API Reference

### `Model.export_sources(output_dir)` ⭐ Recommended

Export compiled model sources to a directory. This is the **preferred method** for exporting sources.

**Parameters:**
- `output_dir` (str | Path): Directory to write files to

**Returns:**
- `Dict[str, Path]`: Mapping of component names to file paths

**Exported Components:**
- `"rhs"`: RHS function source (if available)
- `"events_pre"`: Pre-event handler source (if available)
- `"events_post"`: Post-event handler source (if available)
- `"update_aux"`: Auxiliary variable updater source (if available)
- `"stepper"`: Stepper function source (if available)
- `"jvp"`: Jacobian-vector product source (if available)
- `"jacobian"`: Dense Jacobian function source (if available)

Note: Only components with available source code are exported. The standalone `export_model_sources()` function also exports an `"info"` key with model metadata.

**Example:**
```python
from dynlib import build

model = build("model.toml", stepper="rk4")
files = model.export_sources("./output")

# Access exported files
rhs_file = files["rhs"]  # Path to rhs.py
if "stepper" in files:
    stepper_file = files["stepper"]  # Path to stepper.py
```

### `export_model_sources(model, output_dir)` (Alternative)

Standalone function version of the export functionality. **Consider using `model.export_sources()` instead** for better discoverability and convenience.

**Parameters:**
- `model` (FullModel): Compiled model from `build()`
- `output_dir` (str | Path): Directory to write files to

**Returns:**
- `Dict[str, Path]`: Mapping of component names to file paths

**Example:**
```python
from dynlib import build
from dynlib.compiler.build import export_model_sources

model = build("model.toml", stepper="rk4")
files = export_model_sources(model, "./output")
```

## Tips

1. **Always check source availability first:**
   ```python
   if model.rhs_source:
       model.export_sources("./output")
   ```

2. **Use descriptive output directories:**
   ```python
   model.export_sources(f"./debug_{model.spec_hash[:8]}")
   ```

3. **Combine with version control:**
   ```python
   # Export to git-tracked directory for CI/CD verification
   model.export_sources("./tests/expected_output")
   ```

4. **For permanent storage, use disk_cache=True:**
   ```python
   # Disk cache persists compiled code
   model = build("model.toml", disk_cache=True)
   
   # Export for human inspection
   model.export_sources("./inspect")
   ```
