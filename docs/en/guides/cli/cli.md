# Command-line guide

Dynlib exposes a lightweight command-line interface as two entry points: the `dynlib` console script that is installed with the package and `python -m dynlib.cli`. The CLI mirrors a handful of runtime utilities so you can validate models, introspect the stepper registry, and manage the on-disk JIT cache without writing a Python script. Each verb/noun parser layer supports `--help`, so run `dynlib <command> --help` or `python -m dynlib.cli <command> --help` whenever you need a reminder about the available options.

## Global flags

- `--version`
  Prints the currently installed dynlib version. The CLI discovers the version first via `importlib.metadata` and, when that fails (editable installs, source checkouts, etc.), falls back to reading `pyproject.toml`.

## Model tooling

`dynlib model validate <uri>`

- **Purpose:** Parse and validate a model defined in dynlib's TOML-based DSL. The CLI delegates to `load_model_from_uri`, so it supports the same `builtin://` URIs used in the rest of the project as well as filesystem paths and other registered loaders.
- **Success message:** When the DSL is valid, the command prints `Model OK` along with the model kind (`ode` or `map`), dtype, state count, and the default stepper recorded in `spec.sim.stepper`.
- **Error handling:** Syntax violations, missing fields, or runtime validation problems are surfaced via `DynlibError` with a descriptive message on `stderr` and a non-zero exit code.

Use this command as a quick sanity check before running a simulation, sharing a model, or bundling the spec into another toolchain.

## Steppers registry

`dynlib steppers list [--kind <kind>] [--<cap>] [--jacobian <policy>]`

- **Purpose:** Inspect every registered `StepperMeta`/`StepperCaps` pair. Because the CLI imports `dynlib.steppers` as a side effect, all built-in and registered third-party steppers appear in the listing.
- **Displayed columns:** Each line shows the stepper name, `kind`, `scheme`, `order`, `stiff` hint, and every `StepperCaps` field so you can quickly compare features without reading source code.
- **Kind filter:** Use `--kind ode` or `--kind map` to restrict the list to ODE solvers or discrete maps (the same `Kind` enum used by the runtime).
- **Capability filters:** The CLI dynamically exposes one flag per `StepperCaps` field.
  - Boolean flags (`--dense_output`, `--jit_capable`, `--requires_scipy`, `--variational_stepping`) are _requirements_. When provided, only steppers whose `StepperCaps` set that field to `True` remain in the output.
  - Value flags currently include `--jacobian` (matching the `JacobianPolicy` literal: `none`, `internal`, `optional`, `required`). Provide the exact policy string to filter steppers that declare that Jacobian behavior.
- **Use cases:** This command is handy for confirming which steppers support dense output (e.g., being the basis for animation or variable-step interpolation), identifying the subset that can be JIT-compiled, or quickly checking that a third-party stepper registered the capabilities you expect.

## Cache management

All cache commands delegate to `resolve_cache_root()` so they respect your `[cache]` overrides, `DYN_MODEL_PATH` tag map extensions, or `DYNLIB_CONFIG` environment variable described in [the config file](../modeling/config-file.md).

### `dynlib cache path`

- Prints the cache root directory. Useful when you want to inspect the files on disk, mount the directory into a container, or troubleshoot permission issues.

### `dynlib cache list [--stepper <name>] [--dtype <token>] [--hash <prefix>]`

- **Purpose:** Enumerate every entry under `cache_root/jit/{triplets,steppers,runners}`.
- **Output format:** Each entry prints the family (`triplets`, `steppers`, or `runners`), stepper name, dtype, spec hash, digest, size (human-readable), and filesystem path. Entries that recorded compile-time components also add `components=...`.
- **Filters:**
  - `--stepper` matches the stepper name (case-insensitive).
  - `--dtype` matches the dtype token (also case-insensitive).
  - `--hash` matches a prefix of the model spec hash to pull the artifacts associated with a particular spec.
- **Ordering:** Results are sorted by family, stepper, dtype, and digest so related artifacts appear together.
- **Use cases:** Run this after you switch dtypes/steppers to confirm whether cached kernels exist, or to verify which runners the `disk_cache` flag left behind. Filtering by `--hash` is the fastest way to find the compiled artifact for a model hash, and `--dtype` is helpful when you use mixed precision.

### `dynlib cache clear (--all | --stepper <name> | --dtype <token> | --hash <prefix>) [--dry_run]`

- **Purpose:** Delete cached JIT artifacts you no longer need or to recover from cache corruption after code changes.
- **Safety guard:** You must specify `--all` or at least one of the filter flags. Without a filter the CLI exits with a message and error code `2`.
- **`--all`:** Removes the entire cache root via `shutil.rmtree`. Use this when you want a clean slate (for example, after upgrading dynlib or changing `cache_root`). The command does nothing if the directory is missing.
- **Selective deletion:** Combine `--stepper`, `--dtype`, and/or `--hash` to delete only the matching cache entries. Matching is case-insensitive, and `--hash` works on prefix matches so you can target a commit or spec version even if you only remember part of the hash.
- **`--dry_run`:** Prints the files and directories that _would_ be removed without touching disk. Run this before a destructive operation to double-check the target list.
- **Feedback:** Each deleted cache prints a confirmation line and the command returns a non-zero exit code if any deletion fails.

## Examples

```bash
dynlib model validate docs/models/lorenz.toml
dynlib steppers list --kind ode --jit_capable --variational_stepping
dynlib cache list --hash 9c8a --dtype float64
dynlib cache clear --stepper rk4 --dry_run
```

## Troubleshooting

- When stepping into cache issues, pair `dynlib cache list` with `dynlib cache path` to know which directory you are inspecting.
- If the CLI cannot find a stepper you expect, make sure the module defining it has been imported (the runtime automatically imports `dynlib.steppers` on startup, but third-party steppers must register themselves before running the CLI).
- The `model validate` command raises `DynlibError` for any DSL parsing issue; run it in a shell pipeline (`dynlib model validate ... || true`) when you want to process error output via another tool.

