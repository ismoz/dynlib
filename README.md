# dynlib

Dynamical systems simulation library.

## Quick start

```bash
pip install -e .

# Validate a builtin model or inspect the cache
dynlib model validate builtin://ode/expdecay.toml
dynlib cache list
```

## Command-line interface

Installing dynlib exposes a `dynlib` console script (and `python -m dynlib.cli`) that mirrors the package's
core tooling:

- `dynlib model validate <uri>` parses and validates a model file or URI (including `builtin://` references).
- `dynlib steppers list [--kind ode --jit_capable ...]` lists registered steppers with optional capability filters.
- `dynlib cache {path,list,clear}` inspects or deletes the on-disk JIT cache.

Use `--help` on any subcommand to see available options.
