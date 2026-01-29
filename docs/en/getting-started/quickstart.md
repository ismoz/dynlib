# Quickstart

This page walks you through the first three things every dynlib user does: install the package, sanity-check the CLI, and run a bundled model from Python. After that we point you toward configuring your own model catalog so you can dive into DSL editing or analysis.

## Installation

Use your favorite virtual environment and then install the released wheel. The `dynlib` console script and the `python -m dynlib.cli` entry point both become available.

```bash
python -m pip install dynlib
```

For active development, install from source so you can tweak the code, run tests, and immediately see your changes in the CLI:

```bash
git clone https://github.com/your-username/dynlib.git
cd dynlib
python -m pip install -e .
```

Make sure you activate the virtual environment afterward so both `dynlib` and its dependencies stay isolated from other projects.

## Validate the CLI

The bundled CLI mirrors the Python API and is the fastest way to confirm that the environment works.

- `dynlib --version` or `python -m dynlib.cli --version` shows the installed package version and proves that the entry point scripts are registered.
- `dynlib model validate builtin://ode/lorenz.toml` parses a built-in ODE spec and prints the validated stepper, dtype, and state count. Replace the path with any `builtin://` model under `ode/` or `map/` to try something else.
- `dynlib steppers list --kind ode --jit_capable` lists all available ODE steppers that support JIT, with more filters available (`--stiff`, `--jacobian optional`, etc.) so you can scope the runtime behavior before wiring it into your DSL.
- `dynlib cache path` shows where compiled runners and JIT artifacts live; `dynlib cache list` or `dynlib cache clear --dry_run` helps you inspect or clean the cache if you switch steppers or dtypes.

Each command accepts `--help` for more flags (e.g., `dynlib steppers list --help`), which lets you explore runtime knobs without reading the source.

## Run a built-in model from Python

Use `dynlib.setup` to compile a model, choose a stepper, and get back the `Sim` facade in a single call. The example below loads the bundled Lorenz system, enables JIT, caches the compiled runner, runs for 15 time units, and then plots the `x` and `z` states using built-in plot utilities.

```python
from dynlib import setup
from dynlib.plot import fig, series, export

sim = setup(
    "builtin://ode/lorenz.toml",
    stepper="rk4",
)

sim.run(T=15.0, dt=0.01)
res = sim.results()

print("Recorded states:", res.state_names)
print("Recorded steps:", len(res))
print("Final z value:", res["z"][-1])

ax = fig.single()
series.plot(x=res.t, y=res["x"], ax=ax, label="x")
series.plot(x=res.t, y=res["z"], ax=ax, label="z", xlabel="time")
export.show()
```

`res` is a `ResultsView`, so `res.t`, `res["state_name"]`, `res.event_names()`, and helpers like `res.to_pandas()` (requires `pandas`) work without copying the underlying buffers. Call `sim.run(...)` again with `resume=True`, new `params`, or `record_vars` lists to continue or record subsets of the state vector, and use `sim.model.spec` or `res.state_names` to inspect the DSL metadata.

## Point dynlib at your own models

`builtin://` URIs make it trivial to explore the bundled ODEs (`lorenz`, `vanderpol`, `izhikevich`, etc.) and maps (`logistic`, `henon`, `standard`, …). Once you have your own TOML files, dynlib offers flexible URI resolution:

- Inline TOML strings or `inline:` URIs (see `examples/uri_demo.py`) are parsed immediately and can coexist with file-based models.
- Absolute or relative paths are accepted (`/path/model.toml` or `my_model.toml`), and `model` resolves to `model.toml` in the current directory.
- Tag-based URIs such as `proj://my_model.toml` look up the `proj` tag in `~/.config/dynlib/config.toml` (Linux), `~/Library/Application Support/dynlib/config.toml` (macOS), or `%APPDATA%\dynlib\config.toml` (Windows). You can override that default location with `DYNLIB_CONFIG=/custom/config.toml`.
- To add directories without editing the config file, set `DYN_MODEL_PATH=proj=/extra/models:/more` (Windows uses `;` between tags). The string before `=` becomes the tag name and the comma-separated paths become search roots.

Your config file can look like this:

```toml
[tags]
proj = ["/Users/you/dynlib-models", "~/projects/other-models"]
tests = ["~/src/dynlib/tests/data/models"]

[cache]
cache_root = "/tmp/dynlib-cache"
```

With the `proj` tag registered you can now `setup("proj://decay.toml")`, `dynlib model validate proj://decay.toml`, or refer to that URI inside other specs.

## Next steps

After the quickstart, continue with **Your First Model** to write DSL files, explore **Guides → Modeling/Simulation** for steppers and recording best practices, and read the **Examples** or **Analysis** guides when you need plotting, fixed-point analysis, or Lyapunov observers. If you hit issues, the CLI `dynlib cache list` helps you debug caching, and `dynlib model validate` shows DSL parsing errors with line/column hints.
