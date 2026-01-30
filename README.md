# dynlib

Dynlib is a Python library for **modeling, simulating, and analyzing dynamical systems**.  
Models are described in a TOML-based DSL (Domain-Specific Language) and then executed through a unified runtime—so you can iterate on solvers, parameters, and analyses without rewriting the same Numpy/Matplotlib plumbing for every experiment.

With dynlib, you can define or tweak a model, try different solvers/settings, and visualize behavior quickly. It can be used with notebooks for teaching and demonstration purposes. Created models can be kept in an organized manner and can be shared easily.

## Project status

Dynlib is **alpha-stage** software. APIs may change, and numerical edge cases or bugs can surface. Treat results as exploratory unless you validate them (e.g., alternative steppers, tighter tolerances, smaller step sizes, or analytical checks). If you find suspicious behavior, please open an issue with a minimal reproducer.

## Highlights

### Modeling (TOML DSL)
- Define **ODEs** and **discrete-time maps** using a declarative TOML spec.
- Express equations, parameters, state initialization, and metadata in a consistent format.
- Support for **events**, **auxiliary variables**, **functions/macros**, and **lagging** where applicable.
- Built-in **model registry** and URI loading (including `builtin://...` models).

### Simulation runtime
- Multiple stepper families:
  - ODE: Euler, RK4, RK45, Adams–Bashforth (AB2/AB3), and implicit methods (e.g., SDIRK/TR-BDF2).
  - Maps: dedicated discrete runner(s) including integer-safe modes.
- Runner variants and session introspection utilities for iterative workflows.
- **JIT acceleration** via Numba (optional but highly recommended), plus **disk caching** for compiled runners.
- **Snapshots and resume** support for long or staged simulations.
- Selective recording and result APIs designed for downstream analysis.

### Analysis
Built-in analysis utilities for common dynamical-systems tasks:
- **Bifurcation** and post-processing utilities
- **Basins of attraction** (auto/known variants)
- **Lyapunov exponent** analysis (including runtime observer support)
- **Fixed point / Equilibria** detection
- **Manifold** tracing tools (currently limited to 1D manifolds)
- **Homoclinic/Heteroclinic** orbit tracing and detection
- **Parameter sweep** helpers and trajectory/post-analysis utilities

### Vector fields & plotting (on top of Matplotlib)
Dynlib includes plotting helpers tailored for dynamical systems rather than raw Matplotlib boilerplate:
- Vector field evaluation utilities and **phase-portrait** helpers
- Plot modules for **basins**, **bifurcation diagrams**, **manifolds**, and general dynamics
- Higher-level plotting conveniences: **themes**, **facets**, decorations, and export helpers
- Vector field **animation** support

### CLI
Dynlib ships a small CLI (Command Line Interface) for convenience tasks such as model validation, listing steppers, and inspecting caches.  
The CLI is not required for the Python API.

## Prerequisites
- Python 3.10+
- Matplotlib for plots.
- Numpy for numerical calculations.
- **Numba** is highly recommended for JIT execution:
  - `python -m pip install numba`

## Installation
- `python -m pip install dynlib` or
- `python -m pip install -e .` for editable installs from source

## Quickstart

Sanity-check the CLI and validate a bundled model:

```bash
dynlib --version
dynlib model validate builtin://ode/lorenz.toml
```

Run a built-in model from Python (Lorenz system):

```python
from dynlib import setup
from dynlib.plot import fig, series, export

sim = setup("builtin://ode/lorenz.toml", stepper="rk4")

sim.run(T=15.0, dt=0.01)
res = sim.results()

print("States:", res.state_names)
print("Final z:", res["z"][-1])

ax = fig.single()
series.plot(x=res.t, y=res["x"], ax=ax, label="x")
series.plot(x=res.t, y=res["z"], ax=ax, label="z", xlabel="time")
export.show()
```

Next: see the docs for defining your own TOML models, URI tags (`proj://...`), recording options, and analysis workflows (basins, bifurcation, Lyapunov, fixed points).

## Documentation
Check this link for the project documentation:[Documentation](https://ismoz.github.io/dynlib/)
