# dynlib

Dynlib is a Python library for **modeling, simulating, and analyzing dynamical systems**.  
Models are described in a TOML-based DSL (Domain-Specific Language) and then executed through a unified runtime—so you can iterate on solvers, parameters, and analyses without rewriting the same NumPy/Matplotlib plumbing for every experiment.

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
- **Numba** is highly recommended for JIT execution:
  - `python -m pip install numba`

## Installation
- `pip install dynlib` (when published), or
- `pip install -e .` for editable installs from source

## Documentation
- The documentation relies on `mkdocs`. To regenerate or serve the documentation locally:

1. Install MkDocs and required plugins:
   ```bash
   pip install mkdocs mkdocs-material mkdocs-gen-files mkdocs-literate-nav "mkdocstrings[python]" mkdocstrings mkdocs-static-i18n
   ```

2. Install additional Markdown extensions:
   ```bash
   pip install pymdown-extensions
   ```

3. From the project root, serve the docs:
   ```bash
   mkdocs serve
   ```
   Or build them:
   ```bash
   mkdocs build
   ```

4. To manually update the auto-generated doc files run:
   ```bash
   python tools/gen_model_docs.py
   ```

The generated site will be in the `site/` directory.
