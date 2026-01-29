# dynlib

Dynlib is a Python toolkit for modeling, simulating, and analyzing dynamical systems with a TOML-based DSL.
You describe states, parameters, equations, and auxiliary helpers in declarative specs, then run them through
a unified runtime that keeps steppers, recorders, and analysis utilities consistent across experiments.

## Highlights

- **Modeling DSL:** Use TOML specs for ODEs and discrete maps, express events/aux variables, and reuse macros or config
  registry entries without rewriting solver glue. The Modeling guides (Modeling guide, functions, events, lagging, and
  more) explain DSL design and helpers.
- **Simulation runtime:** Choose from multiple stepper families (Euler, RK4, RK45, Adams–Bashforth, implicit methods,
  map runners) with optional Numba/JIT acceleration, disk caching, snapshots/resume, selective recording, and session
  inspection. Simulation guides cover runner variants, steppers, and configuration.
- **Analysis & plotting:** Built-in utilities for bifurcations, basins, Lyapunov exponents, fixed points, manifold tracing,
  sweeps, and result exploration. Plot helpers deliver phase portraits, bifurcation diagrams, decorations, and export
  workflows that work with Matplotlib under the hood.
- **Optional CLI & tooling:** A CLI serves quick validation, listing steppers, and cache inspection tasks, but the Python
  API is the primary surface. Advanced docs show how the CLI mirrors `dynlib` entry points.

## Start here

1. Read the [Getting Started overview](getting-started/overview.md) to understand the project goals, recommended
   prerequisites, and how the docs are laid out.
2. Follow the [Quickstart](getting-started/quickstart.md) to install dynlib, validate the CLI (if you choose to use it),
   and run a built-in model from Python.
3. Work through [Your First Model](getting-started/first-model.md) to author DSL specs, validate them, and register them
   in a config file.

## Dive deeper

- **Guides → Modeling, Simulation, Plotting, Analysis:** Explore the sections under “Guides” for detailed walkthroughs on
  steppers, DSL features, runner configuration, plotting strategies, and analysis workflows.
- **Examples:** Review curated workflows (analysis catalog, plotting catalog, runtime catalog, and system-specific examples)
  once you know your modeling goals.
- **Reference:** Run `mkdocs build` to regenerate the built-in model docs produced by `tools/gen_model_docs.py`; the
  reference section lists every model in `src/dynlib/models` and exposes the TOML sources.

## Community & project info

- Track ongoing work in `CHANGELOG.md`, report issues via `ISSUES.md`, and consult `TODO.md` for planned improvements.
- The docs site itself is configured via `mkdocs.yml`, so any nav adjustments, plugin tweaks, or generated pages belong
  there.

