# Getting Started

Dynlib is a simulation library for defining, simulating, and analyzing dynamical systems. Models are described in a TOML-based DSL (Domain Specific Language), so the same model definition can drive simulation runs and analysis workflows. Currently, dynlib supports discrete-time maps and ordinary differential equations (ODEs), along with a growing set of analysis utilities.

Dynlib’s goal is to make dynamical-systems work practical and repeatable. Implementation details still matter—but once you already know how to wire up solvers, state arrays, parameter sweeps, plotting, and data management, redoing that plumbing for every new model becomes the bottleneck. Dynlib abstracts the repetitive mechanics so you can focus on the model, parameter regimes, and interpretation, while still retaining control over numerical methods and configuration.

Compared to “plain NumPy + Matplotlib” scripts, dynlib reduces the friction of iterating: you can swap solver settings without rewriting model code, run sweeps and experiments in a consistent way, and move from exploration to structured runs with fewer one-off scripts. This is especially useful in teaching, where students often spend disproportionate effort on array bookkeeping and plotting glue rather than the dynamical concepts. With dynlib, models can be defined and explored interactively in notebooks, keeping attention on phase space behavior, bifurcations, stability, and invariants.

On top of simulation, dynlib includes built-in analysis tools such as bifurcation diagrams, basins of attraction, Lyapunov exponent estimation, manifold tracing, and fixed point detection. It also supports JIT compilation for performance, multiple numerical steppers (Euler, RK4, adaptive methods like RK45), plotting helpers, and workflow features such as disk caching, snapshots for resuming runs, and CLI utilities for model validation and model-library management.

**Disclaimer :** Dynlib is currently in active development. APIs may change, and there may be bugs or numerical edge cases that affect results. If you are using dynlib for research or high-stakes decisions, validate outcomes against trusted references (e.g., alternative solvers, smaller step sizes, analytical checks) and please report issues. 

## Prerequisites

- A working Python environment (3.10+ is recommended). Install into a virtualenv or similar isolation layer so your system site-packages stay clean.
- The package itself (`pip install dynlib`) and the CLI are described in **Quickstart**, which also walks you through validating the CLI and running a built-in model.
- Numba is highly recommended for performance whenever you use JIT-capable runners. Install it with `python -m pip install numba`; it will be picked up automatically when dynlib compiles runners when `jit=True` option is used.

## How to use this section

1. Start with **Quickstart** to install dynlib, sanity-check the CLI (`dynlib --version`, `dynlib model validate`, etc.), and execute one of the bundled models from Python.
2. Move on to **Your First Model** to write a TOML spec, validate it (`dynlib model validate first-model.toml`), and experiment with inline strings before registering specs in your config file.
3. Explore the deeper Modeling and Simulation guides when you need to work with steppers, recorders, and DSL helpers, then review the Results guide for working with outputs and plots.
4. Use the Examples and Analysis sections to see complete workflows, from bifurcation diagrams to neuron models, and plug those ideas back into your own specs.

