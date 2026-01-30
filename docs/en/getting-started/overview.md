# Overview

If you are new to dynlib, make sure to check the [main page](../index.md) for the fundamentals.

## Requirements

- A working Python environment (3.10+ is recommended).
- A virtual environment (virtualenv) or a similar isolation layer for installation, although it is not mandatory.
- The NumPy package for numerical computations (`python -m pip install numpy`).
- The Matplotlib package for plotting (`python -m pip install matplotlib`).
- The package itself (`python -m pip install dynlib`). For installation details, see the [Quickstart](quickstart.md) guide.

!!! important "Numba is strongly recommended for high-performance simulation and analysis (`python -m pip install numba`)."

## How to use this section

1. Follow the **[Quickstart](quickstart.md)** guide to install dynlib, verify the CLI (`dynlib --version`, `dynlib model validate`, etc.), and run one of the built-in models from Python.
2. Move on to **[Your First Model](first-model.md)** to write a TOML model, validate it (`dynlib model validate first-model.toml`), and experiment with inline text definitions.
3. When you need to work with steppers, recorders, and other DSL features, explore the [Modeling guide](../guides/modeling/index.md) and the [Simulation guide](../guides/simulation/index.md).
4. To use simulation outputs and create plots, see the [Simulation Results](../guides/simulation/results.md) and [Plotting](../guides/plotting/index.md) guides.
5. For end-to-end workflows—from bifurcation diagrams to neuron models—check the [Examples](../examples/index.md) and the [Analysis guide](../guides/analysis/index.md) sections.
