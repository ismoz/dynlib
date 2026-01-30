# Dynlib

Dynlib is a simulation library developed for defining, simulating, and analyzing dynamical systems. Models are defined using a TOML-based DSL (Domain-Specific Language), so the same model definition can be used for both simulation and analysis. Dynlib currently supports discrete-time maps and ordinary differential equations (ODEs), and it comes with a growing set of analysis tools.

Dynlib’s goal is to make working with dynamical systems practical and repeatable. Implementation details do matter; however, once you learn how to wire together solvers, state arrays, parameter sweeps, plotting, and data management, rebuilding that scaffolding for every new model becomes a bottleneck. Dynlib abstracts the repetitive mechanical work so you can focus on the model, parameter regimes, and interpretation—without losing control over numerical methods and configuration.

Compared to simulations built with only “NumPy + Matplotlib”, dynlib makes the simulation and analysis workflow much more convenient. With dynlib, you can switch numerical solution methods without rewriting model code, and you can run simulations and analyses with very short scripts. This is especially useful in teaching dynamical systems courses. While students struggle with many implementation details—writing loops, using NumPy and Matplotlib, implementing numerical integration methods—trying to teach dynamical systems often drifts away from the real goal and turns the course into a Python class. With dynlib, you can quickly build a model and focus on simulation and analysis without wrestling with implementation details. With simplified plotting utilities, you can easily produce figures.

In addition to simulation, dynlib includes built-in analysis tools such as bifurcation diagrams, basins of attraction, Lyapunov exponent estimation, manifold tracing, and fixed-point detection. It also provides JIT compilation (Just-In-Time Compilation) and disk caching for performance. It supports multiple families of numerical solvers (Euler, RK4, RK45, TR-BDF2A, etc.). With its model library, you can create new models and access them easily. You can stop and restart simulations (resume) and take a snapshot of the simulation state at any time. Dynlib also provides a simple CLI (command-line interface) so you can perform dynlib-related tasks from the terminal.

!!! warning "Warning: Dynlib is currently under active development. APIs may change, and there may be bugs or numerical edge cases that affect results. If you use dynlib for research or critical decisions, validate results against reliable references (e.g., alternative solvers, smaller step sizes, analytical checks), and please report any issues you encounter."

## Terminology

To understand how dynlib works, you should be familiar with the following terms. Some of these are dynlib-specific.

- **Map:** Discrete-time dynamical systems (such as the logistic map).
- **ODE:** Ordinary Differential Equation system.
- **DSL:** Domain-Specific Language. A simple, readable language in TOML format for defining models.
- **JIT:** Abbreviation of just-in-time compilation. With the help of Numba, compiling Python code enables higher-performance simulation/analysis.
- **Stepper:** The program responsible for computing the next simulation step. ODE steppers implement numerical integration methods (such as Euler, RK4, RK45).
- **Runner:** If simulations are considered “runs”, a runner can be described as the program that executes a simulation run. Dynlib includes multiple specialized runner implementations. Since a runner + stepper combination can be JIT-compiled, you can think of it like a kernel.
- **Wrapper:** Since operations possible in JIT-compiled runners are limited, each runner is used under the control of a wrapper. If the runner is insufficient, the pure-Python wrapper takes over.
- **API:** Application Programming Interface. Defines how you should call a program/library.
- **CLI:** Command-Line Interface.
- **URI:** Represents the address of model TOML files (Uniform Resource Identifier).
- **RHS:** The right-hand side of an equation (Right-Hand Side).
- **Snapshot:** Saving the full state at a particular moment in a simulation, including state variables, parameter values, and simulation settings.

## Highlights

- **Define the model once (TOML DSL):** Write ODEs or discrete maps in a single TOML definition and use the same model across all simulations and analyses.  
  [Modeling](guides/modeling/index.md)

- **Run simulations easily:** Build simulations in a practical way without getting lost in details. Switch numerical methods easily by choosing a stepper. Use JIT acceleration if you want, and enable disk caching for fast builds. Save snapshots and resume simulations from where you left off.  
  [Simulation](guides/simulation/index.md) / [Runtime](examples/runtime.md)

- **Do the core analyses:** Use built-in tools for bifurcation diagrams, basin mapping, Lyapunov exponent computation, fixed-point finding, manifold tracing, and parameter sweeps. Also plot results easily with Matplotlib-based helpers.  
  [Analysis](guides/analysis/index.md) / [Plotting](guides/plotting/index.md)

- **CLI:** Use quick validation and inspection commands from the command line for common tasks.  
  [CLI](guides/cli/cli.md)

## Start Here

1. Read [Getting started overview](getting-started/overview.md) to understand the project goals, recommended prerequisites, and how the documentation is organized.
2. Follow [Quickstart](getting-started/quickstart.md) to install dynlib, validate it with the CLI, and run a built-in model from Python.
3. See [Your First Model](getting-started/first-model.md) to create and validate a model using the DSL.

## Go Deeper

You can get a more detailed view of how to use dynlib by browsing the other guides in the documentation:

- **[Home](index.md)**

### Getting Started
- [Overview](getting-started/overview.md)
- [Quickstart](getting-started/quickstart.md)
- [Your First Model](getting-started/first-model.md)

### Guides

#### CLI
- [CLI](guides/cli/cli.md)

#### Modeling
- [Modeling guide](guides/modeling/index.md)
- [DSL basics](guides/modeling/dsl-basics.md)
- [Equations](guides/modeling/equations.md)
- [Math and macros](guides/modeling/math-and-macros.md)
- [Ternary if](guides/modeling/ternary-if.md)
- [Model registry](guides/modeling/model-registry.md)
- [Auxiliary variables](guides/modeling/aux.md)
- [DSL functions](guides/modeling/functions.md)
- [Events](guides/modeling/events.md)
- [Lagging](guides/modeling/lagging.md)
- [Inline models](guides/modeling/inline-models.md)
- [Configuration file](guides/modeling/config-file.md)
- [Modes](guides/modeling/mods.md)
- [Presets](guides/modeling/presets.md)
- [Simulation defaults](guides/modeling/sim.md)

#### Simulation
- [Simulation guide](guides/simulation/index.md)
- [Basics](guides/simulation/basics.md)
- [Configuration](guides/simulation/configuration.md)
- [Just-In-Time compilation](guides/simulation/jit.md)
- [Presets](guides/simulation/presets.md)
- [Runner variants](guides/simulation/runner-variants.md)
- [Session introspection](guides/simulation/session-introspection.md)
- [Results](guides/simulation/results.md)
- [Steppers](guides/simulation/steppers.md)
- [Snapshots and resume](guides/simulation/snapshots-and-resume.md)
- [Wrapper and runner](guides/simulation/wrapper-and-runner.md)
- [Export sources](guides/simulation/export-sources.md)

#### Plotting
- [Plotting guide](guides/plotting/index.md)
- [Plotting basics](guides/plotting/basics.md)
- [Decorations](guides/plotting/decorations.md)
- [Export plots](guides/plotting/export.md)
- [Cobweb plots](guides/plotting/cobweb.md)
- [Basin plots](guides/plotting/basin-plot.md)
- [Vector fields](guides/plotting/vectorfields.md)
- [Manifold plots](guides/plotting/manifold-plot.md)
- [Themes and facets](guides/plotting/themes-and-facets.md)
- [Bifurcation diagrams](guides/plotting/bifurcation-diagrams.md)

#### Analysis
- [Analysis guide](guides/analysis/index.md)
- [Runtime observers](guides/analysis/observers.md)
- [Lyapunov analysis](guides/analysis/lyapunov.md)
- [Sweep tools](guides/analysis/sweep.md)
- [Post analysis](guides/analysis/post-analysis.md)
- [Fixed points](guides/analysis/fixed-points.md)
- [Basin analysis](guides/analysis/basin.md)
- [Bifurcation diagrams](guides/analysis/bifurcation.md)
- [Manifold analysis](guides/analysis/manifold.md)

### Examples
- [Overview](examples/index.md)
- [Analysis catalog](examples/analysis.md)
- [Plotting catalog](examples/plotting.md)
- [Runtime catalog](examples/runtime.md)
- [State management](examples/state-management.md)
- [Logistic map bifurcation](examples/bifurcation.md)
- [Collatz conjecture](examples/integer-map.md)
- [Izhikevich neuron](examples/izhikevich.md)

### Reference
- [Overview](reference/index.md)
- [Built-in models](reference/models/index.md)

### Project
- [Changelog](project/changelog.md)
- [Issues](project/issues.md)
- [TODO](project/todo.md)
