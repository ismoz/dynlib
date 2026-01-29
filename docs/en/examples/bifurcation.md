# Bifurcation Diagram: Logistic Map

## Overview

This example demonstrates how to create bifurcation diagrams using dynlib's parameter sweep and plotting tools. Bifurcation diagrams visualize how the long-term behavior of a dynamical system changes as a parameter varies, revealing transitions between fixed points, periodic orbits, and chaotic dynamics.

The logistic map `x_{n+1} = r·x_n·(1-x_n)` is a classic example exhibiting the **period-doubling route to chaos** as the parameter `r` increases from 2.5 to 4.0.

## Key Concepts

- **Parameter sweeps**: Computing trajectories across a range of parameter values
- **Bifurcation extraction**: Post-processing trajectories into scatter data for visualization
- **Extraction modes**: Different strategies for capturing system behavior (`all`, `tail`, `final`, `extrema`)
- **Period-doubling cascade**: The Feigenbaum scenario leading to chaos

## The Logistic Map Model

The logistic map is defined as:

$$
x_{n+1} = r \cdot x_n \cdot (1 - x_n)
$$

Key bifurcation points:
- **r < 3.0**: Stable fixed point
- **r = 3.0**: First period-doubling (period-2 orbit emerges)
- **r ≈ 3.449**: Period-4 orbit
- **r ≈ 3.57**: Onset of chaos (Feigenbaum point r∞ ≈ 3.5699)
- **r = 4.0**: Fully developed chaos

## Basic Example

The simplest bifurcation diagram uses the built-in logistic map model:

```python
from dynlib import setup
from dynlib.analysis import sweep
from dynlib.plot import bifurcation_diagram, theme, fig, export
import numpy as np

# Setup model
sim = setup("builtin://map/logistic", stepper="map", jit=True)

# Parameter range
r_values = np.linspace(2.8, 4.0, 5000)

# Run parameter sweep
sweep_result = sweep.traj_sweep(
    sim,
    param="r",
    values=r_values,
    record_vars=["x"],
    N=100,           # iterations per parameter value
    transient=500,   # discard initial transient
)

# Extract bifurcation data
result = sweep_result.bifurcation("x")

# Plot
theme.use("notebook")
ax = fig.single(size=(10, 6))

bifurcation_diagram(
    result,
    xlabel="r",
    ylabel="x*",
    title="Bifurcation Diagram: Logistic Map",
    ax=ax
)

export.show()
```

## Complete Examples in Repository

### 1. **Basic Bifurcation**

```python
--8<-- "examples/bifurcation_logistic_map.py"
```

Demonstrates the standard workflow:
- High-resolution parameter sweep (20,000 points)
- Default extraction mode (all recorded points)
- Simple plotting with minimal customization

### 2. **Mode Comparison**

```python
--8<-- "examples/bifurcation_logistic_map_comparison.py"
```

Compares three extraction modes side-by-side:

```python
extractor = sweep_result.bifurcation("x")

# Mode 1: Final value only
result_final = extractor.final()

# Mode 2: Last 50 points (attractor cloud)
result_tail = extractor.tail(50)

# Mode 3: Local extrema (maxima + minima)
result_extrema = extractor.extrema(tail=100, max_points=30)
```