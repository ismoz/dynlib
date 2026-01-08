"""
Auto basin of attraction calculation for the Henon Map.

Demonstrates the basin_auto function (Persistent Cell-Recurrence Basin Mapping algorithm)
to identify basins of attraction for the 2D Henon map.

basin_auto tries to automatically discover attractors and their basins. Its success highly
depends on the choice of parameters and the nature of the dynamical system. 
"""

from __future__ import annotations
import numpy as np
from dynlib import setup
from dynlib.analysis import basin_auto, print_basin_summary
from dynlib.plot import export, theme, fig, basin_plot
from dynlib.utils import Timer

# Setup Henon map model
print("Setting up Henon map model...")
sim = setup("builtin://map/henon", stepper="map", jit=True)

# Set standard Henon parameters
a_param = 1.4
b_param = 0.3
sim.assign(a=a_param, b=b_param)

print(f"  Parameters: a={a_param}, b={b_param}")
print(f"  State variables: x, y")

# Domain for basin calculation
# Focusing on a region near the Henon attractor
x_min, x_max = -2.5, 2.5
y_min, y_max = -2.5, 2.5

# Grid resolution (increase for higher detail)
grid_nx = 512
grid_ny = 512

print(f"  Grid size: {grid_nx} × {grid_ny} = {grid_nx * grid_ny} points")
print(f"  Domain: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")

# Compute basin of attraction
print("\nComputing basin of attraction...")
print("  Algorithm: PCR-BM (Persistent Cell-Recurrence Basin Mapping)")
print(f"  Grid size: {grid_nx} × {grid_ny} = {grid_nx * grid_ny} points")
print(f"  Domain: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")

with Timer("Basin computation time"):
    result = basin_auto(
        sim,
        ic_grid=[grid_nx, grid_ny],
        ic_bounds=[(x_min, x_max), (y_min, y_max)],
        observe_vars=["x", "y"],  # Use both state variables
        grid_res=[128, 128],  # Cell resolution for recurrence detection
        merge_downsample=4,
        post_detect_samples=512,
        max_samples=2000,  # Number of iterations per initial condition
        transient_samples=500,  # Skip initial transient
        window=128,  # Recurrence detection window size
        u_th=0.95,  # Uniqueness threshold
        recur_windows=3,  # Required recurrence windows
        s_merge=0.3,  # Similarity threshold for merging attractors
        p_in=30,  # Persistence threshold
        b_max=100.0,  # Blowup threshold (lower to catch diverging trajectories)
        outside_limit=50,  # Maximum consecutive outside steps
        parallel_mode="auto",
    )

print("Done!")

# Analyze results
print_basin_summary(result)

# Visualization
print("\nCreating visualization...")
theme.use("notebook")
theme.update(grid=False)

# Create figure
ax = fig.single(size=(10, 8))

basin_plot(
    result,
    ax=ax,
    xlabel="x",
    ylabel="y",
    ylabel_rot=0,
    ypad=15,
    title=f"Basin of Attraction - Henon Map (a={a_param}, b={b_param})",
    titlepad=15,
)

export.show()
print("\nVisualization complete!")
