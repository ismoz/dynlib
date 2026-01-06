"""
Basin of Attraction calculation for the Henon Map.

Demonstrates the basin_auto function (Persistent Cell-Recurrence Basin Mapping algorithm)
to identify basins of attraction for the 2D Henon map.

The Henon map is defined by:
    x' = 1 - a*x² + y
    y' = b*x

For the standard parameters (a=1.4, b=0.3), the map exhibits a strange attractor.
This example computes basins on a 2D grid of initial conditions.

Note: This example uses non-JIT mode to avoid NaN detection issues in the fastpath
that can occur with diverging trajectories.
"""

from __future__ import annotations
import numpy as np
from dynlib import setup
from dynlib.analysis import basin_auto, BLOWUP, OUTSIDE, UNRESOLVED
from dynlib.plot import export, theme, fig, basin_plot

# Setup Henon map model
print("Setting up Henon map model...")
sim = setup("builtin://map/henon", stepper="map", jit=True)

# Set standard Henon parameters
a_param = 1.4
b_param = 0.3
sim.assign(a=a_param, b=b_param)

print(f"  Parameters: a={a_param}, b={b_param}")
print(f"  State variables: x, y")

# Define grid of initial conditions
print("\nGenerating initial condition grid...")

# Domain for basin calculation
# Focusing on a region near the Henon attractor
x_min, x_max = -3, 3
y_min, y_max = -3, 3

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
labels = result.labels.reshape(grid_ny, grid_nx)
n_attractors = len(result.registry)

# Count labels
n_total = len(result.labels)
n_blowup = np.sum(result.labels == BLOWUP)
n_outside = np.sum(result.labels == OUTSIDE)
n_unresolved = np.sum(result.labels == UNRESOLVED)
n_identified = n_total - n_blowup - n_outside - n_unresolved

print("\n" + "="*60)
print("BASIN ANALYSIS RESULTS")
print("="*60)
print(f"Total initial conditions: {n_total}")
print(f"Attractors identified: {n_attractors}")
print(f"Points assigned to attractors: {n_identified} ({100*n_identified/n_total:.1f}%)")
print(f"Blowup trajectories: {n_blowup} ({100*n_blowup/n_total:.1f}%)")
print(f"Outside domain: {n_outside} ({100*n_outside/n_total:.1f}%)")
print(f"Unresolved: {n_unresolved} ({100*n_unresolved/n_total:.1f}%)")

if n_attractors > 0:
    print("\nAttractor details:")
    for attractor in result.registry:
        n_points = np.sum(result.labels == attractor.id)
        print(f"  Attractor {attractor.id}: {n_points} points, "
              f"{len(attractor.fingerprint)} cells in fingerprint")

print("\nAlgorithm parameters:")
for key, value in result.meta.items():
    print(f"  {key}: {value}")

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

# Figure already uses constrained layout from fig.single(); avoid switching engines.

export.show()

print("\nVisualization complete!")
