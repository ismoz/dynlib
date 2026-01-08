"""
Basin of Attraction for Henon Map using Known Attractors.

Demonstrates the simplified basin_known function to identify basins for the
2D Henon map. basin_known requires reference(s) of the attractor(s) to be 
provided beforehand.
"""

from __future__ import annotations

import numpy as np

from dynlib import setup
from dynlib.analysis import basin_known, print_basin_summary, ReferenceRun
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

# Define grid of initial conditions
print("\nGenerating initial condition grid...")

# Domain for basin calculation (region near Henon attractor)
x_min, x_max = -2.5, 2.5
y_min, y_max = -2.5, 2.5

# Grid resolution
grid_nx = 512
grid_ny = 512

print(f"  Grid size: {grid_nx} × {grid_ny} = {grid_nx * grid_ny} points")
print(f"  Domain: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")

# Compute basin of known attractors
print("\nComputing basin of known attractors...")

with Timer("basin_known computation"):
    result = basin_known(
        sim,
        attractors=[
            ReferenceRun(name="Henon attractor", ic=[0.1, 0.1]),
        ],
        ic_grid=[grid_nx, grid_ny],
        ic_bounds=[(x_min, x_max), (y_min, y_max)],
        max_samples=500,
        transient_samples=200,
        signature_samples=500,
        tolerance=0.05,      # 5% of attractor range
        min_match_ratio=0.8,  # 80% of points must match
        escape_bounds=[(-5.0, 5.0), (-5.0, 5.0)],  # Wide bounds for escape detection
        b_max=1e6, # Blowup threshold / None means literal NaN/Inf
        parallel_mode="auto",
        refine=True, # refine is faster for high resolution grids
    )

print("Done!")

# Analyze results
print_basin_summary(result)

# Visualization
print("\nCreating visualization...")
theme.use("notebook")
theme.update(grid=False)

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
