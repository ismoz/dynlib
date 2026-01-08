"""
Benchmark for basin_known with refine=False and refine=True.

Tests performance on different grid sizes: 64x64, 128x128, 256x256, 512x512.
Uses the same parameters and model as basin_henon_known.py.
"""

from __future__ import annotations

import time
import numpy as np

from dynlib import setup
from dynlib.analysis import basin_known, ReferenceRun

# Setup Henon map model
print("Setting up Henon map model...")
sim = setup("builtin://map/henon", stepper="map", jit=True)

# Set standard Henon parameters
a_param = 1.4
b_param = 0.3
sim.assign(a=a_param, b=b_param)

print(f"  Parameters: a={a_param}, b={b_param}")
print(f"  State variables: x, y")

# Domain for basin calculation (region near Henon attractor)
x_min, x_max = -2.5, 2.5
y_min, y_max = -2.5, 2.5

# Common parameters for basin_known
attractors = [
    ReferenceRun(name="Henon attractor", ic=[0.1, 0.1]),
]
ic_bounds = [(x_min, x_max), (y_min, y_max)]
max_samples = 500
transient_samples = 200
signature_samples = 500
tolerance = 0.05      # 5% of attractor range
min_match_ratio = 0.8  # 80% of points must match
escape_bounds = [(-5.0, 5.0), (-5.0, 5.0)]  # Wide bounds for escape detection
b_max = 1e6  # Blowup threshold / None means literal NaN/Inf
parallel_mode = "auto"

# Grid sizes to test
grid_sizes = [64, 128, 256, 512, 1024]

print("\nBenchmarking basin_known with different grid sizes and refine options...")
grid_width = 12
time_width = 12
match_width = 10
print(f"{'Grid Size':<{grid_width}} | {'refine = False':>{time_width}} | {'refine = True':>{time_width}} | {'Match %':>{match_width}}")
print("-" * (grid_width + time_width * 2 + match_width + 9))

# Warm-up
basin_known(
    sim,
    attractors=attractors,
    ic_grid=[grid_sizes[0], grid_sizes[0]],
    ic_bounds=ic_bounds,
    max_samples=max_samples,
    transient_samples=transient_samples,
    signature_samples=signature_samples,
    tolerance=tolerance,
    min_match_ratio=min_match_ratio,
    escape_bounds=escape_bounds,
    b_max=b_max,
    parallel_mode=parallel_mode,
    refine=False,
)

for size in grid_sizes:
    ic_grid = [size, size]
    
    # Measure refine=False
    start_time = time.perf_counter()
    result_false = basin_known(
        sim,
        attractors=attractors,
        ic_grid=ic_grid,
        ic_bounds=ic_bounds,
        max_samples=max_samples,
        transient_samples=transient_samples,
        signature_samples=signature_samples,
        tolerance=tolerance,
        min_match_ratio=min_match_ratio,
        escape_bounds=escape_bounds,
        b_max=b_max,
        parallel_mode=parallel_mode,
        refine=False,
    )
    false_time = time.perf_counter() - start_time
    
    # Measure refine=True
    start_time = time.perf_counter()
    result_true = basin_known(
        sim,
        attractors=attractors,
        ic_grid=ic_grid,
        ic_bounds=ic_bounds,
        max_samples=max_samples,
        transient_samples=transient_samples,
        signature_samples=signature_samples,
        tolerance=tolerance,
        min_match_ratio=min_match_ratio,
        escape_bounds=escape_bounds,
        b_max=b_max,
        parallel_mode=parallel_mode,
        refine=True,
    )
    true_time = time.perf_counter() - start_time
    
    # Compute match percentage
    match_percentage = np.sum(result_false.labels == result_true.labels) / result_false.labels.size * 100
    
    print(f"{f'{size}x{size}':<{grid_width}} | {false_time:>{time_width}.2f} | {true_time:>{time_width}.2f} | {match_percentage:>{match_width}.1f}")

print("\nBenchmark complete!")