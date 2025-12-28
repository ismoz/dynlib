"""
Bifurcation diagram comparison for the logistic map showing different modes.

This example demonstrates the three different bifurcation analysis modes:
1. "final": Shows only the final state (good for fixed points)
2. "tail": Shows multiple samples from the attractor (reveals periodic orbits)
3. "extrema": Shows local extrema (emphasizes periodic structure)
"""

from __future__ import annotations
import numpy as np
from dynlib import setup
from dynlib.analysis import sweep
from dynlib.plot import export, theme, fig

# Setup the builtin logistic map model
sim = setup("builtin://map/logistic", stepper="map", jit=True, disk_cache=True)

# Focus on the interesting region where bifurcations occur
r_values = np.linspace(2.8, 4.0, 1500)

print("Computing bifurcation diagrams in different modes...")

# Runtime sweep (do once)
sweep_result = sweep.traj(
    sim,
    param="r",
    values=r_values,
    record_vars=["x"],
    N=500,
    transient=200,
)

extractor = sweep_result.bifurcation("x")

# Mode 1: "final" - just the final value (good for convergent behavior)
result_final = extractor.final()

# Mode 2: "tail" - multiple samples from attractor
result_tail = extractor.tail(50)

# Mode 3: "extrema" - local extrema (maxima + minima)
result_extrema = extractor.extrema(tail=100, max_points=30, min_peak_distance=1)

print("Done! Creating comparison plot...")

# Configure plot theme
theme.use("notebook")
theme.update(grid=True)

# Create 3-panel comparison
from dynlib.plot import bifurcation_diagram

ax = fig.grid(rows=3, cols=1, size=(10, 10))

# Panel 1: Final mode
bifurcation_diagram(
    result_final,
    marker=".",
    ms=0.5,
    alpha=0.7,
    color="blue",
    ax=ax[0, 0],
    xlim=(2.8, 4.0),
    ylim=(0, 1),
    ylabel="x*",
    title='Mode: "final" (last recorded value)',
    title_fs=12,
)

# Panel 2: Tail mode
bifurcation_diagram(
    result_tail,
    marker=".",
    ms=0.3,
    alpha=0.5,
    color="black",
    ax=ax[1, 0],
    xlim=(2.8, 4.0),
    ylim=(0, 1),
    ylabel="x*",
    title='Mode: "tail" (attractor cloud)',
    title_fs=12,
)

# Panel 3: Extrema mode
bifurcation_diagram(
    result_extrema,
    marker=".",
    ms=0.5,
    alpha=0.6,
    color="red",
    ax=ax[2, 0],
    xlim=(2.8, 4.0),
    ylim=(0, 1),
    xlabel="r",
    ylabel="x*",
    title='Mode: "extrema" (local maxima and minima)',
    title_fs=12,
)

export.show()
