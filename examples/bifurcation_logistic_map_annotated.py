"""
Advanced bifurcation analysis for the logistic map.

This example demonstrates:
1. High-resolution bifurcation diagram
2. Annotating key bifurcation points
3. Zooming into the Feigenbaum cascade region
"""

from __future__ import annotations
import numpy as np
from dynlib import setup
from dynlib.analysis import sweep
from dynlib.plot import export, theme, fig, bifurcation_diagram


# Setup the builtin logistic map model
sim = setup("builtin://map/logistic", stepper="map", jit=True, disk_cache=True)

print("Computing full bifurcation diagram (r ∈ [2.5, 4])...")

# Full diagram
r_full = np.linspace(2.5, 4.0, 2500)
sweep_full = sweep.traj(
    sim,
    param="r",
    values=r_full,
    record_vars=["x"],
    N=100,
    transient=300,
)
result_full = sweep_full.bifurcation("x").tail(80)

print("Computing zoomed diagram (period-doubling cascade region)...")

# Zoomed region: period-doubling cascade
r_zoom = np.linspace(3.4, 3.57, 2000)
sweep_zoom = sweep.traj(
    sim,
    param="r",
    values=r_zoom,
    record_vars=["x"],
    N=200,
    transient=400,
)
result_zoom = sweep_zoom.bifurcation("x")

print("Creating visualization...")

# Configure plot theme
theme.use("notebook")
theme.update(grid=True)

# Create 2-panel plot
ax = fig.grid(rows=2, cols=1, size=(12, 10))

# Panel 1: Full diagram with annotations
bifurcation_diagram(
    result_full,
    alpha=1.0,
    color="black",
    ax=ax[0, 0],
    xlim=(2.5, 4),
    ylim=(0, 1),
    ylabel="x*",
    title="Bifurcation Diagram: Logistic Map (Full Range)",
    title_fs=14,
    ylabel_fs=12,
)

# Add annotations for key bifurcation points
ax[0, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax[0, 0].axvline(x=3.0, color='orange', linestyle='--', alpha=0.3, linewidth=1)
ax[0, 0].axvline(x=3.449, color='green', linestyle='--', alpha=0.3, linewidth=1)
ax[0, 0].axvline(x=3.57, color='blue', linestyle='--', alpha=0.3, linewidth=1)

ax[0, 0].text(1.0, 0.92, 'r=1\n(transcritical)', ha='center', fontsize=8, color='red')
ax[0, 0].text(3.0, 0.92, 'r=3\n(period-2)', ha='center', fontsize=8, color='orange')
ax[0, 0].text(3.449, 0.92, 'r≈3.45\n(period-4)', ha='center', fontsize=8, color='green')
ax[0, 0].text(3.57, 0.92, 'r≈3.57\n(chaos)', ha='center', fontsize=8, color='blue')

# Panel 2: Zoomed into cascade
bifurcation_diagram(
    result_zoom,
    marker=".",
    ms=1.0,
    alpha=1.0,
    color="darkblue",
    ax=ax[1, 0],
    xlim=(3.4, 3.57),
    ylim=(0.3, 0.95),
    xlabel="r",
    ylabel="x*",
    title="Period-Doubling Cascade (Zoomed)",
    title_fs=14,
    xlabel_fs=12,
    ylabel_fs=12,
)

# Highlight the Feigenbaum point
ax[1, 0].axvline(x=3.5699, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax[1, 0].text(3.5699, 0.32, 'Feigenbaum point\nr∞≈3.5699', 
              ha='center', fontsize=9, color='red',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

export.show()
export.savefig(ax, "bifurcation_logistic_map_annotated.png", dpi=300)

print("\nKey bifurcation points:")
print("  r = 1.0    : Transcritical bifurcation (fixed point emerges)")
print("  r = 3.0    : First period-doubling (period-2 orbit)")
print("  r ≈ 3.449  : Period-4 orbit")
print("  r ≈ 3.544  : Period-8 orbit")
print("  r∞ ≈ 3.5699: Feigenbaum point (accumulation of period-doublings)")
print("  r > 3.57   : Chaotic regime with periodic windows")
