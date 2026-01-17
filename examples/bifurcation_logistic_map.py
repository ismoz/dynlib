"""
Bifurcation diagram demonstration for the logistic map.

"""

from __future__ import annotations
import numpy as np
from dynlib import setup
from dynlib.analysis import sweep
from dynlib.plot import export, theme, fig, bifurcation_diagram

r0 = 2.8
rend = 4.0

sim = setup("builtin://map/logistic", stepper="map", jit=True, disk_cache=True)

r_values = np.linspace(r0, rend, 20000)

print("Computing bifurcation diagram...")
print(f"  Parameter: r ∈ [{r0}, {rend}]")
print(f"  Grid points: {len(r_values)}")

sweep_result = sweep.traj_sweep(
    sim,
    param="r",
    values=r_values,
    record_vars=["x"],
    N=100,  
    transient=500,  
    record_interval=1,  
    parallel_mode="auto",
)
# Extract bifurcation data (defaults to .all() mode - all recorded points)
result = sweep_result.bifurcation("x")
# Alternatively: result = sweep_result.bifurcation("x").tail(50)  # for tail mode

print(f"  Total points plotted: {len(result.p)}")
print("Done!")

theme.use("notebook")
theme.update(grid=True)

ax = fig.single(size=(10, 8))

bifurcation_diagram(
    result,
    color="blue",
    xlim=(r0, rend),
    ylim=(0, 1),
    xlabel="r",
    ylabel="x*",
    title="Bifurcation Diagram: Logistic Map",
    xlabel_fs=12,
    ylabel_fs=12,
    title_fs=14,
    ax=ax,
)

export.show()
# Save figure with high dpi will be more smooth
# export.savefig(ax, "bifurcation_logistic_map.png", dpi=600)