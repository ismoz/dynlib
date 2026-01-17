"""
Lyapunov spectrum parameter sweep for the Lorenz system.

Demonstrates sweep.lyapunov_spectrum_sweep for computing the full Lyapunov spectrum
across a range of parameters, and overlays the results with a Lorenz
bifurcation diagram.

For the Lorenz system (sigma=10, beta=8/3):
    - rho < ~24.74: stable fixed points
    - rho > ~24.74: chaotic regime
"""

from __future__ import annotations

import numpy as np
from dynlib import setup
from dynlib.analysis import sweep
from dynlib.plot import series, export, theme, fig, bifurcation_diagram


# Setup simulation
sim = setup("builtin://ode/lorenz", stepper="rk4", jit=True)

sigma, beta = 10.0, 8.0/3.0
initial_state = {"x": 1.0, "y": 1.0, "z": 1.0}

# Sweep configuration
rho_values = np.linspace(0.0, 200.0, 4000)
dt = 0.01
total_time = 50.0
transient = 50.0
record_interval = 1

print(f"Computing Lorenz bifurcation + spectrum for {len(rho_values)} values...")
print(f"  rho range: [{rho_values[0]:.2f}, {rho_values[-1]:.2f}]")
print(f"  T={total_time}, dt={dt}, transient={transient}")

# Bifurcation diagram (z vs rho)
print("\nComputing bifurcation diagram...")
sim.assign(**initial_state, sigma=sigma, rho=rho_values[0], beta=beta)
bif_sweep = sweep.traj_sweep(
    sim,
    param="rho",
    values=rho_values,
    record_vars=["z"],
    T=total_time,
    dt=dt,
    transient=transient,
    record_interval=record_interval,
)
# Local extrema give clean bifurcation diagrams for continuous systems
bif_result = bif_sweep.bifurcation("z").extrema(
    max_points=50,  # Limit points in chaotic regime
    min_peak_distance=8
)

# Lyapunov spectrum sweep
print("\nComputing Lyapunov spectrum...")
sim.reset()
sim.assign(**initial_state, sigma=sigma, rho=rho_values[0], beta=beta)
res = sweep.lyapunov_spectrum_sweep(
    sim,
    param="rho",
    values=rho_values,
    k=3,
    T=total_time,
    dt=dt,
    transient=transient,
    record_interval=record_interval,
    parallel_mode="auto",
)

print("\nSweep completed!")
print(
    "  Spectrum min/max (lambda1): "
    f"[{res.lyap0.min():.4f}, {res.lyap0.max():.4f}]"
)

# ===== Visualization =====
theme.use("notebook")
theme.update(grid=True)

ax = fig.grid(rows=2, cols=1, size=(12, 10), sharex=True)

# Upper panel: Bifurcation diagram
bifurcation_diagram(
    bif_result,
    alpha=0.6,
    color="black",
    ax=ax[0, 0],
    xlabel="rho",
    ylabel="z",
    title="Lorenz Bifurcation Diagram and Lyapunov Spectrum",
    title_fs=14,
    ylabel_fs=12,
)

# Lower panel: Lyapunov spectrum vs parameter
series.multi(
    x=res.values,
    y=[res.lyap0, res.lyap1, res.lyap2],
    names=["$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$"],
    ax=ax[1, 0],
    xlabel="rho",
    ylabel="Lyapunov exponents",
    legend=True,
    hlines=[0.0],
    hlines_kwargs={"color": "gray", "ls": "--", "lw": 1, "alpha": 0.7},
    xlim=(rho_values[0], rho_values[-1]),
)

export.show()
