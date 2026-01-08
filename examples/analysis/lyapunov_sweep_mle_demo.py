"""
Lyapunov exponent parameter sweep for the logistic map.

Demonstrates sweep.lyapunov_mle for computing maximum Lyapunov exponents (MLE)
across a range of parameter values. This reveals the transition from order to chaos
as a continuous function of the control parameter.

For the logistic map x_{n+1} = r*x_n*(1-x_n):
    - r < 3.0: Stable fixed point (λ < 0)
    - 3.0 < r < ~3.57: Periodic orbits (λ = 0)
    - r > ~3.57: Chaotic regime (λ > 0)
    - r = 4.0: Fully chaotic (λ = ln(2) ≈ 0.6931)
"""

from __future__ import annotations
import numpy as np
from dynlib import setup
from dynlib.analysis import sweep
from dynlib.plot import series, export, theme, fig, bifurcation_diagram


# Setup simulation
sim = setup("builtin://map/logistic", jit=True, disk_cache=False)

# Parameter sweep configuration (1000 points won't trigger parallelisation in auto mode)
r_values = np.linspace(2.5, 4.0, 1000)  
record_every = 1

print(f"Computing bifurcation diagram and Lyapunov exponents for {len(r_values)} parameter values...")
print(f"  Parameter range: r ∈ [{r_values[0]:.2f}, {r_values[-1]:.2f}]")
print(f"  Iterations per run: 3000")
print(f"  Transient: 500")
print(f"  Trace recording interval: {record_every}")

# First, compute bifurcation diagram
print("\nComputing bifurcation diagram...")
sweep_bif = sweep.traj(
    sim,
    param="r",
    values=r_values,
    record_vars=["x"],
    N=200,
    transient=300,
)
result_bif = sweep_bif.bifurcation("x")

# Then, run parameter sweep with MLE analysis
print("\nComputing Lyapunov exponents...")
sim.assign(x=0.4)  # Initial condition
res = sweep.lyapunov_mle(
    sim,
    param="r",
    values=r_values,
    N=3000,
    transient=500,
    dt=1.0,
    record_interval=record_every,
    parallel_mode="auto",  # Enable parallelization
)

print(f"\nSweep completed!")
print(f"  Sweep kind: {res.kind}")
print(f"  MLE range: [{res.mle.min():.4f}, {res.mle.max():.4f}]")
print(f"  Chaos onset (λ=0): r ≈ {r_values[np.argmin(np.abs(res.mle))]:.4f}")
print(f"  MLE at r=4.0: {res.mle[-1]:.4f} (theoretical: {np.log(2):.4f})")

# Optional: inspect recorded trace per-parameter (list of arrays)
trace_runs = res.traces.get("mle")
if trace_runs:
    print(f"  Recorded {len(trace_runs)} convergence traces (first length={trace_runs[0].shape[0]})")

# ===== Visualization =====
theme.use("notebook")
theme.update(grid=True, vline_label_placement_pad=0.16)

# Create figure with 2 rows: bifurcation diagram (top) and MLE (bottom)
ax = fig.grid(rows=2, cols=1, size=(12, 10))

# Top panel: Bifurcation diagram
bifurcation_diagram(
    result_bif,
    alpha=0.8,
    color="black",
    ax=ax[0, 0],
    xlim=(2.5, 4.0),
    ylim=(0, 1),
    ylabel="x*",
    title="Bifurcation Diagram and Maximum Lyapunov Exponent",
    title_fs=14,
    ylabel_fs=12,
    vlines=[(3.0, 'r=3\n(period-2)'), (3.57, 'r≈3.57\n(chaos)')],
    vlines_kwargs={'color': 'darkred', 
                   'linestyle': '--', 
                   'alpha': 0.3, 
                   'linewidth': 1,
                   'label_position': 'bottom',},
)

# Bottom panel: MLE vs parameter (sharing same x-axis)
series.plot(
    x=res.values,
    y=res.mle,
    style="continuous",
    ax=ax[1, 0],
    xlabel="r",
    ylabel="λ (MLE)",
    lw=1.5,
    color="darkred",
    xlabel_fs=12,
    ylabel_fs=12,
    vlines=[3.0, 3.57],
    vlines_kwargs={'color': 'orange', 'linestyle': '--', 'alpha': 0.3, 'linewidth': 1},
)

# Add horizontal line at λ=0 (chaos boundary)
ax[1, 0].axhline(0, color="gray", ls="--", lw=1, alpha=0.7)
ax[1, 0].text(2.6, 0.05, "λ = 0 (chaos boundary)", fontsize=10, color="gray")

# Set x-limits to match
ax[1, 0].set_xlim(2.5, 4.0)

export.show()

# ===== Analysis Summary =====
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

# Find chaos onset (where λ crosses zero from negative to positive)
zero_crossings = np.where(np.diff(np.sign(res.mle)) > 0)[0]
if len(zero_crossings) > 0:
    chaos_onset_idx = zero_crossings[0]
    print(f"Chaos onset (first λ=0 crossing): r ≈ {res.values[chaos_onset_idx]:.4f}")

# Count negative, zero, and positive MLE regions
n_negative = np.sum(res.mle < -0.01)
n_zero = np.sum(np.abs(res.mle) <= 0.01)
n_positive = np.sum(res.mle > 0.01)

print(f"\nRegion distribution:")
print(f"  Stable (λ < 0):   {n_negative:3d} points ({n_negative/len(res.mle)*100:.1f}%)")
print(f"  Neutral (λ ≈ 0):  {n_zero:3d} points ({n_zero/len(res.mle)*100:.1f}%)")
print(f"  Chaotic (λ > 0):  {n_positive:3d} points ({n_positive/len(res.mle)*100:.1f}%)")

print(f"\nExtreme values:")
print(f"  Min MLE: {res.mle.min():.6f} at r={res.values[np.argmin(res.mle)]:.4f}")
print(f"  Max MLE: {res.mle.max():.6f} at r={res.values[np.argmax(res.mle)]:.4f}")
