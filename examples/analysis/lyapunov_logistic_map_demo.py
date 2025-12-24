"""
Lyapunov exponent calculation for the logistic map.

Demonstrates the runtime analysis system for computing maximum Lyapunov exponents (MLE)
and Lyapunov spectrum in discrete dynamical systems using the high-level Sim.run() API.

The Lyapunov exponent λ characterizes divergence of nearby trajectories:
    - λ > 0: Chaotic behavior 
    - λ = 0: Periodic orbits
    - λ < 0: Stable fixed points

For the logistic map at r=4: λ = ln(2) ≈ 0.6931
"""

from __future__ import annotations
import numpy as np
from dynlib import setup
from dynlib.analysis.runtime import lyapunov_mle, lyapunov_spectrum
from dynlib.plot import series, export, theme, fig


# Single run with multiple analyses
sim = setup("builtin://map/logistic", jit=True, disk_cache=False)
model = sim.model

record_every = 1

# Run simulation using the high-level Sim API with multiple analyses
print("\nComputing Lyapunov exponents with Sim.run()...")
print(f"  Parameter: r = 4.0")
print(f"  Iterations: 5000")
print(f"  Initial condition: x = 0.4")
print(f"  Analyses: MLE and Spectrum")

sim.assign(x=0.4, r=4.0)
sim.run(
    N=5000,
    dt=1.0,
    record_interval=record_every,
    analysis=[lyapunov_mle(record_interval=record_every), 
              lyapunov_spectrum(k=1, record_interval=record_every)],
)
result = sim.results()

# Extract runtime analysis results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Get analysis result with ergonomic named access
lyap = result.analysis["lyapunov_mle"]
spectrum = result.analysis["lyapunov_spectrum"]

# Direct access to final MLE (auto-computed from trace)
mle = lyap.mle  # Final converged value from trace
log_growth = lyap.log_growth
n_steps = int(lyap.steps)

# Get spectrum results (lyapunov_spectrum stores the final value in the trace)
spectrum_steps = int(spectrum.steps)

theoretical_mle = np.log(2.0)
x_trajectory = result["x"]
lyap_trace = lyap["mle"]  # Full trace array (use bracket for arrays)
spectrum_trace = spectrum["lyap0"]  # Spectrum trace for first exponent

# Note: trace may have one less point than trajectory if recording starts at t0
n_points = min(len(x_trajectory), len(lyap_trace), len(spectrum_trace))
iterations = np.arange(n_points)

# Get final spectrum value from trace (after data is loaded)
# spectrum trace has alternating zeros, so find the last non-zero value
nonzero_indices = np.nonzero(spectrum_trace)[0]
spectrum_mle = spectrum_trace[nonzero_indices[-1]] if len(nonzero_indices) > 0 else 0

print("\nMaximum Lyapunov Exponent (MLE):")
print(f"  Computed MLE:      {mle:.10f}")
print(f"  Theoretical MLE:   {theoretical_mle:.10f} (ln(2))")
print(f"  Relative error:    {abs(mle - theoretical_mle)/theoretical_mle * 100:.4f}%")
print(f"  Total iterations:  {n_steps}")

print("\nLyapunov Spectrum:")
print(f"  Spectrum λ₀:       {spectrum_mle:.10f}")
print(f"  Theoretical λ₀:    {theoretical_mle:.10f} (ln(2))")
print(f"  Relative error:    {abs(spectrum_mle - theoretical_mle)/theoretical_mle * 100:.4f}%")
print(f"  Total iterations:  {spectrum_steps}")

# Visualize
theme.use("notebook")
theme.update(grid=True)

fig_obj = fig.grid(rows=3, cols=1, size=(10, 12))

# Plot trajectory (first 500 iterations)
series.plot(
    x=iterations[:500],
    y=x_trajectory[:500],
    style="line",
    ax=fig_obj[0, 0],
    xlabel="Iteration (n)",
    ylabel="$x_n$",
    title=f"Logistic Map Trajectory (r=4.0)",
    lw=1.0,
    color="C0",
)

# Plot Lyapunov exponent convergence
series.plot(
    x=iterations,
    y=lyap_trace[:n_points],
    style="line",
    ax=fig_obj[1, 0],
    xlabel="Iteration (n)",
    ylabel="$\\lambda$ (MLE)",
    title="Lyapunov Exponent Convergence",
    lw=1.5,
    color="C1",
)

# Add theoretical value
fig_obj[1, 0].axhline(
    y=theoretical_mle,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Theoretical: λ = ln(2) ≈ {theoretical_mle:.4f}'
)
fig_obj[1, 0].legend(fontsize=10)

# Annotate final value
fig_obj[1, 0].text(
    0.98, 0.05,
    f'Computed: λ = {mle:.6f}',
    transform=fig_obj[1, 0].transAxes,
    fontsize=11,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

# Plot Lyapunov spectrum convergence
series.plot(
    x=iterations,
    y=spectrum_trace[:n_points],
    style="line",
    ax=fig_obj[2, 0],
    xlabel="Iteration (n)",
    ylabel="$\\lambda$ (Spectrum)",
    title="Lyapunov Spectrum Convergence",
    lw=1.5,
    color="C2",
)

# Add theoretical value
fig_obj[2, 0].axhline(
    y=theoretical_mle,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Theoretical: λ = ln(2) ≈ {theoretical_mle:.4f}'
)
fig_obj[2, 0].legend(fontsize=10)

# Annotate final value from spectrum
spectrum_final = spectrum_trace[n_points-1] if n_points > 0 else 0
fig_obj[2, 0].text(
    0.98, 0.05,
    f'Computed: λ = {spectrum_final:.6f}',
    transform=fig_obj[2, 0].transAxes,
    fontsize=11,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

export.show()

print("\n" + "="*60)
print("Parameter Scan")
print("="*60)
print(f"{'r':>6s} {'λ (MLE)':>12s} {'Regime':>15s}")
print("-"*60)

test_r_values = [2.5, 3.2, 3.5, 3.83, 4.0]
scan_sim = setup("builtin://map/logistic", stepper="map", jit=False)

for r_test in test_r_values:
    params_test = np.array([r_test], dtype=model.dtype)
    scan_sim.run(
        N=2000,
        dt=1.0,
        record=False,
        record_interval=1,
        max_steps=2000,
        cap_rec=1,
        cap_evt=1,
        ic=np.array([0.4], dtype=model.dtype),
        params=params_test,
        analysis=lyapunov_mle,  # Factory mode: Sim injects model
    )

    lyap_result = scan_sim.results().analysis["lyapunov_mle"]
    mle_test = lyap_result.mle  # Direct final value access
    
    if mle_test < -0.01:
        regime = "Stable"
    elif abs(mle_test) < 0.01:
        regime = "Periodic"
    else:
        regime = "Chaotic"
    
    print(f"{r_test:6.2f} {mle_test:12.6f} {regime:>15s}")
    scan_sim.reset()

print("="*60)
print("\nThe logistic map exhibits the period-doubling route to chaos.")
print("Chaos emerges around r ≈ 3.57, fully developed at r = 4.")
