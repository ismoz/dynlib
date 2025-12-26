"""
Lyapunov exponent calculation for the Lorenz system.

Demonstrates the runtime analysis system for computing maximum Lyapunov exponents (MLE)
and Lyapunov spectrum in continuous dynamical systems using the high-level Sim.run() API.
"""

from __future__ import annotations
from dynlib import setup
from dynlib.analysis.runtime import lyapunov_mle, lyapunov_spectrum
from dynlib.plot import fig, export, phase, series

# 1. Setup simulation
# -------------------
sim = setup("builtin://ode/lorenz", stepper="rk4", jit=True)

# Parameters for standard chaotic regime
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
initial_state = {"x": 1.0, "y": 1.0, "z": 1.0}

# Simulation control
dt = 0.001
total_time = 400
transient = 20
record_interval = 4

print(f"Running Lorenz simulation (T={total_time}, dt={dt})...")

# 2. Run MLE Analysis
# -------------------
print("Computing Maximum Lyapunov Exponent (MLE)...")
sim.assign(**initial_state, sigma=sigma, rho=rho, beta=beta)
sim.run(
    transient=transient,
    T=total_time,
    dt=dt,
    record_interval=record_interval,
    analysis=lyapunov_mle(record_interval=record_interval)
)
res_mle = sim.results()
mle_analysis = res_mle.analysis["lyapunov_mle"]

# 3. Run Spectrum Analysis
# ------------------------
print("Computing Lyapunov Spectrum (3 exponents)...")
sim.reset()
sim.assign(**initial_state, sigma=sigma, rho=rho, beta=beta)
sim.run(
    transient=transient,
    T=total_time,
    dt=dt,
    record_interval=record_interval,
    analysis=lyapunov_spectrum(k=3, record_interval=record_interval)
)
res_spec = sim.results()
spec_analysis = res_spec.analysis["lyapunov_spectrum"]

# 4. Results & Validation
# -----------------------
print("\n" + "="*60)
print("RESULTS & VALIDATION")
print("="*60)

# Theoretical values for Lorenz (standard parameters)
ref_lambda1 = 0.9056
ref_lambda2 = 0.0000
ref_lambda3 = -14.5723

# MLE Results
mle_calc = mle_analysis.mle
mle_err = abs(mle_calc - ref_lambda1) / abs(ref_lambda1) * 100
print(f"\nMaximum Lyapunov Exponent (MLE):")
print(f"  Calculated:  {mle_calc:.4f}")
print(f"  Theoretical: {ref_lambda1:.4f}")
print(f"  Error:       {mle_err:.2f}%")

# Spectrum Results
# Get final values from traces (using attribute access for final value)
l1_calc = spec_analysis.lyap0
l2_calc = spec_analysis.lyap1
l3_calc = spec_analysis.lyap2

# Calculate errors (handle division by zero for lambda2)
l1_err = abs(l1_calc - ref_lambda1) / abs(ref_lambda1) * 100
l3_err = abs(l3_calc - ref_lambda3) / abs(ref_lambda3) * 100

print(f"\nLyapunov Spectrum:")
print(f"  λ₁: {l1_calc:.4f} (Ref: {ref_lambda1:.4f}, Err: {l1_err:.2f}%)")
print(f"  λ₂: {l2_calc:.4f} (Ref: {ref_lambda2:.4f}, Err: N/A)")
print(f"  λ₃: {l3_calc:.4f} (Ref: {ref_lambda3:.4f}, Err: {l3_err:.2f}%)")
print(f"  Sum: {l1_calc + l2_calc + l3_calc:.4f} (Ref: {ref_lambda1 + ref_lambda2 + ref_lambda3:.4f})")

# 5. Visualization
# ----------------
print("\nPlotting results...")

# Create a 3-row, 1-column grid
grid = fig.grid(rows=3, cols=1, size=(8, 12), title="Lorenz System Lyapunov Analysis")
ax_phase, ax_mle, ax_spec = grid[0][0], grid[1][0], grid[2][0]

# Plot 1: Phase Portrait (x vs z)
# Use the trajectory from the spectrum run (it has the same parameters)
x = res_spec["x"]
z = res_spec["z"]
phase.xy(
    x=x,
    y=z,
    ax=ax_phase,
    lw=0.5,
    alpha=0.8,
    title="Phase Portrait (Lorenz Attractor)",
    xlabel="x",
    ylabel="z",
)

# Plot 2: MLE Convergence
# Use explicit trace_time from analysis result
t_mle = mle_analysis.trace_time
mle_trace = mle_analysis["mle"]
series.plot(
    x=t_mle,
    y=mle_trace,
    ax=ax_mle,
    label=f"MLE (λ₁) ≈ {mle_analysis.mle:.3f}",
    title="Maximum Lyapunov Exponent Convergence",
    ylabel="MLE",
    legend=True,
    hlines=[(ref_lambda1, f"Theoretical λ₁ ≈ {ref_lambda1:.3f}")],
    hlines_kwargs={"alpha": 0.5},
)
ax_mle.grid(True, alpha=0.3)

# Plot 3: Lyapunov Spectrum Convergence
# Use explicit trace_time from analysis result
t_spec = spec_analysis.trace_time
# Spectrum trace columns: lyap0, lyap1, lyap2
l1 = spec_analysis["lyap0"]
l2 = spec_analysis["lyap1"]
l3 = spec_analysis["lyap2"]

series.multi(
    x=t_spec,
    y=[l1, l2, l3],
    names=[f"λ₁ ≈ {l1[-1]:.3f}", f"λ₂ ≈ {l2[-1]:.3f}", f"λ₃ ≈ {l3[-1]:.3f}"],
    ax=ax_spec,
    title="Lyapunov Spectrum Convergence",
    xlabel="Time",
    ylabel="Exponents",
    legend=True,
    hlines=[ref_lambda1, ref_lambda2, ref_lambda3],
    hlines_kwargs={"alpha": 0.3},
)

ax_spec.grid(True, alpha=0.3)

# Save figure
# export.savefig(grid, "lyapunov_lorenz_demo.png")
export.show()
