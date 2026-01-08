"""
Demonstration of basin of attraction analysis for the Duffing oscillator with
two fixed point attractors at +1 and -1.

This example uses basin_known analysis utility.
"""
from dynlib import setup
from dynlib.analysis import basin_known, FixedPoint
from dynlib.plot import export, basin_plot
from dynlib.utils import Timer

sim = setup("builtin://ode/duffing", jit=True, disk_cache=False, stepper="rk2")
sim.assign(delta=0.02, alpha=-0.5, beta=0.5, gamma=0.0)

with Timer("Calculation time"):
    results = basin_known(
        sim,
        # radius = 0.3 due to previous knowledge. If you are unconfident,
        # this should be small but the sim will take longer to classify.
        attractors=[FixedPoint(name="+1", loc=[1.0, 0.0], radius=0.3), 
                    FixedPoint(name="-1", loc=[-1.0, 0.0], radius=0.3)],
        ic_grid=[300, 300],
        ic_bounds=[(-1.5, 1.5), (-1.5, 1.5)],
        # For damped, unforced Duffing the true attractors are fixed points.
        # Make recurrence detection trigger only very close to the attractor.
        dt_obs=0.01,
        max_samples=60000,     # T=600
        transient_samples=0,   # Lower transient is better for fixed-points because of the early exit feature
        signature_samples=0,   # No need for a sample
        escape_bounds=[(-2.0, 2.0), (-2.0, 2.0)],  # Wide bounds for escape detection
        b_max=1e6,
    )

print(f"discovered attractors: {len(results.registry)}")
print(f"unique assigned IDs: {len(set(int(x) for x in results.labels.tolist() if int(x) >= 0))}")


# Plotting
basin_plot(results)

export.show()