from dynlib import setup
from dynlib.analysis import basin_auto
from dynlib.plot import export, basin_plot, fig, phase

# Energy Template Oscillator (ETO) with Circular L Curve
sim = setup("builtin://ode/eto-circular", jit=True, disk_cache=True, stepper="rk4")
# Globally stable limit cycle parameters
sim.assign(mu=0.8, a=2.0)

results = basin_auto(
    sim,
    ic_grid=[128, 128],
    ic_bounds=[(-3, 3), (-3, 3)],
    dt_obs=0.01,
    # run long enough to converge + see the cycle
    max_samples=4000,          # 40 time units
    transient_samples=2000,    # 20 time units of transient
    # recurrence / evidence
    window=512,                # ~5.12 time units > 1 period
    recur_windows=3,
    u_th=0.8,                  # less strict; avoids missing clean periodic motion
    post_detect_samples=1024,  # add extra evidence (aim: >= 1–2 periods total)
    # merging
    merge_downsample=2,        # coarser fingerprint grid => more overlap
    s_merge=0.4,               # easier merge for same attractor under phase shifts
    # optional: make persistence assignment more stable
    p_in=12,
    online_max_cells=8192,     # avoid truncating the stored cell set for a large cycle
)

# Create an attractor
sim.run(T=300, transient=100)
attr = sim.results()

# Plotting
ax = fig.single()
phase.xy(x=attr["x"], y=attr["y"], ax=ax)
basin_plot(results, ax=ax)

print("Globally stable limit cycle.")
print("Edges can go outside for high mu values but they should converge back.")

export.show()