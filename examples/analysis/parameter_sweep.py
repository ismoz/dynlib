import numpy as np
from dynlib import setup
from dynlib.plot import series, export
from dynlib.analysis import sweep


DSL = """
inline:
[model]
type="ode"

[states]
x=10.0

[params]
a=-1.0

[equations.rhs]
x = "a * x"
"""

sim = setup(DSL, 
            jit=True,
            disk_cache=False)

values = np.arange(-5.0, -1.0, 1.0)
res=sweep.traj_sweep(sim, param="a", record_vars=["x"], values=values, T=5)

# Plot all trajectories at once (stacked access)
series.multi(x=res.t, y=res["x"], legend=False)
export.show()

# Access individual runs for custom processing
print(f"\nSweep has {len(res.runs)} runs:")
for run in res.runs:
    final_x = run["x"][-1]
    print(f"  a={run.param_value:.1f}: x(T)={final_x:.6f}")

# Access a specific run
run = res.runs[2]  # Third run (a=-3.0)
print(f"\nRun details for a={run.param_value}:")
print(f"  Time points: {len(run.t)}")
print(f"  Initial x: {run['x'][0]:.6f}")
print(f"  Final x: {run['x'][-1]:.6f}")
