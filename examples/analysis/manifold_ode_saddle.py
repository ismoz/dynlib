"""
Example: Tracing stable and unstable manifolds of a 2D saddle point.

Model: x' = x, y' = -y + x^2
Equilibrium at origin (0, 0) with:
  - Unstable eigenvalue λ=+1 (eigenvector along x-axis)
  - Stable eigenvalue λ=-1 (eigenvector along y-axis)

Known analytic solutions:
  - Stable manifold: x = 0 (the y-axis)
  - Unstable manifold: y = x^2/3
"""
from dynlib import setup
from dynlib.analysis import trace_manifold_1d_ode
from dynlib.plot import manifold, theme, export, fig
import numpy as np

# Define the saddle model inline
model_toml = """
inline:
[model]
type = "ode"
name = "saddle2d"

[sim]
t0 = 0.0
T = 10.0
dt = 0.01

[states]
x = 0.1
y = 0.1

[equations.rhs]
x = "x"
y = "-y + x*x"
"""

# Setup simulation
sim = setup(model_toml, jit=True)
bounds = ((-2.5, 2.5), (-2.5, 2.5))

# Trace stable manifold (backward in time)
stable_result = trace_manifold_1d_ode(
    sim,
    fp={"x": 0.0, "y": 0.0},
    kind="stable",
    bounds=bounds,
    dt=0.01,
    max_time=50.0,
    resample_h=0.01,
)

# Trace unstable manifold (forward in time)
unstable_result = trace_manifold_1d_ode(
    sim,
    fp={"x": 0.0, "y": 0.0},
    kind="unstable",
    bounds=bounds,
    dt=0.01,
    max_time=50.0,
    resample_h=0.01,
)

# Validate against analytic solutions
print(f"Stable eigenvalue: {stable_result.eigenvalue:.4f} (expected: -1)")
print(f"Unstable eigenvalue: {unstable_result.eigenvalue:.4f} (expected: +1)")

# Check stable manifold (should be x=0)
stable_x_err = 0.0
for branch in stable_result.branches[0] + stable_result.branches[1]:
    if branch.size > 0:
        stable_x_err = max(stable_x_err, np.max(np.abs(branch[:, 0])))
print(f"Stable manifold max |x| error: {stable_x_err:.3e}")

# Check unstable manifold (should be y=x^2/3)
unstable_err = 0.0
for branch in unstable_result.branches[0] + unstable_result.branches[1]:
    if branch.shape[0] >= 2:
        x_vals = branch[:, 0]
        y_vals = branch[:, 1]
        y_analytic = x_vals**2 / 3.0
        unstable_err = max(unstable_err, np.max(np.abs(y_vals - y_analytic)))
print(f"Unstable manifold max |y - x²/3| error: {unstable_err:.3e}")

# Plot
theme.use("paper")
ax = fig.single(size=(6, 6))

# Stable manifold (blue)
manifold(
    result=stable_result,
    ax=ax,
    color="C0",
    label="W^s (stable)",
    lw=1.5,
)

# Unstable manifold (red)
manifold(
    result=unstable_result,
    ax=ax,
    color="C3",
    label="W^u (unstable)",
    lw=1.5,
)

# Analytic unstable manifold for comparison (dashed black)
x_analytic = np.linspace(bounds[0][0], bounds[0][1], 500)
y_analytic = x_analytic**2 / 3.0
ax.plot(x_analytic, y_analytic, "k--", lw=1.0, label="y = x²/3 (analytic)")

# Mark the equilibrium
ax.plot(0, 0, "ko", ms=8)

ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Stable/Unstable Manifolds: x'=x, y'=-y+x²")
ax.legend(loc="upper left")

export.show()
