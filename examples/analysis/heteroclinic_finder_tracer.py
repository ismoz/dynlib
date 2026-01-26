"""
Example: Heteroclinic finder/tracer (manifold finder/tracer) on an inline model.

Workflow:
1) Use heteroclinic_finder to locate a parameter value with a heteroclinic orbit.
2) Use heteroclinic_tracer to trace and plot the orbit at the found parameter.

This example demonstrates the simplified API with:
- List inputs for fixed points (no need for np.array)
- window parameter as list of tuples
- preset parameter for common configurations ("fast", "default", "precise")

Important: Matching Finder and Tracer Settings
-----------------------------------------------
The tracer's hit_radius should be chosen carefully based on the finder's accuracy:
- If finder reports gap_found ~ 1e-4, use hit_radius ~ 0.01 to 0.05
- If finder reports gap_found ~ 1e-6, use hit_radius ~ 0.001 to 0.01
- Rule of thumb: hit_radius ≈ 5-50x the finder's gap_found value
- Use the same preset for both finder and tracer for consistency

The finder and tracer may have slightly different numerical behavior, so the
tracer might not achieve the exact same precision as the finder's gap measurement.
If the tracer fails with status 'window' or gets close but doesn't hit the target,
either increase hit_radius or tighten the finder's gap_tol/x_tol tolerances.

Advanced Configuration
----------------------
For fine-grained control, you can use the full configuration classes:

    from dynlib.analysis import (
        HeteroclinicRK45Config,      # RK45 integrator settings (dt0, atol, rtol, etc.)
        HeteroclinicBranchConfig,    # Branch tracing settings (eps, t_max, window, etc.)
        HeteroclinicFinderConfig2D,  # Finder config for 2D systems
        HeteroclinicFinderConfigND,  # Finder config for N-dimensional systems
    )

    # Example: Custom RK45 and branch configuration
    rk_cfg = HeteroclinicRK45Config(dt0=1e-4, atol=1e-12, rtol=1e-9, max_steps=5_000_000)
    branch_cfg = HeteroclinicBranchConfig(
        t_max=500.0,
        r_blow=500.0,
        window_min=[-20.0, -20.0],
        window_max=[20.0, 20.0],
        rk=rk_cfg,
    )
    finder_cfg = HeteroclinicFinderConfigND(
        trace_u=branch_cfg,
        trace_s=branch_cfg,
        scan_n=121,
        max_bisect=80,
        gap_tol=1e-6,
    )

    result = heteroclinic_finder(sim, ..., cfg=finder_cfg)

You can also use `trace_cfg` for unified branch configuration:

    result = heteroclinic_finder(
        sim, ...,
        trace_cfg=branch_cfg,  # applies to both unstable and stable branches
        gap_tol=1e-6,
    )
"""
import numpy as np

from dynlib import setup
from dynlib.analysis import (
    heteroclinic_finder,
    heteroclinic_tracer,
)
from dynlib.plot import theme, fig, export, manifold


MODEL = """
inline:
[model]
type = "ode"
label = "heteroclinic-demo"

[sim]
t0 = 0.0
T = 10.0
dt = 0.01

[states]
u = 0.0
v = 0.0

[params]
c = 0.0
a = 0.3

[equations.rhs]
u = "v"
v = "-c*v - u*(1-u)*(u-a)"

[equations.jacobian]
expr = [
    ["0", "1"],
    ["3*u*u - 2*(1+a)*u + a", "-c"]
]
"""


sim = setup(MODEL, jit=True, disk_cache=False)

# NOTE: Stricter hit_radius requires stricter config values!
#       With 0.02 you will see a gap near fixed points.
hit_radius = 0.02

# ============================================================================
# SIMPLIFIED API: Using preset and flattened kwargs
# ============================================================================

# Simple case: just specify essential parameters, uses "default" preset
result = heteroclinic_finder(
    sim,
    param="c",
    param_min=-0.5,
    param_max=1.0,
    param_init=0.0,
    source_eq_guess=[0.0, 0.0],  # Lists work! No need for np.array
    target_eq_guess=[1.0, 0.0],
    # Simplified kwargs:
    preset="default",  # or "fast", "precise"
    window=[(-10, 10), (-10, 10)],  # state-space bounds as (min, max) tuples
    scan_n=61,
    max_bisect=60,
    gap_tol=1e-4,
    x_tol=1e-4,
    t_max=200.0,
    r_blow=200.0,
)

print("success:", result.success)
print("c_found:", result.param_found)
print("fail:", result.info["fail"])
print("sign_u, sign_s:", result.info["sign_u"], result.info["sign_s"])
print("best_by_abs_gap:", result.info["best_by_abs_gap"])
print("best_by_q:", result.info["best_by_q"])
if result.success and result.miss is not None:
    print(
        "gap_found:",
        result.info["gap_found"],
        "gap_tol_eff_found:",
        result.info["gap_tol_eff_found"],
        "q_found:",
        result.info["q_found"],
    )

if (not result.success) or result.miss is None:
    raise SystemExit("Heteroclinic finder did not converge; see info above.")

# Trace the orbit using the simplified API
# Note: hit_radius should be adjusted based on the accuracy of the finder's gap (gap_found).
# If the finder's gap is ~1e-4, a hit_radius of 1e-3 may be too strict for the tracer.
# Reasonable values: 0.01 to 0.05 for typical cases, or 5-10x the finder's gap_found.
trace = heteroclinic_tracer(
    sim,
    param="c",
    param_value=result.param_found,
    source_eq=result.miss.source_eq,  # Can also use lists: [0.0, 0.0]
    target_eq=result.miss.target_eq,
    sign_u=result.miss.sign_u,
    preset="default",  # Use same preset as finder for consistency
    window=[(-10, 10), (-10, 10)],
    t_max=200.0,
    hit_radius=hit_radius,  # Adjusted to be more realistic given gap_found ~ 2e-4
)

print("trace success: ", trace.success)
print("status: ", trace.meta.status)
print("event: ", None if trace.meta.event is None else trace.meta.event.kind)

if not trace.success:
    # Diagnose why tracing failed
    if trace.branch_pos and len(trace.branch_pos) > 0:
        traj = trace.branch_pos[0]
        dist_to_B = np.linalg.norm(traj[-1] - result.miss.target_eq)
        print(f"\nTracing failed with status '{trace.meta.status}'.")
        print(f"Trajectory reached distance {dist_to_B:.6f} from target_eq.")
        print(f"This exceeds hit_radius={hit_radius}.")
        print("Consider increasing hit_radius or tightening finder tolerances.")

if not trace.success:
    # Diagnose why tracing failed
    if trace.branch_pos and len(trace.branch_pos) > 0:
        traj = trace.branch_pos[0]
        dist_to_B = np.linalg.norm(traj[-1] - result.miss.target_eq)
        print(f"\nTracing failed with status '{trace.meta.status}'.")
        print(f"Trajectory reached distance {dist_to_B:.6f} from target_eq.")
        print(f"This exceeds hit_radius={hit_radius}.")
        print("Consider increasing hit_radius or tightening finder tolerances.")

# Only plot if we have valid trajectory data
if trace.branch_pos and len(trace.branch_pos) > 0:
    theme.use("paper")
    ax = fig.single()
    
    # Add title indicating whether full heteroclinic connection was achieved
    title_suffix = "" if trace.success else " (partial, did not reach B within hit_radius)"
    
    manifold(
        result=trace,
        ax=ax,
        color="C0",
        lw=1.2,
        label="Wu(A)" + (" → B" if trace.success else " (incomplete)"),
        xlim=(-0.5, 1.5),
        ylim=(-0.2, 0.4),
        xlabel="u",
        ylabel="v",
        title=f"Heteroclinic orbit at c={result.param_found:.10f}{title_suffix}",
        aspect="equal",
    )
    ax.scatter([result.miss.source_eq[0]], [result.miss.source_eq[1]], s=60, label="A")
    ax.scatter([result.miss.target_eq[0]], [result.miss.target_eq[1]], s=60, label="B")
    ax.scatter([result.miss.x_u_cross[0]], [result.miss.x_u_cross[1]], s=30, label="cross (Wu)")
    ax.scatter([result.miss.x_s_cross[0]], [result.miss.x_s_cross[1]], s=30, label="cross (Ws)")
    ax.legend(loc="best")
    
    """
    cross (Wu) (or x_u_cross): This is the point where the unstable manifold (Wu) of
    equilibrium point A crosses a section plane centered at equilibrium point B.

    cross (Ws) (or x_s_cross): This is the point where the stable manifold (Ws) of
    equilibrium point B crosses the same section plane.

    When a heteroclinic orbit exists, these two crossing points should be very close
    (ideally the same point), meaning the unstable manifold of A connects to the stable
    manifold of B. The algorithm searches for parameter values where this "miss distance"
    is minimized.
    """
    
    export.show()
else:
    print("\nNo trajectory data to plot.")
