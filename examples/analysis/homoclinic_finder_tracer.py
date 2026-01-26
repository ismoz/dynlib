"""
Example: Homoclinic finder/tracer (manifold finder/tracer) on an inline model.

Workflow:
1) Use homoclinic_finder to locate a parameter value with a homoclinic orbit.
2) Use homoclinic_tracer to trace and plot the orbit at the found parameter.

This example demonstrates the simplified API with:
- List inputs for equilibrium guess (no need for np.array)
- window parameter as list of tuples
- preset parameter for common configurations ("fast", "default", "precise")

Important: Matching Finder and Tracer Settings
-----------------------------------------------
The tracer should use the same section/radius settings as the finder:
- r_sec (section radius) and t_min_event (minimum event time) should match
  the finder's configuration to ensure consistent return-section logic.
- If tracing fails with status 'no_cross', try increasing t_max or loosening
  the finder's gap_tol/x_tol tolerances.

Advanced Configuration
----------------------
For fine-grained control, you can use the full configuration classes:

    from dynlib.analysis import (
        HomoclinicRK45Config,      # RK45 integrator settings (dt0, atol, rtol, etc.)
        HomoclinicBranchConfig,    # Branch tracing settings (eps, t_max, r_sec, etc.)
        HomoclinicFinderConfig,    # Finder config
    )

    # Example: Custom RK45 and branch configuration
    rk_cfg = HomoclinicRK45Config(dt0=1e-4, atol=1e-12, rtol=1e-9, max_steps=5_000_000)
    branch_cfg = HomoclinicBranchConfig(
        t_max=2000.0,
        r_blow=200.0,
        r_sec=1e-2,
        window_min=[-3.0, -3.0],
        window_max=[3.0, 3.0],
        rk=rk_cfg,
    )
    finder_cfg = HomoclinicFinderConfig(
        trace=branch_cfg,
        scan_n=81,
        max_bisect=60,
        gap_tol=1e-6,
        x_tol=1e-4,
    )

    result = homoclinic_finder(sim, ..., cfg=finder_cfg)

You can also use `trace_cfg` for unified branch configuration:

    result = homoclinic_finder(
        sim, ...,
        trace_cfg=branch_cfg,  # applies to the unstable branch
        gap_tol=1e-6,
    )
"""
from dynlib import setup
from dynlib.analysis import (
    homoclinic_finder,
    homoclinic_tracer,
)
from dynlib.plot import theme, fig, export, manifold


MODEL = """
inline:
[model]
type = "ode"
label = "homoclinic-demo"

[sim]
t0 = 0.0
T = 10.0
dt = 0.01

[states]
x = 0.0
y = 0.0

[params]
mu = -0.86

[equations.rhs]
x = "y"
y = "mu*y + x - x*x + x*y"

[equations.jacobian]
expr = [
    ["0", "1"],
    ["1 - 2*x + y", "mu + x"]
]
"""


sim = setup(MODEL, jit=True, disk_cache=False)

# ============================================================================
# SIMPLIFIED API: Using preset and flattened kwargs
# ============================================================================

# Simple case: just specify essential parameters, uses "default" preset
result = homoclinic_finder(
    sim,
    param="mu",
    param_min=-1.2,
    param_max=-0.6,
    param_init=-1.0,
    eq_guess=[0.0, 0.0],  # Lists work! No need for np.array
    # Simplified kwargs:
    preset="default",  # or "fast", "precise"
    window=[(-3, 3), (-3, 3)],  # state-space bounds as (min, max) tuples
    scan_n=61,
    max_bisect=60,
    gap_tol=1e-4,
    x_tol=1e-4,
    t_max=2000.0,
    r_blow=200.0,
)

print("success:", result.success)
print("mu_found:", result.param_found)
print("fail:", result.info["fail"])
print("best_by_abs_gap:", result.info["best_by_abs_gap"])
print("best_by_q:", result.info["best_by_q"])
if result.success and result.miss is not None:
    print(
        "gap_found:",
        result.info["gap_found"],
        "q_found:",
        result.info["q_found"],
        "t_cross:",
        result.info["t_cross"],
    )

if (not result.success) or result.miss is None:
    raise SystemExit("Homoclinic finder did not converge; see info above.")

# Trace the orbit using the simplified API
trace = homoclinic_tracer(
    sim,
    param="mu",
    param_value=result.param_found,
    eq=result.miss.eq,          # Can also use lists: [0.0, 0.0]
    sign_u=result.miss.sign_u,
    preset="default",  # Use same preset as finder for consistency
    window=[(-3, 3), (-3, 3)],
    t_max=2000.0,
    r_blow=200.0,
    r_sec=result.miss.r_sec,
    t_min_event=result.miss.t_min,
)

print("trace success: ", trace.success)
print("status: ", trace.meta.status)
print("event: ", None if trace.meta.event is None else trace.meta.event.kind)

if not trace.success:
    print(f"\nTracing failed with status '{trace.meta.status}'.")
    print("Consider increasing t_max or loosening gap_tol/x_tol.")

# Only plot if we have valid trajectory data
traj = trace.branch_pos[0] if trace.branch_pos and len(trace.branch_pos) > 0 else None
if traj is not None and len(traj) > 1:
    theme.use("paper")
    ax = fig.single()

    title_suffix = "" if trace.success else " (partial, no return-section hit)"

    manifold(
        result=trace,
        ax=ax,
        color="C0",
        lw=1.2,
        label="Homoclinic excursion" + (" (closed)" if trace.success else " (incomplete)"),
        xlim=(-1.6, 1.6),
        ylim=(-1.0, 1.0),
        xlabel="x",
        ylabel="y",
        title=f"Homoclinic orbit at mu={result.param_found:.10f}{title_suffix}",
        aspect="equal",
    )
    ax.scatter([result.miss.eq[0]], [result.miss.eq[1]], s=60, label="Equilibrium")
    ax.scatter([trace.meta.x_cross[0]], [trace.meta.x_cross[1]], s=30, label="Section cross")
    ax.legend(loc="best")

    """
    Section cross: This is the point where the unstable manifold trajectory
    re-enters the return section defined by r_sec around the equilibrium.
    When a homoclinic orbit exists, the crossing should align with the unstable
    direction, making the signed return value g close to zero.
    """

    export.show()
else:
    print("\nNo trajectory data to plot.")
