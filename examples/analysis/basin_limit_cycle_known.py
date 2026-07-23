from dynlib import setup
from dynlib.analysis import basin_known, basin_axis, ReferenceRun
from dynlib.plot import export, basin_plot, fig, phase

def main():
    # Energy Template Oscillator (ETO) with Circular L Curve
    sim = setup("builtin://ode/eto-circular", jit=True, disk_cache=True, stepper="rk4")
    # Globally stable limit cycle parameters
    sim.assign(mu=0.8, a=2.0)

    results = basin_known(
        sim,
        attractors=[
            ReferenceRun(name="Limit Cycle Attractor", ic=[1.0, 0.0]),
        ],
        ic={
            "x": basin_axis(-3, 3, n=128),
            "y": basin_axis(-3, 3, n=128),
        },
        dt_obs=0.01,
        # run long enough to converge + see the cycle
        signature_samples=8000, # samples are in steps not time
        max_samples=4000,
        transient_samples=2000, 
        tolerance=0.05,      # 5% of attractor range
        min_match_ratio=0.8,  # 80% of points must match
        escape_bounds=[(-5.0, 5.0), (-5.0, 5.0)],  # Wide bounds for escape detection
        b_max=1e6, # Blowup threshold / None means literal NaN/Inf
    )

    # Create an attractor
    sim.run(T=300, transient=100)
    attr = sim.results()

    # Plotting
    ax = fig.single()
    phase.xy(x=attr["x"], y=attr["y"], ax=ax)
    basin_plot(results, ax=ax)

    export.show()

if __name__ == "__main__":
    main()