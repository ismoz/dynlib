from dynlib import setup
from dynlib.plot import series, export, fig, phase
from dynlib.utils import Timer

stepper = "tr-bdf2a"
mu = 1000.0


def main():
    sim = setup("builtin://ode/vanderpol", 
                stepper=stepper, 
                jit=True,
                disk_cache=False)

    sim.assign(mu=mu)
    sim.config(dt=5e-4, max_steps=6_500_000)
    with Timer("run simulation"):
        sim.run(T=3000.0)
    res = sim.results()

    series.line(x=res.t, y=res["x"],
                title=f"Van der Pol Oscillator (μ={mu})",
                xlabel="Time",
                ylabel="x",
                ylim=(-3, 3),
                )

    phase.xy(x=res["x"], y=res["y"])
    export.show()


if __name__ == "__main__":
    main()