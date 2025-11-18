from dynlib import setup
from dynlib.plot import series, export, fig, phase


stepper = "bdf2"
mu = 1000.0

sim = setup("builtin://ode/vanderpol", 
            stepper=stepper, 
            jit=True)

sim.assign(mu=mu)
sim.config(dt=5e-5, max_steps=6_500_000)
sim.run(T=3000.0)
res = sim.results()

series.plot(x=res.t, y=res["x"],
            title=f"Van der Pol Oscillator (μ={mu})",
            xlabel="Time",
            ylabel="x",
            ylim=(-3, 3),
            )

phase.xy(x=res["x"], y=res["y"])
export.show()