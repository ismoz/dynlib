from dynlib import setup
from dynlib.plot import series, phase, export

model = '''
inline:
[model]
type="map"
name="Logistic Map"

[states]
x=0.1

[params]
r=4.0

[equations.rhs]
x = "r * x * (1 - x)"
'''

sim = setup(model, stepper="map", jit=True, disk_cache=True)
sim.run(N=192, transient=40)
sim.run(resume=True, N=400)

res=sim.results()
series.plot(x=res.step, y=res["x"], title="Logistic Map", 
            xlabel="Iteration", 
            ylabel="x",
            style="line",
            lw=1.2)

phase.return_map(x=res["x"], title="Return Map", 
                 xlabel="x[n]", 
                 ylabel="x[n+1]",
                 ms=1.2)

export.show()