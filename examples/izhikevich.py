from dynlib import setup
from dynlib.plot import series, export


DSL = '''
inline:
[model]
type = "ode"

[states]
v = -65.0
u = -13.0

[params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0
I = 10.0

[equations]
expr = """
dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
du = a * (b * v - u)
"""

[events.reset]
cond = "v >= 30.0"
phase = "post"
action = """
v = c
u = u + d
"""
'''

sim = setup(DSL, stepper="euler", jit=False, dtype="float32")
sim.stepper_config(dt=0.01)
sim.assign(I=0.0)
sim.run(T=100.0, transient=50.0)
sim.assign(I=5.0)
sim.run(T=400.0, resume=True)
sim.assign(I=10.0)
sim.run(T=700.0, resume=True)
sim.assign(I=15.0)
sim.run(T=1000.0, resume=True)

res = sim.results()

series.plot(x=res.t, y=res["v"], 
            title="Membrane Potential (v)")
export.show()

print(sim.list_snapshots())