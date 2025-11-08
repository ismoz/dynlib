from dynlib import build, Sim
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

model = build(DSL, stepper_name="euler", jit=False, model_dtype="float32")
sim = Sim(model)
sim.run(t_end=600.0, dt=0.01, cap_rec=10000)

res = sim.results()

series.plot(x=res.t, y=res["v"], title="Membrane Potential (v)")
export.show()