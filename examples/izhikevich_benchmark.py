from dynlib import build, Sim
from dynlib.plot import series, export
from dynlib.utils import Timer


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

T = 1.0e6

with Timer("build model"):
    model = build(DSL, stepper="euler", jit=False, dtype="float32")

with Timer("build model jit"):
    model_jit = build(DSL, stepper="euler", jit=True, dtype="float32")

sim = Sim(model)
sim_jit = Sim(model_jit)

with Timer("jit False"):
    sim.run(T=T, dt=0.01, cap_rec=10000, record=False)

with Timer("jit True"):
    sim_jit.run(T=T, dt=0.01, cap_rec=10000, record=False)
