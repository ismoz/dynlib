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
sim.config(dt=0.01)

I0, T0 = 0.0, 100.0
I1, T1 = 5.0, 400.0
I2, T2 = 10.0, 700.0
I3, T3 = 15.0, 1000.0

sim.assign(I=I0)
sim.run(T=T0, transient=50.0)
sim.assign(I=I1)
sim.run(T=T1, resume=True)
sim.assign(I=I2)
sim.run(T=T2, resume=True)
sim.assign(I=I3)
sim.run(T=T3, resume=True)

res = sim.results()

series.plot(x=res.t, y=res["v"],
            ylim=(-80, 50),
            title="Membrane Potential (v)",
            bands=[(0,T0,"b"), (T0,T1,"m"), (T1,T2,"g"), (T2,T3,"r")],
            vlines=[(0, "I=0"), (T0, "I=5"), (T1, "I=10"), (T2, "I=15")],
            vlines_color="red")    
export.show()

print(sim.list_snapshots())