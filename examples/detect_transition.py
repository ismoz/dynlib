from dynlib import setup
from dynlib.plot import series, export

"""
This example demonstrate lagging mechanism to detect trasition 
from negative to positive value in one of the state variables 
of a chaotic system.
"""

lorenz = '''
inline:
[model]
type="ode"
name="lorenz"

[states]
x=0.1
y=0.1
z=0.1

[params]
sigma=10.0
rho=28.0
beta="8/3"

[equations.rhs]
x = "sigma * (y - x)"
y = "x * (rho - z) - y"
z = "x * y - beta * z"

[events.detect]
cond = "cross_up(x, 0)"
phase = "post"
log = ["t"]

'''

sim = setup(lorenz, stepper="rk4", jit=False)
sim.config(dt=0.01)
sim.run(T=50.0, transient=10.0)

res = sim.results()
ev = res.event("detect")

series.plot(x=res.t,
            y=res["x"],
            vlines=ev.t,
            xlabel='Time',
            ylabel='x',
            title='Lorenz system: x variable with transition detection')

export.show()