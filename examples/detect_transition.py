from dynlib import setup
from dynlib.plot import series, export, theme

"""
This example demonstrate lagging mechanism to detect trasition 
from negative to positive value in one of the state variables 
of a chaotic system.
"""

detect_mod = '''
inline:
[mod]
name = "detect_transition"

[mod.add.events.detect]
cond = "cross_up(x, 0)"
phase = "post"
log = ["t"]
'''

sim = setup("builtin://ode/lorenz", stepper="rk4", jit=False, mods=[detect_mod])
sim.config(dt=0.01)
sim.run(T=50.0, transient=10.0)

res = sim.results()
ev = res.event("detect")

theme.use("paper")

series.plot(x=res.t,
            y=res["x"],
            vlines=ev.t,
            xlabel='Time',
            xlabel_fs=11, # Theme override
            ylabel='x',
            title='Lorenz system: x variable with transition detection')

export.show()