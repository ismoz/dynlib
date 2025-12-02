import numpy as np
from dynlib import setup
from dynlib.plot import series, export
from dynlib.analysis import sweep


DSL = """
inline:
[model]
type="ode"

[states]
x=10.0

[params]
a=-1.0

[equations.rhs]
x = "a * x"
"""

sim = setup(DSL, 
            jit=True,
            disk_cache=False)

values = np.arange(-5.0, -1.0, 1.0)
res=sweep.traj(sim, param="a", record_vars=["x"], values=values, T=5)

series.multi(x=res.t, series=res["x"], legend=False)
export.show()
