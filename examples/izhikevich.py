from dynlib import setup
from dynlib.plot import series, export, fig


sim = setup("builtin://ode/izhikevich", 
            stepper="rk4", 
            jit=False, 
            dtype="float32")

I0, T0 = 0.0, 300.0
I1, T1 = 5.0, 600.0
I2, T2 = 10.0, 900.0
I3, T3 = 15.0, 1200.0
I4, T4 = 10.0, 1500.0

sim.config(dt=0.01)
sim.assign(I=I0)
sim.run(T=T0, transient=50.0)
sim.assign(I=I1)
sim.run(T=T1, resume=True)
sim.assign(I=I2)
sim.run(T=T2, resume=True)
sim.assign(I=I3)
sim.run(T=T3, resume=True)
sim.apply_preset("bursting")
sim.assign(I=I4)
sim.run(T=T4, resume=True)

res = sim.results()

ax = fig.single(size=(8, 4))
series.plot(x=res.t, y=res["v"],
            ax=ax,
            ylim=(-80, 50),
            title="Membrane Potential (v)",
            bands=[(0,T0,"b"), (T0,T1,"m"), (T1,T2,"g"), (T2,T3,"r"), (T3,T4,"c")],
            vlines=[(0, "I=0"), (T0, "I=5"), (T1, "I=10"), (T2, "I=15"), (T3, "I=10")],
            vlines_color="red")    
export.show()

print("SNAPSHOTS: ", sim.list_snapshots())
print("Snapshot Parameter Vector: ", sim.param_vector(source="snapshot"))
print("Snapshot Parameter Dictionary: ", sim.param_dict(source="snapshot"))