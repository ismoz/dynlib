# Izhikevich Neuron Example

## Overview

This example walks through simulating the Izhikevich spiking neuron and visualizing its membrane potential as we step through different input currents, including a burst-inducing preset. It highlights dynlib's ability to resume simulations, toggle presets, and annotate plots so that you can clearly see how changing drive currents alters the firing pattern.

## Key Concepts

- **Current stepping**: run a single simulation while updating the injected current and resuming state, capturing both transients and new attractors.
- **Apply presets**: call `sim.apply_preset("bursting")` to switch the intrinsic parameters (`c`, `d`, etc.) that reshape the neuron's excitability.
- **Snapshot tooling**: use `sim.list_snapshots()` plus `sim.param_vector`/`param_dict` with `source="snapshot"` to inspect the stored configuration after running through different regimes.
- **Annotated time series**: `series.plot` supports `vbands` and `vlines` so you can label regimes where the injected current changes.

## The Izhikevich Model

The builtin model packages the canonical 2D system:

$$
dv = 0.04\,v^2 + 5.0\,v + 140.0 - u + I
$$
$$
du = a\,(b\,v - u)
$$

with a hard reset event when `v >= v_th` (default `30.0`), setting `v = c` and `u = u + d`. The default parameters (`a=0.02`, `b=0.2`, `c=-65`, `d=8`, `I=10`) reproduce regular spiking; the `bursting` preset reduces `c`/`d` to produce fast bursts that emerge as the current ramps up.

## Basic Example

```python
from dynlib import setup
from dynlib.plot import series, fig, export

sim = setup("builtin://ode/izhikevich", stepper="rk4", jit=True, dtype="float32")

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
series.plot(
    x=res.t,
    y=res["v"],
    ax=ax,
    ylim=(-80, 50),
    title="Membrane Potential (v)",
    vbands=[(0, T0, "b"), (T0, T1, "m"), (T1, T2, "g"), (T2, T3, "r"), (T3, T4, "c")],
    vlines=[(0, "I=0"), (T0, "I=5"), (T1, "I=10"), (T2, "I=15"), (T3, "I=10")],
    vlines_color="red",
)
export.show()

print("SNAPSHOTS: ", sim.list_snapshots())
print("Snapshot Parameter Vector: ", sim.param_vector(source="snapshot"))
print("Snapshot Parameter Dictionary: ", sim.param_dict(source="snapshot"))
```

`series.plot` overlays vertical bands/lines to flag each current step, while the run/profile calls demonstrate how `resume=True` keeps the stateful simulation continuous as current changes. After plotting the membrane potential, the snapshot helpers summarize the stored parameter sets for later analysis.

## Complete Examples in Repository

### 1. **Izhikevich Neuron**

```python
--8<-- "examples/izhikevich.py"
```

- Full driving-current sequence with five regimes and the `bursting` preset applied just before the largest steps.
- Demonstrates how to configure `vbands`/`vlines` on a `series.plot` trace.
- Prints the snapshots, parameter vector, and dictionary so you can audit the recorded presets.

### 2. **Izhikevich Benchmark**

```python
--8<-- "examples/izhikevich_benchmark.py"
```

- Uses `dynlib.build`/`Sim` with inline DSL to recreate the same ODE and reset event.
- Measures runtime differences between JIT and non-JIT builds using `dynlib.utils.Timer`.
- Runs `sim.run` with `cap_rec=10000` and `record=False` to isolate performance rather than memory usage.
