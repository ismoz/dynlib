"""Demonstration of TrajectoryAnalyzer helpers on a damped oscillator."""

from __future__ import annotations
import numpy as np
from dynlib import setup


MODEL = """
inline:
[model]
type="ode"
label="Damped oscillator with aux energy"

[states]
x=1.0
v=0.0

[params]
omega=2.0
zeta=0.15

[equations.rhs]
x = "v"
v = "-2*zeta*omega*v - omega**2 * x"

[aux]
energy = "0.5 * (v**2 + (omega * x)**2)"
"""


def _round_dict(values: dict[str, float], digits: int = 3) -> dict[str, float]:
    return {k: round(float(v), digits) for k, v in values.items()}


def main() -> None:
    sim = setup(MODEL, stepper="rk4", jit=False, disk_cache=False)
    sim.config(dt=0.01, max_steps=150_000)
    sim.run(T=15.0, record_vars=["x", "v", "aux.energy"])
    res = sim.results()

    print("\n--- Single-variable analysis (x) ---")
    x = res.analyze("x")
    print("summary:", _round_dict(x.summary()))
    t_peak, peak = x.argmax()
    print(f"peak x at t={t_peak:.3f}: {peak:.3f}")
    crossings = x.zero_crossings(direction="up")
    print("first three zero up-crossings (s):", np.round(crossings[:3], 3))
    print(f"time with x >= 0.5: {x.time_above(0.5):.3f}s")
    print(f"time with x < -0.5: {x.time_below(-0.5):.3f}s")

    print("\n--- Multi-variable analysis (x, v) ---")
    mv = res.analyze(["x", "v"])
    print("argmax by var:", {k: (round(t, 3), round(v, 3)) for k, (t, v) in mv.argmax().items()})
    print("range by var:", _round_dict(mv.range()))
    print("mean by var:", _round_dict(mv.mean()))

    print("\n--- Aux variable analysis (energy) ---")
    energy = res.analyze("energy")  # auto-detected aux without explicit prefix
    print("energy min/max:", (round(energy.min(), 3), round(energy.max(), 3)))
    print("energy range:", round(energy.range(), 3))
    below_times = energy.crossing_times(threshold=0.1, direction="down")
    print("first drop below 0.1:", round(below_times[0], 3) if len(below_times) else "never")

    print("\n--- Automatic variable selection ---")
    auto = res.analyze()
    print("recorded vars picked by res.analyze():", auto.vars)


if __name__ == "__main__":
    main()
