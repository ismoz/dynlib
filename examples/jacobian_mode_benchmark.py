"""Benchmark adaptive steppers with and without analytic Jacobian on Van der Pol (jit=True).

Switches Jacobian handling via the stepper config `jacobian_mode` 
("internal" = Numerical Finite Difference , "external" = Analytic).
"""
from __future__ import annotations

import statistics
import time

from dynlib import setup


MODEL_URI = "builtin://ode/vanderpol"
T_END = 20.0
N_RUNS = 5

STEPPERS = ["bdf2", "bdf2a", "sdirk2", "tr-bdf2a"]

def make_sim(jacobian_mode: str, stepper: str = "tr-bdf2a"):
    sim = setup(
        MODEL_URI,
        stepper=stepper,
        jit=True,
        disk_cache=True,
    )
    sim.config(jacobian_mode=jacobian_mode)
    return sim


def time_run(sim, *, T: float, runs: int) -> list[float]:
    """Run the simulation multiple times and return elapsed times (seconds)."""
    # Warmup (compilation and workspace setup)
    sim.run(T=T, record=False)
    sim.reset()

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sim.run(T=T, record=False)
        times.append(time.perf_counter() - t0)
        sim.reset()
    return times


def summarize(label: str, samples: list[float]) -> str:
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return f"{label:<12} {mean*1000:8.3f} ms ± {stdev*1000:6.3f} (n={len(samples)})"


def main(stepper: str = "tr-bdf2a"):
    sim_fd = make_sim("internal", stepper=stepper)
    times_fd = time_run(sim_fd, T=T_END, runs=N_RUNS)

    sim_an = make_sim("external", stepper=stepper)
    times_an = time_run(sim_an, T=T_END, runs=N_RUNS)

    print(f"\n{stepper.upper()} Jacobian Benchmark (Van der Pol, jit=True)")
    print(f"Model={MODEL_URI}, T_end={T_END}, runs={N_RUNS}, recording disabled\n")
    print(summarize("FD", times_fd))
    print(summarize("Analytic", times_an))

    speedup = statistics.mean(times_fd) / statistics.mean(times_an)
    note = "✓ Analytic faster" if speedup > 1.0 else "✗ Analytic slower"
    print(f"\nSpeedup (FD / Analytic): {speedup:.3f}x   {note}")


if __name__ == "__main__":
    for stepper in STEPPERS:
        main(stepper=stepper)