"""List accuracy of adaptive ODE steppers and their change with tolerance"""

from dynlib import setup, list_steppers
import math
import os

a = 1.0
x0 = 10.0
T = 1.0

def actual(t):
    return x0 * math.exp(-a * t)

tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
MAX_STEPS = 100_000_000

adaptive_steppers = list_steppers(kind="ode", jit_capable=True, time_control="adaptive")

def run_tests(steppers, results, failed):
    for stepper in steppers:
        results[stepper] = []
        failed[stepper] = []
        try:
            sim = setup("builtin://ode/expdecay.toml", stepper=stepper, jit=True, disk_cache=False)
            sim.assign(a=a, x=x0)
            for tol in tols:
                rtol = tol
                atol = tol * 1e-3
                sim.config(rtol=rtol, atol=atol)
                try:
                    sim.run(T=T, record=False, max_steps=MAX_STEPS)
                    error = abs(sim.state("x") - actual(T))
                    step_count = sim._session_state.step_count
                    results[stepper].append((tol, error, step_count))
                    sim.reset()
                except Exception as e:
                    failed[stepper].append((tol, str(e)))
                    sim.reset()
        except Exception as e:
            # If setup fails, mark all as failed
            for tol in tols:
                failed[stepper].append((tol, str(e)))

def write_results(f, steppers, results, failed, title):
    f.write(f"{title}\n")
    f.write("=" * len(title) + "\n\n")
    for stepper in steppers:
        f.write(f"Stepper: {stepper}\n")
        f.write(f"{'step_count':>10} {'tol':>10} {'error':>15}\n")
        f.write("-" * 35 + "\n")
        for tol, error, step_count in results[stepper]:
            f.write(f"{step_count:>10} {tol:>10.2e} {error:>15.4e}\n")
        if failed[stepper]:
            f.write("Failed tols:\n")
            for tol, err in failed[stepper]:
                f.write(f"  tol={tol:.2e}: {err}\n")
        f.write("\n")

    # Per tol ranking
    for tol in tols:
        successful = []
        failed_tol = []
        for stepper in steppers:
            res = [r for r in results[stepper] if r[0] == tol]
            if res:
                successful.append((stepper, res[0][1]))  # stepper, error
            else:
                fail = [f for f in failed[stepper] if f[0] == tol]
                if fail:
                    failed_tol.append((stepper, fail[0][1]))
        # Sort successful by error ascending
        successful.sort(key=lambda x: x[1])
        f.write(f"tol={tol:.2e}\n")
        for i, (stepper, error) in enumerate(successful, 1):
            f.write(f"{i}) {stepper} = {error:.4e}\n")
        if failed_tol:
            f.write("Failed steppers:\n")
            for stepper, err in failed_tol:
                f.write(f"  {stepper}: {err}\n")
        f.write("\n")

results_adaptive = {}
failed_adaptive = {}
run_tests(adaptive_steppers, results_adaptive, failed_adaptive)

# Write to file
fname = os.path.splitext(__file__)[0] + ".txt"
with open(fname, "w") as f:
    write_results(f, adaptive_steppers, results_adaptive, failed_adaptive, "Adaptive steppers tolerance sweep")