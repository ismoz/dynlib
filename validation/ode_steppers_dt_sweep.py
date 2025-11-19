"""List accuracy of ODE steppers and their change with dt"""

from dynlib import setup, list_steppers
import math
import os

a = 1.0
x0 = 10.0
T = 1.0

def actual(t):
    return x0 * math.exp(-a * t)

dts = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
MAX_STEPS = 100_000_000

fixed_steppers = list_steppers(kind="ode", jit_capable=True, time_control="fixed")
adaptive_steppers = list_steppers(kind="ode", jit_capable=True, time_control="adaptive")

def run_tests(steppers, results, failed):
    for stepper in steppers:
        results[stepper] = []
        failed[stepper] = []
        try:
            sim = setup("builtin://ode/expdecay.toml", stepper=stepper, jit=True, disk_cache=False)
            sim.assign(a=a, x=x0)
            for dt in dts:
                try:
                    sim.run(T=T, dt=dt, record=False, max_steps=MAX_STEPS)
                    error = abs(sim.state("x") - actual(T))
                    step_count = sim._session_state.step_count
                    results[stepper].append((dt, error, step_count))
                    sim.reset()
                except Exception as e:
                    failed[stepper].append((dt, str(e)))
                    sim.reset()
        except Exception as e:
            # If setup fails, mark all as failed
            for dt in dts:
                failed[stepper].append((dt, str(e)))

def write_results(f, steppers, results, failed, title):
    f.write(f"{title}\n")
    f.write("=" * len(title) + "\n\n")
    for stepper in steppers:
        f.write(f"Stepper: {stepper}\n")
        f.write(f"{'step_count':>10} {'dt':>10} {'error':>15}\n")
        f.write("-" * 35 + "\n")
        for dt, error, step_count in results[stepper]:
            f.write(f"{step_count:>10} {dt:>10.2e} {error:>15.4e}\n")
        if failed[stepper]:
            f.write("Failed dts:\n")
            for dt, err in failed[stepper]:
                f.write(f"  dt={dt:.2e}: {err}\n")
        f.write("\n")

    # Per dt ranking
    for dt in dts:
        successful = []
        failed_dt = []
        for stepper in steppers:
            res = [r for r in results[stepper] if r[0] == dt]
            if res:
                successful.append((stepper, res[0][1]))  # stepper, error
            else:
                fail = [f for f in failed[stepper] if f[0] == dt]
                if fail:
                    failed_dt.append((stepper, fail[0][1]))
        # Sort successful by error ascending
        successful.sort(key=lambda x: x[1])
        f.write(f"dt={dt:.2e}\n")
        for i, (stepper, error) in enumerate(successful, 1):
            f.write(f"{i}) {stepper} = {error:.4e}\n")
        if failed_dt:
            f.write("Failed steppers:\n")
            for stepper, err in failed_dt:
                f.write(f"  {stepper}: {err}\n")
        f.write("\n")

results_fixed = {}
failed_fixed = {}
run_tests(fixed_steppers, results_fixed, failed_fixed)

results_adaptive = {}
failed_adaptive = {}
run_tests(adaptive_steppers, results_adaptive, failed_adaptive)

# Write to file
fname = os.path.splitext(__file__)[0] + ".txt"
with open(fname, "w") as f:
    write_results(f, fixed_steppers, results_fixed, failed_fixed, "Fixed-step steppers")
    write_results(f, adaptive_steppers, results_adaptive, failed_adaptive, "Adaptive-step steppers")

