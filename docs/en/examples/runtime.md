# Runtime utilities and diagnostics

## Overview

The runtime utilities in this suite exercise dynlib's low-level control hooks: `setup()` can drive a broad range of steppers, stop conditions, and event observers, while helper APIs reveal what the compiler built or what a trajectory is doing in real time. These scripts demonstrate how to sanity-check numerical accuracy, cancel simulations early, detect interesting transitions, and print the DSL equations so you can audit what is actually running inside `Sim`.

## Example scripts

### Stepper accuracy ranking

```python
--8<-- "examples/accuracy_demo.py"
```
Runs every registered ODE stepper against two analytic solutions (exponential decay and the harmonic oscillator) with a tiny constant `dt`. Each run computes an RMS relative error so you can sort steppers by accuracy, log failures, and compare how each integrator behaves when your time step is near its stability limit.

### Early exit macros

```python
--8<-- "examples/early_exit_demo.py"
```
Explores the DSL `stop` macros such as `cross_up`, `in_interval`, and `decreasing`. For several logistic-map variants the script runs up to 100 steps but halts as soon as the condition becomes true, printing the exit reason, executed steps, and the values just before/after the trigger. It highlights how stopping early keeps the run deterministic while saving time.

### Transition detection via events

```python
--8<-- "examples/detect_transition.py"
```
Adds an event that fires when the Lorenz `x` variable crosses zero from below. The run records the timestamp of each `detect` event, draws vertical lines on the `x(t)` series plot, and demonstrates how to inspect `res.event("detect")` to drive annotations or downstream logic in chaotic regimes.

### Printing compiled equations

```python
--8<-- "examples/print_equations_demo.py"
```
Builds the Henon map and Lorenz system with `jit=False` and simply prints the RHS and Jacobian tables that the compiler derives from Toml or builtin specs. This is handy when you need to confirm that `dynlib` sees the model you expect before running expensive simulations.
