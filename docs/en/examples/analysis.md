# Analysis examples

## Overview

The `examples/analysis/` folder showcases how to apply dynlib's diagnostic and geometric toolkits. You will find scripts that (1) map basins of attraction with persistence or targeted classifiers, (2) compute Lyapunov exponents and spectra, (3) trace stable/unstable manifolds and homoclinic/heteroclinic orbits, and (4) sweep parameters while atomically analyzing every recorded trajectory.

## Basin mapping and classification

### Henon map basin discovery

```python
--8<-- "examples/analysis/basin_henon_auto.py"
```
Runs `basin_auto` on the built-in Henon map with a 512×512 grid, PCR-BM detection across both `x`/`y`, and timer instrumentation. The script prints attractor metadata, summarizes the result, and then hands the `BasinResult` to `basin_plot` so you can see how the automatic fingerprinting divides the plane.

### Basin of known attractors

```python
--8<-- "examples/analysis/basin_henon_known.py"
```
Demonstrates `basin_known` + `ReferenceRun` to classify the Henon attractor during refinement. It feeds pre-computed attractor fingerprints into the known-attractor library, runs on a 512×512 grid, and shows how `signature_samples`, `tolerance`, and `min_match_ratio` influence the assignment.

### Limit-cycle basins

```python
--8<-- "examples/analysis/basin_limit_cycle_auto.py"
```

```python
--8<-- "examples/analysis/basin_limit_cycle_known.py"
```
The first script uses `basin_auto` on the Energy Template Oscillator to capture a globally stable limit cycle via recurrence detection and post-detection persistence. The second repeats the experiment with `basin_known`/`ReferenceRun`, showcasing how to compare phase-space points with a reference trajectory while still refining the grid.

### Refinement benchmark

```python
--8<-- "examples/analysis/basin_refine_benchmark.py"
```
Bi-directional benchmark toggles `refine` on `basin_known` across grid sizes from 64×64 up to 1024×1024, printing wall-clock time and label agreement so you can see the performance payoff before enabling coarse-to-fine passes.

## Lyapunov exponents and spectral sweeps

### Logistic-map Lyapunov demo

```python
--8<-- "examples/analysis/lyapunov_logistic_map_demo.py"
```
Runs the logistic map at `r=4`, attaches both the MLE and spectrum runtime observers, and plots the trajectory, Lyapunov convergence, and final spectrum trace while printing the error against `ln(2)`. The script also prints a quick scan over several `r` values to classify stability vs. chaos.

### Lorenz-system Lyapunov demo

```python
--8<-- "examples/analysis/lyapunov_lorenz_demo.py"
```
Repeats the approach for the continuous Lorenz attractor. It computes an `rk4` run with `lyapunov_mle_observer` and another with `lyapunov_spectrum_observer`, prints the error to the reference `(0.9056, 0.0, -14.57)` spectrum, and draws the attractor plus convergence traces.

### Parameter sweeps with Lyapunov statistics

```python
--8<-- "examples/analysis/lyapunov_sweep_mle_demo.py"
```
uses `sweep.lyapunov_mle_sweep` for the logistic map, pairing the bifurcation extraction with an MLE sweep over 1000 `r` values so you can see chaos onset from λ crossing zero.

```python
--8<-- "examples/analysis/lyapunov_sweep_spectrum_demo.py"
```
sweeps the Lorenz `rho` parameter with `sweep.lyapunov_spectrum_sweep`, overlays the z-plane bifurcation, and plots all three Lyapunov exponents across the range.

## Manifolds and orbit finders

### Manifold tracing

```python
--8<-- "examples/analysis/manifold_henon.py"
```
traces the 1D stable and unstable manifolds of the Henon map fixed point using `trace_manifold_1d_map`, shows both branches on a single plot, and overlays the fixed point.

```python
--8<-- "examples/analysis/manifold_ode_saddle.py"
```
builds a simple saddle ODE, traces the manifolds via `trace_manifold_1d_ode`, compares against analytic formulas (`x=0`, `y=x²/3`), and reports eigenvalue errors.

### Homoclinic & heteroclinic finders

```python
--8<-- "examples/analysis/homoclinic_finder_tracer.py"
```

```python
--8<-- "examples/analysis/heteroclinic_finder_tracer.py"
```
Each script runs the corresponding finder/tracer pair on an inline 2D model. They demonstrate the simplified API (`preset`, tuple-style `window`, optional `x_tol`/`gap_tol`), print finder diagnostics (gap size, status), and plot the traced orbit in the return section so you can visualize whether the unstable manifold connects back.

## Sweeps and trajectory helpers

### Parameter sweeps

```python
--8<-- "examples/analysis/parameter_sweep.py"
```
Uses `analysis.sweep.traj_sweep` to run the toy ODE `x' = a x` for several `a` values, stacks the trajectories with `series.multi`, and accesses each `SweepRun` to print the final state and step details.

### Trajectory analyzer

```python
--8<-- "examples/analysis/trajectory_analysis_demo.py"
```
Runs a damped oscillator with an auxiliary energy variable, then calls `res.analyze()` to show `summary()`, `argmax()`, `zero_crossings()`, and `time_above/Below` for single- and multi-variable selections. It also inspects the automatically selected auxiliary variables to demonstrate how `TrajectoryAnalyzer` finds recorded names.
