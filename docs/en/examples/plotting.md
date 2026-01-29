# Plotting examples

## Overview

These demos capture the plotting helpers that sit on top of dynlib's simulation stack. They show how to build time series, phase portraits, return maps, vector fields, histograms, and animations with consistent styling and the `dynlib.plot` API.

## Time series and phase portraits

### Logistic map diagnostics

```python
--8<-- "examples/logistic_map.py"
```
Builds the builtin logistic map, runs with a transient, and then plots the time series, return map, and cobweb diagram. The script also prints the fixed points found by `sim.model.fixed_points(seeds=...)` so you can compare analytic predictions with the numerically discovered attractors.

### Van der Pol oscillator

```python
--8<-- "examples/vanderpol.py"
```
Runs the stiff `builtin://ode/vanderpol` model with the `tr-bdf2a` stepper, times the run with `dynlib.utils.Timer`, and plots both the time series and phase portrait. It demonstrates how to tune `dt`/`max_steps` for long runs while keeping the figure creation simple.

## Plot helper gallery

### Basic plotting primitives

```python
--8<-- "examples/plot/plot_demo.py"
```
Presents one figure with six subplots to showcase `series.stem`, `series.step`, `utils.hist`, `phase.xy`, and the `series.plot` styles `map` and `mixed`. This is a quick reference for how each helper handles line, stem, histogram, and map-style data.

### Theme presets

```python
--8<-- "examples/plot/themes_demo.py"
```
Iterates through the `notebook`, `paper`, `talk`, `dark`, and `mono` themes, renders a sample figure, and saves PNGs so you can inspect how each preset affects colors, gridlines, and typography.

### Faceted histograms

```python
--8<-- "examples/plot/facet.py"
```
Uses `plot.facet.wrap` to build a grid of histogram panels for multiple categories. Each axis receives its own data slice plus titles/labels so you can explore the data-distribution helper without manually creating `plt.subplots`.

### Vector field helper

```python
--8<-- "examples/plot/vectorfield_demo.py"
```
Shows `plot.vectorfield` with a spiral model, getter handles that update parameters (`a`, `b`), nullclines, and a custom color mapping for speed. It demonstrates how the returned handle can redraw the vector field when you tweak parameters interactively.

## Animated & swept vector fields

### Vector field animation demo

```python
--8<-- "examples/plot/vectorfield_animate_demo.py"
```
Uses `plot.vectorfield_animate` to step the `a` parameter through several profiles while normalizing and color-mapping the speed. The animation can preview in notebooks or be saved later with Matplotlib writers.

### Dense vector field animation

```python
--8<-- "examples/plot/vectorfield_animation.py"
```
Builds a sin/cos-based vector field and sweeps the frequency `k` across 300 frames, giving you an extensive example of how to keep the `anim` handle alive so it doesn't get garbage-collected before you save.

### High-dimensional slices

```python
--8<-- "examples/plot/vectorfield_highdim_demo.py"
```
Projects the 3D Lorenz vector field onto two different 2D slices (`x/y` and `y/z`) with adjustable fixed-state values. The demo lets you click either panel to trigger a trajectory trace and shows how to reuse the returned handle to update the slice without recreating axes.

### Vector field sweeps

```python
--8<-- "examples/plot/vectorfield_sweep_demo.py"
```
Sweeps the spiral model over a list of `a` values, arranges the resulting fields in a grid, and shows how `plot.vectorfield_sweep` automatically handles layout, nullclines, and annotations so you can compare parameter regimes at a glance.
