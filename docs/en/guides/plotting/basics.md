# Plotting Basics

Dynlib's plotting module (`dynlib.plot`) provides a set of high-level helpers that wrap Matplotlib to align with dynlib's analysis workflows. These tools handle common plotting tasks for dynamical systems, such as time series, phase portraits, and manifolds, while ensuring consistent styling and layout. All helpers accept NumPy arrays, lists, pandas Series, or slices from `Results` objects, automatically converting them as needed.

This guide covers the basics of using these plotting helpers, from creating figures to customizing plots.

## Getting Started

To use dynlib's plotting tools, import the necessary modules:

```python
from dynlib.plot import fig, series, phase, utils
```

Here's a quick example that demonstrates creating a grid of subplots and plotting different types of data:

```python
# Create a 2x2 grid of subplots
axes = fig.grid(rows=2, cols=2, size=(10, 8))

# Plot a time series
series.plot(x=t, y=x_traj, label="x(t)", ax=axes[0, 0])

# Plot discrete data as stems
series.stem(x=k, y=impulse_response, ax=axes[0, 1])

# Plot a phase portrait with equilibrium points
phase.xy(x=x_traj, y=y_traj, equil=[(x_eq, y_eq)], ax=axes[1, 0])

# Display a 2D image with a colorbar
utils.image(Z, extent=[0, 10, 0, 1], colorbar=True, ax=axes[1, 1])
```

Dynlib's `fig` helpers manage figure creation and layout automatically, so you can focus on the data rather than Matplotlib's boilerplate.

## Creating Figures and Subplots

Dynlib provides convenient functions to create figures and subplots with consistent styling:

- `fig.single()`: Creates a single subplot.
- `fig.grid(rows=2, cols=2)`: Creates a grid of subplots.
- `fig.wrap(n=5, cols=3)`: Creates a grid that wraps a specified number of subplots into columns, hiding any unused axes.
- `fig.single3D()`: Creates a single 3D subplot.

These functions accept parameters like `title`, `size`, `scale`, `sharex`, and `sharey` for customization. They return Matplotlib axes objects that you can pass to plotting helpers.

For faceting plots over data categories, use `plot.facet.wrap(keys, cols=3)`, which yields axes and keys for iteration.

Example:

```python
# Create a single subplot
ax = fig.single(size=(8, 6))
series.plot(x=t, y=data, ax=ax)

# Create a grid for multiple plots
axes = fig.grid(rows=1, cols=3)
for i, dataset in enumerate(datasets):
    series.plot(x=t, y=dataset, ax=axes[i])
```

## Styling and Decorations

Dynlib's plotting helpers support consistent styling through presets and decorations.

### Style Presets

Style presets define how data is visualized (e.g., lines, markers, or both). Available presets include `"continuous"`, `"discrete"`, `"line"`, `"scatter"`, and others. You can pass a preset name or a custom style dictionary.

Example:

```python
# Use a preset
series.plot(x=t, y=data, style="continuous", ax=ax)

# Customize with overrides
series.plot(x=t, y=data, style={"ls": "--", "marker": "x"}, color="red", ax=ax)
```

### Decorations

Add vertical or horizontal lines and bands to highlight features:

- `vlines`: Vertical lines (e.g., `vlines=[5, (10, "threshold")]`)
- `hlines`: Horizontal lines
- `vbands`: Vertical bands
- `hbands`: Horizontal bands

Labels are positioned automatically and respect axis limits.

Example:

```python
series.plot(x=t, y=data, vlines=[(5, "start"), 10], hbands=[(0, 1, "region")], ax=ax)
```

### Axis Control

Control axis limits with `xlim`, `ylim`, and `zlim` (for 3D). Helpers automatically apply consistent labels, fonts, and rotations.

## Plotting Time Series

Use `series` helpers for time-based plots:

- `series.plot(x, y, ...)`: Standard line plot for continuous or discrete data.
- `series.stem(x, y, ...)`: Stem plot for discrete samples.
- `series.step(x, y, ...)`: Step plot for piecewise-constant data.
- `series.multi(data, ...)`: Plot multiple series at once.

These helpers support all styling and decoration options.

Examples:

```python
# Simple time series
series.plot(x=t, y=x_traj, label="Position", ax=ax)

# Multiple series
data = {"x": x_traj, "y": y_traj}
series.multi(data, styles={"x": "continuous", "y": "discrete"}, ax=ax)

# Stem plot for impulses
series.stem(x=k, y=impulse, ax=ax)
```

## Plotting Phase Portraits

Phase-space plots visualize relationships between state variables:

- `phase.xy(x, y, ...)`: 2D phase portrait.
- `phase.xyz(x, y, z, ...)`: 3D phase portrait.
- `phase.multi(x_list, y_list, ...)`: Multiple trajectories on one plot.
- `phase.return_map(x, step, ...)`: Return map for maps.

Mark equilibria with `equil` and customize labels.

Examples:

```python
# 2D phase portrait
phase.xy(x=x_traj, y=y_traj, equil=[(0, 0)], ax=ax)

# Return map
phase.return_map(x=trajectory, step=1, equil=[(fixed_point,)], ax=ax)
```

## Utility Plots

Additional helpers for common visualizations:

- `utils.hist(data, ...)`: Histogram of 1D data.
- `utils.image(data, ...)`: 2D image plot.

Both support styling and can include colorbars.

Example:

```python
# Histogram
utils.hist(data, bins=50, density=True, ax=ax)

# Image with colorbar
utils.image(matrix, extent=[0, 1, 0, 1], colorbar=True, ax=ax)
```

## Exporting and displaying figures

Import `export` from `dynlib.plot` whenever you need to present or save a figure. It re-exports Matplotlib's `savefig` helpers with dynlib-aware defaults, so you can pass entire axes grids or figure handles without extra boilerplate. Call `export.show()` once your plotting script is complete (handy in notebooks or scripts) or `export.savefig(fig_or_ax, "plots/my-fig", fmts=("png", "pdf"))` to write multiple formats. See the dedicated [Exporting plots guide](export.md) for details on format selection, metadata, and working with grid containers.

## Plotting Manifolds

For 1D manifolds, use `plot.manifold(segments, ...)` or pass `result` objects with branches. Specify components and styles for different groups.

Example:

```python
plot.manifold(result.branches, components=(0, 1), ax=ax)
```

## Tips and Best Practices

- Always use `ax=` when creating multi-panel figures to control where plots appear.
- Leverage style presets for quick styling, then override as needed.
- Decorations like lines and bands work across most helpers for consistent annotations.
- Helpers automatically handle data conversion, so mix NumPy arrays, lists, and dynlib results freely.
- For images with colorbars, access the colorbar via `ax._last_colorbar` for further customization.

This should get you started with dynlib's plotting tools. For more advanced features, check the API reference.
