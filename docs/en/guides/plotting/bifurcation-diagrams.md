# Bifurcation Diagrams

Bifurcation diagrams visualize how the long-term behavior of a dynamical system changes as a parameter is varied. Dynlib provides the `bifurcation_diagram()` function to create scatter-style plots of bifurcation data, optimized for the dense point clouds typical of bifurcation analysis.

## Basic Usage

The `bifurcation_diagram()` function accepts bifurcation data in two formats:

1. A `BifurcationExtractor`/`BifurcationResult` (usually returned by `SweepResult.bifurcation()`)
2. A tuple of `(parameter_values, state_values)` arrays

See the **Input Data Formats** section below for more on extractor helpers and raw arrays.

```python
from dynlib.plot import bifurcation_diagram, theme, fig, export

# Assuming you have bifurcation data from analysis
# result = sweep_result.bifurcation("x")  # returns an extractor (defaults to .all())

# Create the plot
ax = fig.single(size=(10, 6))
bifurcation_diagram(
    result,  # BifurcationResult or (p, y) tuple
    xlabel="r",
    ylabel="x*",
    title="Bifurcation Diagram",
    ax=ax
)

export.show()
```

## Input Data Formats

### Bifurcation extractors and results

`SweepResult.bifurcation("x")` returns a `BifurcationExtractor`, which implements the same thin interface as `BifurcationResult` (exposing `.p`, `.y`, `.param_name`, `.meta`, and `.mode`). You can pass the extractor directly to `bifurcation_diagram()` or call helper methods such as `.all()`, `.tail()`, `.extrema()`, or `.final()` to get a concrete `BifurcationResult`.

```python
from dynlib.analysis.sweep import traj_sweep

sweep_result = traj_sweep(sim, param="r", values=r_values, record_vars=["x"], ...)
result = sweep_result.bifurcation("x")             # extractor defaults to the "all" mode
# result = result.tail(50)                          # optional: focus on the last 50 points

bifurcation_diagram(result)  # xlabel="r", ylabel="x", title based on mode
```

`bifurcation_diagram()` automatically pulls:
- Parameter values from `result.p`
- State values from `result.y`
- Axis labels from `result.param_name` and `result.meta["var"]` (when metadata is available)
- Title from `result.mode` (e.g., `"all"`, `"tail"`, `"extrema"`)

### Raw Arrays

For custom data or external bifurcation calculations:

```python
import numpy as np

# Raw parameter and state value arrays
r_values = np.array([...])  # parameter values
x_values = np.array([...])  # corresponding state values

bifurcation_diagram(
    (r_values, x_values),
    xlabel="r",
    ylabel="x*",
    title="Custom Bifurcation Data"
)
```

## Plot Customization

### Styling Options

Bifurcation diagrams use scatter-style plotting with optimized defaults:

```python
bifurcation_diagram(
    result,
    color="blue",           # Marker color
    marker=",",             # Pixel marker (default)
    ms=0.5,                 # Marker size (ignored for pixel markers)
    alpha=0.5,              # Transparency (default)
    label="Logistic Map"    # Legend label
)
```

Override the defaults for different visual styles:

```python
# Larger, more visible markers
bifurcation_diagram(
    result,
    marker=".",
    ms=1.0,
    alpha=1.0,
    color="darkred"
)
```

### Axis Control

Set axis limits and labels:

```python
bifurcation_diagram(
    result,
    xlim=(2.5, 4.0),       # Parameter range
    ylim=(0, 1),            # State range
    xlabel="r",             # Parameter axis label
    ylabel="x*",            # State axis label
    title="Logistic Map Bifurcations"
)
```

### Font Sizes and Layout

Customize text appearance:

```python
bifurcation_diagram(
    result,
    xlabel_fs=12,           # X-axis label font size
    ylabel_fs=12,           # Y-axis label font size
    title_fs=14,            # Title font size
    xtick_fs=10,            # X-axis tick font size
    ytick_fs=10             # Y-axis tick font size
)
```

## Annotations and Highlights

### Vertical Lines

Add vertical lines to mark important parameter values:

```python
bifurcation_diagram(
    result,
    vlines=[
        3.0,                                    # Simple line at r=3
        (3.449, "Period-4 bifurcation"),        # Line with label
        (3.5699, "Feigenbaum point")             # Another labeled line
    ],
    vlines_color="red",
    vlines_kwargs={
        "linestyle": "--",
        "alpha": 0.7,
        "linewidth": 1
    }
)
```

### Advanced Styling

Control vertical line appearance:

```python
bifurcation_diagram(
    result,
    vlines=[(3.0, "r=3"), (3.57, "Chaos onset")],
    vlines_kwargs={
        "linestyle": ":",
        "alpha": 0.5,
        "label_rotation": 90,      # Rotate labels
        "label_position": "top"    # Position labels above/below
    }
)
```

## Complete Example

Here's a comprehensive example showing multiple customization options:

```python
from dynlib.plot import bifurcation_diagram, theme, fig, export

# Configure theme
theme.use("notebook")
theme.update(grid=True)

# Create figure
ax = fig.single(size=(12, 8))

# Plot with full customization
bifurcation_diagram(
    result,
    color="black",
    alpha=0.8,
    xlim=(2.5, 4.0),
    ylim=(0, 1),
    xlabel="r",
    ylabel="x*",
    title="Logistic Map: Period-Doubling Cascade",
    xlabel_fs=14,
    ylabel_fs=14,
    title_fs=16,
    vlines=[
        (3.0, "Period-2"),
        (3.449, "Period-4"),
        (3.5699, "Feigenbaum point")
    ],
    vlines_kwargs={
        "color": "red",
        "linestyle": "--",
        "alpha": 0.6,
        "linewidth": 1.5
    },
    ax=ax
)

export.show()
```

## Tips for Effective Bifurcation Plots

1. **Resolution**: Use high-resolution parameter sweeps (10,000+ points) for smooth diagrams
2. **Transients**: Ensure sufficient transient time to reach attractors
3. **Markers**: Pixel markers (`,`) work well for dense data; use larger markers for sparse data
4. **Alpha**: Lower alpha values help visualize point density in dense regions
5. **Annotations**: Use vertical lines to highlight bifurcation points and transitions
6. **Zooming**: For complex cascades, consider plotting zoomed regions separately

The `bifurcation_diagram()` function integrates seamlessly with dynlib's analysis workflow, automatically handling the conversion from trajectory sweeps to visual bifurcation diagrams.</content>
<parameter name="filePath">/home/ismail/remote/PYTHON/my-packages/dynlib/docs/guides/plotting/bifurcation-diagrams.md
