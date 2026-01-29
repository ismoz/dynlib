# Themes & Facets

Dynlib's plotting system provides powerful theming and faceting capabilities to create consistent, publication-ready figures. This guide covers how to use `plot.theme` for styling and `plot.fig`/`plot.facet` for multi-panel layouts.

## Theming Overview

Themes in dynlib control the visual appearance of all plots, ensuring consistency across figures. The theme system manages:

- Font sizes and families
- Line widths and marker styles
- Color palettes
- Grid and background settings
- Spacing and margins

Themes are applied globally and affect all subsequent plots until changed.

### Built-in Presets

Dynlib includes several predefined themes optimized for different use cases:

- **notebook**: Default theme for interactive Jupyter notebooks with balanced styling.
- **paper**: Clean theme for publications, with subtle grids disabled and optimized font sizes.
- **talk**: High-contrast theme for presentations, with larger elements and bolder lines.
- **dark**: Dark background theme with adjusted colors for better visibility.
- **mono**: Monochrome theme using grayscale colors.

### Using Themes

Set a theme at the beginning of your plotting script:

```python
from dynlib.plot import theme

# Use a preset
theme.use("paper")

# Or customize on top of a preset
theme.use("notebook", tokens={"scale": 1.2, "grid": False})
```

### Customizing Themes

You can modify theme settings without changing presets:

```python
# Temporarily adjust settings
theme.update(tokens={"fontsize_title": 16, "line_w": 2.0})

# Or push/pop for scoped changes
theme.push(tokens={"palette": "mono"})
# ... create plots ...
theme.pop()  # Reverts to previous theme
```

### Color Palettes

Dynlib supports multiple color palettes:

- **classic**: Standard Matplotlib colors
- **cbf**: Colorblind-friendly palette (recommended for accessibility)
- **mono**: Grayscale palette

Register custom palettes:

```python
theme.register_palette("my_colors", ["#ff0000", "#00ff00", "#0000ff"])
theme.use("notebook", tokens={"palette": "my_colors"})
```

### Theme Tokens

Themes are controlled by tokens that specify individual styling properties. Key tokens include:

- **scale**: Overall size multiplier
- **fontsize_***: Font sizes for different elements (base, label, title, etc.)
- **line_w**: Line width
- **marker_size**: Marker size
- **grid**: Whether to show grid lines
- **palette**: Color palette name
- **background**: "light" or "dark"

Access current token values:

```python
current_scale = theme.get("scale")
font_size = theme.get("fontsize_title")
```

## Figure Grids and Layouts

Dynlib provides high-level helpers for creating figure layouts that work seamlessly with themes.

### Basic Figure Creation

```python
from dynlib.plot import fig

# Single plot
ax = fig.single(title="My Plot")

# Grid of subplots
axes = fig.grid(rows=2, cols=3, title="Parameter Sweep")

# 3D plot
ax_3d = fig.single3D(title="3D Trajectory")

# Flexible grid that wraps subplots
axes = fig.wrap(n=7, cols=3)  # Creates 3x3 grid, hides last 2 axes
```

All `fig` helpers accept parameters for customization:

- `size`: Figure size as (width, height) tuple
- `scale`: Size multiplier
- `sharex`/`sharey`: Whether to share axes
- `title`: Figure title

### Integration with Plotting

Pass created axes to plotting functions:

```python
from dynlib.plot import fig, series

ax = fig.single()
series.plot(x=time, y=signal, ax=ax, label="Signal")
```

## Faceting for Multi-Panel Figures

Faceting automatically creates grids and iterates over data categories, perfect for parameter sweeps or grouped data.

### Basic Faceting

```python
from dynlib.plot import facet, series

# Data for different parameters
data = {
    "r=2.5": trajectory_r25,
    "r=3.0": trajectory_r30,
    "r=3.5": trajectory_r35,
}

# Create faceted plot
for ax, param in facet.wrap(data.keys(), cols=2, title="Bifurcation Analysis"):
    traj = data[param]
    series.plot(x=time, y=traj, ax=ax, title=param)
```

The `facet.wrap` function:

- Takes an iterable of keys (categories)
- Creates a grid with specified number of columns
- Yields (axis, key) pairs for iteration
- Automatically handles layout and hides unused axes

### Faceting Parameters

- `cols`: Number of columns in the grid
- `title`: Overall figure title
- `size`, `scale`: Figure sizing
- `sharex`/`sharey`: Axis sharing

### Advanced Faceting Example

```python
import numpy as np
from dynlib.plot import facet, series, theme

# Parameter sweep
r_values = np.linspace(2.8, 4.0, 12)

theme.use("paper")
for ax, r in facet.wrap(r_values, cols=4, title="Logistic Map Bifurcations"):
    # Simulate trajectory for this r
    x = 0.1
    traj = [x]
    for _ in range(100):
        x = r * x * (1 - x)
        traj.append(x)
    
    series.plot(x=range(len(traj)), y=traj, ax=ax, title=f"r={r:.1f}")
```

## Best Practices

1. **Set themes early**: Apply themes at the start of your script before creating any figures.

2. **Use consistent palettes**: Choose colorblind-friendly palettes like "cbf" for better accessibility.

3. **Leverage faceting**: For parameter sweeps or grouped data, faceting reduces boilerplate code.

4. **Customize thoughtfully**: Use `theme.update()` for small adjustments rather than creating entirely new themes.

5. **Scope changes**: Use `theme.push()`/`pop()` for temporary theme modifications.

6. **Size appropriately**: Use the `scale` parameter or theme tokens to adjust figure sizes for different outputs (screen vs. print).

Themes and faceting work together to make your dynlib plots professional and consistent, whether for exploratory analysis or publication figures.
