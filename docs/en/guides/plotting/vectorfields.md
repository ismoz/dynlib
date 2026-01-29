# Vector Fields and Vector Field Animations

Vector fields visualize the direction and magnitude of change in dynamical systems. In dynlib, vector fields are computed by evaluating the system's right-hand side (RHS) equations on a 2D grid of points, showing how trajectories would evolve from each point.

Most snippets in this guide assume `from dynlib import build, plot`. When we need numerical sequences for sweeps or animations, you will also see `numpy` being used—import it as `import numpy as np` in those contexts.

## Basic Vector Field Plotting

The core function for plotting vector fields is `plot.vectorfield()`. It evaluates your model's equations on a grid and displays the resulting vectors.

### Simple Example

```python
from dynlib import build, plot

# Define a simple 2D system
model_uri = """
inline:
[model]
type = "ode"

[states]
x = 0.0
y = 0.0

[params]
a = 1.0
b = -1.0

[equations.rhs]
x = "a * x + y"
y = "b * x + y"
"""

# You can also pass a Sim object created by setup()
model = build(model_uri)
plot.theme.use("notebook")

# Plot the vector field
plot.vectorfield(
    model,
    xlim=(-2, 2),
    ylim=(-2, 2),
    grid=(25, 25)
)

plot.export.show()
```

When passing DSL inline to `build()` (or `setup()`), start the string with `inline:` (as shown above) so dynlib knows to treat the content as an embedded model definition instead of a path.

## Vector Field Options

### Grid and Limits

- `xlim`, `ylim`: Tuples specifying the plot boundaries (default: `(-1, 1)`)
- `grid`: Tuple of `(nx, ny)` specifying grid resolution (default: `(20, 20)`)

Higher grid values give smoother, more detailed plots but take longer to compute.

### Variable Selection

For systems with more than 2 variables, specify which 2 to plot:

```python
# For a 3D Lorenz system
plot.vectorfield(
    model,
    vars=("x", "y"),  # Plot x vs y
    fixed={"z": 10.0},  # Fix z at 10
    xlim=(-20, 20),
    ylim=(-30, 30)
)
```

### Vector Normalization

- `normalize=True`: Scale all vectors to unit length, showing only direction
- `normalize=False` (default): Show actual magnitudes

```python
# Compare normalized vs magnitude-preserving
plot.vectorfield(model, normalize=True)   # Shows flow directions
plot.vectorfield(model, normalize=False)  # Shows flow speeds
```

### Coloring Options

#### Single Color
```python
plot.vectorfield(model, color="blue")
```

The `color` argument flows directly into Matplotlib, so you can use any named color, hex string, or RGBA tuple for a consistent palette.

#### Speed-Based Coloring
Color vectors by their magnitude:

```python
plot.vectorfield(
    model,
    speed_color=True,
    speed_cmap="plasma",
    normalize=False  # Speed coloring works best with actual magnitudes
)
```

Pass `speed_norm` to fix the full range manually (or let a sweep compute a shared norm with `share_speed_norm=True`) so you can compare different plots on the same scale.

### Plot Modes

- `mode="quiver"` (default): Arrow/quiver plot
- `mode="stream"`: Streamline plot using matplotlib's streamplot

```python
# Streamlines can be smoother for dense flows
plot.vectorfield(model, mode="stream", speed_color=True)
```

### Nullclines

Nullclines show where the system has zero velocity in x or y directions:

```python
plot.vectorfield(
    model,
    nullclines=True,
    nullcline_style={"colors": ["red", "blue"], "linewidths": 1.5}
)
```

Nullclines are computed on a denser grid by default for accuracy.

Use `nullcline_grid` when you need even finer contours or resizing relative to the main grid.

## Interactive Features

`plot.vectorfield` returns a `VectorFieldHandle`, so the same call that draws the arrows also gives you a programmatic handle you can update, simulate, or clear. Pass `interactive=True` to hook into the click/tap callbacks described below, or call `handle.update()` to redraw with new params/fixed states without recreating the plot.

Enable interactive plotting to explore trajectories:

```python
handle = plot.vectorfield(
    model,
    interactive=True,
    T=10.0,  # Trajectory duration
    trajectory_style={"color": "red", "linewidth": 2}
)
```

**Interactive Controls:**
- **Click** anywhere on the plot to launch a trajectory from that point
- Press **N** to toggle nullclines on/off
- Press **C** to clear drawn trajectories
`handle.clear_trajectories()` also removes collected paths if you want to reset programmatically.

## Parameter Sweeps

Use `plot.vectorfield_sweep()` to compare vector fields across parameter values. It returns a `VectorFieldSweep` containing `.handles`, `.axes`, and `.colorbar` so you can adjust individual facets after plotting or grab the shared `speed_norm` used for coloring. Pass `param`/`values` for a simple 1D sweep, or provide the `sweep` mapping when you need custom overrides for parameters and fixed states; the `target` argument chooses whether each sweep `value` edits params (default) or fixed states.

```python
plot.vectorfield_sweep(
    model,
    param="a",  # Parameter to vary
    values=[-1.0, 0.0, 1.0, 2.0],  # Values to test
    xlim=(-3, 3),
    ylim=(-3, 3),
    cols=2,  # 2 columns in the grid
    normalize=True,
    facet_titles="a = {value:.1f}"  # Custom titles
)

```

Shared normalization (`share_speed_norm=True`) keeps the coloring consistent across all facets, and `add_colorbar=True` draws a single legend for the speed coloring when one is available.

## Vector Field Animations

Animate how vector fields change with parameters using `plot.vectorfield_animate()`:

```python
import numpy as np

# Animate parameter changes
anim = plot.vectorfield_animate(
    model,
    param="a",
    values=np.linspace(-2, 2, 100),  # 100 frames
    fps=15,
    title_func=lambda v, idx: f"Parameter a = {v:.2f}",
    normalize=True,
    speed_color=True
)

# Save animation
anim.animation.save("vectorfield_animation.gif", writer="pillow")
```

`plot.vectorfield_animate()` returns a `VectorFieldAnimation` that keeps both the live `VectorFieldHandle` (accessible via `.handle`) and the underlying `matplotlib.animation.FuncAnimation` so you can update the handle manually or save the animation later. Provide either `frames`, `values` (with `param`), or `duration`/`fps`, and use `params_func`, `fixed_func`, or `title_func` when you need custom overrides for each frame.

### Animation Options

- `fps`: Frames per second (default: 15)
- `interval`: Milliseconds between frames (alternative to fps)
- `title_func`: Function to generate titles for each frame
- `repeat`: Whether animation loops (default: True)
- `blit`: Use blitting for smoother animation (may not work in all backends)

## Advanced Usage

### Updating Plots Dynamically

The `vectorfield()` function returns a `VectorFieldHandle` that allows dynamic updates:

```python
handle = plot.vectorfield(model, params={"a": 1.0})

# Update parameters without replotting
handle.update(params={"a": 2.0})

# Update fixed values
handle.update(fixed={"z": 15.0})
```

### Custom Evaluation

For low-level control, use `plot.eval_vectorfield()` to get raw vector data:

```python
X, Y, U, V = plot.eval_vectorfield(
    model,
    xlim=(-2, 2),
    ylim=(-2, 2),
    grid=(50, 50),
    normalize=True
)

# Use with matplotlib directly
import matplotlib.pyplot as plt
plt.quiver(X, Y, U, V)
```

Pass `return_speed=True` when you need the magnitude grid (e.g., to color with a colormap or compare normalized vs non-normalized speeds).

### Higher-Dimensional Systems

For systems with >2 dimensions, project onto 2D slices:

```python
# Lorenz system: x-y plane with z fixed
plot.vectorfield(
    lorenz_model,
    vars=("x", "y"),
    fixed={"z": 25.0}
)

# Same system: y-z plane with x fixed
plot.vectorfield(
    lorenz_model,
    vars=("y", "z"),
    fixed={"x": 0.0}
)
```

When you slice higher-dimensional systems, make sure every other state is pinned with `fixed` so the evaluation stays within the desired plane.

## Performance Considerations

- **Grid size**: Larger grids give better resolution but slower computation
- **Normalization**: Normalized plots compute faster (no magnitude calculation)
- **Nullclines**: Computed on separate grid; use `nullcline_grid` to control density
- **JIT compilation**: Set `jit=True` for repeated evaluations of the same model
- **Disk caching**: Pass `disk_cache=True` when building from a URI to reuse compiled artifacts between runs

## Common Patterns

### Phase Portrait with Trajectories

```python
ax = plot.fig.single()
handle = plot.vectorfield(
    model,
    ax=ax,
    normalize=True,
    nullclines=True
)

# Add specific trajectories
from dynlib import Sim
sim = Sim(model)
sim.run(t_end=20.0, state_ic=[1.0, 0.0])
plot.series(sim, ax=ax, vars=("x", "y"))
```

### Bifurcation Analysis Setup

```python
# Build the sweep range with numpy
import numpy as np

# Sweep parameter and observe qualitative changes
plot.vectorfield_sweep(
    model,
    param="r",
    values=np.linspace(0, 1, 9),
    normalize=True,
    speed_color=True,
    title="Bifurcation in vector field structure"
)
```

### Animation with Custom Parameter Functions

```python
# Parameter updates can reuse NumPy for oscillations
import numpy as np

# Oscillating parameter
def param_func(frame_idx):
    return {"a": 2.0 * np.sin(2 * np.pi * frame_idx / 50)}

plot.vectorfield_animate(
    model,
    frames=50,
    params_func=param_func,
    fps=10
)
```
