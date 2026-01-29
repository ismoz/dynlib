# Cobweb Plots

Cobweb plots are a powerful visualization tool for analyzing the behavior of one-dimensional discrete dynamical systems (maps). They provide an intuitive way to understand how iterations of a function evolve over time and help identify important dynamical features like fixed points, periodic orbits, and chaotic behavior.

## How Cobweb Plots Work

A cobweb plot visualizes the iterative process of applying a function repeatedly. For a function $f(x)$, starting from an initial value $x_0$, the iteration generates a sequence:

$$x_{n+1} = f(x_n)$$

The cobweb plot represents this iteration geometrically by:

1. **Function curve**: Plotting $y = f(x)$
2. **Identity line**: Plotting $y = x$ (shown as a dashed line)
3. **Iteration path**: Drawing a "staircase" that shows how each iteration moves from $(x_n, x_n)$ to $(x_n, x_{n+1})$ to $(x_{n+1}, x_{n+1})$

The staircase is constructed by:
- Drawing a vertical line from $(x_n, x_n)$ to $(x_n, f(x_n))$
- Drawing a horizontal line from $(x_n, f(x_n))$ to $(f(x_n), f(x_n))$

This creates a path that zigzags between the function curve and the identity line, visually representing the iterative process.

## Basic Usage

```python
from dynlib.plot import cobweb

# Using a simple function
def logistic(x, r=4.0):
    return r * x * (1 - x)

cobweb(
    f=logistic,
    x0=0.1,      # initial condition
    steps=50,     # number of iterations
    xlim=(0, 1),  # x-axis limits
)
```

## Working with Dynlib Models

Cobweb plots work seamlessly with dynlib models:

```python
from dynlib import setup
from dynlib.plot import cobweb

model = """
inline:
[model]
type="map"
name="Logistic Map"

[states]
x=0.1

[params]
r=4.0

[equations.rhs]
x = "r * x * (1 - x)"
"""

sim = setup(model, stepper="map")
cobweb(
    f=sim.model,  # Pass the model directly
    x0=0.1,
    xlim=(0, 1),
    steps=50,
)
```
When passing DSL inline to `setup()` (or `build()`), start the string with `inline:` so dynlib treats it as an embedded model definition instead of a path.

## Key Parameters

### Function Specification
- `f`: The function or model to iterate. Can be:
  - A callable function `f(x)` or `f(x, r)`
  - A dynlib Model object with a `map()` method
  - A Sim object (will use `sim.model`)

### Iteration Control
- `x0`: Initial value for the iteration
- `steps`: Number of iteration steps to plot (default: 50)
- `t0`: Starting time index (default: 0.0)
- `dt`: Time step size (default: 1.0)

### Model-Specific Options
- `state`: For multi-dimensional models, specify which state variable to use (by name or index)
- `fixed`: Dictionary of fixed parameter/state values
- `r`: Override for the 'r' parameter (common in bifurcation analysis)

### Plot Styling
- `xlim`/`ylim`: Axis limits (auto-calculated if not specified)
- `color`: Color for the function curve
- `identity_color`: Color for the identity line (y=x)
- `stair_color`: Color for the iteration staircase
- `lw`: Line width for the function curve
- `stair_lw`: Line width for the staircase
- `alpha`: Transparency

### Labels and Appearance
- `xlabel`/`ylabel`: Axis labels
- `title`: Plot title
- `legend`: Whether to show legend (default: True)

## Interpreting Cobweb Plots

### Fixed Points
Fixed points occur where $x = f(x)$. In the cobweb plot, these appear as intersection points between the function curve and the identity line.

### Stability
- **Stable fixed points**: The staircase spirals inward toward the fixed point
- **Unstable fixed points**: The staircase spirals outward away from the fixed point

### Periodic Orbits
Periodic behavior appears as closed loops in the cobweb plot. A period-2 orbit, for example, creates a rectangular path that the staircase follows repeatedly.

### Chaos
Chaotic behavior is indicated by the staircase never settling into a regular pattern, often filling regions of the plot densely.

## Advanced Examples

### Multi-Parameter Analysis
```python
# Analyze different r values
r_values = [2.5, 3.2, 3.5, 4.0]

for r in r_values:
    cobweb(
        f=logistic,
        x0=0.1,
        r=r,  # Override r parameter
        xlim=(0, 1),
        title=f"Logistic Map (r={r})",
    )
```

### Multi-State Models (iterate a single variable)

Cobweb plots still visualize a single state trajectory even when the underlying map has many states. Set `state` to the variable you want to follow and keep the rest constant via `fixed`. This effectively reduces the system to a 1D map for the chosen state while the other states stay pinned at the provided values.

```python
from dynlib import setup

multi_state_model = """
inline:
[model]
type = "map"
name = "Two-State Map"

[states]
x = 0.5
y = 1.0

[params]
a = 3.0

[equations.rhs]
x = "a * x * (1 - x) + 0.1 * y"
y = "0.9 * y"
"""

sim = setup(multi_state_model, stepper="map")
cobweb(
    f=sim.model,
    x0=sim.state["x"],
    state="x",
    fixed={"y": 1.0},
    steps=100,
)
```

Because the plot still shows just the selected `state`, cobweb plots remain limited to one-dimensional maps; multi-state systems are visualized by slicing out one coordinate at a time.

### Custom Styling
```python
cobweb(
    f=sim.model,
    x0=0.1,
    xlim=(0, 1),
    color="blue",
    identity_color="red",
    stair_color="green",
    stair_lw=1.5,
    alpha=0.8,
    title="Custom Styled Cobweb Plot",
)
```

## Limitations

- Cobweb plots are designed for 1D maps only; multi-state models must be reduced to a single iterated state using `state`/`fixed`.
- Models using `lag()` functions are not supported (cannot evaluate safely)
- Requires discrete map models (`spec.kind == 'map'`)

## Related Functions

- `plot.return_map()`: Plot return maps ($x_n$ vs $x_{n+1}$)
- `plot.series()`: Plot time series of the iteration
- `plot.phase()`: Phase space plots for higher-dimensional systems
