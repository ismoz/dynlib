# Basin Plots

Basin plots expose which attractor each initial condition settles toward by coloring a 2D grid of parameter space. `plot.basin_plot()` turns the categorical labels produced by the basin-of-attraction analysis utilities into a `pcolormesh` so you can instantly see the structure of the basins, the special outcomes, and how the color legend relates to each attractor.

## What the plot shows

Each grid cell corresponds to one initial condition defined during the analysis. The value stored for that cell is an integer label:

- **Attractor IDs** (0, 1, …) mark trajectories that converged onto a known attractor.
- **Special values** (`BLOWUP`, `OUTSIDE`, `UNRESOLVED`) flag diverging trajectories, escapes from the region of interest, or initial conditions that did not reach a decision within the computation budget.

`basin_plot()` maps those integers to colors and emits a colorbar that labels the special outcomes first and then the attractors in index order.

## Preparing your data

Pass the `BasinResult` returned by `analysis.basin_auto()` (or `analysis.basin_known()`) directly to `basin_plot()`. The helper reads `res.labels` for the categorical grid and uses `res.meta` to infer grid dimensions (`ic_grid`), bounds (`ic_bounds`), observed variables (`observe_vars`), and attractor metadata (`attractor_labels`/`attractor_names`).

If you computed the labels yourself, pass them with `labels=`; 1D arrays require a `grid=(nx, ny)` shape so the helper can reshape to 2D, whereas pre-shaped 2D arrays can be supplied directly. Alternatively, provide explicit `x` and `y` coordinates that match the label array.

```python
from dynlib import setup
from dynlib.analysis import basin_auto
from dynlib.plot import basin_plot

sim = setup("models/henon.map", stepper="map")
res = basin_auto(sim, ic_grid=[256, 256], ic_bounds=[(-2, 2), (-2, 2)])
basin_plot(res)
```

## Controlling the colormap

`basin_plot()` builds a single colormap that starts with the special outcomes (default order `[BLOWUP, OUTSIDE, UNRESOLVED]`) followed by the attractor IDs. Override the defaults with:

- `special_order`: swap or drop the special IDs.
- `special_colors`: supply one color per special label; defaults fall back to grayscale pairs derived from the current Matplotlib palette if you request more entries.
- `special_labels`: rename the special entries that appear on the colorbar (e.g., `{"blowup": "Diverged"}`).
- `attractor_cmap`: switch from `"hsv"` to any Matplotlib colormap (or pass a `Colormap` instance) for the attractors.
- `attractor_colors`: give explicit colors instead of sampling a colormap.
- `attractor_labels`: customize the attractor names on the colorbar; they default to `meta["attractor_labels"]`, `meta["attractor_names"]`, or `A0`, `A1`, ….

```python
basin_plot(
    res,
    special_colors=["#1a1a1a", "#444444", "#777777"],
    attractor_cmap="viridis",
    attractor_labels=["Period-1", "Period-2"],
    colorbar_label="Outcome",
)
```

`basin_plot()` raises if you ask for fewer colors than there are labels, so make sure your palette matches the number of special outcomes and attractors in the result.

## Axis, bounds, and annotations

Axis limits, labels, and tick styling are handled by the usual `plot` helpers:

- `bounds` or `res.meta["ic_bounds"]` specify the `(x_min, x_max)`/`(y_min, y_max)` ranges used when generating the initial conditions. If you supply `x` and `y` arrays instead, `bounds` is ignored.
- `xlabel`, `ylabel`, and `title` behave like `matplotlib.axes.Axes` labels. When the result metadata contains `observe_vars`, those names automatically populate `xlabel`/`ylabel` unless you override them.
- `xlim`, `ylim`, `aspect`, `xlabel_fs`, `ylabel_fs`, `xtick_fs`, `ytick_fs`, `xlabel_rot`, `ylabel_rot`, `title_fs`, `titlepad`, `xpad`, and `ypad` let you fine-tune the appearance.
- The helper accepts an existing `ax=` so you can place the basin map into a multi-panel figure produced by `plot.fig()`/`plot.theme()` or Matplotlib directly.

Because the plot uses `pcolormesh`, `shading` defaults to `"auto"` and `alpha` can be used to fade the grid if you intend to overlay contours or basins from another dataset.

## Colorbar tuning

Set `colorbar=False` to omit it. Otherwise, the helper automatically hooks up ticks for the special entries followed by the attractor IDs. Use:

- `colorbar_label`, `colorbar_label_rotation`, and `colorbar_labelpad` to adjust the axis title.
- `colorbar_kwargs` to forward extra settings (e.g. `{"fraction": 0.05}`) to `plt.colorbar()`.

`plot.basin_plot()` stores the created `Colorbar` on `ax._last_colorbar`, mirroring other plotting helpers.

## Tips

- Provide `grid` when passing only flattened labels; the helper must know how to reshape them into `(ny, nx)` for `pcolormesh`.
- `res.meta["ic_grid"]` and `res.meta["ic_bounds"]` are frequently populated by the analysis routines, so you rarely need to repeat them when plotting.
- To visualize a different slice of a multi-parameter result, slice the labels array before calling `basin_plot()` and update `bounds` accordingly (the helper does not automatically reshape multidimensional slices).
- If you want to highlight attractor names instead of IDs, always supply `attractor_labels` so the colorbar ticks read clearly regardless of the order of your attractor registry.

`basin_plot()` always returns the `Axes` object so you can continue annotating the figure with Matplotlib commands or the shared decoration arguments (`vlines`, `hlines`, `vbands`, `hbands`) described in [Plot Decorations](decorations.md).
