# Manifold Plots

`plot.manifold()` renders 1D manifold traces (stable/unstable branches, heteroclinic connections, homoclinic loops) as 2D projections with consistent styling and legend handling. It is the plotting companion to the manifold analysis utilities, so you can focus on extracting the manifolds elsewhere and rely on this helper to visualize whatever segments those analyses produce.

## What the helper draws

`plot.manifold()` expects sequences of `(state_x, state_y)` samples that follow a manifold branch. It accepts raw segments or higher-level results and projects the selected state components onto `x`/`y` axes. Each supplied branch appears as either a `LineCollection` (for plain lines) or individual `plot` calls (when markers are requested), and any branch with a non-`None` label automatically enters the legend.

## Feeding the helper

- **Segments:** Pass `segments=[arr1, arr2, …]` where each `arr` is a `(steps, states)` array. The helper enforces that each array has at least two rows and enough columns to cover the requested `components`.
- **Branches:** Supply a `branches` tuple like `(branch_pos, branch_neg)` when you manage positive/negative branches yourself. Each branch list can contain multiple segments.
- **Result objects:** Most manifold-analysis results expose a `branches` attribute compatible with `plot.manifold()`, so you can pass them directly:
  - `dynlib.analysis.trace_manifold_1d_map(...)` / `trace_manifold_1d_ode(...)` return `ManifoldTraceResult` with two branch lists (positive/negative).
  - `dynlib.analysis.heteroclinic_tracer(...)` and `dynlib.analysis.homoclinic_tracer(...)` return `HeteroclinicTraceResult` and `HomoclinicTraceResult` respectively; both expose `.branches` (the latter branches contain the single traced orbit) plus `.kind`/`.meta` so `plot.manifold()` labels them automatically.
  - If you wrap your own segments in a custom structure, make sure it provides a `branches` attribute that resolves to a two-tuple of sequences before passing it as `result=…`.

```python
from dynlib import setup
from dynlib.analysis import trace_manifold_1d_map, heteroclinic_tracer
from dynlib.plot import fig, manifold

sim = setup("models/henon.map", stepper="map")
unstable = trace_manifold_1d_map(sim, kind="unstable", branch_len=500)
hex_trace = heteroclinic_tracer(sim, source_eq="E0", target_eq="E1", preset="default")

ax = fig.single()
manifold(result=unstable, components=(0, 1), label="Unstable manifold", ax=ax)
manifold(result=hex_trace, components=(0, 1), style="discrete", label="Heteroclinic orbit", ax=ax)
```

## Styling branches

`style`, `color`, `lw`, `ls`, `marker`, `ms`, and `alpha` behave like any other `plot` helper, but `style` also accepts the built-in presets:

- `"continuous"`, `"flow"` / `"cont"`: solid line without markers (ideal for ODE-generated branches).
- `"discrete"`, `"map"`: marker-only (good for discrete-time manifolds).
- `"mixed"` / `"connected"`: markers connected by lines.
- `"line"` / `"scatter"`: explicit shorthand for lines-only or markers-only.

Per-branch overrides are handled with `groups`. Each group is either a mapping (`{"segments": …, "label": …, "style": …}`) or a tuple `(segments, label?, style?)` (? means optional; can be `None`). The helper inherits the global `style` but mixes in preset overrides or explicit mappings for each group, letting you color the stable branch differently from the unstable branch or the heteroclinic trace.

## Choosing projections and axes

Use `components=(i, j)` to select which state indices (`int`s) to draw (e.g., `(0, 1)` for the first two states). Components must be distinct and within the dimensionality of the supplied segments.

Axis labels, limits, and aspect ratios are controlled by:

- `xlabel`, `ylabel`, `title`, `xlabel_fs`, `ylabel_fs`, `title_fs`
- `xlim`, `ylim`, `aspect`, `xlabel_rot`, `ylabel_rot`
- `xpad`, `ypad`, `titlepad` for extra spacing

If your result provides metadata (e.g., `result.meta` from a `ManifoldTraceResult`), you can reuse it to annotate the plot title/labels before calling `manifold()`.

`plot.manifold()` returns the `Axes`, so you can layer decorations (vertical/horizontal lines, bands) as described in [Plot Decorations](decorations.md) or integrate the manifold into multi-panel figures created by `fig.grid()`/`plot.fig()`.

## Legend and grouping tips

- Set `label=` on the helper or via `groups` to tag each branch. The legend only appears if at least one label is set and `legend=True` (default).
- Use `groups` to overlay branch fragments with different styles or colors (for example, highlight the first segment of a heteroclinic trace with a thicker line while keeping the remainder subtle).
- When plotting multiple manifolds together (e.g., stable vs. unstable), pass `legend=True` only once in the final call to avoid duplicate handles.

## Tips

- Slice large result arrays before plotting if you only want a windowed view; the helper respects the provided segments exactly.
- Combine `plot.manifold()` with other plot helpers (phase portraits, time series) by passing the same `ax=` and controlling `legend`.
- Because `linecollection` is used when no markers are requested, `alpha`/`linewidth` apply uniformly to entire segments.
- If you want to reuse the same style across calls, keep a dictionary and update it for each group (`style={"color": "C0"}`) rather than repeating the preset string.

`plot.manifold()` mirrors the appearance and workflow of the analysis utilities, so once you have a `ManifoldTraceResult`, `HeteroclinicTraceResult`, or `HomoclinicTraceResult`, you can document and style the manifold without additional data massaging.
