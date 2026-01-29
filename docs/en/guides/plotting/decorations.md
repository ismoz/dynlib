# Plot Decorations

Dynlib plotting helpers expose a single set of decoration arguments that every high-level renderer forwards through `_apply_decor()` in `src/dynlib/plot/_primitives.py`. The helper accepts the same parameters from `series.plot()`, `series.step()` and similar entry points, so decorations behave identically everywhere.

## Vertical lines (`vlines`)

- Pass either a list of scalar values or `(x, label)` tuples to `vlines`. Tuples are rendered with labels next to the corresponding line without needing a separate `text()` call.
- You can supply `vlines_kwargs` (or the convenience `vlines_color`/`vlines_kwargs` pairing found on most helpers) to adjust appearance. Default kwargs are `color='black'`, `linestyle='--'`, `linewidth=1`, `alpha=0.7`.
- Label positioning is controlled by four special kwargs that `_apply_decor()` intercepts before forwarding the rest to `ax.axvline()`:
  - `label_position`: one of `'top'`, `'bottom'`, `'center'`. Determines whether the text anchors near the top/bottom/center of the axis before applying offsets.
  - `placement_pad`: adds additional offset along the axis (fraction of axis height if `<1`, data units otherwise) when computing the text anchor point.
  - `label_pad`: moves the label perpendicular to the line (i.e., horizontally), again interpreting values `<1` as axis fractions and `>=1` as data units.
  - `label_rotation`/`label_color`: override the rotation (default 90°) and text color (defaults to the line color).

Example:
```python
series.plot(
    x=t,
    y=x_traj,
    vlines=[(3.0, "period-2"), 3.57],
    vlines_kwargs={
        "label_position": "bottom",
        "placement_pad": 0.08,
        "label_pad": 0.05,
        "label_rotation": 90,
        "linestyle": ":",
        "color": "firebrick",
    },
)
```

## Horizontal lines (`hlines`)

- Works the same way as `vlines`, but for y coordinates. Labels can be provided via `(y, label)` tuples.
- `hlines_color` exists for the common case of only changing the color; it merges into `hlines_kwargs` before `_apply_decor()` runs.
- Special kwargs are similar but mirror the horizontal geometry:
  - `label_position`: `'left'`, `'right'`, or `'center'` to choose which side of the axis the label hugs.
  - `placement_pad`: shifts the anchor point along the x axis (`<1` = axis fraction, `>=1` = data units).
  - `label_pad`: offsets the label perpendicular to the line (moves it vertically) with the same axis-vs-data-unit interpretation.
  - `label_rotation` defaults to `0` degrees for horizontal text, and `label_color` again defaults to the line color.

Example:
```python
series.plot(
    x=t,
    y=x_traj,
    hlines=[(0.25, "low"), (0.75, "high")],
    hlines_kwargs={
        "label_position": "left",
        "placement_pad": 0.1,
        "label_pad": 0.02,
        "label_color": "navy",
        "linestyle": "-",
        "alpha": 0.6,
    },
)
```

## Vertical bands (`vbands`)

- Pass a list of `(start, end)` tuples (optionally with a third entry for color) to shade vertical regions: `(start, end)` uses the default color `C0`, `(start, end, "teal")` overrides it.
- `_apply_decor()` enforces `start < end` and renders using `ax.axvspan(start, end, color=color, alpha=0.1)`.

Example:
```python
series.plot(
    x=t,
    y=x_traj,
    vbands=[(2.5, 2.9, "gold"), (3.4, 3.6)],
)
```

## Horizontal bands (`hbands`)

- Behaves like `vbands` but uses `ax.axhspan()` to fill horizontal strips. Tuples may include an optional color.
- The helper also checks `start < end` before plotting and defaults to `color='C0'` with `alpha=0.1`.

Example:
```python
series.plot(
    x=t,
    y=x_traj,
    hbands=[(0, 0.2), (0.8, 1.0, "lightcoral")],
)
```

## Summary of special kwargs (all decorating helpers)

- `vlines_kwargs` / `hlines_kwargs` accept the usual Matplotlib line arguments plus these label-placement helpers: `label_position`, `placement_pad`, `label_pad`, `label_rotation`, `label_color`.
- `placement_pad` and `label_pad` treat values `<1` as fractions of the relevant axis span and `>=1` as data units, so you can switch between relative and absolute offsets.
- Tuple-based line definitions (value + label) trigger automatic text rendering; you can still supply `label_color` and `label_rotation` to customize each label globally.
- Bands (`vbands`, `hbands`) only accept `(start, end)` or `(start, end, color)` and raise `ValueError` when the tuple length is incorrect or when `start >= end`.

If you need other axes annotations (e.g., manual `text()` calls or extra artists), you can mix them with these decorations; `_apply_decor()` only runs once per axes and leaves any other artists untouched.
