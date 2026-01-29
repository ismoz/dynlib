# Exporting plots

The `dynlib.plot.export` helper wraps the Matplotlib `savefig`/`show` workflow so your scripts keep the same consistent defaults regardless of whether you draw a single panel, grid layout, or higher-level container.

## Core functions

### `export.savefig(fig_or_ax, path, *, fmts=("png",), dpi=300, transparent=False, pad=0.01, metadata=None, bbox_inches="tight")`

- `fig_or_ax` accepts a figure, axes, or any dynlib layout object (`fig.grid`, `AxesGrid`, etc.). The helper finds the underlying figure automatically so you can call it right after plotting without grabbing `fig.figure`.
- `path` can either include an extension (e.g., `"plots/phase.png"`) or omit it (e.g., `"plots/phase"`). When you pass an extension, the helper writes only that format; omit the extension and specify `fmts` to save multiple formats in one go.
- `fmts` defaults to `("png",)` unless you inferred formats from the path. The helper normalizes, deduplicates, and lowercases the values you provide so you can pass `(".PNG", ".pdf")` without extra parsing.
- The remaining keyword arguments mirror Matplotlib's `savefig`. Use `dpi` for resolution, `transparent` for alpha backgrounds, `pad` to add whitespace, and `metadata` to embed search-friendly tags. When you rely on dynlib's `fig` helpers with `constrained_layout=True`, the helper automatically avoids applying a tight bounding box that could clip decorations.

### `export.show()`

Call `export.show()` at the end of a script, notebook cell, or in any interactive session to trigger Matplotlib's `plt.show()`. It follows dynlib's styling, so figure numbering and layouts behave the same way whether you use the CLI or import the helpers in a script.

## Best practices

- Keep `export` imports next to your plotting helpers: `from dynlib.plot import fig, series, export`. That way you can consistently call `export.show()` after every figure set.
- When saving multiple formats, leave the extension off `path` and rely on `fmts`. For example, `export.savefig(ax, "figures/lorenz", fmts=("svg","png"))` writes `lorenz.svg` and `lorenz.png` with the same dpi/pad settings.
- Pass dynlib containers like `axes = fig.grid(...)` or the return value of helpers such as `plot.vectorfield()` directly to `savefig`; `export` traverses the container to locate the figure automatically.
- Use `metadata` (a dict of string keys/values) for searchable keywords or author info when generating publication-ready images.

For a worked example and more formatting notes, return to the [Plotting basics guide](basics.md) or explore the rest of the plotting docs.
