# src/dynlib/plot/_fig.py
from __future__ import annotations
import math
from typing import Iterable, List
import matplotlib.pyplot as plt
import numpy as np
from . import _theme

_BASE_FIG_SIZE = (6.0, 4.0)

class AxesGrid(List[List[plt.Axes]]):
    def __init__(self, iterable=(), *, figure: plt.Figure | None = None):
        super().__init__(iterable)
        self.figure = figure

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
            return super().__getitem__(row).__getitem__(col)
        return super().__getitem__(item)

def _resolve_figsize(size: tuple[float, float] | None, scale: float | None) -> tuple[float, float]:
    base_scale = float(_theme.get("scale"))
    if size is not None:
        width, height = size
    else:
        width = _BASE_FIG_SIZE[0] * base_scale
        height = _BASE_FIG_SIZE[1] * base_scale
    if scale is not None:
        width *= scale
        height *= scale
    return float(width), float(height)

def single(*, title: str | None = None, size: tuple[float, float] | None = None, scale: float | None = None) -> plt.Axes:
    fig, ax = plt.subplots(figsize=_resolve_figsize(size, scale), layout="constrained")
    if title:
        fig.suptitle(title)
    return ax

def single3D(*, title: str | None = None, size: tuple[float, float] | None = None, scale: float | None = None) -> plt.Axes:
    fig = plt.figure(figsize=_resolve_figsize(size, scale), constrained_layout=True)
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    return ax

def grid(
    *,
    rows: int,
    cols: int,
    sharex: bool | str = False,
    sharey: bool | str = False,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    scale: float | None = None,
) -> AxesGrid:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers.")
    fig, axes = plt.subplots(
        rows, cols,
        sharex=sharex, sharey=sharey,
        figsize=_resolve_figsize(size, scale),
        layout="constrained",
    )
    if title:
        fig.suptitle(title)
    axes_array = np.asarray(axes, dtype=object).reshape(rows, cols)
    return AxesGrid(
        [[axes_array[r, c] for c in range(cols)] for r in range(rows)],
        figure=fig,
    )

def wrap(
    *,
    n: int,
    cols: int,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    scale: float | None = None,
    sharex: bool | str = False,
    sharey: bool | str = False,
) -> AxesGrid:
    if n <= 0:
        raise ValueError("n must be positive.")
    if cols <= 0:
        raise ValueError("cols must be positive.")
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        sharex=sharex, sharey=sharey,
        figsize=_resolve_figsize(size, scale),
        layout="constrained",
    )
    axes_array = np.atleast_2d(axes)
    flat_axes: list[plt.Axes] = [ax for ax in axes_array.ravel()]
    for idx, ax in enumerate(flat_axes):
        if idx >= n:
            ax.set_visible(False)
    if title:
        fig.suptitle(title)

    grid_axes: AxesGrid = AxesGrid([], figure=fig)
    for r in range(rows):
        row_axes: list[plt.Axes] = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                row_axes.append(axes_array[r, c])
        if row_axes:
            grid_axes.append(row_axes)
    return grid_axes

__all__ = ["single", "single3D", "grid", "wrap"]
