"""Basin of attraction plotting utilities."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap

from dynlib.analysis import BLOWUP, OUTSIDE, UNRESOLVED

from . import _theme
from ._primitives import (
    _apply_labels,
    _apply_limits,
    _apply_tick_fontsizes,
    _apply_tick_rotation,
    _get_ax,
)

__all__ = ["basin_plot"]


def _coerce_grid_2d(grid: Sequence[int] | int | None) -> tuple[int, int] | None:
    if grid is None:
        return None
    if isinstance(grid, (int, np.integer)):
        return int(grid), int(grid)
    grid_tuple = tuple(int(x) for x in grid)
    if len(grid_tuple) != 2:
        raise ValueError("grid must be an int or a length-2 sequence for 2D basin plots.")
    return grid_tuple


def _resolve_labels(res, labels: np.ndarray | None, grid: Sequence[int] | int | None) -> tuple[np.ndarray, dict[str, object]]:
    if labels is None:
        if hasattr(res, "labels"):
            labels = res.labels
        else:
            labels = res
    labels_arr = np.asarray(labels)
    meta: dict[str, object] = {}
    if hasattr(res, "meta"):
        meta = dict(getattr(res, "meta") or {})

    grid_tuple = _coerce_grid_2d(grid)
    if grid_tuple is None:
        meta_grid = meta.get("ic_grid")
        if meta_grid is not None:
            grid_tuple = _coerce_grid_2d(meta_grid)
    if labels_arr.ndim == 1:
        if grid_tuple is None:
            raise ValueError("1D basin labels require grid=(nx, ny) to reshape.")
        nx, ny = grid_tuple
        # Basin labels are flattened from (nx, ny) with indexing='ij'
        # Need to reshape to (nx, ny) then transpose to (ny, nx) for pcolormesh
        shape = (int(nx), int(ny))
        if labels_arr.size != shape[0] * shape[1]:
            raise ValueError("grid size does not match labels length.")
        labels_arr = labels_arr.reshape(shape).T
    elif labels_arr.ndim == 2:
        if grid_tuple is not None:
            nx, ny = grid_tuple
            # For 2D labels, verify they match expected transpose shape (ny, nx)
            shape = (int(ny), int(nx))
            if labels_arr.shape != shape:
                # Maybe they're in (nx, ny) format, try transposing
                if labels_arr.shape == (int(nx), int(ny)):
                    labels_arr = labels_arr.T
                else:
                    raise ValueError("grid size does not match labels shape.")
    else:
        raise ValueError("labels must be 1D or 2D for basin_plot.")

    return labels_arr, meta


def _resolve_xy(
    labels: np.ndarray,
    *,
    bounds: Sequence[tuple[float, float]] | None,
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = labels.shape
    if x is None and y is None:
        if bounds is None:
            x_vals = np.arange(nx, dtype=float)
            y_vals = np.arange(ny, dtype=float)
            return x_vals, y_vals
        if len(bounds) != 2:
            raise ValueError("bounds must be a length-2 sequence for 2D basin plots.")
        (x_min, x_max), (y_min, y_max) = bounds
        x_vals = np.linspace(float(x_min), float(x_max), nx)
        y_vals = np.linspace(float(y_min), float(y_max), ny)
        return x_vals, y_vals

    if x is None or y is None:
        raise ValueError("x and y must be provided together.")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x_arr.size != nx or y_arr.size != ny:
        raise ValueError("x/y lengths must match labels shape.")
    return x_arr, y_arr


def _resolve_attractor_ids(res, labels: np.ndarray) -> list[int]:
    registry = getattr(res, "registry", None)
    if registry:
        return [int(attr.id) for attr in registry]
    unique_ids = np.unique(labels[labels >= 0])
    return [int(x) for x in unique_ids.tolist()]


def _resolve_attractor_labels(
    meta: Mapping[str, object],
    attr_ids: Sequence[int],
    attractor_label: str,
) -> list[str]:
    labels = meta.get("attractor_labels")
    if labels is None:
        labels = meta.get("attractor_names")
    if labels is None:
        return [f"{attractor_label}{attr_id}" for attr_id in attr_ids]
    if isinstance(labels, Mapping):
        return [str(labels.get(attr_id, f"{attractor_label}{attr_id}")) for attr_id in attr_ids]
    try:
        labels_list = list(labels)
    except TypeError:
        return [f"{attractor_label}{attr_id}" for attr_id in attr_ids]
    if len(labels_list) != len(attr_ids):
        return [f"{attractor_label}{attr_id}" for attr_id in attr_ids]
    return [str(label) for label in labels_list]


def basin_plot(
    res,
    *,
    grid: Sequence[int] | int | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    labels: np.ndarray | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    ax=None,
    # color and label control
    special_order: Sequence[int] | None = None,
    special_colors: Sequence[Any] | None = None,
    special_labels: Mapping[int, str] | None = None,
    attractor_cmap: str | Colormap | None = "hsv",
    attractor_colors: Sequence[Any] | None = None,
    attractor_label: str = "A",
    attractor_labels: Sequence[str] | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = "Basin Classification",
    colorbar_label_rotation: float | None = 270.0,
    colorbar_labelpad: float | None = 20.0,
    colorbar_kwargs: Mapping[str, Any] | None = None,
    shading: str = "auto",
    alpha: float | None = None,
    aspect: str | None = "equal",
    # axes config
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlabel_rot: float | None = None,
    ylabel_rot: float | None = None,
    xpad: float | None = None,
    ypad: float | None = None,
    titlepad: float | None = None,
    xlabel_fs: float | None = None,
    ylabel_fs: float | None = None,
    title_fs: float | None = None,
    xtick_fs: float | None = None,
    ytick_fs: float | None = None,
):
    """
    Plot 2D basin-of-attraction results with categorical color mapping.

    Parameters
    ----------
    res : BasinResult or array-like
        BasinResult returned by ``analysis.basin_auto`` or raw labels array.
    grid : Sequence[int] | int, optional
        ``ic_grid`` used in basin_auto (nx, ny). Required for 1D labels
        unless present in ``res.meta['ic_grid']``.
    bounds : Sequence[tuple[float, float]], optional
        ``ic_bounds`` used for the IC grid. When provided, sets the axis scale,
        and defaults from ``res.meta['ic_bounds']`` when available.
    labels : np.ndarray, optional
        Override labels to plot (useful when res is not a BasinResult).
    x, y : np.ndarray, optional
        Explicit axis coordinates. Must be 1D arrays matching labels shape.
    Attractor labels fall back to ``res.meta['attractor_labels']`` or
    ``res.meta['attractor_names']`` when provided and ``attractor_labels`` is None.
    """
    labels_arr, meta = _resolve_labels(res, labels, grid)
    if bounds is None:
        meta_bounds = meta.get("ic_bounds")
        if meta_bounds is not None:
            bounds = meta_bounds
    x_vals, y_vals = _resolve_xy(labels_arr, bounds=bounds, x=x, y=y)
    attr_ids = _resolve_attractor_ids(res, labels_arr)

    special_order = list(special_order) if special_order is not None else [BLOWUP, OUTSIDE, UNRESOLVED]
    if len(set(special_order)) != len(special_order):
        raise ValueError("special_order must contain unique label ids.")

    default_labels = {
        BLOWUP: "Blowup",
        OUTSIDE: "Outside",
        UNRESOLVED: "Unresolved",
    }
    if special_labels:
        default_labels.update(special_labels)

    special_values = []
    special_map: dict[int, float] = {}
    for idx, label in enumerate(special_order):
        value = float(idx - len(special_order))
        special_map[int(label)] = value
        special_values.append(value)

    attr_id_to_index = {attr_id: idx for idx, attr_id in enumerate(attr_ids)}

    cmap_data = np.full(labels_arr.shape, np.nan, dtype=float)
    for label, value in special_map.items():
        cmap_data[labels_arr == label] = value
    for attr_id, idx in attr_id_to_index.items():
        cmap_data[labels_arr == attr_id] = float(idx)

    if np.isnan(cmap_data).any():
        unknown = np.unique(labels_arr[np.isnan(cmap_data)])
        raise ValueError(f"Unrecognized basin labels encountered: {unknown.tolist()}")

    default_special_colors = ["#2C2C2C", "#606060", "#A0A0A0"]
    if special_colors is None:
        if len(special_order) <= len(default_special_colors):
            special_colors = default_special_colors[: len(special_order)]
        else:
            special_colors = [plt.cm.Greys(i / (len(special_order) + 1)) for i in range(1, len(special_order) + 1)]
    elif len(special_colors) < len(special_order):
        raise ValueError("special_colors must have at least one entry per special label.")

    if attractor_colors is None:
        if attr_ids:
            cmap_obj = plt.get_cmap(attractor_cmap) if not isinstance(attractor_cmap, Colormap) else attractor_cmap
            attractor_colors = [
                cmap_obj(i / max(len(attr_ids), 1))
                for i in range(len(attr_ids))
            ]
        else:
            attractor_colors = []
    elif len(attractor_colors) < len(attr_ids):
        raise ValueError("attractor_colors must have at least one entry per attractor.")

    colors = list(special_colors) + list(attractor_colors)
    if not colors:
        raise ValueError("No colors available for basin_plot.")

    boundaries = []
    if special_values:
        boundaries.append(special_values[0] - 0.5)
        boundaries.extend([val + 0.5 for val in special_values])
    else:
        boundaries.append(-0.5)
    boundaries.extend([i + 0.5 for i in range(len(attr_ids))])

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))

    plot_ax = _get_ax(ax)
    im_kwargs: dict[str, Any] = {"shading": shading, "cmap": cmap, "norm": norm}
    if alpha is not None:
        im_kwargs["alpha"] = float(alpha)

    im = plot_ax.pcolormesh(x_vals, y_vals, cmap_data, **im_kwargs)

    if colorbar:
        ticks = special_values + list(range(len(attr_ids)))
        tick_labels = [default_labels.get(int(label), str(label)) for label in special_order]
        if attractor_labels is None:
            tick_labels += _resolve_attractor_labels(meta, attr_ids, attractor_label)
        else:
            if len(attractor_labels) != len(attr_ids):
                raise ValueError("attractor_labels must match number of attractors.")
            tick_labels += list(attractor_labels)

        cbar_kwargs = dict(colorbar_kwargs or {})
        cbar = plt.colorbar(im, ax=plot_ax, ticks=ticks, **cbar_kwargs)
        cbar.ax.set_yticklabels(tick_labels)
        if colorbar_label:
            label_kwargs: dict[str, Any] = {}
            if colorbar_label_rotation is not None:
                label_kwargs["rotation"] = float(colorbar_label_rotation)
            if colorbar_labelpad is not None:
                label_kwargs["labelpad"] = float(colorbar_labelpad)
            cbar.set_label(colorbar_label, **label_kwargs)
        setattr(plot_ax, "_last_colorbar", cbar)

    if xlabel is None and isinstance(meta.get("observe_vars"), (list, tuple)):
        obs = meta.get("observe_vars")
        if obs:
            xlabel = str(obs[0])
        if ylabel is None and len(obs) > 1:
            ylabel = str(obs[1])

    if xlim is None and bounds is not None:
        xlim = (float(bounds[0][0]), float(bounds[0][1]))
    if ylim is None and bounds is not None:
        ylim = (float(bounds[1][0]), float(bounds[1][1]))

    _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
    _apply_labels(
        plot_ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xpad=xpad,
        ypad=ypad,
        titlepad=titlepad,
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        title_fs=title_fs,
    )
    _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
    _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
    if aspect is not None:
        plot_ax.set_aspect(aspect)
    return plot_ax
