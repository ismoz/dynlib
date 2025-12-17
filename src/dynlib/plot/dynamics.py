"""Domain-specific plots for dynamical systems (public API: top-level re-exports)."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _theme
from ._primitives import (
    _apply_labels,
    _apply_limits,
    _apply_tick_fontsizes,
    _apply_tick_rotation,
    _get_ax,
    _style_kwargs,
    _resolve_unary_map_k,
)

__all__ = ["cobweb"]


def cobweb(
    *,
    f: Any,
    x0: float,
    steps: int = 50,
    t0: float = 0.0,
    dt: float = 1.0,
    state: str | int | None = None,
    fixed: dict[str, float] | None = None,
    r: float | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    ax=None,
    # styling
    color: str | None = None,
    lw: float | None = None,
    ls: str | None = None,
    alpha: float | None = None,
    # specific cobweb parts
    identity_color: str | None = None,
    stair_color: str | None = None,
    stair_lw: float | None = 0.5,
    stair_ls: str | None = None,
    # labels
    xlabel: str | None = "x",
    ylabel: str | None = "f(x)",
    title: str | None = None,
    legend: bool = True,
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
    Cobweb plot for analyzing 1D discrete map dynamics.

    Visualizes iteration trajectories by drawing vertical and horizontal
    lines between the function curve and identity line.
    """
    g = _resolve_unary_map_k(f, state=state, fixed=fixed, r=r, t0=t0, dt=dt)

    # orbit
    x = float(x0)
    orbit = [x]
    for k in range(steps):
        x = g(k, x)
        orbit.append(x)
    orbit_arr = np.asarray(orbit, dtype=float)

    # auto limits
    lo = float(np.min(orbit_arr))
    hi = float(np.max(orbit_arr))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    auto_lim = (lo - pad, hi + pad)

    # xlim
    if xlim is None:
        xlim_resolved = auto_lim
    else:
        xlim_resolved = (
            xlim[0] if xlim[0] is not None else auto_lim[0],
            xlim[1] if xlim[1] is not None else auto_lim[1],
        )

    # ylim
    if ylim is None:
        ylim_resolved = xlim_resolved
    else:
        ylim_resolved = (
            ylim[0] if ylim[0] is not None else xlim_resolved[0],
            ylim[1] if ylim[1] is not None else xlim_resolved[1],
        )

    # sample f at k=0 (t=t0)
    xs = np.linspace(xlim_resolved[0], xlim_resolved[1], 400)
    ys = np.asarray([g(0, x) for x in xs], dtype=float)

    plot_ax = _get_ax(ax)
    style_args = _style_kwargs(color=color, lw=lw, ls=ls, marker=None, ms=None, alpha=alpha)
    plot_ax.plot(xs, ys, label="f(x)", **style_args)

    # identity line
    id_kw: dict[str, Any] = {"linestyle": "--"}
    if identity_color is not None:
        id_kw["color"] = identity_color
    else:
        id_kw.setdefault("color", "gray")
    plot_ax.plot(xs, xs, label="y = x", **id_kw)

    # staircase
    stair_kw: dict[str, Any] = {}
    if stair_color is not None:
        stair_kw["color"] = stair_color
    elif "color" in style_args:
        stair_kw["color"] = style_args["color"]
    else:
        stair_kw.setdefault("color", "black")
    if stair_lw is not None:
        stair_kw["linewidth"] = float(stair_lw)
    elif "linewidth" in style_args:
        stair_kw["linewidth"] = style_args["linewidth"]
    else:
        stair_kw.setdefault("linewidth", 0.5)
    if stair_ls is not None:
        stair_kw["linestyle"] = stair_ls
    elif "linestyle" in style_args:
        stair_kw["linestyle"] = style_args["linestyle"]

    for start, end in zip(orbit_arr[:-1], orbit_arr[1:]):
        plot_ax.plot([start, start], [start, end], **stair_kw)
        plot_ax.plot([start, end], [end, end], **stair_kw)

    _apply_limits(plot_ax, xlim=xlim_resolved, ylim=ylim_resolved)
    if legend:
        plot_ax.legend()

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
    return plot_ax

