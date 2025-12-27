# src/dynlib/plot/bifurcation.py
"""Bifurcation diagram plotting utilities."""

from __future__ import annotations

from typing import Any

from . import _theme
from ._primitives import (
    _apply_decor,
    _apply_labels,
    _apply_limits,
    _apply_tick_fontsizes,
    _apply_tick_rotation,
    _ensure_array,
    _get_ax,
    _style_kwargs,
)

__all__ = ["bifurcation_diagram"]


def _coerce_bifurcation_inputs(res: Any, *, xlabel: str | None, ylabel: str | None, title: str | None):
    if hasattr(res, "p") and hasattr(res, "y"):
        p_vals = _ensure_array(res.p)
        y_vals = _ensure_array(res.y)
        if xlabel is None:
            xlabel = getattr(res, "param_name", None)
        if ylabel is None and getattr(res, "meta", None):
            ylabel = res.meta.get("var")
        if title is None and getattr(res, "mode", None):
            title = f"mode={res.mode}"
        return p_vals, y_vals, xlabel, ylabel, title

    if not isinstance(res, (tuple, list)) or len(res) != 2:
        raise TypeError("Pass a BifurcationResult or (p, y) tuple/array.")
    p_vals = _ensure_array(res[0])
    y_vals = _ensure_array(res[1])
    return p_vals, y_vals, xlabel, ylabel, title


def bifurcation_diagram(
    res,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str | None = None,
    ms: float | None = None,
    lw: float | None = None,
    ls: str | None = None,
    alpha: float | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
    xlabel_rot: float | None = None,
    ylabel_rot: float | None = None,
    ax=None,
    xpad: float | None = None,
    ypad: float | None = None,
    titlepad: float | None = None,
    xlabel_fs: float | None = None,
    ylabel_fs: float | None = None,
    title_fs: float | None = None,
    xtick_fs: float | None = None,
    ytick_fs: float | None = None,
    vlines: list[float | tuple[float, str]] | None = None,
    vlines_color: str | None = None,
    vlines_kwargs: dict[str, Any] | None = None,
):
    """
    Scatter-style bifurcation diagram plot.

    Accepts a BifurcationResult or raw ``(p, y)`` arrays.
    
    Bifurcation-specific defaults (can be overridden by explicit arguments):
    - marker: "," (pixel marker)
    - marker_size: 0.5 (does not matter for marker=",")
    - alpha: 0.5
    """
    p_vals, y_vals, xlabel, ylabel, title = _coerce_bifurcation_inputs(
        res, xlabel=xlabel, ylabel=ylabel, title=title
    )

    if p_vals.shape != y_vals.shape:
        raise ValueError(f"p and y must have the same shape; got {p_vals.shape} vs {y_vals.shape}")

    # Apply bifurcation-specific theme defaults (only if not explicitly provided)
    bifurc_defaults = {}
    if marker is None:
        bifurc_defaults["marker"] = ","
    if ms is None:
        bifurc_defaults["marker_size"] = 0.5
    if alpha is None:
        bifurc_defaults["alpha"] = 0.5
    
    with _theme.temp(**bifurc_defaults):
        plot_ax = _get_ax(ax)
        style = _style_kwargs(color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        style.setdefault("linestyle", "")
        plot_ax.plot(p_vals, y_vals, label=label, **style)

        if label and legend:
            plot_ax.legend()

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
        
        # Apply decorations (vlines, etc.)
        if vlines_color is not None:
            if vlines_kwargs is None:
                vlines_kwargs = {}
            vlines_kwargs = dict(vlines_kwargs)
            vlines_kwargs["color"] = vlines_color
        
        _apply_decor(
            plot_ax,
            vlines=vlines,
            vlines_kwargs=vlines_kwargs,
        )
    
    return plot_ax
