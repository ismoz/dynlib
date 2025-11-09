# src/dynlib/plot/_primitives.py
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from . import _theme
from ._fig import _resolve_figsize


# ============================================================================
# Style Presets for Different System Types
# ============================================================================

STYLE_PRESETS: dict[str, dict[str, Any]] = {
    # For continuous systems (flows/ODEs)
    "continuous": {"linestyle": "-", "marker": "", "markersize": 0},
    "cont": {"linestyle": "-", "marker": "", "markersize": 0},
    "flow": {"linestyle": "-", "marker": "", "markersize": 0},

    # For discrete systems (maps)
    "discrete": {"linestyle": "", "marker": "o", "markersize": 4},
    "map": {"linestyle": "", "marker": "o", "markersize": 4},

    # Mixed styles
    "mixed": {"linestyle": "-", "marker": "o", "markersize": 4},
    "connected": {"linestyle": "-", "marker": "o", "markersize": 4},

    # Other useful presets
    "scatter": {"linestyle": "", "marker": "o", "markersize": 6},
    "line": {"linestyle": "-", "marker": "", "markersize": 0},
}


# ----------------------------------------------------------------------------
# Figure/Axes helpers
# ----------------------------------------------------------------------------

def _get_ax(ax=None, *, projection: str | None = None) -> plt.Axes:
    if ax is not None:
        return ax
    subplot_kw = {"projection": projection} if projection else None
    _fig, created_ax = plt.subplots(
        figsize=_resolve_figsize(None, None),
        subplot_kw=subplot_kw,
        layout="constrained",
    )
    return created_ax


def _ensure_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


# ----------------------------------------------------------------------------
# Value/time resolution (array-only; no Results/type coupling)
# ----------------------------------------------------------------------------

def _resolve_time(t: Any) -> np.ndarray:
    return _ensure_array(t)


def _resolve_value(value: Any) -> np.ndarray:
    if isinstance(value, str):
        raise TypeError("String keys are not supported by plot primitives. Pass arrays directly.")
    return _ensure_array(value)


# ----------------------------------------------------------------------------
# Model-aware utility for cobweb plots (kept here; array-only elsewhere)
# ----------------------------------------------------------------------------

def _resolve_unary_map_k(
    obj,
    *,
    state: str | int | None,
    fixed: Mapping[str, float] | None,
    r: float | None,
    t0: float,
    dt: float,
):
    """
    Return g(k, x) -> x_next.

    Supports:
      - callable: f(x) or f(x, r)   (k/t ignored)
      - Model:    map(k, t, y, p)   (project to chosen state; freeze others)
      - Sim:      unwrap .model
    """

    # Unwrap Sim-like
    model = getattr(obj, "model", None)
    if model is not None:
        obj = model

    # 1) Direct callable
    if callable(obj) and not hasattr(obj, "map"):
        fn = obj

        def g_callable(k_iter: int, x: float) -> float:
            if r is None:
                return float(fn(float(x)))
            try:
                return float(fn(float(x), float(r)))
            except TypeError:
                return float(fn(float(x)))

        return g_callable

    # 2) Model with map(k, t, y, p)
    map_fn = getattr(obj, "map", None)
    if callable(map_fn):
        if not hasattr(obj, "_state_names") or not hasattr(obj, "_state_index"):
            raise TypeError("Model lacks state metadata.")
        # determine target state index
        if isinstance(state, int):
            tgt = int(state)
            if tgt < 0 or tgt >= len(obj._state_names):
                raise IndexError("state index out of range.")
        else:
            if state is None and len(obj._state_names) == 1:
                tgt = 0
            elif state is None:
                raise ValueError("Multi-dimensional model requires 'state' (name or index).")
            else:
                nm = str(state)
                if nm not in obj._state_index:
                    raise KeyError(f"Unknown state '{nm}'.")
                tgt = obj._state_index[nm]

        y_base = obj.ic
        p_ns = obj.p

        # apply fixed overrides (states/params; supports prefixes)
        if fixed:
            for k, v in fixed.items():
                if k.startswith("state__") or k.startswith("y__"):
                    nm = k.split("__", 1)[1]
                    y_base[obj._state_index[nm]] = float(v)
                elif k.startswith("param__") or k.startswith("p__"):
                    nm = k.split("__", 1)[1]
                    setattr(p_ns, nm, float(v))
                else:
                    if k in obj._state_index:
                        y_base[obj._state_index[k]] = float(v)
                    elif k in obj._param_index:
                        setattr(p_ns, k, float(v))
                    else:
                        raise KeyError(f"Unknown fixed key '{k}'.")

        has_r = hasattr(p_ns, "r")

        def g_model(k_iter: int, x: float) -> float:
            t_k = float(t0 + k_iter * dt)
            y = y_base.copy()
            y[tgt] = float(x)
            if r is not None and has_r:
                old_r = getattr(p_ns, "r")
                setattr(p_ns, "r", float(r))
                try:
                    y_next = map_fn(int(k_iter), t_k, y, p_ns)
                finally:
                    setattr(p_ns, "r", old_r)
            else:
                y_next = map_fn(int(k_iter), t_k, y, p_ns)
            return float(np.asarray(y_next, dtype=obj._dtype)[tgt])

        return g_model

    raise TypeError("cobweb: 'f' must be callable g(x) or a Model with map(k, t, y, p), or a Sim exposing .model.")


# ----------------------------------------------------------------------------
# Styling helpers
# ----------------------------------------------------------------------------

def _apply_ticks(ax: plt.Axes) -> None:
    if getattr(ax, "name", "") == "3d":
        return
    tick_n = int(_theme.get("tick_n"))
    if tick_n > 0:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_n))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_n))
    if bool(_theme.get("minor_ticks")):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def _apply_margins(ax: plt.Axes) -> None:
    if getattr(ax, "name", "") == "3d":
        return
    ax.margins(x=float(_theme.get("xmargin")), y=float(_theme.get("ymargin")))


def _apply_limits(
    ax: plt.Axes,
    *,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    zlim: tuple[float | None, float | None] | None = None,
) -> None:
    """Apply axis limits to the plot."""
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None and hasattr(ax, "set_zlim"):
        ax.set_zlim(zlim)


def _apply_labels(
    ax: plt.Axes,
    *,
    xlabel: str | None,
    ylabel: str | None,
    title: str | None,
    xpad: float | None = None,
    ypad: float | None = None,
    titlepad: float | None = None,
    xlabel_fs: float | None = None,
    ylabel_fs: float | None = None,
    title_fs: float | None = None,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            labelpad=float(xpad) if xpad is not None else None,
            fontsize=float(xlabel_fs) if xlabel_fs is not None else None,
        )
    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            labelpad=float(ypad) if ypad is not None else None,
            fontsize=float(ylabel_fs) if ylabel_fs is not None else None,
        )
    if title is not None:
        ax.set_title(
            title,
            pad=float(titlepad) if titlepad is not None else None,
            fontsize=float(title_fs) if title_fs is not None else None,
        )
    _apply_ticks(ax)
    _apply_margins(ax)


def _apply_tick_fontsizes(ax: plt.Axes, *, xtick_fs: float | None, ytick_fs: float | None) -> None:
    if xtick_fs is not None:
        for tick in ax.get_xticklabels():
            tick.set_fontsize(float(xtick_fs))
    if ytick_fs is not None:
        for tick in ax.get_yticklabels():
            tick.set_fontsize(float(ytick_fs))


def _apply_time_decor(
    ax: plt.Axes,
    events: list[tuple[float, str]] | None,
    bands: list[tuple[float, float]] | None,
    events_kwargs: Mapping[str, Any] | None = None,
) -> None:
    if bands:
        for start, end in bands:
            ax.axvspan(start, end, color="C0", alpha=0.1)
    if events:
        transform = ax.get_xaxis_transform()
        yoff = 1.0 + float(_theme.get("event_label_pad"))
        # Default styling for event vertical lines
        default_ev_kw: dict[str, Any] = {
            "color": "black",
            "linestyle": "--",
            "linewidth": 1.0,
            "alpha": 0.7,
        }
        merged_ev_kw = {**default_ev_kw, **dict(events_kwargs)} if events_kwargs else default_ev_kw
        for x, label in events:
            ax.axvline(x, **merged_ev_kw)
            if label:
                ax.text(
                    x,
                    yoff,
                    label,
                    rotation=90,
                    va="bottom",
                    ha="center",
                    transform=transform,
                    clip_on=False,
                    rotation_mode="anchor",
                )


def _resolve_style(
    style: str | dict[str, Any] | None = None,
    *,
    color: str | None = None,
    lw: float | None = None,
    ls: str | None = None,
    marker: str | None = None,
    ms: float | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """
    Resolve style from preset name or custom dict, with explicit overrides.
    Priority: explicit args > dict > preset > theme defaults
    """
    result: dict[str, Any] = {}

    if isinstance(style, str):
        if style in STYLE_PRESETS:
            result.update(STYLE_PRESETS[style])
        else:
            raise ValueError(
                f"Unknown style preset '{style}'. Available: {', '.join(sorted(STYLE_PRESETS.keys()))}"
            )
    elif isinstance(style, dict):
        result.update(style)

    if color is not None:
        result["color"] = color
    if lw is not None:
        result["linewidth"] = float(lw)
    if ls is not None:
        result["linestyle"] = ls
    if marker is not None:
        result["marker"] = marker
    if ms is not None:
        result["markersize"] = float(ms)
    if alpha is not None:
        result["alpha"] = alpha

    if "linewidth" not in result:
        result["linewidth"] = float(_theme.get("line_w"))
    if "marker" not in result:
        default_marker = _theme.get("marker")
        if default_marker:
            result["marker"] = default_marker
    if "markersize" not in result:
        result["markersize"] = float(_theme.get("marker_size"))
    if "alpha" not in result:
        default_alpha = _theme.get("alpha")
        if default_alpha is not None:
            result["alpha"] = default_alpha

    return result


def _style_kwargs(
    *,
    color: str | None = None,
    lw: float | None = None,
    ls: str | None = None,
    marker: str | None = None,
    ms: float | None = None,
    alpha: float | None = None,
) -> dict:
    kw: dict[str, Any] = {}
    if color is not None:
        kw["color"] = color
    default_lw = float(_theme.get("line_w"))
    lw_val = default_lw if lw is None else float(lw)
    if lw_val is not None:
        kw["linewidth"] = lw_val
    if ls is not None:
        kw["linestyle"] = ls
    default_marker = _theme.get("marker")
    marker_val = default_marker if marker is None else marker
    if marker_val:
        kw["marker"] = marker_val
    default_ms = float(_theme.get("marker_size"))
    ms_val = default_ms if ms is None else float(ms)
    if ms_val is not None:
        kw["markersize"] = ms_val
    default_alpha = _theme.get("alpha")
    alpha_val = default_alpha if alpha is None else alpha
    if alpha_val is not None:
        kw["alpha"] = alpha_val
    return kw


def _style_from_mapping(style: Mapping[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key in ("color", "lw", "ls", "marker", "ms", "alpha"):
        if key in style:
            filtered[key] = style[key]
    if not filtered:
        return {}
    return _style_kwargs(**filtered)


def _scatter_kwargs(style_kw: dict[str, Any]) -> dict[str, Any]:
    scatter_kw: dict[str, Any] = {}
    for key, value in style_kw.items():
        if key == "linewidth":
            scatter_kw["linewidths"] = value
        elif key == "linestyle":
            continue
        elif key == "markersize":
            scatter_kw["s"] = float(value) ** 2
        else:
            scatter_kw[key] = value
    return scatter_kw


def _apply_tick_rotation(ax: plt.Axes, *, xlabel_rot: float | None, ylabel_rot: float | None, theme=_theme) -> None:
    tick_x = float(theme.get("tick_label_rot_x"))
    tick_y = float(theme.get("tick_label_rot_y"))
    if tick_x:
        for label in ax.get_xticklabels():
            label.set_rotation(tick_x)
    if tick_y:
        for label in ax.get_yticklabels():
            label.set_rotation(tick_y)
    if xlabel_rot is not None:
        ax.xaxis.label.set_rotation(float(xlabel_rot))
    if ylabel_rot is not None:
        ax.yaxis.label.set_rotation(float(ylabel_rot))


def _coerce_series(
    series: Mapping[str, Any] | list[tuple[Any, ...]]
) -> list[tuple[str, Any, Mapping[str, Any]]]:
    if isinstance(series, Mapping):
        return [(name, values, {}) for name, values in series.items()]

    normalized: list[tuple[str, Any, Mapping[str, Any]]] = []
    for entry in series:
        if not isinstance(entry, tuple):
            raise TypeError("Series entries must be tuples.")
        if len(entry) == 2:
            name, values = entry
            style: Mapping[str, Any] = {}
        elif len(entry) == 3:
            name, values, style = entry
            if not isinstance(style, Mapping):
                raise TypeError("Style entries must be mappings.")
        else:
            raise ValueError("Series tuples must have 2 or 3 elements.")
        normalized.append((name, values, style))
    return normalized


# ----------------------------------------------------------------------------
# Public plotting primitives (array-only)
# ----------------------------------------------------------------------------

class _SeriesPlot:
    def plot(
        self,
        *,
        x,
        y,
        label: str | None = None,
        style: str | dict[str, Any] | None = "continuous",
        color: str | None = None,
        lw: float | None = None,
        ls: str | None = None,
        marker: str | None = None,
        ms: float | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        events: list[tuple[float, str]] | None = None,
        events_color: str | None = None,
        events_kwargs: Mapping[str, Any] | None = None,
        bands: list[tuple[float, float]] | None = None,
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
    ) -> plt.Axes:
        """Plot a series (x vs y). Array-only API."""
        x_vals = _resolve_time(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)

        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot(x_vals, y_vals, label=label, **style_args)

        if label:
            plot_ax.legend()
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        # Merge simple events_color into events_kwargs for convenience
        merged_events_kwargs: Mapping[str, Any] | None
        if events_color is not None:
            base_ev_kw = {} if events_kwargs is None else dict(events_kwargs)
            base_ev_kw["color"] = events_color
            merged_events_kwargs = base_ev_kw
        else:
            merged_events_kwargs = events_kwargs

        _apply_time_decor(plot_ax, events, bands, merged_events_kwargs)
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def stem(
        self,
        *,
        x,
        y,
        label: str | None = None,
        color: str | None = None,
        lw: float | None = None,
        marker: str | None = None,
        ms: float | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        events: list[tuple[float, str]] | None = None,
        events_color: str | None = None,
        events_kwargs: Mapping[str, Any] | None = None,
        bands: list[tuple[float, float]] | None = None,
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
    ) -> plt.Axes:
        """Stem plot (array-only)."""
        x_vals = _resolve_time(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)

        style_args = _style_kwargs(color=color, lw=lw, ls=None, marker=marker, ms=ms, alpha=alpha)
        markerline, stemlines, baseline = plot_ax.stem(  # noqa: F841
            x_vals, y_vals, linefmt="-", markerfmt="o", basefmt=" "
        )
        if label:
            markerline.set_label(label)
        if style_args:
            if "color" in style_args:
                markerline.set_color(style_args["color"])
                stemlines.set_color(style_args["color"])
            if "linewidth" in style_args:
                stemlines.set_linewidth(style_args["linewidth"])
            if "marker" in style_args:
                markerline.set_marker(style_args["marker"])
            if "markersize" in style_args:
                markerline.set_markersize(style_args["markersize"])
            if "alpha" in style_args:
                markerline.set_alpha(style_args["alpha"])
                stemlines.set_alpha(style_args["alpha"])

        if label:
            plot_ax.legend()
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        merged_events_kwargs: Mapping[str, Any] | None
        if events_color is not None:
            base_ev_kw = {} if events_kwargs is None else dict(events_kwargs)
            base_ev_kw["color"] = events_color
            merged_events_kwargs = base_ev_kw
        else:
            merged_events_kwargs = events_kwargs

        _apply_time_decor(plot_ax, events, bands, merged_events_kwargs)
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def step(
        self,
        *,
        x,
        y,
        label: str | None = None,
        color: str | None = None,
        lw: float | None = None,
        ls: str | None = None,
        marker: str | None = None,
        ms: float | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
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
    ) -> plt.Axes:
        """Step plot (array-only). Values are held until next point."""
        x_vals = _resolve_time(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)

        style_args = _style_kwargs(color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.step(x_vals, y_vals, where="post", label=label, **style_args)

        if label:
            plot_ax.legend()
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def multi(
        self,
        *,
        x,
        series: Mapping[str, Any] | list[tuple[Any, ...]],
        styles: Mapping[str, Mapping[str, Any]] | None = None,
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
    ) -> plt.Axes:
        """Plot multiple named series against x. Array-only."""
        x_vals = _resolve_time(x)
        plot_ax = _get_ax(ax)
        base_styles = styles or {}
        for name, values, inline_style in _coerce_series(series):
            y_vals = _resolve_value(values)
            combined_style: dict[str, Any] = {}
            if name in base_styles:
                combined_style.update(base_styles[name])
            if inline_style:
                combined_style.update(inline_style)
            style_args = _style_from_mapping(combined_style)
            plot_ax.plot(x_vals, y_vals, label=name, **style_args)
        if legend:
            plot_ax.legend()
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax


class _PhasePlot:
    def xy(
        self,
        *,
        x,
        y,
        style: str | dict[str, Any] | None = "continuous",
        color: str | None = None,
        lw: float | None = None,
        ls: str | None = None,
        marker: str | None = None,
        ms: float | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        ax=None,
        equil: list[tuple[float, float]] | None = None,
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
    ) -> plt.Axes:
        """Plot 2D trajectory through phase space (y vs x). Array-only."""
        x_vals = _resolve_value(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)
        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot(x_vals, y_vals, **style_args)
        if equil:
            for ex, ey in equil:
                plot_ax.plot(ex, ey, marker="o", linestyle="None")
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def xyz(
        self,
        *,
        x,
        y,
        z,
        style: str | dict[str, Any] | None = "continuous",
        color: str | None = None,
        lw: float | None = None,
        ls: str | None = None,
        marker: str | None = None,
        ms: float | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        zlim: tuple[float | None, float | None] | None = None,
        labels: tuple[str | None, str | None, str | None] | None = None,
        title: str | None = None,
        ax=None,
        xpad: float | None = None,
        ypad: float | None = None,
        zpad: float | None = None,
        title_y: float | None = None,
        xlabel_fs: float | None = None,
        ylabel_fs: float | None = None,
        zlabel_fs: float | None = None,
        title_fs: float | None = None,
    ):
        """Plot a 3D trajectory (x, y, z). Array-only."""
        x_vals = _resolve_value(x)
        y_vals = _resolve_value(y)
        z_vals = _resolve_value(z)
        plot_ax = _get_ax(ax, projection="3d")
        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot3D(x_vals, y_vals, z_vals, **style_args)
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim, zlim=zlim)
        if labels:
            xlabel, ylabel, zlabel = labels
            if xlabel:
                plot_ax.set_xlabel(
                    xlabel,
                    labelpad=float(xpad) if xpad is not None else None,
                    fontsize=float(xlabel_fs) if xlabel_fs is not None else None,
                )
            if ylabel:
                plot_ax.set_ylabel(
                    ylabel,
                    labelpad=float(ypad) if ypad is not None else None,
                    fontsize=float(ylabel_fs) if ylabel_fs is not None else None,
                )
            if zlabel:
                plot_ax.set_zlabel(
                    zlabel,
                    labelpad=float(zpad) if zpad is not None else None,
                    fontsize=float(zlabel_fs) if zlabel_fs is not None else None,
                )
        if title:
            title_y_pos = float(title_y) if title_y is not None else 1.02
            plot_ax.set_title(title, y=title_y_pos, fontsize=float(title_fs) if title_fs is not None else None)
        return plot_ax

    def return_map(
        self,
        *,
        x,
        step: int = 1,
        style: str | dict[str, Any] | None = "map",
        color: str | None = None,
        lw: float | None = None,
        ls: str | None = None,
        marker: str | None = None,
        ms: float | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        ax=None,
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
    ) -> plt.Axes:
        """Plot return map: x[n] vs x[n+step]. Array-only."""
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        x_vals = _resolve_value(x)
        if len(x_vals) <= step:
            raise ValueError(f"Series length ({len(x_vals)}) must be > step ({step})")

        x_n = x_vals[:-step]
        x_n_lag = x_vals[step:]

        plot_ax = _get_ax(ax)
        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot(x_n, x_n_lag, **style_args)

        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)

        if xlabel is None:
            xlabel = "$x[n]$"
        if ylabel is None:
            ylabel = "$x[n+1]$" if step == 1 else f"$x[n+{step}]$"

        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax


class _AnalysisPlot:
    """Specialized analysis and diagnostic plots for dynamical systems."""
    def hist(
        self,
        y,
        *,
        bins: int = 50,
        density: bool = False,
        color: str | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
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
    ) -> plt.Axes:
        """Histogram of data values. Array-only."""
        data = _resolve_value(y)
        plot_ax = _get_ax(ax)
        hist_kwargs: dict[str, Any] = {"bins": bins, "density": density}
        if color is not None:
            hist_kwargs["color"] = color
        if alpha is not None:
            hist_kwargs["alpha"] = alpha
        plot_ax.hist(data, **hist_kwargs)
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def cobweb(
        self,
        *,
        f,
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
        xpad: float | None = None,
        ypad: float | None = None,
        titlepad: float | None = None,
        xlabel_fs: float | None = None,
        ylabel_fs: float | None = None,
        title_fs: float | None = None,
        xtick_fs: float | None = None,
        ytick_fs: float | None = None,
    ) -> plt.Axes:
        """Cobweb plot for 1-D discrete map analysis."""
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
        plot_ax.plot(xs, ys, label="f(x)", **style_args)  # f(x)

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
        plot_ax.legend()

        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax


class _UtilsPlot:
    """General-purpose plotting utilities."""
    def image(
        self,
        Z,
        *,
        extent=None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        colorbar: bool = False,
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
    ) -> plt.Axes:
        """Display 2D array as an image/heatmap."""
        data = _ensure_array(Z)
        plot_ax = _get_ax(ax)
        im_kwargs: dict[str, Any] = {"aspect": "auto", "extent": extent, "origin": "lower"}
        if alpha is not None:
            im_kwargs["alpha"] = alpha
        im = plot_ax.imshow(data, **im_kwargs)
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        if colorbar:
            cbar = plot_ax.figure.colorbar(im, ax=plot_ax)
            setattr(plot_ax, "_last_colorbar", cbar)
        return plot_ax


# Create module-level instances
series = _SeriesPlot()
phase = _PhasePlot()
analysis = _AnalysisPlot()
utils = _UtilsPlot()

__all__ = ["series", "phase", "analysis", "utils"]
