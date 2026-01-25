# src/dynlib/plot/_primitives.py
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from dynlib.runtime.workspace import make_runtime_workspace

from . import _theme
from ._fig import _resolve_figsize


# ============================================================================
# Style Presets for Different System Types
# ============================================================================

STYLE_PRESETS: dict[str, dict[str, Any]] = {
    # For continuous systems (flows/ODEs)
    # Note: Only visual pattern (line/marker presence), not sizes/widths (those come from theme)
    "continuous": {"linestyle": "-", "marker": ""},
    "cont": {"linestyle": "-", "marker": ""},
    "flow": {"linestyle": "-", "marker": ""},

    # For discrete systems (maps)
    "discrete": {"linestyle": "", "marker": "o"},
    "map": {"linestyle": "", "marker": "o"},

    # Mixed styles
    "mixed": {"linestyle": "-", "marker": "o"},
    "connected": {"linestyle": "-", "marker": "o"},

    # Other useful presets
    "scatter": {"linestyle": "", "marker": "o"},
    "line": {"linestyle": "-", "marker": ""},
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
        if bool(getattr(obj, "equations_use_lag", False)):
            raise RuntimeError("cobweb: model equations use lag(), so the helper cannot evaluate them safely.")
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

    # 3) v2 FullModel / ModelSpec path (no .map helper; use compiled rhs)
    spec = getattr(obj, "spec", None)
    rhs_fn = getattr(obj, "rhs", None)
    if spec is not None and callable(rhs_fn):
        if getattr(spec, "kind", None) != "map":
            raise TypeError("cobweb requires a discrete map model (spec.kind == 'map').")

        if bool(getattr(obj, "equations_use_lag", getattr(spec, "equations_use_lag", False))):
            raise RuntimeError("cobweb: model equations use lag(), so the helper cannot evaluate them safely.")

        state_names = tuple(getattr(spec, "states", ()))
        if not state_names:
            raise TypeError("Model spec missing states.")
        state_index = {name: idx for idx, name in enumerate(state_names)}

        param_names = tuple(getattr(spec, "params", ()))
        param_index = {name: idx for idx, name in enumerate(param_names)}

        if isinstance(state, int):
            tgt = int(state)
            if tgt < 0 or tgt >= len(state_names):
                raise IndexError("state index out of range.")
        else:
            if state is None and len(state_names) == 1:
                tgt = 0
            elif state is None:
                raise ValueError("Multi-dimensional model requires 'state' (name or index).")
            else:
                nm = str(state)
                if nm not in state_index:
                    raise KeyError(f"Unknown state '{nm}'.")
                tgt = state_index[nm]

        dtype = np.dtype(getattr(obj, "dtype", float))
        y_base = np.array(getattr(spec, "state_ic", ()), dtype=dtype, copy=True)
        params_base = np.array(getattr(spec, "param_vals", ()), dtype=dtype, copy=True)

        # apply fixed overrides (states/params; supports prefixes)
        if fixed:
            for k, v in fixed.items():
                val = float(v)
                if k.startswith("state__") or k.startswith("y__"):
                    nm = k.split("__", 1)[1]
                    if nm not in state_index:
                        raise KeyError(f"Unknown state '{nm}'.")
                    y_base[state_index[nm]] = val
                elif k.startswith("param__") or k.startswith("p__"):
                    nm = k.split("__", 1)[1]
                    if nm not in param_index:
                        raise KeyError(f"Unknown param '{nm}'.")
                    params_base[param_index[nm]] = val
                else:
                    if k in state_index:
                        y_base[state_index[k]] = val
                    elif k in param_index:
                        params_base[param_index[k]] = val
                    else:
                        raise KeyError(f"Unknown fixed key '{k}'.")

        r_idx = param_index.get("r")

        lag_state_info = getattr(obj, "lag_state_info", None)
        if lag_state_info is None:
            lag_map = getattr(spec, "lag_map", None) or {}
            lag_state_info = tuple(
                (state_index[name], int(depth), int(offset), int(head_index))
                for name, (depth, offset, head_index) in lag_map.items()
                if name in state_index
            )

        y_seed = y_base.copy()
        params_seed = params_base.copy()
        runtime_ws_seed = make_runtime_workspace(
            lag_state_info=lag_state_info,
            dtype=dtype,
        )

        def _fresh_runtime_ws():
            if (
                runtime_ws_seed.lag_ring.size == 0
                and runtime_ws_seed.lag_head.size == 0
            ):
                return runtime_ws_seed
            return type(runtime_ws_seed)(
                np.array(runtime_ws_seed.lag_ring, copy=True),
                np.array(runtime_ws_seed.lag_head, copy=True),
                runtime_ws_seed.lag_info,
                np.array(runtime_ws_seed.aux_values, copy=True),
                np.array(runtime_ws_seed.stop_flag, copy=True),
                np.array(runtime_ws_seed.stop_phase_mask, copy=True),
            )

        def g_model(k_iter: int, x: float) -> float:
            t_k = float(t0 + k_iter * dt)
            y = y_seed.copy()
            y[tgt] = float(x)
            params_arr = params_seed.copy()
            if r is not None and r_idx is not None:
                params_arr[r_idx] = float(r)
            y_next = np.empty_like(y_seed)
            runtime_ws = _fresh_runtime_ws()
            rhs_fn(t_k, y, y_next, params_arr, runtime_ws)
            return float(y_next[tgt])

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
        kwargs = {"labelpad": float(xpad)} if xpad is not None else {}
        if xlabel_fs is not None:
            kwargs["fontsize"] = float(xlabel_fs)
        ax.set_xlabel(xlabel, **kwargs)
    if ylabel is not None:
        kwargs = {"labelpad": float(ypad)} if ypad is not None else {}
        if ylabel_fs is not None:
            kwargs["fontsize"] = float(ylabel_fs)
        ax.set_ylabel(ylabel, **kwargs)
    if title is not None:
        kwargs = {"pad": float(titlepad)} if titlepad is not None else {}
        if title_fs is not None:
            kwargs["fontsize"] = float(title_fs)
        ax.set_title(title, **kwargs)
    _apply_ticks(ax)
    _apply_margins(ax)


def _apply_tick_fontsizes(ax: plt.Axes, *, xtick_fs: float | None, ytick_fs: float | None) -> None:
    if xtick_fs is not None:
        for tick in ax.get_xticklabels():
            tick.set_fontsize(float(xtick_fs))
    if ytick_fs is not None:
        for tick in ax.get_yticklabels():
            tick.set_fontsize(float(ytick_fs))


def _apply_decor(
    ax: plt.Axes,
    *,
    vlines: list[float | tuple[float, str]] | None = None,
    vbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
    vlines_kwargs: Mapping[str, Any] | None = None,
    hlines: list[float | tuple[float, str]] | None = None,
    hbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
    hlines_kwargs: Mapping[str, Any] | None = None,
) -> None:
    # --- Vertical Bands ---
    if vbands:
        for band in vbands:
            if len(band) == 2:
                start, end = band
                color = "C0"
            elif len(band) == 3:
                start, end, color = band
            else:
                raise ValueError(f"vband tuple must have 2 or 3 elements, got {len(band)}")
            if start >= end:
                raise ValueError(f"vband start must be less than end, got start={start}, end={end}")
            ax.axvspan(start, end, color=color, alpha=0.1)

    # --- Horizontal Bands ---
    if hbands:
        for band in hbands:
            if len(band) == 2:
                start, end = band
                color = "C0"
            elif len(band) == 3:
                start, end, color = band
            else:
                raise ValueError(f"hband tuple must have 2 or 3 elements, got {len(band)}")
            if start >= end:
                raise ValueError(f"hband start must be less than end, got start={start}, end={end}")
            ax.axhspan(start, end, color=color, alpha=0.1)

    # --- Vertical Lines ---
    if vlines is not None:
        # Draw vlines as usual (data x, span full y-range)
        default_vl_kw: dict[str, Any] = {
            "color": "black",
            "linestyle": "--",
            "linewidth": 1.0,
            "alpha": 0.7,
        }
        merged_vl_kw = {**default_vl_kw, **dict(vlines_kwargs)} if vlines_kwargs else default_vl_kw

        # Use current y-limits to place labels INSIDE the axes, in data coords
        ymin, ymax = ax.get_ylim()
        # vertical_pad controls WHERE vertically to place the label (top/bottom positioning)
        # This is for placement, not offset from the line
        vertical_pad = merged_vl_kw.get('placement_pad', float(_theme.get("vline_label_placement_pad")))  # Default from theme
        # horizontal_pad controls the OFFSET from the vertical line (perpendicular distance)
        # This uses the theme default vline_label_pad
        horizontal_pad = merged_vl_kw.get('label_pad', float(_theme.get("vline_label_pad")))
        label_position = merged_vl_kw.get('label_position', 'top')

        # Calculate y_text based on position
        if label_position == 'top':
            # If vertical_pad < 1: interpret as fraction of axis height down from the top
            # If vertical_pad >= 1: interpret as data-units down from the top
            if vertical_pad < 1.0:
                y_text = ymax - vertical_pad * (ymax - ymin)
            else:
                y_text = ymax - vertical_pad
            va = "top"
        elif label_position == 'bottom':
            # If vertical_pad < 1: interpret as fraction of axis height up from the bottom
            # If vertical_pad >= 1: interpret as data-units up from the bottom
            if vertical_pad < 1.0:
                y_text = ymin + vertical_pad * (ymax - ymin)
            else:
                y_text = ymin + vertical_pad
            va = "bottom"
        elif label_position == 'center':
            # Center vertically in the axis
            y_text = (ymin + ymax) / 2.0
            va = "center"
        else:
            raise ValueError(f"label_position must be 'top', 'bottom', or 'center', got {label_position}")

        for item in vlines:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                x, label = item
            else:
                x = float(item)
                label = ""
            # Remove label-specific and placement kwargs before passing to axvline
            vline_kw = {k: v for k, v in merged_vl_kw.items() 
                        if not k.startswith('label_') and k != 'placement_pad'}
            ax.axvline(x, **vline_kw)
            if label:
                label_rotation = merged_vl_kw.get('label_rotation', 90)
                label_color = merged_vl_kw.get('label_color', merged_vl_kw.get('color'))
                text_kwargs = {}
                if label_color is not None:
                    text_kwargs['color'] = label_color
                # Apply horizontal offset: if < 1, treat as fraction of x-range; else as data units
                xmin_cur, xmax_cur = ax.get_xlim()
                if abs(horizontal_pad) < 1.0:
                    x_offset = horizontal_pad * (xmax_cur - xmin_cur)
                else:
                    x_offset = horizontal_pad
                x_text = x + x_offset
                ax.text(
                    x_text,
                    y_text,
                    label,
                    rotation=label_rotation,
                    va=va,        # anchor based on position
                    ha="center",
                    transform=ax.transData,  # <-- data coordinates now
                    clip_on=True,    # keep it inside the axes
                    rotation_mode="anchor",
                    **text_kwargs,
                )

    # --- Horizontal Lines ---
    if hlines is not None:
        default_hl_kw: dict[str, Any] = {
            "color": "black",
            "linestyle": "--",
            "linewidth": 1.0,
            "alpha": 0.7,
        }
        merged_hl_kw = {**default_hl_kw, **dict(hlines_kwargs)} if hlines_kwargs else default_hl_kw

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # horizontal_pad controls WHERE horizontally to place the label (left/right positioning)
        # This is for placement, not offset from the line
        horizontal_pad = merged_hl_kw.get('placement_pad', float(_theme.get("hline_label_placement_pad")))  # Default from theme
        # vertical_pad controls the OFFSET from the horizontal line (perpendicular distance)
        # This uses the theme default hline_label_pad
        vertical_pad = merged_hl_kw.get('label_pad', float(_theme.get("hline_label_pad")))
        label_position = merged_hl_kw.get('label_position', 'right')

        # Calculate x_text based on position
        if label_position == 'right':
            # If horizontal_pad < 1: interpret as fraction of axis width from right
            # If horizontal_pad >= 1: interpret as data-units from right
            if horizontal_pad < 1.0:
                x_text = xmax - horizontal_pad * (xmax - xmin)
            else:
                x_text = xmax - horizontal_pad
            ha = "right"
        elif label_position == 'left':
            # If horizontal_pad < 1: interpret as fraction of axis width from left
            # If horizontal_pad >= 1: interpret as data-units from left
            if horizontal_pad < 1.0:
                x_text = xmin + horizontal_pad * (xmax - xmin)
            else:
                x_text = xmin + horizontal_pad
            ha = "left"
        elif label_position == 'center':
            # Center horizontally in the axis
            x_text = (xmin + xmax) / 2.0
            ha = "center"
        else:
            raise ValueError(f"label_position must be 'left', 'right', or 'center', got {label_position}")

        for item in hlines:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                y, label = item
            else:
                y = float(item)
                label = ""
            # Remove label-specific and placement kwargs before passing to axhline
            hline_kw = {k: v for k, v in merged_hl_kw.items() 
                        if not k.startswith('label_') and k != 'placement_pad'}
            ax.axhline(y, **hline_kw)
            if label:
                label_rotation = merged_hl_kw.get('label_rotation', 0)
                label_color = merged_hl_kw.get('label_color', merged_hl_kw.get('color'))
                text_kwargs = {}
                if label_color is not None:
                    text_kwargs['color'] = label_color
                # Apply vertical offset: if < 1, treat as fraction of y-range; else as data units
                ymin_cur, ymax_cur = ax.get_ylim()
                if abs(vertical_pad) < 1.0:
                    y_offset = vertical_pad * (ymax_cur - ymin_cur)
                else:
                    y_offset = vertical_pad
                y_text = y + y_offset
                ax.text(
                    x_text,
                    y_text,
                    label,
                    rotation=label_rotation,
                    va="center",  # center on the offset position
                    ha=ha,
                    transform=ax.transData,
                    clip_on=True,
                    **text_kwargs,
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
    
    Priority hierarchy (highest to lowest):
      1. Explicit function arguments (color=, lw=, ls=, marker=, ms=, alpha=)
      2. Style dict values (if style is a dict)
      3. Style preset values (if style is a preset name like "continuous")
      4. Theme defaults (line_w, marker_size, etc.)
    
    Note: Style presets define visual PATTERNS (line/marker presence),
          while themes define rendering PROPERTIES (sizes, widths, colors).
    """
    result: dict[str, Any] = {}

    # Apply style preset or dict (visual pattern)
    if isinstance(style, str):
        if style in STYLE_PRESETS:
            result.update(STYLE_PRESETS[style])
        else:
            raise ValueError(
                f"Unknown style preset '{style}'. Available: {', '.join(sorted(STYLE_PRESETS.keys()))}"
            )
    elif isinstance(style, dict):
        result.update(style)

    # Apply explicit overrides (highest priority)
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

    # Fill in missing values from theme (lowest priority)
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


def _style_overrides_from_mapping(style: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if "color" in style:
        overrides["color"] = style["color"]
    if "lw" in style:
        overrides["linewidth"] = float(style["lw"])
    if "ls" in style:
        overrides["linestyle"] = style["ls"]
    if "marker" in style:
        overrides["marker"] = style["marker"]
    if "ms" in style:
        overrides["markersize"] = float(style["ms"])
    if "alpha" in style:
        overrides["alpha"] = style["alpha"]
    return overrides


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
    y: Mapping[str, Any] | list[tuple[Any, ...]] | np.ndarray | list[np.ndarray],
    names: Sequence[str] | None = None,
) -> list[tuple[str, Any, Mapping[str, Any]]]:
    # 1) Mapping: existing behaviour
    if isinstance(y, Mapping):
        return [(name, values, {}) for name, values in y.items()]

    # 2) ndarray: new path
    if isinstance(y, np.ndarray):
        arr = np.asarray(y)
        if arr.ndim == 1:
            name = names[0] if names else "y"
            return [(name, arr, {})]
        if arr.ndim == 2:
            M, K = arr.shape
            if names is not None and len(names) != K:
                raise ValueError(f"names length ({len(names)}) must match y.shape[1] ({K})")
            result: list[tuple[str, Any, Mapping[str, Any]]] = []
            for j in range(K):
                name = names[j] if names is not None else f"y{j}"
                result.append((name, arr[:, j], {}))
            return result
        raise ValueError("ndarray y must be 1D or 2D")

    # 3) list of ndarrays: new path
    if isinstance(y, list) and y and isinstance(y[0], np.ndarray):
        if names is not None and len(names) != len(y):
            raise ValueError(f"names length ({len(names)}) must match number of y ({len(y)})")
        result: list[tuple[str, Any, Mapping[str, Any]]] = []
        for i, arr in enumerate(y):
            name = names[i] if names is not None else f"y{i}"
            result.append((name, arr, {}))
        return result

    # 4) list-of-tuples: existing behaviour
    normalized: list[tuple[str, Any, Mapping[str, Any]]] = []
    for entry in y:
        if not isinstance(entry, tuple):
            raise TypeError("Y entries must be tuples.")
        if len(entry) == 2:
            name, values = entry
            style: Mapping[str, Any] = {}
        elif len(entry) == 3:
            name, values, style = entry
            if not isinstance(style, Mapping):
                raise TypeError("Style entries must be mappings.")
        else:
            raise ValueError("Y tuples must have 2 or 3 elements.")
        normalized.append((name, values, style))
    return normalized


def _coerce_segments(segments: Any) -> list[np.ndarray]:
    if segments is None:
        return []
    if isinstance(segments, np.ndarray):
        arr = np.asarray(segments)
        if arr.ndim != 2:
            raise ValueError("segments array must be 2D")
        return [arr]
    if isinstance(segments, (list, tuple)):
        out: list[np.ndarray] = []
        for seg in segments:
            arr = np.asarray(seg)
            if arr.ndim != 2:
                raise ValueError("segments entries must be 2D arrays")
            out.append(arr)
        return out
    raise TypeError("segments must be an array or a sequence of arrays")


def _coerce_manifold_groups(
    *,
    segments: Any = None,
    branches: Any = None,
    result: Any = None,
    groups: Any = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    if groups is not None:
        out: list[dict[str, Any]] = []
        for entry in groups:
            if isinstance(entry, Mapping):
                if "segments" not in entry:
                    raise ValueError("group mapping must include 'segments'")
                out.append(
                    {
                        "segments": _coerce_segments(entry["segments"]),
                        "label": entry.get("label"),
                        "style": entry.get("style"),
                    }
                )
            elif isinstance(entry, (list, tuple)):
                if len(entry) < 1 or len(entry) > 3:
                    raise ValueError("group tuples must be (segments, label?, style?)")
                out.append(
                    {
                        "segments": _coerce_segments(entry[0]),
                        "label": entry[1] if len(entry) >= 2 else None,
                        "style": entry[2] if len(entry) == 3 else None,
                    }
                )
            else:
                raise TypeError("groups must contain mappings or tuples")
        return out

    if result is not None:
        branches = getattr(result, "branches", None)
        if branches is None:
            raise TypeError("result must provide a 'branches' attribute")
        if label is None:
            label = getattr(result, "kind", None)

    if branches is not None:
        if not isinstance(branches, (list, tuple)) or len(branches) != 2:
            raise ValueError("branches must be a 2-tuple of segment lists")
        label_main = label
        return [
            {"segments": _coerce_segments(branches[0]), "label": label_main, "style": None},
            {"segments": _coerce_segments(branches[1]), "label": None, "style": None},
        ]

    if segments is None:
        raise ValueError("Provide segments, branches, result, or groups")
    return [{"segments": _coerce_segments(segments), "label": label, "style": None}]


# ----------------------------------------------------------------------------
# Public plotting primitives (array-only)
# ----------------------------------------------------------------------------

def manifold(
    *,
    segments=None,
    branches=None,
    result=None,
    groups=None,
    components: tuple[int, int] = (0, 1),
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
    ax=None,
    legend: bool = True,
    aspect: str | None = None,
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
    """
    Plot 1D manifolds represented as segmented trajectories.

    Accepts raw segments, branch tuples, or result objects with a ``branches``
    attribute (e.g., ManifoldTraceResult). For higher-dimensional systems,
    select which components to plot via ``components``.
    """
    if len(components) != 2:
        raise ValueError("components must be a 2-tuple like (0, 1)")
    c0, c1 = (int(components[0]), int(components[1]))
    if c0 == c1:
        raise ValueError("components entries must be distinct")

    plot_ax = _get_ax(ax)
    base_style = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
    groups_list = _coerce_manifold_groups(
        segments=segments, branches=branches, result=result, groups=groups, label=label
    )

    for group in groups_list:
        segs = [seg for seg in group["segments"] if seg.shape[0] >= 2]
        if not segs:
            continue
        max_dim = max(c0, c1)
        for seg in segs:
            if seg.shape[1] <= max_dim:
                raise ValueError("segment dimensionality is smaller than components")

        style_args = dict(base_style)
        group_style = group.get("style")
        if group_style is not None:
            if isinstance(group_style, str):
                if group_style not in STYLE_PRESETS:
                    raise ValueError(
                        f"Unknown style preset '{group_style}'. Available: {', '.join(sorted(STYLE_PRESETS.keys()))}"
                    )
                style_args.update(STYLE_PRESETS[group_style])
            elif isinstance(group_style, Mapping):
                style_args.update(_style_overrides_from_mapping(group_style))
                passthrough = {
                    k: v
                    for k, v in group_style.items()
                    if k not in ("color", "lw", "ls", "marker", "ms", "alpha")
                }
                style_args.update(passthrough)
            else:
                raise TypeError("group style must be a preset name or a mapping")

        marker_val = style_args.get("marker")
        wants_markers = bool(marker_val) and marker_val != "None"
        label_val = group.get("label")

        if wants_markers:
            for idx, seg in enumerate(segs):
                lbl = label_val if (label_val and idx == 0) else None
                plot_ax.plot(seg[:, c0], seg[:, c1], label=lbl, **style_args)
        else:
            lines = [seg[:, (c0, c1)] for seg in segs]
            lc_kwargs: dict[str, Any] = {}
            if "color" in style_args:
                lc_kwargs["colors"] = style_args["color"]
            if "linewidth" in style_args:
                lc_kwargs["linewidths"] = style_args["linewidth"]
            if "linestyle" in style_args:
                ls_val = style_args["linestyle"]
                if ls_val not in ("", "None", None):
                    lc_kwargs["linestyles"] = ls_val
            if "alpha" in style_args:
                lc_kwargs["alpha"] = style_args["alpha"]
            lc = LineCollection(lines, label=label_val, **lc_kwargs)
            plot_ax.add_collection(lc)

    if any(group.get("label") for group in groups_list) and legend:
        plot_ax.legend()

    _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
    if aspect is not None:
        plot_ax.set_aspect(aspect)
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
        vlines: list[float | tuple[float, str]] | None = None,
        vlines_color: str | None = None,
        vlines_kwargs: Mapping[str, Any] | None = None,
        vbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
        hlines: list[float | tuple[float, str]] | None = None,
        hlines_color: str | None = None,
        hlines_kwargs: Mapping[str, Any] | None = None,
        hbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
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
        """
        Plot a single series as y versus x.
        
        The fundamental time-series plotting primitive. Supports continuous
        and discrete system visualization.
        
        Args:
            x: Time or independent variable array
            y: Values array
            label: Legend label for the series
            style: Style preset name or custom dict (e.g., 'continuous', 'discrete', 'mixed')
            color: Line/marker color
            lw: Line width
            ls: Line style ('-', '--', '-.', ':')
            marker: Marker style ('o', 's', '^', etc.)
            ms: Marker size
            alpha: Transparency (0-1)
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            vlines: Vertical lines at specified x values or (x, label) tuples
            vlines_color: Color for vertical lines
            vlines_kwargs: Additional kwargs for vertical lines
            vbands: Vertical shaded regions as (start, end) or (start, end, color) tuples
            hlines: Horizontal lines at specified y values or (y, label) tuples
            hlines_color: Color for horizontal lines
            hlines_kwargs: Additional kwargs for horizontal lines
            hbands: Horizontal shaded regions as (start, end) or (start, end, color) tuples
            legend: Whether to show legend if label is provided
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            ax: Existing axes to plot on
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object
        """
        x_vals = _resolve_time(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)

        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot(x_vals, y_vals, label=label, **style_args)

        if label and legend:
            plot_ax.legend()
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        # Merge simple vlines_color into vlines_kwargs for convenience
        merged_vlines_kwargs: Mapping[str, Any] | None
        if vlines_color is not None:
            base_vl_kw = {} if vlines_kwargs is None else dict(vlines_kwargs)
            base_vl_kw["color"] = vlines_color
            merged_vlines_kwargs = base_vl_kw
        else:
            merged_vlines_kwargs = vlines_kwargs

        merged_hlines_kwargs: Mapping[str, Any] | None
        if hlines_color is not None:
            base_hl_kw = {} if hlines_kwargs is None else dict(hlines_kwargs)
            base_hl_kw["color"] = hlines_color
            merged_hlines_kwargs = base_hl_kw
        else:
            merged_hlines_kwargs = hlines_kwargs

        _apply_decor(
            plot_ax,
            vlines=vlines,
            vbands=vbands,
            vlines_kwargs=merged_vlines_kwargs,
            hlines=hlines,
            hbands=hbands,
            hlines_kwargs=merged_hlines_kwargs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def stem(
        self,
        *,
        x,
        y,
        label: str | None = None,
        style: str | dict[str, Any] | None = None,
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
        vlines: list[float | tuple[float, str]] | None = None,
        vlines_color: str | None = None,
        vlines_kwargs: Mapping[str, Any] | None = None,
        vbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
        hlines: list[float | tuple[float, str]] | None = None,
        hlines_color: str | None = None,
        hlines_kwargs: Mapping[str, Any] | None = None,
        hbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
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
        """
        Stem plot showing discrete data points with vertical lines to baseline.
        
        Particularly useful for discrete-time systems and impulse responses.
        
        Args:
            x: Time or independent variable array
            y: Values array
            label: Legend label for the series
            style: Style preset name or custom dict (e.g., 'discrete', 'map')
            color: Color for stems and markers
            lw: Line width for stems
            marker: Marker style ('o', 's', '^', etc.)
            ms: Marker size
            alpha: Transparency (0-1)
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            vlines: Vertical lines at specified x values or (x, label) tuples
            vlines_color: Color for vertical lines
            vlines_kwargs: Additional kwargs for vertical lines
            vbands: Vertical shaded regions as (start, end) or (start, end, color) tuples
            hlines: Horizontal lines at specified y values or (y, label) tuples
            hlines_color: Color for horizontal lines
            hlines_kwargs: Additional kwargs for horizontal lines
            hbands: Horizontal shaded regions as (start, end) or (start, end, color) tuples
            legend: Whether to show legend if label is provided
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            ax: Existing axes to plot on
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object
        """
        x_vals = _resolve_time(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)

        style_args = _resolve_style(style, color=color, lw=lw, ls=None, marker=marker, ms=ms, alpha=alpha)
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

        if label and legend:
            plot_ax.legend()
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        merged_vlines_kwargs: Mapping[str, Any] | None
        if vlines_color is not None:
            base_vl_kw = {} if vlines_kwargs is None else dict(vlines_kwargs)
            base_vl_kw["color"] = vlines_color
            merged_vlines_kwargs = base_vl_kw
        else:
            merged_vlines_kwargs = vlines_kwargs

        merged_hlines_kwargs: Mapping[str, Any] | None
        if hlines_color is not None:
            base_hl_kw = {} if hlines_kwargs is None else dict(hlines_kwargs)
            base_hl_kw["color"] = hlines_color
            merged_hlines_kwargs = base_hl_kw
        else:
            merged_hlines_kwargs = hlines_kwargs

        _apply_decor(
            plot_ax,
            vlines=vlines,
            vbands=vbands,
            vlines_kwargs=merged_vlines_kwargs,
            hlines=hlines,
            hbands=hbands,
            hlines_kwargs=merged_hlines_kwargs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def step(
        self,
        *,
        x,
        y,
        label: str | None = None,
        style: str | dict[str, Any] | None = None,
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
        vlines: list[float | tuple[float, str]] | None = None,
        vlines_color: str | None = None,
        vlines_kwargs: Mapping[str, Any] | None = None,
        vbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
        hlines: list[float | tuple[float, str]] | None = None,
        hlines_color: str | None = None,
        hlines_kwargs: Mapping[str, Any] | None = None,
        hbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
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
        """
        Step plot where values are held constant until the next point.
        
        Useful for discrete-time signals or piecewise-constant functions.
        
        Args:
            x: Time or independent variable array
            y: Values array
            label: Legend label for the series
            style: Style preset name or custom dict (e.g., 'continuous', 'discrete')
            color: Line/marker color
            lw: Line width
            ls: Line style ('-', '--', '-.', ':')
            marker: Marker style ('o', 's', '^', etc.)
            ms: Marker size
            alpha: Transparency (0-1)
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            vlines: Vertical lines at specified x values or (x, label) tuples
            vlines_color: Color for vertical lines
            vlines_kwargs: Additional kwargs for vertical lines
            vbands: Vertical shaded regions as (start, end) or (start, end, color) tuples
            hlines: Horizontal lines at specified y values or (y, label) tuples
            hlines_color: Color for horizontal lines
            hlines_kwargs: Additional kwargs for horizontal lines
            hbands: Horizontal shaded regions as (start, end) or (start, end, color) tuples
            legend: Whether to show legend if label is provided
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            ax: Existing axes to plot on
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object
        """
        x_vals = _resolve_time(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)

        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.step(x_vals, y_vals, where="post", label=label, **style_args)

        if label and legend:
            plot_ax.legend()
        _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
        _apply_labels(
            plot_ax, xlabel=xlabel, ylabel=ylabel, title=title,
            xpad=xpad, ypad=ypad, titlepad=titlepad,
            xlabel_fs=xlabel_fs, ylabel_fs=ylabel_fs, title_fs=title_fs,
        )
        
        merged_vlines_kwargs: Mapping[str, Any] | None
        if vlines_color is not None:
            base_vl_kw = {} if vlines_kwargs is None else dict(vlines_kwargs)
            base_vl_kw["color"] = vlines_color
            merged_vlines_kwargs = base_vl_kw
        else:
            merged_vlines_kwargs = vlines_kwargs

        merged_hlines_kwargs: Mapping[str, Any] | None
        if hlines_color is not None:
            base_hl_kw = {} if hlines_kwargs is None else dict(hlines_kwargs)
            base_hl_kw["color"] = hlines_color
            merged_hlines_kwargs = base_hl_kw
        else:
            merged_hlines_kwargs = hlines_kwargs

        _apply_decor(
            plot_ax,
            vlines=vlines,
            vbands=vbands,
            vlines_kwargs=merged_vlines_kwargs,
            hlines=hlines,
            hbands=hbands,
            hlines_kwargs=merged_hlines_kwargs,
        )
        _apply_tick_rotation(plot_ax, xlabel_rot=xlabel_rot, ylabel_rot=ylabel_rot, theme=_theme)
        _apply_tick_fontsizes(plot_ax, xtick_fs=xtick_fs, ytick_fs=ytick_fs)
        return plot_ax

    def multi(
        self,
        *,
        x,
        y: Mapping[str, Any] | list[tuple[Any, ...]] | np.ndarray | list[np.ndarray],
        names: Sequence[str] | None = None,
        styles: Mapping[str, Mapping[str, Any]] | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        vlines: list[float | tuple[float, str]] | None = None,
        vlines_color: str | None = None,
        vlines_kwargs: Mapping[str, Any] | None = None,
        vbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
        hlines: list[float | tuple[float, str]] | None = None,
        hlines_color: str | None = None,
        hlines_kwargs: Mapping[str, Any] | None = None,
        hbands: list[tuple[float, float] | tuple[float, float, str]] | None = None,
        legend: bool = False,
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
        """
        Plot multiple named series against a common x-axis.
        
        Useful for comparing multiple state variables or different trajectories.
        
        Args:
            x: Time or independent variable array (shared by all series)
            y: Either a dict {name: values}, list of (name, values) or
                   (name, values, style_dict) tuples, a numpy ndarray, or list of ndarrays
            names: Optional sequence of names for ndarray y columns
            styles: Optional dict mapping series names to style dicts
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            vlines: Vertical lines at specified x values or (x, label) tuples
            vlines_color: Color for vertical lines
            vlines_kwargs: Additional kwargs for vertical lines
            vbands: Vertical shaded regions as (start, end) or (start, end, color) tuples
            hlines: Horizontal lines at specified y values or (y, label) tuples
            hlines_color: Color for horizontal lines
            hlines_kwargs: Additional kwargs for horizontal lines
            hbands: Horizontal shaded regions as (start, end) or (start, end, color) tuples
            legend: Whether to show legend
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            ax: Existing axes to plot on
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object
        """
        x_vals = _resolve_time(x)
        plot_ax = _get_ax(ax)
        base_styles = styles or {}
        for name, values, inline_style in _coerce_series(y, names=names):
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
        
        merged_vlines_kwargs: Mapping[str, Any] | None
        if vlines_color is not None:
            base_vl_kw = {} if vlines_kwargs is None else dict(vlines_kwargs)
            base_vl_kw["color"] = vlines_color
            merged_vlines_kwargs = base_vl_kw
        else:
            merged_vlines_kwargs = vlines_kwargs

        merged_hlines_kwargs: Mapping[str, Any] | None
        if hlines_color is not None:
            base_hl_kw = {} if hlines_kwargs is None else dict(hlines_kwargs)
            base_hl_kw["color"] = hlines_color
            merged_hlines_kwargs = base_hl_kw
        else:
            merged_hlines_kwargs = hlines_kwargs

        _apply_decor(
            plot_ax,
            vlines=vlines,
            vbands=vbands,
            vlines_kwargs=merged_vlines_kwargs,
            hlines=hlines,
            hbands=hbands,
            hlines_kwargs=merged_hlines_kwargs,
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
        ax=None,
        equil: list[tuple[float, float]] | None = None,
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
    ) -> plt.Axes:
        """
        Plot 2D trajectory through phase space (y versus x).
        
        Useful for visualizing system dynamics in state space, showing
        trajectories, limit cycles, and equilibrium points.
        
        Args:
            x: First state variable array
            y: Second state variable array
            label: Legend label for the trajectory
            style: Style preset name or custom dict (e.g., 'continuous', 'discrete')
            color: Line/marker color
            lw: Line width
            ls: Line style ('-', '--', '-.', ':')
            marker: Marker style ('o', 's', '^', etc.)
            ms: Marker size
            alpha: Transparency (0-1)
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            ax: Existing axes to plot on
            equil: List of equilibrium points as (x, y) tuples to mark
            legend: Whether to show legend if label is provided
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object
        """
        x_vals = _resolve_value(x)
        y_vals = _resolve_value(y)
        plot_ax = _get_ax(ax)
        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot(x_vals, y_vals, label=label, **style_args)
        if equil:
            for ex, ey in equil:
                plot_ax.plot(ex, ey, marker="o", linestyle="None")
        if label and legend:
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

    def xyz(
        self,
        *,
        x,
        y,
        z,
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
        zlim: tuple[float | None, float | None] | None = None,
        labels: tuple[str | None, str | None, str | None] | None = None,
        title: str | None = None,
        ax=None,
        legend: bool = True,
        xpad: float | None = None,
        ypad: float | None = None,
        zpad: float | None = None,
        titlepad: float | None = None,
        xlabel_fs: float | None = None,
        ylabel_fs: float | None = None,
        zlabel_fs: float | None = None,
        title_fs: float | None = None,
        xtick_fs: float | None = None,
        ytick_fs: float | None = None,
        ztick_fs: float | None = None,
    ):
        """
        Plot 3D trajectory through phase space.
        
        Useful for visualizing dynamics of 3D systems like Lorenz attractor,
        Rössler system, or other chaotic systems.
        
        Args:
            x: First state variable array
            y: Second state variable array
            z: Third state variable array
            label: Legend label for the trajectory
            style: Style preset name or custom dict (e.g., 'continuous', 'discrete')
            color: Line/marker color
            lw: Line width
            ls: Line style ('-', '--', '-.', ':')
            marker: Marker style ('o', 's', '^', etc.)
            ms: Marker size
            alpha: Transparency (0-1)
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            zlim: Z-axis limits as (min, max)
            labels: Tuple of (xlabel, ylabel, zlabel) for axis labels
            title: Plot title
            ax: Existing 3D axes to plot on
            legend: Whether to show legend if label is provided
            xpad: X-axis label padding
            ypad: Y-axis label padding
            zpad: Z-axis label padding
            titlepad: Title padding (vertical position adjustment)
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            zlabel_fs: Z-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            ztick_fs: Z-axis tick label font size
            
        Returns:
            Matplotlib 3D axes object
        """
        x_vals = _resolve_value(x)
        y_vals = _resolve_value(y)
        z_vals = _resolve_value(z)
        plot_ax = _get_ax(ax, projection="3d")
        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot3D(x_vals, y_vals, z_vals, label=label, **style_args)
        if label and legend:
            plot_ax.legend()
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
            title_y_pos = float(titlepad) if titlepad is not None else 1.02
            plot_ax.set_title(title, y=title_y_pos, fontsize=float(title_fs) if title_fs is not None else None)
        
        # Apply tick font sizes for 3D plot
        if xtick_fs is not None:
            for tick in plot_ax.get_xticklabels():
                tick.set_fontsize(float(xtick_fs))
        if ytick_fs is not None:
            for tick in plot_ax.get_yticklabels():
                tick.set_fontsize(float(ytick_fs))
        if ztick_fs is not None:
            for tick in plot_ax.get_zticklabels():
                tick.set_fontsize(float(ztick_fs))
        
        return plot_ax

    def multi(
        self,
        *,
        x,
        y,
        labels: Sequence[str] | None = None,
        styles: Mapping[str, Mapping[str, Any]] | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        legend: bool = False,
        xlabel_rot: float | None = None,
        ylabel_rot: float | None = None,
        ax=None,
        equil: list[tuple[float, float]] | None = None,
        xpad: float | None = None,
        ypad: float | None = None,
        titlepad: float | None = None,
        xlabel_fs: float | None = None,
        ylabel_fs: float | None = None,
        title_fs: float | None = None,
        xtick_fs: float | None = None,
        ytick_fs: float | None = None,
    ) -> plt.Axes:
        """
        Plot multiple 2D phase trajectories on the same axes.
        
        Useful for visualizing parameter sweeps in phase space, showing how
        trajectories evolve across different parameter values or initial conditions.
        
        Args:
            x: X-coordinate data - can be:
               - List of arrays (one per trajectory)
               - 2D numpy array of shape (M, N) where M is number of trajectories
               - List from sweep result accessor like result["x"]
            y: Y-coordinate data - same structure as x
            labels: Optional sequence of labels for each trajectory
            styles: Optional dict mapping label names to style dicts
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            legend: Whether to show legend
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            ax: Existing axes to plot on
            equil: List of equilibrium points as (x, y) tuples to mark
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object
            
        Examples:
            >>> # With sweep results
            >>> sweep_res = sweep.traj_sweep(sim, param="r", values=[...], record_vars=["x", "y"])
            >>> phase.multi(x=sweep_res["x"], y=sweep_res["y"], 
            ...             labels=[f"r={v:.2f}" for v in sweep_res.values])
            
            >>> # With explicit arrays
            >>> x_trajs = [traj1_x, traj2_x, traj3_x]
            >>> y_trajs = [traj1_y, traj2_y, traj3_y]
            >>> phase.multi(x=x_trajs, y=y_trajs, labels=["IC1", "IC2", "IC3"])
        """
        plot_ax = _get_ax(ax)
        
        # Normalize x and y to list of arrays
        # Handle both sweep result format (N, M) and list format
        if isinstance(x, np.ndarray):
            x_arr = np.asarray(x)
            if x_arr.ndim == 1:
                x_list = [x_arr]
            elif x_arr.ndim == 2:
                # Check if it's sweep format (N, M) or trajectory format (M, N)
                # Sweep format: time is leading axis, so shape is (N_time, M_params)
                # We want M trajectories, so split along axis=1
                if x_arr.shape[1] <= x_arr.shape[0]:
                    # Likely sweep format: (N_time, M_params)
                    x_list = [x_arr[:, i] for i in range(x_arr.shape[1])]
                else:
                    # Likely trajectory format: (M_params, N_time)
                    x_list = [x_arr[i, :] for i in range(x_arr.shape[0])]
            else:
                raise ValueError("x array must be 1D or 2D")
        elif isinstance(x, list):
            x_list = [_ensure_array(xi) for xi in x]
        else:
            raise TypeError("x must be ndarray or list of arrays")
        
        if isinstance(y, np.ndarray):
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_list = [y_arr]
            elif y_arr.ndim == 2:
                # Use same logic as x
                if y_arr.shape[1] <= y_arr.shape[0]:
                    y_list = [y_arr[:, i] for i in range(y_arr.shape[1])]
                else:
                    y_list = [y_arr[i, :] for i in range(y_arr.shape[0])]
            else:
                raise ValueError("y array must be 1D or 2D")
        elif isinstance(y, list):
            y_list = [_ensure_array(yi) for yi in y]
        else:
            raise TypeError("y must be ndarray or list of arrays")
        
        if len(x_list) != len(y_list):
            raise ValueError(f"x and y must have same number of trajectories ({len(x_list)} vs {len(y_list)})")
        
        # Generate labels if not provided
        if labels is None:
            label_list = [f"traj{i}" for i in range(len(x_list))]
        else:
            label_list = list(labels)
            if len(label_list) != len(x_list):
                raise ValueError(f"labels length ({len(label_list)}) must match number of trajectories ({len(x_list)})")
        
        # Plot each trajectory
        base_styles = styles or {}
        for i, (x_vals, y_vals, label) in enumerate(zip(x_list, y_list, label_list)):
            # Get style for this trajectory
            style_dict = base_styles.get(label, {})
            style_args = _style_from_mapping(style_dict)
            plot_ax.plot(x_vals, y_vals, label=label, **style_args)
        
        # Mark equilibrium points if provided
        if equil:
            for ex, ey in equil:
                plot_ax.plot(ex, ey, marker="o", linestyle="None", color="red", markersize=8, zorder=10)
        
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

    def return_map(
        self,
        *,
        x,
        step: int = 1,
        label: str | None = None,
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
        equil: list[float] | None = None,
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
    ) -> plt.Axes:
        """
        Plot return map showing x[n+step] versus x[n].
        
        Useful for analyzing discrete maps, finding fixed points, and
        detecting periodic orbits. Fixed points appear on the identity line.
        
        Args:
            x: Time series array
            step: Step size for the return map (default=1 for first return)
            label: Legend label for the series
            style: Style preset name or custom dict (default='map' for discrete)
            color: Line/marker color
            lw: Line width
            ls: Line style ('-', '--', '-.', ':')
            marker: Marker style ('o', 's', '^', etc.)
            ms: Marker size
            alpha: Transparency (0-1)
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label (default: '$x[n]$')
            ylabel: Y-axis label (default: '$x[n+1]$' or '$x[n+step]$')
            title: Plot title
            ax: Existing axes to plot on
            equil: List of fixed point values to mark on identity line
            legend: Whether to show legend if label is provided
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object
        """
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        x_vals = _resolve_value(x)
        if len(x_vals) <= step:
            raise ValueError(f"Series length ({len(x_vals)}) must be > step ({step})")

        x_n = x_vals[:-step]
        x_n_lag = x_vals[step:]

        plot_ax = _get_ax(ax)
        style_args = _resolve_style(style, color=color, lw=lw, ls=ls, marker=marker, ms=ms, alpha=alpha)
        plot_ax.plot(x_n, x_n_lag, label=label, **style_args)
        
        if equil:
            for eq_val in equil:
                plot_ax.plot(eq_val, eq_val, marker="o", linestyle="None", color="red", markersize=8, zorder=10)

        if label and legend:
            plot_ax.legend()

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


class _UtilsPlot:
    """General-purpose plotting utilities."""
    def hist(
        self,
        y,
        *,
        bins: int = 50,
        density: bool = False,
        label: str | None = None,
        color: str | None = None,
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
    ) -> plt.Axes:
        """
        Histogram of data values showing distribution.

        Useful for analyzing stationary distributions, invariant measures,
        and statistical properties of dynamical systems.
        """
        data = _resolve_value(y)
        plot_ax = _get_ax(ax)
        hist_kwargs: dict[str, Any] = {"bins": bins, "density": density}
        if color is not None:
            hist_kwargs["color"] = color
        if alpha is not None:
            hist_kwargs["alpha"] = alpha
        if label is not None:
            hist_kwargs["label"] = label
        plot_ax.hist(data, **hist_kwargs)

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
        return plot_ax

    def image(
        self,
        Z,
        *,
        extent=None,
        label: str | None = None,
        alpha: float | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        ylim: tuple[float | None, float | None] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        colorbar: bool = False,
        legend: bool = False,
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
        """
        Display 2D array as an image or heatmap.
        
        Useful for bifurcation diagrams, Poincaré sections, basin boundaries,
        Lyapunov exponent maps, and other 2D visualizations.
        
        Args:
            Z: 2D data array to display
            extent: Axis extent as [left, right, bottom, top] for scaling
            label: Legend label (rarely used for images)
            alpha: Transparency (0-1)
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            colorbar: Whether to show colorbar
            legend: Whether to show legend if label is provided
            xlabel_rot: X-axis label rotation angle
            ylabel_rot: Y-axis label rotation angle
            ax: Existing axes to plot on
            xpad: X-axis label padding
            ypad: Y-axis label padding
            titlepad: Title padding
            xlabel_fs: X-axis label font size
            ylabel_fs: Y-axis label font size
            title_fs: Title font size
            xtick_fs: X-axis tick label font size
            ytick_fs: Y-axis tick label font size
            
        Returns:
            Matplotlib axes object (with _last_colorbar attribute if colorbar=True)
        """
        data = _ensure_array(Z)
        plot_ax = _get_ax(ax)
        im_kwargs: dict[str, Any] = {"aspect": "auto", "extent": extent, "origin": "lower"}
        if alpha is not None:
            im_kwargs["alpha"] = alpha
        if label is not None:
            im_kwargs["label"] = label
        im = plot_ax.imshow(data, **im_kwargs)
        
        if label and legend:
            plot_ax.legend()
        
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
utils = _UtilsPlot()

__all__ = ["series", "phase", "utils", "manifold"]
