# src/dynlib/plot/_theme.py
from __future__ import annotations

from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any, Dict

import matplotlib as mpl
from cycler import cycler

_BASE_TOKENS: Dict[str, Any] = {
    "scale": 1.0,
    "font": "DejaVu Sans",
    "mono_font": "DejaVu Sans Mono",
    "line_w": 1.8,
    "marker": "",
    "marker_size": 4.0,
    "alpha": 1.0,
    "tick_n": 5,
    "tick_len": 4.0,
    "tick_w": 0.9,
    "tick_label_rot_x": 0.0,
    "tick_label_rot_y": 0.0,
    "label_pad": 6.0,
    "title_pad": 6.0,
    "tick_pad": 3.0,
    "minor_ticks": False,
    "axis_w": 1.2,
    "xmargin": 0.02,
    "ymargin": 0.05,
    "vline_label_pad": 0.07,
    "grid": True,
    "grid_alpha": 0.3,
    "palette": "cbf",
    "color_cycle": "cbf",
    "background": "light",
    "usetex": False,
    "legend_loc": "best",
    "legend_frame": False,
}

_PALETTES: Dict[str, list[str]] = {
    "classic": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    "cbf": [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ],
    "mono": [
        "#111111",
        "#444444",
        "#777777",
        "#aaaaaa",
        "#cccccc",
        "#eeeeee",
    ],
}

_PRESETS: Dict[str, Dict[str, Any]] = {
    "notebook": {
        "scale": 1.0,
        "grid": True,
        "palette": "cbf",
        "color_cycle": "cbf",
        "background": "light",
    },
    "paper": {
        "scale": 0.9,
        "grid": False,
        "tick_n": 6,
        "line_w": 1.2,
        "axis_w": 1.0,
        "marker_size": 5.0,
    },
    "talk": {
        "scale": 1.4,
        "grid": True,
        "line_w": 2.4,
        "tick_len": 5.0,
        "tick_w": 1.1,
    },
    "dark": {
        "background": "dark",
        "grid": True,
        "grid_alpha": 0.2,
        "palette": "cbf",
        "color_cycle": "cbf",
        "tick_w": 1.1,
        "axis_w": 1.4,
    },
    "mono": {
        "palette": "mono",
        "color_cycle": "mono",
        "grid": False,
        "background": "light",
    },
}

_FONT_SIZES = {
    "font.size": 11.0,
    "axes.labelsize": 11.0,
    "axes.titlesize": 13.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "figure.titlesize": 14.0,
}


def _resolve_palette(name: str) -> list[str]:
    if name not in _PALETTES:
        raise ValueError(f"Unknown palette '{name}'. Available: {', '.join(sorted(_PALETTES))}.")
    return _PALETTES[name]


def _current_background(alpha: float = 1.0) -> tuple[str, str, str, str]:
    tokens = _MANAGER.tokens
    if tokens["background"] == "dark":
        axes_face = "#111111"
        figure_face = "#0a0a0a"
        text = "#f2f2f2"
        grid_color = "#dddddd"
    else:
        axes_face = "#ffffff"
        figure_face = "#ffffff"
        text = "#111111"
        grid_color = "#444444"
    return axes_face, figure_face, text, grid_color


@dataclass
class _ThemeManager:
    tokens: Dict[str, Any]

    def __post_init__(self) -> None:
        self._stack: list[Dict[str, Any]] = []

    def use(self, preset: str = "notebook") -> None:
        if preset not in _PRESETS:
            raise ValueError(f"Unknown theme preset '{preset}'. Available: {', '.join(sorted(_PRESETS))}.")
        self.tokens = {**_BASE_TOKENS, **_PRESETS[preset]}
        self._apply()

    def update(self, **tokens: Any) -> None:
        unknown = sorted(set(tokens) - set(_BASE_TOKENS))
        if unknown:
            raise ValueError(f"Unknown theme tokens: {', '.join(unknown)}.")
        if "color_cycle" in tokens and "palette" not in tokens:
            tokens = {**tokens, "palette": tokens["color_cycle"]}
        elif "palette" in tokens and "color_cycle" not in tokens:
            tokens = {**tokens, "color_cycle": tokens["palette"]}
        self.tokens.update(tokens)
        self._apply()

    def push(self, overrides: Dict[str, Any]) -> None:
        self._stack.append(dict(self.tokens))
        self.update(**overrides)

    def pop(self) -> None:
        if not self._stack:
            return
        self.tokens = self._stack.pop()
        self._apply()

    def _apply(self) -> None:
        rc = mpl.rcParams
        tokens = self.tokens

        scale = float(tokens["scale"])
        for key, base in _FONT_SIZES.items():
            rc[key] = base * scale

        rc["lines.linewidth"] = float(tokens["line_w"])
        rc["lines.markersize"] = float(tokens["marker_size"])
        rc["lines.marker"] = tokens["marker"]

        rc["font.family"] = [tokens["font"]]
        rc["font.monospace"] = [tokens["mono_font"]]
        rc["text.usetex"] = bool(tokens["usetex"])

        rc["axes.linewidth"] = float(tokens["axis_w"])

        rc["xtick.major.size"] = float(tokens["tick_len"])
        rc["ytick.major.size"] = float(tokens["tick_len"])
        rc["xtick.major.width"] = float(tokens["tick_w"])
        rc["ytick.major.width"] = float(tokens["tick_w"])
        rc["xtick.minor.visible"] = bool(tokens["minor_ticks"])
        rc["ytick.minor.visible"] = bool(tokens["minor_ticks"])

        rc["axes.labelpad"] = float(tokens["label_pad"])
        rc["axes.titlepad"] = float(tokens["title_pad"])
        rc["xtick.major.pad"] = float(tokens["tick_pad"])
        rc["ytick.major.pad"] = float(tokens["tick_pad"])

        axes_face, figure_face, text_color, grid_color = _current_background()
        rc["axes.facecolor"] = axes_face
        rc["figure.facecolor"] = figure_face
        rc["text.color"] = text_color
        rc["axes.labelcolor"] = text_color
        rc["xtick.color"] = text_color
        rc["ytick.color"] = text_color
        rc["axes.edgecolor"] = text_color

        grid = tokens["grid"]
        if not grid:
            rc["axes.grid"] = False
        else:
            rc["axes.grid"] = True
            if grid in ("x", "y"):
                rc["axes.grid.axis"] = grid
            else:
                rc["axes.grid.axis"] = "both"
        rc["grid.alpha"] = float(tokens["grid_alpha"])
        rc["grid.color"] = grid_color

        palette_name = tokens.get("color_cycle", tokens["palette"])
        palette = _resolve_palette(palette_name)
        rc["axes.prop_cycle"] = cycler(color=palette)

        # Make legends follow theme decisions
        rc["legend.loc"] = tokens["legend_loc"]
        rc["legend.frameon"] = bool(tokens["legend_frame"])

        # 3D axes readability tweaks (respect background)
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            axes_face, figure_face, text_color, grid_color = _current_background()
            rc["axes3d.grid"] = True
            rc["axes3d.xaxis.panecolor"] = axes_face
            rc["axes3d.yaxis.panecolor"] = axes_face
            rc["axes3d.zaxis.panecolor"] = axes_face
        except Exception:
            pass

    def get(self, key: str) -> Any:
        return self.tokens[key]


class temp(ContextDecorator):
    def __init__(self, **tokens: Any):
        self._tokens = tokens

    def __enter__(self):
        _MANAGER.push(self._tokens)
        return self

    def __exit__(self, *exc):
        _MANAGER.pop()
        return False


_MANAGER = _ThemeManager(tokens=dict(_BASE_TOKENS))
_MANAGER.use("notebook")


def use(preset: str = "notebook") -> None:
    _MANAGER.use(preset)


def update(**tokens: Any) -> None:
    _MANAGER.update(**tokens)


def get(token: str) -> Any:
    return _MANAGER.get(token)


__all__ = ["use", "update", "temp", "get"]