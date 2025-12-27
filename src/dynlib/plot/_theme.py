# src/dynlib/plot/_theme.py
from __future__ import annotations

from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import matplotlib as mpl
from cycler import cycler

# A complete schema of tokens we manage. Presets can be sparse; missing keys are filled from here.
_BASE_TOKENS: Dict[str, Any] = {
    "scale": 1.0,
    "fontsize_base": 11.0,
    "fontsize_label": 12.0,
    "fontsize_title": 13.0,
    "fontsize_tick": 10.0,
    "fontsize_legend": 10.0,
    "fontsize_figtitle": 14.0,
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
    "vline_label_pad": 0.01,
    "hline_label_pad": 0.1,
    "vline_label_placement_pad": 0.05,
    "hline_label_placement_pad": 0.05,
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


@dataclass(frozen=True)
class ThemeSpec:
    """
    Sparse declaration of a theme layer.

    - tokens: patch over BASE_TOKENS
    - rc: raw matplotlib rcParams to apply after tokens
    - inherits: name of another preset to apply before this one
    - palettes: palette definitions available when this theme is active
    """
    name: str
    tokens: Mapping[str, Any] | None = None
    rc: Mapping[str, Any] | None = None
    inherits: str | None = None
    palettes: Mapping[str, list[str]] | None = None


_PRESETS: Dict[str, ThemeSpec] = {
    # Base/default theme (matches previous "notebook" defaults)
    "notebook": ThemeSpec(
        name="notebook",
    ),
    "paper": ThemeSpec(
        name="paper",
        inherits="notebook",
        tokens={
            "scale": 1.0,
            "grid": False,
            "tick_n": 6,
            "line_w": 1.4,
            "axis_w": 1.4,
            "marker_size": 3.0,
            "fontsize_label": 12.0,
            "fontsize_title": 12.0,
            "fontsize_tick": 10.0,
            "fontsize_legend": 10.0,
            "fontsize_figtitle": 13.0,
        },
    ),
    "talk": ThemeSpec(
        name="talk",
        inherits="notebook",
        tokens={
            "scale": 1.4,
            "grid": True,
            "line_w": 2.4,
            "tick_len": 5.0,
            "tick_w": 1.1,
        },
    ),
    "dark": ThemeSpec(
        name="dark",
        inherits="notebook",
        tokens={
            "background": "dark",
            "grid": True,
            "grid_alpha": 0.2,
            "palette": "cbf",
            "color_cycle": "cbf",
            "tick_w": 1.1,
            "axis_w": 1.4,
        },
    ),
    "mono": ThemeSpec(
        name="mono",
        inherits="notebook",
        tokens={
            "palette": "mono",
            "color_cycle": "mono",
            "grid": False,
            "background": "light",
        },
    ),
}


def _merge_dicts(target: MutableMapping[str, Any], source: Mapping[str, Any] | None) -> None:
    if not source:
        return
    target.update(source)


def _normalize_palette_name(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """Keep palette and color_cycle in sync after merges."""
    palette = tokens.get("palette")
    color_cycle = tokens.get("color_cycle")
    if palette and not color_cycle:
        tokens["color_cycle"] = palette
    elif color_cycle and not palette:
        tokens["palette"] = color_cycle
    return tokens


def _resolve_palette(name: str, palette_bank: Mapping[str, list[str]]) -> list[str]:
    if name not in palette_bank:
        raise ValueError(f"Unknown palette '{name}'. Available: {', '.join(sorted(palette_bank))}.")
    return palette_bank[name]


def _current_background(tokens: Mapping[str, Any]) -> tuple[str, str, str, str]:
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


class _ThemeManager:
    def __init__(
        self,
        base_tokens: Mapping[str, Any],
        presets: Mapping[str, ThemeSpec],
        palette_bank: Mapping[str, list[str]],
    ) -> None:
        self._base_tokens: Dict[str, Any] = dict(base_tokens)
        self._presets: Dict[str, ThemeSpec] = dict(presets)
        self._user_palettes: Dict[str, list[str]] = {}
        self._stack: list[
            tuple[ThemeSpec, Dict[str, Any], Dict[str, Any]]
        ] = []

        self._active_spec: ThemeSpec = self._presets["notebook"]
        self._user_tokens: Dict[str, Any] = {}
        self._user_rc: Dict[str, Any] = {}
        self._base_palette_bank: Dict[str, list[str]] = dict(palette_bank)

        self._apply_current()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def use(self, preset: str | ThemeSpec = "notebook", *, tokens: Mapping[str, Any] | None = None,
            rc: Mapping[str, Any] | None = None) -> None:
        spec = self._resolve_spec_ref(preset)
        self._active_spec = spec
        self._user_tokens = dict(tokens or {})
        self._user_rc = dict(rc or {})
        self._apply_current()

    def update(self, *, tokens: Mapping[str, Any] | None = None, rc: Mapping[str, Any] | None = None, **legacy_tokens: Any) -> None:
        # Allow legacy update(**tokens) style
        merged_tokens: Dict[str, Any] = dict(tokens or {})
        _merge_dicts(merged_tokens, legacy_tokens)
        self._user_tokens.update(self._validate_tokens(merged_tokens))
        _merge_dicts(self._user_rc, rc)
        self._apply_current()

    def push(self, *, tokens: Mapping[str, Any] | None = None, rc: Mapping[str, Any] | None = None, **legacy_tokens: Any) -> None:
        # Save current state
        self._stack.append(
            (
                self._active_spec,
                dict(self._user_tokens),
                dict(self._user_rc),
            )
        )
        self.update(tokens=tokens, rc=rc, **legacy_tokens)

    def pop(self) -> None:
        if not self._stack:
            return
        self._active_spec, self._user_tokens, self._user_rc = self._stack.pop()
        self._apply_current()

    def get(self, key: str) -> Any:
        return self.tokens[key]

    def rc(self, key: str) -> Any:
        return mpl.rcParams[key]

    def register_palette(self, name: str, colors: Iterable[str]) -> None:
        self._user_palettes[str(name)] = list(colors)
        self._apply_current()

    def register_preset(self, spec: ThemeSpec) -> None:
        self._presets[spec.name] = spec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_tokens(self, tokens: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = sorted(set(tokens) - set(self._base_tokens))
        if unknown:
            raise ValueError(f"Unknown theme tokens: {', '.join(unknown)}.")
        return dict(tokens)

    def _resolve_spec_ref(self, ref: str | ThemeSpec) -> ThemeSpec:
        if isinstance(ref, ThemeSpec):
            return ref
        if ref not in self._presets:
            raise ValueError(f"Unknown theme preset '{ref}'. Available: {', '.join(sorted(self._presets))}.")
        return self._presets[ref]

    def _iter_layers(self, spec: ThemeSpec) -> Iterable[ThemeSpec]:
        seen = set()
        current = spec
        while current is not None:
            if current.name in seen:
                raise ValueError("Theme inheritance loop detected.")
            seen.add(current.name)
            yield current
            if current.inherits is None:
                break
            current = self._presets.get(current.inherits)
            if current is None:
                raise ValueError(f"Unknown parent theme '{spec.inherits}' for theme '{spec.name}'.")

    def _resolve_tokens_and_rc(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, list[str]]]:
        tokens: Dict[str, Any] = dict(self._base_tokens)
        rc: Dict[str, Any] = {}
        palette_bank: Dict[str, list[str]] = {**self._base_palette_bank, **self._user_palettes}

        # Apply inheritance chain (base <- parent <- child)
        for layer in reversed(list(self._iter_layers(self._active_spec))):
            if layer.palettes:
                palette_bank.update({name: list(colors) for name, colors in layer.palettes.items()})
            _merge_dicts(tokens, self._validate_tokens(layer.tokens or {}))
            _merge_dicts(rc, layer.rc)

        # Apply user overrides last
        _merge_dicts(tokens, self._user_tokens)
        _merge_dicts(rc, self._user_rc)

        _normalize_palette_name(tokens)
        return tokens, rc, palette_bank

    def _apply_current(self) -> None:
        tokens, rc_overrides, palette_bank = self._resolve_tokens_and_rc()
        self.tokens = tokens  # Keep resolved tokens available to the module
        self._apply(tokens, rc_overrides, palette_bank)

    def _apply(self, tokens: Mapping[str, Any], rc_overrides: Mapping[str, Any], palette_bank: Mapping[str, list[str]]) -> None:
        rc = mpl.rcParams

        scale = float(tokens["scale"])

        # Apply scale to all fontsize tokens
        rc["font.size"] = float(tokens["fontsize_base"]) * scale
        rc["axes.labelsize"] = float(tokens["fontsize_label"]) * scale
        rc["axes.titlesize"] = float(tokens["fontsize_title"]) * scale
        rc["xtick.labelsize"] = float(tokens["fontsize_tick"]) * scale
        rc["ytick.labelsize"] = float(tokens["fontsize_tick"]) * scale
        rc["legend.fontsize"] = float(tokens["fontsize_legend"]) * scale
        rc["figure.titlesize"] = float(tokens["fontsize_figtitle"]) * scale

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

        axes_face, figure_face, text_color, grid_color = _current_background(tokens)
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
        palette = _resolve_palette(palette_name, palette_bank)
        rc["axes.prop_cycle"] = cycler(color=palette)

        # Make legends follow theme decisions
        rc["legend.loc"] = tokens["legend_loc"]
        rc["legend.frameon"] = bool(tokens["legend_frame"])

        # Apply custom rc overrides last so users can touch anything
        if rc_overrides:
            mpl.rcParams.update(rc_overrides)

        # 3D axes readability tweaks (respect background)
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            axes_face, _, text_color, _ = _current_background(tokens)
            rc["axes3d.grid"] = True
            rc["axes3d.xaxis.panecolor"] = axes_face
            rc["axes3d.yaxis.panecolor"] = axes_face
            rc["axes3d.zaxis.panecolor"] = axes_face
            rc["axes3d.xaxis.labelpad"] = float(tokens["label_pad"])
            rc["axes3d.yaxis.labelpad"] = float(tokens["label_pad"])
            rc["axes3d.zaxis.labelpad"] = float(tokens["label_pad"])
            rc["axes3d.edgecolor"] = text_color
        except Exception:
            pass


class temp(ContextDecorator):
    def __init__(self, *, tokens: Mapping[str, Any] | None = None, rc: Mapping[str, Any] | None = None, **legacy_tokens: Any):
        merged_tokens: Dict[str, Any] = dict(tokens or {})
        _merge_dicts(merged_tokens, legacy_tokens)
        self._tokens = merged_tokens
        self._rc = dict(rc or {})

    def __enter__(self):
        _MANAGER.push(tokens=self._tokens, rc=self._rc)
        return self

    def __exit__(self, *exc):
        _MANAGER.pop()
        return False


_MANAGER = _ThemeManager(
    base_tokens=_BASE_TOKENS,
    presets=_PRESETS,
    palette_bank=_PALETTES,
)


def use(preset: str | ThemeSpec = "notebook", *, tokens: Mapping[str, Any] | None = None, rc: Mapping[str, Any] | None = None) -> None:
    _MANAGER.use(preset, tokens=tokens, rc=rc)


def update(*, tokens: Mapping[str, Any] | None = None, rc: Mapping[str, Any] | None = None, **legacy_tokens: Any) -> None:
    _MANAGER.update(tokens=tokens, rc=rc, **legacy_tokens)


def get(token: str) -> Any:
    return _MANAGER.get(token)


def register_palette(name: str, colors: Iterable[str]) -> None:
    _MANAGER.register_palette(name, colors)


def register_preset(spec: ThemeSpec) -> None:
    _MANAGER.register_preset(spec)


__all__ = ["use", "update", "temp", "get", "register_palette", "register_preset", "ThemeSpec"]
