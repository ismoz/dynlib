# src/dynlib/plot/_export.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

def _as_fig(obj) -> plt.Figure:
    # Fast path: already a Figure-like object.
    if hasattr(obj, "savefig") and hasattr(obj, "get_constrained_layout"):
        return obj

    # Matplotlib Axes expose `.figure`.
    if hasattr(obj, "figure") and getattr(obj, "figure") is not None:
        return obj.figure  # Axes -> Figure

    # Some grid/container types expose `.fig`.
    if hasattr(obj, "fig") and getattr(obj, "fig") is not None:
        return obj.fig

    # Fallback for custom containers (e.g. dynlib.plot._fig.AxesGrid):
    # find a Figure via the first contained Axes.
    try:
        for item in obj:
            if item is None:
                continue
            fig = _as_fig(item)
            if fig is not None:
                return fig
    except TypeError:
        pass

    return obj  # last resort: assume Figure

_FMTS_DEFAULT: Any = object()

def savefig(
    fig_or_ax,
    path: str | Path,
    *,
    fmts: tuple[str, ...] | Any = _FMTS_DEFAULT,
    dpi: int = 300,
    transparent: bool = False,
    pad: float = 0.01,
    metadata: dict[str, str] | None = None,
    bbox_inches: str | None = "tight",
) -> list[Path]:
    """
    Save figure (or axes.figure) to <path>.<fmt> for each fmt in fmts.

    Format inference rules:
    - If `path` has an extension and `fmts` is not provided, infer `fmts` from
      the extension and write exactly that format.
    - If `fmts` is provided, `path` must not include an extension.
    Returns the list of written paths in order.
    """
    fig = _as_fig(fig_or_ax)
    target = Path(path)
    if target.suffix:
        if fmts is _FMTS_DEFAULT:
            fmts = (target.suffix.lstrip("."),)
            target = target.with_suffix("")
        else:
            raise ValueError(
                "Do not pass both a file extension in `path` and `fmts`. "
                "Use `path='name'` with `fmts=(...)`, or `path='name.ext'` "
                "without `fmts`."
            )
    elif fmts is _FMTS_DEFAULT:
        fmts = ("png",)

    # If constrained layout is used, don't apply tight bbox_inches to avoid clipping
    if fig.get_constrained_layout() and bbox_inches == "tight":
        bbox_inches = None

    # normalize fmts: lower, dedupe while preserving order
    seen: set[str] = set()
    norm_fmts: list[str] = []
    for f in fmts:
        f2 = str(f).lower().lstrip(".")
        if f2 and f2 not in seen:
            seen.add(f2)
            norm_fmts.append(f2)

    if not norm_fmts:
        raise ValueError("fmts must contain at least one non-empty format.")

    target.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata or {}

    written: list[Path] = []
    for fmt in norm_fmts:
        outfile = target.with_suffix(f".{fmt}")
        fig.savefig(
            outfile,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad,
            metadata=meta,
        )
        written.append(outfile)
    return written

def show() -> None:
    plt.show()

__all__ = ["savefig", "show"]
