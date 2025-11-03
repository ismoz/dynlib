# src/dynlib/plot/_export.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt

def _as_fig(obj) -> plt.Figure:
    if hasattr(obj, "figure") and obj.figure is not None:
        return obj.figure  # Axes -> Figure
    return obj  # assume Figure

def savefig(
    fig_or_ax,
    path: str | Path,
    *,
    fmts: tuple[str, ...] = ("png",),
    dpi: int = 300,
    transparent: bool = False,
    pad: float = 0.01,
    metadata: dict[str, str] | None = None,
    bbox_inches: str | None = "tight",
) -> list[Path]:
    """
    Save figure (or axes.figure) to <path>.<fmt> for each fmt in fmts.
    Returns the list of written paths in order.
    """
    fig = _as_fig(fig_or_ax)
    target = Path(path)
    if target.suffix:
        target = target.with_suffix("")

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
