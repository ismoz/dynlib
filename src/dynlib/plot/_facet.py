# src/dynlib/plot/_facet.py
from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import Any
from . import _fig
import matplotlib.pyplot as plt

def wrap(keys: Iterable[Any], *, cols: int = 3, title: str | None = None) -> Iterator[tuple[plt.Axes, Any]]:
    keys_list = list(keys)
    if not keys_list:
        return
    axes_grid = _fig.wrap(n=len(keys_list), cols=cols, title=title)
    flat_axes = [ax for row in axes_grid for ax in row]
    for ax, key in zip(flat_axes, keys_list):
        yield ax, key

__all__ = ["wrap"]
