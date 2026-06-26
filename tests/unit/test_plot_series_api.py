# tests/unit/test_plot_series_api.py
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import dynlib.plot as plot
from dynlib.plot import series


def test_series_line_draws_single_series():
    ax = series.line(x=[0, 1], y=[1, 2], label="x")
    try:
        assert ax.lines[0].get_label() == "x"
    finally:
        plt.close(ax.figure)


def test_trace_shortcut_delegates_to_series_line():
    ax = plot.trace(x=[0, 1], y=[2, 3], color="red")
    try:
        assert ax.lines[0].get_color() == "red"
    finally:
        plt.close(ax.figure)


def test_series_plot_is_removed():
    assert not hasattr(series, "plot")
