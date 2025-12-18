# tests/unit/test_plot_export_savefig.py
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pytest

from dynlib.plot import export


def _make_fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_savefig_infers_format_from_extension(tmp_path):
    fig = _make_fig()
    try:
        out = export.savefig(fig, tmp_path / "plot.pdf")
    finally:
        plt.close(fig)

    assert out == [tmp_path / "plot.pdf"]
    assert (tmp_path / "plot.pdf").exists()


def test_savefig_multiple_formats_without_extension(tmp_path):
    fig = _make_fig()
    try:
        out = export.savefig(fig, tmp_path / "plot", fmts=("pdf", "png"))
    finally:
        plt.close(fig)

    assert out == [tmp_path / "plot.pdf", tmp_path / "plot.png"]
    assert (tmp_path / "plot.pdf").exists()
    assert (tmp_path / "plot.png").exists()


def test_savefig_rejects_extension_and_fmts_together(tmp_path):
    fig = _make_fig()
    try:
        with pytest.raises(ValueError, match="extension.*fmts|fmts.*extension"):
            export.savefig(fig, tmp_path / "plot.pdf", fmts=("png",))
    finally:
        plt.close(fig)

