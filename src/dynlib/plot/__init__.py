# src/dynlib/plot/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

from . import _theme as theme
from . import _fig as fig
from ._primitives import series, phase, utils, manifold
from ._facet import wrap as facet_wrap
from . import _export as export
from .bifurcation import bifurcation_diagram
from .basin import basin_plot
from .dynamics import cobweb
from .vectorfield import (
    vectorfield,
    eval_vectorfield,
    VectorFieldHandle,
    vectorfield_sweep,
    VectorFieldSweep,
    vectorfield_animate,
    VectorFieldAnimation,
)

if TYPE_CHECKING:
    # Expose the concrete type of the instances to type-checkers/editors
    # so they can provide method/parameter hints. This import is type-only
    # and won't affect runtime.
    from ._primitives import _SeriesPlot as _SeriesPlot  # type: ignore
    from ._primitives import _PhasePlot as _PhasePlot  # type: ignore
    from ._primitives import _UtilsPlot as _UtilsPlot  # type: ignore

    series: _SeriesPlot
    phase: _PhasePlot
    utils: _UtilsPlot

__all__ = [
    "theme",
    "fig",
    "series",
    "phase",
    "utils",
    "manifold",
    "facet",
    "export",
    "cobweb",
    "bifurcation_diagram",
    "basin_plot",
    "return_map",
    "vectorfield",
    "eval_vectorfield",
    "VectorFieldHandle",
    "vectorfield_sweep",
    "VectorFieldSweep",
    "vectorfield_animate",
    "VectorFieldAnimation",
]


class _FacetModule:
    wrap = staticmethod(facet_wrap)


facet = _FacetModule()


def return_map(*args, **kwargs):
    return phase.return_map(*args, **kwargs)
