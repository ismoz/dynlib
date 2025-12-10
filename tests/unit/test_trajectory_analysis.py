from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest

from dynlib.runtime.results import Results
from dynlib.runtime.results_api import ResultsView


def _make_results_view(
    *,
    state_names: tuple[str, ...] | list[str],
    aux_names: tuple[str, ...] | list[str],
    t: list[float] | tuple[float, ...],
    y: np.ndarray | list[list[float]] | None = None,
    aux: np.ndarray | list[list[float]] | None = None,
) -> ResultsView:
    """Construct a minimal ResultsView for testing analysis helpers."""
    t_arr = np.asarray(t, dtype=np.float64)
    n = int(t_arr.shape[0])
    state_names = tuple(state_names)
    aux_names = tuple(aux_names)

    y_arr = np.asarray(y if y is not None else np.zeros((len(state_names), n)), dtype=np.float64)
    if y_arr.size and y_arr.shape != (len(state_names), n):
        y_arr = y_arr.reshape(len(state_names), n)

    aux_arr = None
    if aux_names:
        aux_arr = np.asarray(aux if aux is not None else np.zeros((len(aux_names), n)), dtype=np.float64)
        if aux_arr.size and aux_arr.shape != (len(aux_names), n):
            aux_arr = aux_arr.reshape(len(aux_names), n)

    evt_cap = 1
    raw = Results(
        T=t_arr,
        Y=y_arr,
        AUX=aux_arr,
        STEP=np.arange(n, dtype=np.int64),
        FLAGS=np.zeros((n,), dtype=np.int32),
        EVT_CODE=np.zeros((evt_cap,), dtype=np.int32),
        EVT_INDEX=-np.ones((evt_cap,), dtype=np.int32),
        EVT_LOG_DATA=np.zeros((evt_cap, 1), dtype=np.float64),
        n=n,
        m=0,
        status=0,
        final_state=np.zeros((len(state_names),), dtype=np.float64),
        final_params=np.zeros((0,), dtype=np.float64),
        t_final=float(t_arr[-1]) if n else 0.0,
        final_dt=0.0,
        step_count_final=int(n - 1) if n else 0,
        final_workspace={"runtime": {}, "stepper": {}},
        state_names=list(state_names),
        aux_names=list(aux_names),
    )
    spec = SimpleNamespace(
        states=state_names,
        aux={name: None for name in aux_names},
        events=(),
        tag_index={},
    )
    return ResultsView(raw, spec)


def test_analyze_prefers_states_when_present() -> None:
    view = _make_results_view(
        state_names=("x",),
        aux_names=("energy",),
        t=[0.0, 1.0],
        y=[[0.0, 1.0]],
        aux=[[2.0, 3.0]],
    )
    analyzer = view.analyze()
    assert analyzer.vars == ("x",)
    assert analyzer.max()["x"] == pytest.approx(1.0)


def test_analyze_falls_back_to_aux_only_recordings() -> None:
    view = _make_results_view(
        state_names=(),
        aux_names=("energy",),
        t=[0.0, 1.0, 2.0],
        aux=[[1.0, 2.0, 1.5]],
    )
    analyzer = view.analyze()
    assert analyzer.vars == ("energy",)
    assert analyzer.argmax()["energy"] == pytest.approx((1.0, 2.0))


def test_analyze_errors_when_nothing_recorded() -> None:
    view = _make_results_view(state_names=(), aux_names=(), t=[])
    with pytest.raises(ValueError, match="No recorded variables available"):
        view.analyze()


def test_percentile_rejects_out_of_range() -> None:
    view = _make_results_view(
        state_names=("x",),
        aux_names=(),
        t=[0.0, 1.0],
        y=[[0.0, 1.0]],
    )
    analyzer = view.analyze("x")
    with pytest.raises(ValueError):
        analyzer.percentile(-1)
    with pytest.raises(ValueError):
        analyzer.percentile(101)


def test_time_above_and_below_interpolate_crossings() -> None:
    view = _make_results_view(
        state_names=("x",),
        aux_names=(),
        t=[0.0, 1.0, 2.0],
        y=[[-1.0, 1.0, -1.0]],
    )
    analyzer = view.analyze("x")
    assert analyzer.time_above(0.0) == pytest.approx(1.0)
    assert analyzer.time_below(0.0) == pytest.approx(1.0)


def test_multivar_time_above_and_crossings_use_cached_analyzers() -> None:
    view = _make_results_view(
        state_names=("x", "y"),
        aux_names=(),
        t=[0.0, 1.0, 2.0],
        y=[[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]],
    )
    mv = view.analyze(("x", "y"))
    assert mv.time_above(0.0) == {"x": pytest.approx(2.0), "y": pytest.approx(1.0)}
    crossings = mv.crossing_times(0.0, direction="up")["y"]
    assert crossings.shape == (1,)
    assert crossings[0] == pytest.approx(0.5)
