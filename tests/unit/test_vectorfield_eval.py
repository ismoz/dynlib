import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tomllib

from dynlib.plot.vectorfield import eval_vectorfield, vectorfield
from dynlib.compiler.build import build
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec


def _linear_model():
    toml_str = """
[model]
type = "ode"
stepper = "rk4"

[sim]
t0 = 0.0
dt = 0.1

[states]
x = 0.0
y = 0.0
z = 1.0

[params]
a = 1.0
b = 2.0

[equations.rhs]
x = "a * x + z"
y = "b * y"
z = "0.0"
"""
    data = tomllib.loads(toml_str)
    spec = build_spec(parse_model_v2(data))
    return build(spec, jit=False, disk_cache=False)


def test_eval_vectorfield_values_change_with_params_and_fixed():
    model = _linear_model()
    X, Y, U, V = eval_vectorfield(
        model,
        xlim=(-1, 1),
        ylim=(0, 1),
        grid=(3, 2),
        params={"a": 2.0},
        fixed={"z": 1.0},
    )
    xs = np.linspace(-1, 1, 3, dtype=model.dtype)
    ys = np.linspace(0, 1, 2, dtype=model.dtype)
    expected_U = np.vstack([2.0 * xs + 1.0 for _ in ys])
    expected_V = np.vstack([np.full_like(xs, 2.0 * y) for y in ys])
    assert U.shape == (2, 3)
    assert V.shape == (2, 3)
    np.testing.assert_allclose(U, expected_U)
    np.testing.assert_allclose(V, expected_V)


def test_vectorfield_handle_update_recomputes():
    model = _linear_model()
    ax = plt.axes()
    handle = vectorfield(
        model,
        ax=ax,
        xlim=(-1, 1),
        ylim=(-1, 1),
        grid=(2, 2),
        normalize=False,
        nullclines=False,
    )
    handle.update(params={"a": 3.0}, redraw=False)
    xs = np.linspace(-1, 1, 2, dtype=model.dtype)
    ys = np.linspace(-1, 1, 2, dtype=model.dtype)
    expected_U = np.vstack([3.0 * xs + 1.0 for _ in ys])
    expected_V = np.vstack([np.full_like(xs, 2.0 * y) for y in ys])
    np.testing.assert_allclose(handle.U, expected_U)
    np.testing.assert_allclose(handle.V, expected_V)
    plt.close(ax.figure)


def _flatten_contour_segments(cs):
    segs = cs.allsegs[0]
    if not segs:
        raise AssertionError("Nullcline contour has no segments.")
    return np.concatenate(segs, axis=0)


def test_nullclines_use_dense_grid_and_straight_lines():
    model = _linear_model()
    ax = plt.axes()
    handle = vectorfield(
        model,
        ax=ax,
        xlim=(-2, 2),
        ylim=(-1, 1),
        grid=(5, 5),
        normalize=True,
        nullclines=True,
    )
    assert len(handle.nullcline_artists) == 2

    dx_cs, dy_cs = handle.nullcline_artists
    dx_pts = _flatten_contour_segments(dx_cs)
    dy_pts = _flatten_contour_segments(dy_cs)

    # dx/dt = a * x + z => x = -1. Expect nearly vertical line at x = -1.
    assert np.allclose(np.mean(dx_pts[:, 0]), -1.0, atol=1e-3)
    assert np.std(dx_pts[:, 0]) < 5e-3

    # dy/dt = b * y => y = 0 horizontal line.
    assert np.allclose(np.mean(dy_pts[:, 1]), 0.0, atol=1e-3)
    assert np.std(dy_pts[:, 1]) < 5e-3

    plt.close(ax.figure)


# Test that attempting to evaluate a map model raises TypeError
def test_map_model_rejected():
    with pytest.raises(TypeError):
        eval_vectorfield("tests/data/models/logistic_map.toml", grid=(5, 5))


def test_toggle_nullclines_reuses_cached_values():
    model = _linear_model()
    ax = plt.axes()
    handle = vectorfield(model, ax=ax, nullclines=True)
    cached_u = handle.nullcline_U
    cached_v = handle.nullcline_V
    handle.toggle_nullclines()  # disable
    assert not handle.nullclines_enabled
    handle.toggle_nullclines()  # enable, should reuse cached grid/values
    assert handle.nullclines_enabled
    assert handle.nullcline_U is cached_u
    assert handle.nullcline_V is cached_v
    assert len(handle.nullcline_artists) == 2
    plt.close(ax.figure)


def test_interactive_click_runs_simulation_and_can_clear():
    model = _linear_model()
    ax = plt.axes()
    handle = vectorfield(model, ax=ax, xlim=(-1, 1), ylim=(-1, 1))
    initial_lines = len(ax.lines)
    line = handle.simulate_at(0.25, -0.5, T=0.5)
    assert line in handle.traj_lines
    assert len(ax.lines) == initial_lines + 1
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    assert pytest.approx(xdata[0]) == 0.25
    assert pytest.approx(ydata[0]) == -0.5
    handle.clear_trajectories()
    assert len(handle.traj_lines) == 0
    assert len(ax.lines) == initial_lines
    plt.close(ax.figure)


def test_stepper_override_used_when_building_model():
    inline_model = """
inline:
[model]
type = "ode"
stepper = "rk4"

[sim]
t0 = 0.0
dt = 0.1
t_end = 0.5

[states]
x = 0.0
y = 0.0

[params]
a = 1.0

[equations.rhs]
x = "a * x - y"
y = "x + a * y"
"""
    ax = plt.axes()
    handle = vectorfield(inline_model, ax=ax, stepper="euler", interactive=False)
    assert getattr(handle.model, "stepper_name", None) == "euler"
    plt.close(ax.figure)
