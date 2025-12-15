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
