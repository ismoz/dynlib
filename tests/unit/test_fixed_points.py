import math
from pathlib import Path

import numpy as np
import tomllib

from dynlib import build
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec


def _build_logistic_map():
    test_dir = Path(__file__).resolve().parents[1]
    base_model = test_dir / "data" / "models" / "logistic_map.toml"
    return build(str(base_model), jit=False, disk_cache=False)


def _build_lorenz_model():
    toml_str = """
[model]
type = "ode"
label = "Lorenz"
stepper = "rk4"

[states]
x = 0.0
y = 1.0
z = 0.0

[params]
sigma = 10.0
rho = 28.0
beta = 2.6666666666666665

[equations.rhs]
x = "sigma * (y - x)"
y = "x * (rho - z) - y"
z = "x * y - beta * z"

[equations.jacobian]
expr = [
    ["-sigma", "sigma", "0.0"],
    ["rho - z", "-1.0", "-x"],
    ["y", "x", "-beta"],
]
"""
    data = tomllib.loads(toml_str)
    spec = build_spec(parse_model_v2(data))
    return build(spec, jit=False, disk_cache=False)


def test_fixed_points_logistic_map():
    model = _build_logistic_map()
    seeds = np.array([[0.0], [0.9], [0.6]], dtype=float)
    result = model.fixed_points(params={"r": 2.5}, seeds=seeds, tol=1e-12)
    assert result.points.shape == (2, 1)
    points = np.sort(result.points[:, 0])
    np.testing.assert_allclose(points, np.array([0.0, 0.6]), atol=1e-6)

    assert result.stability is not None
    for point, label in zip(result.points[:, 0], result.stability):
        if np.isclose(point, 0.0, atol=1e-6):
            assert label == "unstable"
        elif np.isclose(point, 0.6, atol=1e-6):
            assert label == "stable"
        else:
            raise AssertionError(f"Unexpected fixed point {point}")


def test_fixed_points_lorenz_equilibria():
    model = _build_lorenz_model()
    rho = 28.0
    beta = 2.6666666666666665
    eq = math.sqrt(beta * (rho - 1.0))
    seeds = np.array(
        [
            [0.0, 0.0, 0.0],
            [eq, eq, rho - 1.0],
            [-eq, -eq, rho - 1.0],
        ],
        dtype=float,
    )
    result = model.fixed_points(seeds=seeds, tol=1e-12, classify=False)
    assert result.points.shape == (3, 3)

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [eq, eq, rho - 1.0],
            [-eq, -eq, rho - 1.0],
        ],
        dtype=float,
    )
    for target in expected:
        dist = np.min(np.linalg.norm(result.points - target, axis=1))
        assert dist <= 1e-6
