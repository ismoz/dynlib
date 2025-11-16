# tests/unit/test_lag_system.py
import numpy as np
import pytest
import tomllib

from dynlib.compiler.build import build
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.runtime.sim import Sim


LAGGED_MAP_TOML = """
[model]
type = "map"
dtype = "float64"

[states]
x = 0.1

[params]
r = 3.5
alpha = 0.6

[equations.rhs]
x = "r * (alpha * lag_x() + (1 - alpha) * lag_x(2)) * (1 - (alpha * lag_x() + (1 - alpha) * lag_x(2)))"

[sim]
t0 = 0.0
t_end = 12.0
dt = 1.0
record = true
record_interval = 1
stepper = "map"
"""


def _build_lagged_map_model(jit: bool = False):
    doc = tomllib.loads(LAGGED_MAP_TOML)
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    return build(spec, stepper="map", jit=jit, disk_cache=False)


def _build_two_state_ode_model(
    *,
    lag_target: str | None = None,
    lag_depth: int = 0,
    jit: bool = False,
):
    """Helper for continuous runners with optional per-state lag usage."""
    lag_depth = int(lag_depth)
    x_expr = "-a * x + 0.25 * y"
    y_expr = "0.5 * x - b * y"
    if lag_target == "x":
        x_expr += f" + 0.05 * lag_x({lag_depth})"
    elif lag_target == "y":
        y_expr += f" + 0.05 * lag_y({lag_depth})"
    doc = {
        "model": {"type": "ode", "dtype": "float64"},
        "states": {"x": 0.25, "y": -0.1},
        "params": {"a": 0.4, "b": 0.1},
        "equations": {"rhs": {"x": x_expr, "y": y_expr}},
        "sim": {
            "t0": 0.0,
            "t_end": 2.0,
            "dt": 0.1,
            "record": True,
            "record_interval": 1,
            "stepper": "rk4",
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    return build(spec, stepper="rk4", jit=jit, disk_cache=False)


def _build_two_state_map_model(
    *,
    lag_target: str | None = None,
    lag_depth: int = 0,
    jit: bool = False,
):
    """Helper for discrete runners with optional per-state lag usage."""
    lag_depth = int(lag_depth)
    x_expr = "0.4 * x + 0.05 * y"
    y_expr = "0.3 * y - 0.02 * x"
    if lag_target == "x":
        x_expr += f" + 0.1 * lag_x({lag_depth})"
    elif lag_target == "y":
        y_expr += f" + 0.1 * lag_y({lag_depth})"
    doc = {
        "model": {"type": "map", "dtype": "float64"},
        "states": {"x": 0.2, "y": -0.05},
        "params": {"alpha": 0.3},
        "equations": {"rhs": {"x": x_expr, "y": y_expr}},
        "sim": {
            "t0": 0.0,
            "t_end": 10.0,
            "dt": 1.0,
            "record": True,
            "record_interval": 1,
            "stepper": "map",
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    return build(spec, stepper="map", jit=jit, disk_cache=False)


def _run_and_capture(model, *, run_kwargs: dict[str, object]):
    """Run a model and capture (T, Y) snapshots for comparison."""
    sim = Sim(model)
    sim.run(record=True, record_interval=1, **run_kwargs)
    results = sim.raw_results()
    return (
        np.array(results.T_view, copy=True),
        np.array(results.Y_view, copy=True),
    )


@pytest.mark.parametrize("jit", [True, False])
def test_lagged_map_buffers_track_history(jit):
    model = _build_lagged_map_model(jit=jit)
    sim = Sim(model)
    steps = 12
    sim.run(N=steps, record=True, record_interval=1)
    results = sim.raw_results()
    ws = results.final_workspace["runtime"]  # Access runtime workspace for lag buffers
    ss = ws["lag_ring"]
    iw0 = ws["lag_head"]
    lag_info = model.lag_state_info or []

    for state_idx, depth, ss_offset, iw0_index in lag_info:
        # Ensure we have enough history to check
        available = min(depth - 1, results.n - 1)
        if available == 0:
            continue
        head = int(iw0[iw0_index])
        state_series = results.Y[state_idx, :results.n]
        for k in range(1, available + 1):
            slot = ss_offset + ((head - k) % depth)
            expected = state_series[-(k + 1)]
            np.testing.assert_allclose(
                ss[slot], expected, rtol=1e-12, atol=1e-12, err_msg=f"lag buffer mismatch for state idx {state_idx} k={k}"
            )


@pytest.mark.parametrize("jit", [True, False])
def test_lag_buffers_survive_resume_and_match_fresh_run(jit):
    model = _build_lagged_map_model(jit=jit)
    sim = Sim(model)
    sim.run(N=5, record=True, record_interval=1)
    sim.run(N=8, record=True, record_interval=1, resume=True)
    resumed = sim.raw_results()

    fresh_model = _build_lagged_map_model(jit=jit)
    sim_fresh = Sim(fresh_model)
    sim_fresh.run(N=8, record=True, record_interval=1)
    fresh = sim_fresh.raw_results()

    np.testing.assert_allclose(resumed.Y_view[:, :resumed.n], fresh.Y_view[:, :fresh.n], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(resumed.T_view[:resumed.n], fresh.T_view[:fresh.n], rtol=1e-12, atol=1e-12)


def test_continuous_runner_lag_metadata_is_per_instance():
    model_a = _build_two_state_ode_model(lag_target="x", lag_depth=3, jit=False)
    t_ref, y_ref = _run_and_capture(model_a, run_kwargs={"T": 2.0})

    model_b = _build_two_state_ode_model(lag_target="y", lag_depth=5, jit=False)
    _run_and_capture(model_b, run_kwargs={"T": 2.0})

    t_new, y_new = _run_and_capture(model_a, run_kwargs={"T": 2.0})
    np.testing.assert_allclose(t_new, t_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_new, y_ref, rtol=1e-12, atol=1e-12)


def test_continuous_runner_without_lags_ignores_foreign_metadata():
    model_no_lag = _build_two_state_ode_model(jit=False)
    t_ref, y_ref = _run_and_capture(model_no_lag, run_kwargs={"T": 2.0})

    model_with_lag = _build_two_state_ode_model(lag_target="x", lag_depth=4, jit=False)
    _run_and_capture(model_with_lag, run_kwargs={"T": 2.0})

    t_new, y_new = _run_and_capture(model_no_lag, run_kwargs={"T": 2.0})
    np.testing.assert_allclose(t_new, t_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_new, y_ref, rtol=1e-12, atol=1e-12)


def test_discrete_runner_lag_metadata_is_per_instance():
    model_a = _build_two_state_map_model(lag_target="x", lag_depth=3, jit=False)
    t_ref, y_ref = _run_and_capture(model_a, run_kwargs={"N": 10})

    model_b = _build_two_state_map_model(lag_target="y", lag_depth=5, jit=False)
    _run_and_capture(model_b, run_kwargs={"N": 10})

    t_new, y_new = _run_and_capture(model_a, run_kwargs={"N": 10})
    np.testing.assert_allclose(t_new, t_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_new, y_ref, rtol=1e-12, atol=1e-12)


def test_discrete_runner_without_lags_ignores_foreign_metadata():
    model_no_lag = _build_two_state_map_model(jit=False)
    t_ref, y_ref = _run_and_capture(model_no_lag, run_kwargs={"N": 10})

    model_with_lag = _build_two_state_map_model(lag_target="x", lag_depth=4, jit=False)
    _run_and_capture(model_with_lag, run_kwargs={"N": 10})

    t_new, y_new = _run_and_capture(model_no_lag, run_kwargs={"N": 10})
    np.testing.assert_allclose(t_new, t_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_new, y_ref, rtol=1e-12, atol=1e-12)
