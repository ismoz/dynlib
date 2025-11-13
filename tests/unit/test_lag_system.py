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
x = "r * (alpha * prev_x() + (1 - alpha) * lag_x(2)) * (1 - (alpha * prev_x() + (1 - alpha) * lag_x(2)))"

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


@pytest.mark.parametrize("jit", [True, False])
def test_lagged_map_buffers_track_history(jit):
    model = _build_lagged_map_model(jit=jit)
    sim = Sim(model)
    steps = 12
    sim.run(N=steps, record=True, record_interval=1)
    results = sim.raw_results()
    ws = results.final_stepper_ws
    ss = ws["ss"]
    iw0 = ws["iw0"]
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
