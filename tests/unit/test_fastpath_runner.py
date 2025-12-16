import numpy as np
import pytest
import tomllib

from dynlib.runtime.fastpath import FixedStridePlan
from dynlib.runtime.fastpath.runner import (
    fastpath_for_sim,
    fastpath_batch_for_sim,
)
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build


def _build_simple_model(jit: bool = False) -> Model:
    toml_str = """
[model]
type = "ode"
stepper = "rk4"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
record = true
stepper = "rk4"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"
"""
    data = tomllib.loads(toml_str)
    spec = build_spec(parse_model_v2(data))
    full_model = build(spec, stepper=spec.sim.stepper, jit=jit)
    return Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        workspace_sig=full_model.workspace_sig,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        update_aux=full_model.update_aux,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
        rhs_source=full_model.rhs_source,
        events_pre_source=full_model.events_pre_source,
        events_post_source=full_model.events_post_source,
        update_aux_source=full_model.update_aux_source,
        stepper_source=full_model.stepper_source,
        lag_state_info=full_model.lag_state_info,
        uses_lag=full_model.uses_lag,
        equations_use_lag=full_model.equations_use_lag,
        make_stepper_workspace=full_model.make_stepper_workspace,
    )


@pytest.fixture
def simple_sim():
    return Sim(_build_simple_model(jit=False))


def test_run_fastpath_transient(simple_sim):
    plan = FixedStridePlan(stride=1)
    ic = simple_sim.state_vector(source="session", copy=True)
    params = simple_sim.param_vector(source="session", copy=True)

    res = fastpath_for_sim(
        simple_sim,
        plan=plan,
        t0=0.0,
        T=0.3,
        N=None,
        dt=0.1,
        record_vars=["x"],
        transient=0.1,
        record_interval=1,
        max_steps=100,
        ic=ic,
        params=params,
    )

    assert res is not None
    assert res.ok
    # Warm-up should advance start time to roughly the transient boundary
    assert float(res.t[0]) == pytest.approx(0.1, rel=0, abs=1e-9)
    assert float(res.t[-1]) == pytest.approx(0.3, rel=0, abs=1e-9)


def test_sim_fastpath_helper(simple_sim):
    t_before = float(simple_sim._session_state.t_curr)
    step_before = int(simple_sim._session_state.step_count)

    res = simple_sim.fastpath(T=0.2, dt=0.1, record_vars=["x"])
    assert res.ok
    np.testing.assert_allclose(res.t, np.array([0.0, 0.1, 0.2]))
    assert res["x"].shape == (3,)

    # Session state should remain unchanged by fastpath runs
    assert simple_sim._session_state.t_curr == pytest.approx(t_before)
    assert simple_sim._session_state.step_count == step_before


def test_batch_fastpath_sim_matches_single(simple_sim):
    plan = FixedStridePlan(stride=1)
    ic = simple_sim.state_vector(source="session", copy=True)
    params = simple_sim.param_vector(source="session", copy=True)

    single = fastpath_for_sim(
        simple_sim,
        plan=plan,
        t0=0.0,
        T=0.2,
        N=None,
        dt=0.1,
        record_vars=["x"],
        transient=0.0,
        record_interval=1,
        max_steps=100,
        ic=ic,
        params=params,
    )
    assert single is not None

    batch_views = fastpath_batch_for_sim(
        simple_sim,
        plan=plan,
        t0=0.0,
        T=0.2,
        N=None,
        dt=0.1,
        record_vars=["x"],
        transient=0.0,
        record_interval=1,
        max_steps=100,
        ic=np.stack([ic, ic], axis=0),
        params=np.stack([params, params], axis=0),
        parallel_mode="none",
    )
    assert batch_views is not None
    assert len(batch_views) == 2
    np.testing.assert_allclose(batch_views[0]["x"], single["x"])
    np.testing.assert_allclose(batch_views[1]["x"], single["x"])


def test_batch_fastpath_sim_param_variation(simple_sim):
    plan = FixedStridePlan(stride=1)
    ic = simple_sim.state_vector(source="session", copy=True)
    params = simple_sim.param_vector(source="session", copy=True)

    params_fast = params.copy()
    params_fast[0] = 2.0  # different decay rate

    views = fastpath_batch_for_sim(
        simple_sim,
        plan=plan,
        t0=0.0,
        T=0.3,
        N=None,
        dt=0.1,
        record_vars=["x"],
        transient=0.0,
        record_interval=1,
        max_steps=100,
        ic=np.stack([ic, ic], axis=0),
        params=np.stack([params, params_fast], axis=0),
        parallel_mode="none",
    )
    assert views is not None
    assert len(views) == 2

    slow = views[0]["x"]
    fast = views[1]["x"]
    assert slow.shape == fast.shape
    # With larger decay, the second series should be smaller by the end
    assert fast[-1] < slow[-1]
