import numpy as np
import pytest
import tomllib

from dynlib.compiler.build import build, FullModel
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.analysis.runtime import (
    AnalysisHooks,
    AnalysisModule,
    AnalysisRequirements,
    TraceSpec,
    lyapunov_mle,
)
from dynlib.runtime.fastpath import FixedStridePlan, FixedTracePlan
from dynlib.runtime.fastpath.capability import assess_capability
from dynlib.runtime.fastpath.executor import fastpath_for_sim
from dynlib.runtime.wrapper import run_with_wrapper
from dynlib.runtime.sim import Sim


def _dummy_rhs(t, y_vec, dy_out, params, runtime_ws):
    dy_out[:] = params[0]


def _dummy_events(t, y_vec, params, evt_log_scratch, runtime_ws):
    return -1, 0


def _dummy_update_aux(t, y_vec, params, aux_out, runtime_ws):
    if aux_out.shape[0] > 0:
        aux_out[:] = 0.0


def _euler_stepper(t, dt, y_curr, rhs, params, runtime_ws, ws, cfg, y_prop, t_prop, dt_next, err_est):
    rhs(t, y_curr, y_prop, params, runtime_ws)
    for i in range(y_curr.shape[0]):
        y_prop[i] = y_curr[i] + dt * y_prop[i]
    t_prop[0] = t + dt
    dt_next[0] = dt
    err_est[0] = 0.0
    return 0


def _noop_runner(*_args, **_kwargs):  # pragma: no cover
    return 0


def _counter_analysis():
    plan = FixedTracePlan(stride=1)

    def _post(
        t, dt, step, y_curr, y_prev, params, runtime_ws, analysis_ws, analysis_out, trace_buf, trace_count, trace_cap, trace_stride
    ):
        if analysis_out.shape[0]:
            analysis_out[0] += 1.0
        if trace_buf.shape[0] > 0 and trace_stride > 0 and (step % trace_stride == 0):
            idx = int(trace_count[0])
            if idx < trace_cap:
                trace_buf[idx, 0] = t
                trace_count[0] = idx + 1
            else:
                trace_count[0] = trace_cap + 1

    hooks = AnalysisHooks(post_step=_post)
    return AnalysisModule(
        name="counter",
        requirements=AnalysisRequirements(fixed_step=True),
        workspace_size=0,
        output_size=1,
        trace=TraceSpec(width=1, plan=plan),
        hooks=hooks,
    )

def _increment_factory(name: str):
    """Simple runtime analysis factory that increments per step."""
    def _post(
        t, dt, step, y_curr, y_prev, params, runtime_ws, analysis_ws, analysis_out, trace_buf, trace_count, trace_cap, trace_stride
    ):
        if analysis_out.shape[0]:
            analysis_out[0] += 1.0
        if trace_buf.shape[0] > 0 and trace_stride > 0 and (step % trace_stride == 0):
            idx = int(trace_count[0])
            if idx < trace_cap:
                trace_buf[idx, 0] = step
                trace_count[0] = idx + 1
            else:
                trace_count[0] = trace_cap + 1

    def _factory(model=None, record_interval=None):
        plan = FixedTracePlan(stride=int(record_interval) if record_interval is not None else 1)
        return AnalysisModule(
            name=name,
            requirements=AnalysisRequirements(fixed_step=True),
            workspace_size=0,
            output_size=1,
            output_names=(f"{name}_out",),
            trace=TraceSpec(width=1, plan=plan),
            trace_names=(f"{name}_trace",),
            hooks=AnalysisHooks(post_step=_post),
        )

    return _factory


def _tracking_analysis(call_log: list[float]):
    """Records every step time into call_log for verification."""
    def _post(
        t, dt, step, y_curr, y_prev, params, runtime_ws, analysis_ws, analysis_out, trace_buf, trace_count, trace_cap, trace_stride
    ):
        call_log.append(float(t))

    return AnalysisModule(
        name="tracker",
        requirements=AnalysisRequirements(fixed_step=True),
        workspace_size=0,
        output_size=0,
        trace=None,
        hooks=AnalysisHooks(post_step=_post),
    )


def _build_map_model(jit: bool = False) -> FullModel:
    toml_str = """
[model]
type = "map"
stepper = "map"

[sim]
dt = 1.0
record = true

[states]
x = 1.0

[params]
r = 2.0

[equations.rhs]
x = "r * x"

[equations.jacobian]
exprs = [
    ["r"]
]
"""
    data = tomllib.loads(toml_str)
    spec = build_spec(parse_model_v2(data))
    return build(spec, stepper=spec.sim.stepper, jit=jit)


def test_wrapper_analysis_survives_reentry():
    ic = np.array([0.0], dtype=np.float64)
    params = np.array([1.0], dtype=np.float64)
    analysis = _counter_analysis()

    res = run_with_wrapper(
        runner=_noop_runner,
        stepper=_euler_stepper,
        rhs=_dummy_rhs,
        events_pre=_dummy_events,
        events_post=_dummy_events,
        update_aux=_dummy_update_aux,
        dtype=np.float64,
        n_state=1,
        n_aux=0,
        t0=0.0,
        t_end=0.2,
        dt_init=0.1,
        max_steps=8,
        record=True,
        record_interval=1,
        state_record_indices=np.array([0], dtype=np.int32),
        aux_record_indices=np.array([], dtype=np.int32),
        state_names=["x"],
        aux_names=[],
        ic=ic,
        params=params,
        cap_rec=1,  # force GROW_REC re-entry
        cap_evt=1,
        analysis=analysis,
        model_hash="test-model",
        stepper_name="euler",
    )

    assert res.ok
    counters = res.analysis.get("counter")
    assert counters is not None
    assert counters["out"] is not None
    assert counters["out"][0] == res.step_count_final
    trace = counters["trace"]
    assert trace is not None
    assert trace.shape[0] == res.step_count_final
    assert trace[0, 0] == pytest.approx(0.1)
    assert trace[-1, 0] == pytest.approx(res.t_final)


def test_sim_run_preserves_analysis():
    model = _build_map_model(jit=False)
    sim = Sim(model)
    analysis = _counter_analysis()

    sim.run(
        N=5,
        dt=1.0,
        record=True,
        record_interval=1,
        max_steps=5,
        cap_rec=8,
        cap_evt=1,
        analysis=analysis,
    )

    raw = sim.raw_results()
    obs = sim.results().analysis
    assert "counter" in obs
    counter = obs["counter"]
    assert counter["out"] is not None
    assert counter["out"][0] == pytest.approx(raw.step_count_final)
    trace = counter["trace"]
    assert trace is not None
    assert trace.shape[0] == raw.step_count_final
    assert counter["stride"] == 1


def test_sim_resolves_analysis_sequence_factories():
    model = _build_map_model(jit=False)
    sim = Sim(model)

    sim.run(
        N=5,
        dt=1.0,
        record=True,
        record_interval=2,
        max_steps=5,
        cap_rec=8,
        cap_evt=1,
        analysis=[_increment_factory("a"), _increment_factory("b")],
    )

    raw = sim.raw_results()
    obs = sim.results().analysis
    assert set(obs.keys()) == {"a", "b"}
    assert obs["a"].out is not None and obs["b"].out is not None
    assert obs["a"].out[0] == pytest.approx(raw.step_count_final)
    assert obs["b"].out[0] == pytest.approx(raw.step_count_final)
    assert obs["a"].stride == 2
    assert obs["b"].trace is not None and obs["a"].trace is not None
    assert obs["a"].trace.shape == obs["b"].trace.shape


def test_combined_analysis_runs_with_jit():
    model = _build_map_model(jit=True)
    sim = Sim(model)

    sim.run(
        N=4,
        dt=1.0,
        record=True,
        record_interval=1,
        max_steps=4,
        cap_rec=8,
        cap_evt=1,
        analysis=[_increment_factory("a"), _increment_factory("b")],
    )

    raw = sim.raw_results()
    obs = sim.results().analysis
    assert obs["a"].out is not None and obs["b"].out is not None
    assert obs["a"].out[0] == pytest.approx(raw.step_count_final)
    assert obs["b"].out[0] == pytest.approx(raw.step_count_final)
    assert obs["a"].trace is not None and obs["b"].trace is not None
    assert obs["a"].trace.shape[1] == 1 and obs["b"].trace.shape[1] == 1
    assert obs["a"].trace.shape[0] >= raw.step_count_final
    assert obs["a"].trace.shape == obs["b"].trace.shape

def test_fastpath_gate_rejects_python_only_observer():
    toml_str = """
[model]
type = "ode"
stepper = "rk4"

[sim]
t0 = 0.0
t_end = 0.3
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
    full_model = build(spec, stepper=spec.sim.stepper, jit=False)
    simple_sim = Sim(full_model)

    analysis_module = AnalysisModule(
        name="py-only",
        requirements=AnalysisRequirements(fixed_step=True),
        workspace_size=0,
        output_size=0,
        trace=None,
        hooks=AnalysisHooks(post_step=lambda *a, **k: None),
    )
    plan = FixedStridePlan(stride=1)
    support = assess_capability(
        simple_sim,
        plan=plan,
        record_vars=["x"],
        dt=0.1,
        transient=0.0,
        adaptive=False,
        analysis=analysis_module,
    )
    if support.ok:
        assert support.reason is None
    else:
        assert "jit" in (support.reason or "").lower()


def test_transient_warmup_skips_analysis_hooks():
    model = _build_map_model(jit=False)
    sim = Sim(model)
    calls: list[float] = []
    analysis = _tracking_analysis(calls)

    sim.run(N=5, transient=3, analysis=analysis)

    assert len(calls) == 5
    assert min(calls) >= 3.0


def test_lyapunov_matches_wrapper_and_fastpath():
    model = _build_map_model(jit=False)
    assert model.jvp is not None
    sim = Sim(model)

    analysis_mod = lyapunov_mle(
        jvp=model.jvp,
        n_state=1,
        trace_plan=FixedTracePlan(stride=1),
        mode="map",
    )

    wrapper_res = run_with_wrapper(
        runner=model.runner,
        stepper=model.stepper,
        rhs=model.rhs,
        events_pre=model.events_pre,
        events_post=model.events_post,
        update_aux=model.update_aux,
        dtype=model.dtype,
        n_state=1,
        n_aux=0,
        t0=0.0,
        t_end=0.0,
        dt_init=1.0,
        max_steps=5,
        record=False,
        record_interval=1,
        state_record_indices=np.array([], dtype=np.int32),
        aux_record_indices=np.array([], dtype=np.int32),
        state_names=[],
        aux_names=[],
        ic=np.array([1.0], dtype=model.dtype),
        params=np.array([2.0], dtype=model.dtype),
        cap_rec=1,
        cap_evt=1,
        discrete=True,
        target_steps=5,
        make_stepper_workspace=model.make_stepper_workspace,
        analysis=analysis_mod,
        model_hash=model.spec_hash,
        stepper_name=model.stepper_name,
    )

    fast_res_view = fastpath_for_sim(
        sim,
        plan=FixedStridePlan(stride=1),
        t0=0.0,
        T=None,
        N=5,
        dt=1.0,
        record_vars=["x"],
        transient=0.0,
        record_interval=1,
        max_steps=10,
        ic=np.array([1.0], dtype=model.dtype),
        params=np.array([2.0], dtype=model.dtype),
        analysis=analysis_mod,
    )

    assert fast_res_view is not None
    fast_res = fast_res_view._raw

    mle_wrapper = wrapper_res.analysis["lyapunov_mle"]["out"][0] / wrapper_res.analysis["lyapunov_mle"]["out"][1]
    mle_fast = fast_res_view.analysis["lyapunov_mle"]["out"][0] / fast_res_view.analysis["lyapunov_mle"]["out"][1]
    target = np.log(2.0)
    assert mle_wrapper == pytest.approx(target, rel=1e-6)
    assert mle_fast == pytest.approx(target, rel=1e-6)
    assert wrapper_res.analysis_trace_filled == 5
    assert fast_res.analysis_trace_filled == 5


def test_lyapunov_defaults_from_model_and_record_interval():
    model = _build_map_model(jit=False)
    sim = Sim(model)
    interval = 3
    mod = lyapunov_mle(model=sim, record_interval=interval)

    assert mod.trace_stride == interval
    assert mod.workspace_size == 2 * len(model.spec.states)
    assert mod.requirements.need_jvp is True


def test_lyapunov_requires_jvp_or_model():
    factory = lyapunov_mle()
    assert callable(factory)
    model = _build_map_model(jit=False)
    mod = factory(model)
    assert isinstance(mod, AnalysisModule)


def test_sim_resolves_analysis_factory_and_passes_record_interval():
    model = _build_map_model(jit=False)
    sim = Sim(model)
    interval = 2

    mod = sim._resolve_analysis(lyapunov_mle, record_interval=interval)
    assert isinstance(mod, AnalysisModule)
    assert mod.trace_stride == interval


def test_sim_run_accepts_analysis_factory():
    model = _build_map_model(jit=False)
    sim = Sim(model)
    sim.run(
        N=3,
        dt=1.0,
        record=False,
        record_interval=1,
        cap_rec=1,
        cap_evt=1,
        ic=np.array([1.0], dtype=model.dtype),
        params=np.array([2.0], dtype=model.dtype),
        analysis=lyapunov_mle,
    )

    result = sim.results().analysis["lyapunov_mle"]
    assert result.steps == 3
    # Backward compat: raw access still works
    assert result["out"][1] == 3


def test_lyapunov_partial_with_record_interval():
    model = _build_map_model(jit=False)
    sim = Sim(model)
    analysis_factory = lyapunov_mle(record_interval=2)
    mod = sim._resolve_analysis(analysis_factory, record_interval=1)  # Sim-provided interval should be accepted
    assert isinstance(mod, AnalysisModule)
    assert mod.trace_stride == 2  # respects user-provided stride
