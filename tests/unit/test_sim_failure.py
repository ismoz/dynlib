from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import tomllib

from dynlib.compiler.build import build
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.runtime.results import Results
from dynlib.runtime.runner_api import Status
from dynlib.runtime.sim import Sim
from dynlib.compiler.build import FullModel


def _load_model(toml_name: str) -> FullModel:
    data_dir = Path(__file__).resolve().parent.parent / "data" / "models"
    with open(data_dir / toml_name, "rb") as fh:
        data = tomllib.load(fh)
    spec = build_spec(parse_model_v2(data))
    return build(spec, stepper=spec.sim.stepper, jit=True)


@pytest.fixture(scope="module")
def decay_model() -> FullModel:
    return _load_model("decay.toml")


@pytest.fixture()
def fresh_sim(decay_model: FullModel) -> Sim:
    return Sim(decay_model)


def _make_result(sim: Sim, status: Status, *, t_final: float = 0.0, step_count: int = 0) -> Results:
    n_state = sim._n_state
    dtype = sim._dtype
    log_cols = max(1, sim._max_log_width)
    return Results(
        T=np.zeros((1,), dtype=np.float64),
        Y=np.zeros((n_state, 1), dtype=dtype),
        AUX=None,
        STEP=np.zeros((1,), dtype=np.int64),
        FLAGS=np.zeros((1,), dtype=np.int32),
        EVT_CODE=np.zeros((1,), dtype=np.int32),
        EVT_INDEX=-np.ones((1,), dtype=np.int32),
        EVT_LOG_DATA=np.zeros((1, log_cols), dtype=dtype),
        n=0,
        m=0,
        status=int(status),
        final_state=np.zeros((n_state,), dtype=dtype),
        final_params=np.array(sim.model.spec.param_vals, dtype=dtype, copy=True),
        t_final=float(t_final),
        final_dt=float(sim._nominal_dt),
        step_count_final=int(step_count),
        final_workspace={"runtime": {}, "stepper": {}},
        state_names=[],
        aux_names=[],
    )


def test_transient_stepfail_aborts_run(monkeypatch: pytest.MonkeyPatch, fresh_sim: Sim) -> None:
    calls: list[bool] = []

    def fake_execute_run(self: Sim, **kwargs):
        calls.append(bool(kwargs.get("record")))
        return _make_result(self, Status.STEPFAIL, t_final=kwargs["t_end"])

    monkeypatch.setattr(Sim, "_execute_run", fake_execute_run)

    with pytest.raises(RuntimeError, match="transient warm-up"):
        fresh_sim.run(T=1.0, transient=0.1)

    assert calls == [False]
    summary = fresh_sim.session_state_summary()
    assert summary["step"] == 0
    with pytest.raises(RuntimeError):
        fresh_sim.raw_results()


def test_recorded_run_stepfail_aborts_run(monkeypatch: pytest.MonkeyPatch, fresh_sim: Sim) -> None:
    calls: list[bool] = []

    def fake_execute_run(self: Sim, **kwargs):
        calls.append(bool(kwargs.get("record")))
        return _make_result(self, Status.STEPFAIL, t_final=kwargs["t_end"])

    monkeypatch.setattr(Sim, "_execute_run", fake_execute_run)

    with pytest.raises(RuntimeError, match="recorded run"):
        fresh_sim.run(T=0.5)

    assert calls == [True]
    summary = fresh_sim.session_state_summary()
    assert summary["step"] == 0
    with pytest.raises(RuntimeError):
        fresh_sim.raw_results()
