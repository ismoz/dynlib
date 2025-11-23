# tests/unit/test_sim_results_api.py
"""Unit tests for Sim.results() and Sim.raw_results()."""
from __future__ import annotations
from pathlib import Path
import tomllib
import pytest

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model
from dynlib.runtime.results import Results
from dynlib.runtime.results_api import ResultsView


def _load_decay_model(jit: bool = True) -> Model:
    """Compile a small decay model used across tests."""
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay.toml"
    with open(model_path, "rb") as fh:
        data = tomllib.load(fh)

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
        lag_state_info=full_model.lag_state_info,
        uses_lag=full_model.uses_lag,
        equations_use_lag=full_model.equations_use_lag,
        make_stepper_workspace=full_model.make_stepper_workspace,
    )


@pytest.fixture
def simple_sim() -> Sim:
    return Sim(_load_decay_model(jit=True))


def test_results_requires_run(simple_sim: Sim) -> None:
    with pytest.raises(RuntimeError):
        simple_sim.results()
    with pytest.raises(RuntimeError):
        simple_sim.raw_results()


def test_results_and_raw_access(simple_sim: Sim) -> None:
    simple_sim.run()
    raw = simple_sim.raw_results()
    view_a = simple_sim.results()
    view_b = simple_sim.results()  # cached

    assert isinstance(raw, Results)
    assert isinstance(view_a, ResultsView)
    assert view_a is view_b
    assert raw is simple_sim.raw_results()
    assert raw.n == view_a.t.shape[0]
