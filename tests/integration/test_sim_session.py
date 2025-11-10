from __future__ import annotations

from pathlib import Path
import tomllib
import numpy as np
import pytest

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model


def _load_model(toml_name: str) -> Model:
    data_dir = Path(__file__).parent.parent / "data" / "models"
    with open(data_dir / toml_name, "rb") as fh:
        data = tomllib.load(fh)
    spec = build_spec(parse_model_v2(data))
    full_model = build(spec, stepper=spec.sim.stepper, jit=True)
    return Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        struct=full_model.struct,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
    )


def test_resume_stitches_without_duplicates():
    sim = Sim(_load_model("decay_with_event.toml"))
    sim.run(t_end=1.0)
    sim.run(t_end=2.0, resume=True)
    res = sim.raw_results()
    t = res.T_view
    step = res.STEP_view
    assert np.all(np.diff(t) > 0.0), "Time axis should be strictly increasing after resume"
    assert np.all(np.diff(step) > 0), "STEP should be strictly increasing across runs"
    evt_idx = res.EVT_INDEX_view
    if evt_idx.size:
        non_negative = evt_idx[evt_idx >= 0]
        assert np.all(np.diff(non_negative) >= 0), "Event ownership index must be non-decreasing"


def test_record_off_then_resume_records_with_offset():
    sim = Sim(_load_model("decay.toml"))
    sim.run(t_end=0.5, record=False)
    prev_steps = sim.session_state_summary()["step"]
    sim.run(t_end=1.0, record=True, resume=True)
    res = sim.raw_results()
    assert len(res.STEP_view) > 0, "Second run should record samples"
    assert res.STEP_view[-1] > prev_steps, "Recorded STEP axis must extend past prior session steps"


def test_snapshot_reset_clears_results():
    sim = Sim(_load_model("decay.toml"))
    sim.run(t_end=0.5)
    sim.create_snapshot("checkpoint", description="after first segment")
    sim.run(t_end=1.0, resume=True)
    assert sim.raw_results().n > 0
    sim.reset("checkpoint")
    summary = sim.session_state_summary()
    assert summary["t"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        sim.raw_results()
