from __future__ import annotations

from pathlib import Path
import tomllib
import numpy as np
import pytest

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build, FullModel
from dynlib.runtime.sim import Sim


def _load_model(toml_name: str) -> FullModel:
    data_dir = Path(__file__).parent.parent.parent / "data" / "models"
    with open(data_dir / toml_name, "rb") as fh:
        data = tomllib.load(fh)
    spec = build_spec(parse_model_v2(data))
    return build(spec, stepper=spec.sim.stepper, jit=True)


@pytest.fixture(scope="module")
def decay_model() -> FullModel:
    return _load_model("decay_with_event.toml")


def test_resume_stitches_without_duplicates():
    sim = Sim(_load_model("decay_with_event.toml"))
    sim.run(T=1.0)
    sim.run(T=2.0, resume=True)
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
    sim.run(T=0.5, record=False)
    prev_steps = sim.session_state_summary()["step"]
    sim.run(T=1.0, record=True, resume=True)
    res = sim.raw_results()
    assert len(res.STEP_view) > 0, "Second run should record samples"
    assert res.STEP_view[-1] > prev_steps, "Recorded STEP axis must extend past prior session steps"


def test_snapshot_reset_clears_results():
    sim = Sim(_load_model("decay.toml"))
    sim.run(T=0.5)
    sim.create_snapshot("checkpoint", description="after first segment")
    sim.run(T=1.0, resume=True)
    assert sim.raw_results().n > 0
    sim.reset("checkpoint")
    summary = sim.session_state_summary()
    assert summary["t"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        sim.raw_results()


def test_segments_single_run(decay_model):
    sim = Sim(decay_model)
    sim.run(T=0.5)
    res = sim.results()
    assert len(res.segment) == 1
    seg = res.segment["run#0"]
    np.testing.assert_allclose(seg.t, res.t)
    first_state = res.state_names[0]
    np.testing.assert_allclose(seg[first_state], res[first_state])


def test_segments_resume_bounds(decay_model):
    sim = Sim(decay_model)
    sim.run(T=0.4)
    seg_initial = sim.results().segment[0]
    initial_len = len(seg_initial.t)
    assert initial_len > 0
    sim.run(T=0.8, resume=True)
    res = sim.results()
    assert len(res.segment) == 2
    lengths = [len(res.segment[i].t) for i in range(len(res.segment))]
    total = len(res.t)
    assert sum(lengths) in (total, total - 1)


def test_segments_tags_and_events(decay_model):
    sim = Sim(decay_model)
    sim.run(T=0.2, tag="baseline")
    sim.run(T=0.6, resume=True, tag="I=12")
    res = sim.results()
    assert "baseline" in res.segment
    assert "I=12" in res.segment
    assert "run#0" in res.segment
    assert "run#1" in res.segment
    seg_named = res.segment["I=12"]
    assert seg_named.name == "I=12"
    raw = sim.raw_results()
    codes_full = raw.EVT_CODE_view[: raw.m]
    codes_seg, idx_seg, logs_seg = seg_named.events()
    assert codes_seg.shape[0] == seg_named.meta.evt_len
    if seg_named.meta.evt_len:
        start = seg_named.meta.evt_start
        stop = start + seg_named.meta.evt_len
        np.testing.assert_array_equal(codes_seg, codes_full[start:stop])
    assert logs_seg.shape[0] == seg_named.meta.evt_len
    assert idx_seg.shape[0] == seg_named.meta.evt_len


def test_segments_reset_clears_history(decay_model):
    sim = Sim(decay_model)
    sim.run(T=0.3)
    sim.run(T=0.6, resume=True)
    assert len(sim.results().segment) == 2
    sim.reset()
    sim.run(T=0.1)
    res = sim.results()
    assert len(res.segment) == 1
    assert "run#0" in res.segment


def test_segments_no_recording_skips_segment(decay_model):
    sim = Sim(decay_model)
    sim.run(T=0.3, record=False)
    res = sim.results()
    assert len(res.segment) == 0


def test_segments_renaming_unique(decay_model):
    sim = Sim(decay_model)
    sim.run(T=0.25)
    sim.run(T=0.5, resume=True)
    sim.name_segment("run#0", "baseline")
    sim.name_last_segment("baseline")  # should auto-suffix
    res = sim.results()
    assert "baseline" in res.segment
    assert "baseline-2" in res.segment
    assert "run#1" in res.segment
    assert res.segment["baseline"].name == "baseline"
    assert res.segment["baseline-2"].name == "baseline-2"
    if len(res.state_names) >= 2:
        pair = res.segment["baseline-2"][list(res.state_names[:2])]
        assert pair.shape[0] == len(res.segment["baseline-2"].t)
