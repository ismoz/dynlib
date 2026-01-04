import numpy as np
import pytest
from pathlib import Path

from dynlib import Status, build
from dynlib.errors import ModelLoadError
from dynlib.runtime.sim import Sim


def _build_logistic_map_with_stop(*, tmp_path: Path, cond: str, jit: bool = False):
    test_dir = Path(__file__).resolve().parents[1]
    base_model = test_dir / "data" / "models" / "logistic_map.toml"

    mod_file = tmp_path / "stop_mod.toml"
    mod_file.write_text(
        """
[mod]
name = "stop"

[mod.set.sim]
stop = """ + repr(cond) + """
"""
    )

    return build(str(base_model), mods=[str(mod_file)], jit=jit)


def _build_logistic_map_with_stop_table(*, tmp_path: Path, cond: str, phase: str):
    test_dir = Path(__file__).resolve().parents[1]
    base_model = test_dir / "data" / "models" / "logistic_map.toml"

    mod_file = tmp_path / "stop_mod.toml"
    mod_file.write_text(
        f"""
[mod]
name = "stop"

[mod.set.sim]
stop = {{ cond = {cond!r}, phase = {phase!r} }}
"""
    )

    return build(str(base_model), mods=[str(mod_file)], jit=False)


def test_stop_early_exit_wrapper_success(tmp_path: Path):
    model = _build_logistic_map_with_stop(tmp_path=tmp_path, cond="x > 0.8")
    sim = Sim(model)

    sim.run(N=100)
    raw = sim.raw_results()

    assert raw.ok is True
    assert raw.exited_early is True
    assert int(raw.status) == int(Status.EARLY_EXIT)

    # Logistic map with r=3.5, x0=0.1 crosses 0.8 at step 6.
    assert int(raw.step_count_final) == 6
    assert raw.n == 7  # t0 record + 6 iterations


def test_stop_early_exit_fastpath_success(tmp_path: Path):
    model = _build_logistic_map_with_stop(tmp_path=tmp_path, cond="x > 0.8")
    sim = Sim(model)

    res = sim.fastpath(N=100)

    assert res.ok is True
    assert res.exited_early is True
    assert int(res.status) == int(Status.EARLY_EXIT)
    assert int(res.step_count_final) == 6
    assert int(res.n) == 7


def test_stop_early_exit_wrapper_jit_success(tmp_path: Path):
    pytest.importorskip("numba")
    model = _build_logistic_map_with_stop(tmp_path=tmp_path, cond="x > 0.8", jit=True)
    sim = Sim(model)

    sim.run(N=100)
    raw = sim.raw_results()

    assert raw.ok is True
    assert raw.exited_early is True
    assert int(raw.status) == int(Status.EARLY_EXIT)
    assert int(raw.step_count_final) == 6
    assert raw.n == 7


def test_stop_early_exit_fastpath_jit_success(tmp_path: Path):
    pytest.importorskip("numba")
    model = _build_logistic_map_with_stop(tmp_path=tmp_path, cond="x > 0.8", jit=True)
    sim = Sim(model)

    res = sim.fastpath(N=100)

    assert res.ok is True
    assert res.exited_early is True
    assert int(res.status) == int(Status.EARLY_EXIT)
    assert int(res.step_count_final) == 6
    assert int(res.n) == 7


@pytest.mark.parametrize("phase", ["pre", "both"])
def test_stop_phase_rejects_pre_and_both(tmp_path: Path, phase: str):
    with pytest.raises(ModelLoadError, match="phase must be 'post'"):
        _build_logistic_map_with_stop_table(tmp_path=tmp_path, cond="x > 0.8", phase=phase)
