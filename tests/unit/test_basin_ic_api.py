from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import importlib
import numpy as np
import pytest

from dynlib import setup
from dynlib.analysis import (
    BasinResult,
    FixedPoint,
    ReferenceRun,
    basin_axis,
    basin_known,
    basin_summary,
    build_known_attractors_psc,
    basin_points,
    basin_values,
)
from dynlib.analysis.basin import Attractor, _resolve_basin_ic
from dynlib.plot import basin_plot


MODEL = """
inline:
[model]
type = "ode"
name = "basin-ic-test"

[states]
x = 1.0
y = 2.0
z = 3.0

[params]
a = 0.0

[equations]
expr = '''
dx = -x
dy = -y
dz = -z
'''
"""


@pytest.fixture
def sim():
    return setup(MODEL, jit=False, disk_cache=False, stepper="rk4")


def test_basin_ic_mapping_broadcasts_fixed_and_defaults(sim):
    ic, meta = _resolve_basin_ic(
        sim,
        {
            "y": basin_axis(-1.0, 1.0, n=3),
            "z": 0.5,
        },
        sim.model.dtype,
    )

    assert ic.shape == (3, 3)
    np.testing.assert_allclose(ic[:, 0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(ic[:, 1], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(ic[:, 2], [0.5, 0.5, 0.5])
    assert meta["ic_vars"] == ("y",)
    assert meta["ic_fixed"]["x"] == 1.0
    assert meta["ic_fixed"]["z"] == 0.5


def test_basin_ic_axis_order_and_center_sampling(sim):
    ic, meta = _resolve_basin_ic(
        sim,
        {
            "z": basin_axis(0.0, 10.0, n=2, sample="center"),
            "x": basin_axis(-1.0, 1.0, n=3),
        },
        sim.model.dtype,
    )

    assert meta["ic_grid"] == (2, 3)
    assert meta["ic_vars"] == ("z", "x")
    assert meta["ic_axis_values"][0] == (2.5, 7.5)
    np.testing.assert_allclose(ic[:, 2], [2.5, 2.5, 2.5, 7.5, 7.5, 7.5])
    np.testing.assert_allclose(ic[:, 0], [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])


def test_basin_values_preserves_explicit_values(sim):
    ic, meta = _resolve_basin_ic(
        sim,
        {"x": basin_values([2.0, -1.0, 0.25])},
        sim.model.dtype,
    )

    np.testing.assert_allclose(ic[:, 0], [2.0, -1.0, 0.25])
    assert meta["ic_axis_values"][0] == (2.0, -1.0, 0.25)
    assert meta["ic_refinable"] is False


def test_basin_summary_compacts_ic_axis_values():
    result = BasinResult(
        labels=np.array([], dtype=np.int64),
        registry=[],
        meta={
            "ic_axis_values": (
                (-4.979166666666667, -4.9375, 4.979166666666666),
                (-1.0, 0.0, 1.0),
            ),
            "ic_axis_sample": ("center", "edge"),
            "ic_vars": ("phi", "x"),
        },
    )

    text = basin_summary(result, meta_keys=["ic_axis_values"], sort_meta=False)

    assert "  ic_axis_values:\n" in text
    assert "    phi: 3 values, min=-4.97917, max=4.97917, sample=center" in text
    assert "    x: 3 values, min=-1, max=1, sample=edge" in text
    assert "(-4.979166666666667, -4.9375, 4.979166666666666)" not in text


def test_basin_points_maps_named_columns_to_state_order(sim):
    points = [[10.0, 20.0], [30.0, 40.0]]
    ic, meta = _resolve_basin_ic(
        sim,
        basin_points(points, vars=["z", "x"]),
        sim.model.dtype,
    )

    np.testing.assert_allclose(ic, [[20.0, 2.0, 10.0], [40.0, 2.0, 30.0]])
    assert meta["ic_kind"] == "points"
    assert meta["ic_vars"] == ("z", "x")


def test_basin_ic_rejects_bad_specs(sim):
    with pytest.raises(ValueError, match="Unknown state"):
        _resolve_basin_ic(sim, {"bad": 1.0}, sim.model.dtype)
    with pytest.raises(TypeError, match="raw IC arrays"):
        _resolve_basin_ic(sim, np.zeros((2, 3)), sim.model.dtype)
    with pytest.raises(ValueError, match="positive"):
        basin_axis(0.0, 1.0, n=0)


def test_old_grid_keywords_are_removed(sim):
    with pytest.raises(TypeError, match="ic_grid"):
        basin_known(
            sim,
            attractors=[FixedPoint(name="origin", loc=[0.0, 0.0])],
            ic_grid=[2, 2],
            ic_bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        )


def test_reference_run_timing_overrides_global_defaults(sim):
    known = build_known_attractors_psc(
        sim,
        [
            ReferenceRun("short", [1.0, 2.0, 3.0], transient_samples=1, signature_samples=2),
            ReferenceRun("default", [1.0, 2.0, 3.0]),
        ],
        dt_obs=0.01,
        transient_samples=3,
        signature_samples=4,
    )

    assert known.meta["reference_timing"] == (
        {"transient_samples": 1, "signature_samples": 2, "max_steps": 4},
        {"transient_samples": 3, "signature_samples": 4, "max_steps": 8},
    )
    assert known.trajectories[0].shape == (4, 3)
    assert known.trajectories[1].shape == (6, 3)


def test_reference_run_requires_positive_resolved_signature_samples(sim):
    with pytest.raises(ValueError, match="signature_samples must be positive"):
        build_known_attractors_psc(
            sim,
            [ReferenceRun("bad", [1.0, 2.0, 3.0])],
            dt_obs=0.01,
            signature_samples=0,
        )

    build_known_attractors_psc(
        sim,
        [ReferenceRun("ok", [1.0, 2.0, 3.0], signature_samples=1)],
        dt_obs=0.01,
        signature_samples=0,
    )


def test_basin_known_refine_preserves_fixed_state(monkeypatch, sim):
    basin_known_mod = importlib.import_module("dynlib.analysis.basin_known")

    calls: list[np.ndarray] = []

    def fake_classify(*, ic_arr, **_kwargs):
        calls.append(ic_arr.copy())
        if len(calls) == 1:
            labels = np.arange(ic_arr.shape[0]) % 2
            return labels.astype(np.int64)
        return np.zeros((ic_arr.shape[0],), dtype=np.int64)

    monkeypatch.setattr(basin_known_mod, "_classify_batch_core_inner", fake_classify)

    result = basin_known(
        sim,
        attractors=[FixedPoint(name="origin", loc=[0.0, 0.0])],
        ic={
            "x": basin_axis(-1.0, 1.0, n=32),
            "y": basin_axis(-1.0, 1.0, n=32),
            "z": 0.25,
        },
        observe_vars=["x", "y"],
        dt_obs=0.01,
        max_samples=4,
        signature_samples=0,
        transient_samples=0,
        refine=True,
        coarse_factor=4,
        parallel_mode="none",
    )

    assert result.meta["ic_grid"] == (32, 32)
    assert len(calls) >= 2
    for arr in calls:
        assert arr.shape[1] == 3
        np.testing.assert_allclose(arr[:, 2], 0.25)


def test_basin_plot_uses_ic_metadata_for_axes(sim):
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    result = BasinResult(
        labels=labels,
        registry=[
            Attractor(id=0, fingerprint=set(), cells=set()),
            Attractor(id=1, fingerprint=set(), cells=set()),
        ],
        meta={
            "ic_grid": (3, 2),
            "ic_bounds": ((-1.0, 1.0), (2.5, 7.5)),
            "ic_vars": ("x", "z"),
            "ic_axis_values": ((-1.0, 0.0, 1.0), (2.5, 7.5)),
        },
    )

    ax = basin_plot(result, colorbar=False)

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "z"
    assert tuple(ax.get_xlim()) == pytest.approx((-1.0, 1.0))
    assert tuple(ax.get_ylim()) == pytest.approx((2.5, 7.5))
