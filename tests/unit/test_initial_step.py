import numpy as np
import pytest

from dynlib.runtime.initial_step import (
    WRMSConfig,
    choose_initial_dt_wrms,
    make_wrms_config_from_stepper,
)
from dynlib.steppers.base import StepperMeta


def _rhs_linear(t, y_vec, dy_out, params, runtime_ws):  # pragma: no cover - helper
    dy_out[:] = params[0] * y_vec


def test_choose_initial_dt_wrms_respects_bounds():
    y0 = np.array([1.0, 2.0], dtype=np.float64)
    params = np.array([1.0], dtype=np.float64)
    cfg = WRMSConfig(atol=1e-6, rtol=1e-3, order=2, safety=1.0, min_dt=0.2, max_dt=0.2)

    dt = choose_initial_dt_wrms(
        rhs=_rhs_linear,
        runtime_ws=None,
        t0=0.0,
        y0=y0,
        params=params,
        t_end=10.0,
        cfg=cfg,
    )
    assert pytest.approx(dt) == 0.2


def test_make_wrms_config_from_stepper_dict_source():
    meta = StepperMeta(name="rk", kind="ode", time_control="adaptive", order=4)
    cfg = make_wrms_config_from_stepper(
        meta,
        {"atol": 1e-7, "rtol": 1e-4},
        safety=0.5,
        min_dt=1e-8,
        max_dt=0.5,
    )
    assert cfg is not None
    assert cfg.atol == pytest.approx(1e-7)
    assert cfg.rtol == pytest.approx(1e-4)
    assert cfg.order == 4
    assert cfg.safety == 0.5
    assert cfg.max_dt == 0.5


def test_make_wrms_config_from_stepper_rejects_missing_fields():
    meta = StepperMeta(name="rk", kind="ode", time_control="adaptive", order=3)
    assert make_wrms_config_from_stepper(meta, {"atol": 1e-6}) is None
    assert make_wrms_config_from_stepper(
        StepperMeta(name="map", kind="map", time_control="fixed"),
        {"atol": 1e-6, "rtol": 1e-3},
    ) is None
