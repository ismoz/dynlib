# tests/steppers/ode/test_euler_basic.py
"""
Integration tests: end-to-end Euler simulations using end-user API.

Tests verify:
- Analytic solution matching for dx/dt = -a*x
- Transient warm-up behavior (reuse of state)
- Event mutations work correctly
- Growth paths are triggered with small capacities
- Buffer growth does not change recorded trajectory
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynlib import setup

# tests/steppers/ode/test_euler_basic.py
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")
DECAY_WITH_EVENT_MODEL = str(DATA_DIR / "decay_with_event.toml")


def test_euler_decay_analytic():
    """
    dx/dt = -a*x, x(0)=1, a=1
    Analytic: x(t) = exp(-t)

    We integrate with Euler (dt=0.1, T=2.0) and require modest accuracy.
    """
    sim = setup(DECAY_MODEL, stepper="euler", jit=True)

    T = 2.0
    dt = 0.1
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()

    assert res.n > 0

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # Euler: O(dt) global error; for dt=0.1 over T=2 this is loose but OK.
    assert rel_error < 0.15, f"Euler too far from analytic: {x_final} vs {x_analytic}"


def test_euler_transient_warmup_reuses_state():
    """
    Transient warm-up should advance the state before recording and
    then restart the time axis at 0 for the recorded run.

    We compare:
      - reference: run from t=0 to T_total=3.0
      - transient: warm-up 1.0 (unrecorded), then record T=2.0

    Recorded trajectory after transient should match reference
    trajectory shifted by 1.0 in time.
    """
    T_total = 3.0
    dt = 0.1

    # Reference run: no transient, T_total
    sim_ref = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_ref.run(T=T_total, dt=dt, record=True)
    res_ref = sim_ref.raw_results()

    # Transient run: warm-up 1.0, then record 2.0
    sim_tr = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_tr.run(T=2.0, dt=dt, transient=1.0, record=True)
    res_tr = sim_tr.raw_results()

    # Recorded time should restart at 0 and end at 2.0
    assert res_tr.T_view[0] == pytest.approx(0.0)
    assert res_tr.T_view[-1] == pytest.approx(2.0)

    # Drop first 1.0 units from reference and compare
    start_idx = int(np.searchsorted(res_ref.T_view, 1.0 - 1e-12, side="left"))
    ref_t = res_ref.T_view[start_idx:] - 1.0
    ref_y = res_ref.Y_view[:, start_idx:]

    assert ref_t.shape[0] == res_tr.n
    np.testing.assert_allclose(res_tr.T_view, ref_t, atol=1e-12)
    np.testing.assert_allclose(res_tr.Y_view, ref_y, atol=1e-12)


def test_euler_with_event_resets_state():
    """
    Use a model with an event that resets x when it crosses a threshold.

    We only check qualitative behavior: state must occasionally jump up,
    otherwise pure decay would be monotone decreasing.
    """
    sim = setup(DECAY_WITH_EVENT_MODEL, stepper="euler", jit=True)

    sim.run(T=2.0, dt=0.1, record=True)
    res = sim.raw_results()

    x = res.Y_view[0, : res.n]
    assert x.size > 2

    # Count noticeable upward jumps (resets)
    jumps = 0
    for i in range(1, x.size):
        if x[i] > x[i - 1] + 0.1:
            jumps += 1

    assert jumps >= 1, f"Expected at least one reset-like jump, got {jumps}"


def test_euler_record_buffer_growth_triggered():
    """
    Start with tiny cap_rec to force geometric growth.

    Growth must not crash and we must end up with more than initial
    capacity worth of records.
    """
    sim = setup(DECAY_MODEL, stepper="euler", jit=True)

    sim.run(T=2.0, dt=0.1, cap_rec=4, record_interval=1, record=True)
    res = sim.raw_results()

    # dt=0.1, T=2.0 -> ~20 steps; so > 4 records if growth worked.
    assert res.n > 4
    assert res.T_view[0] == pytest.approx(0.0)
    assert res.Y_view[0, 0] == pytest.approx(1.0)


def test_euler_growth_vs_reference_trajectory():
    """
    Forcing record-buffer growth must not change the recorded trajectory.

    Compare:
      - reference: large cap_rec (no growth)
      - small-cap: tiny cap_rec (forces growth)
    """
    T = 2.0
    dt = 0.1

    # Reference
    sim_ref = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_ref.run(T=T, dt=dt, cap_rec=128, record_interval=1, record=True)
    res_ref = sim_ref.raw_results()

    # Tiny capacity
    sim_small = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_small.run(T=T, dt=dt, cap_rec=3, record_interval=1, record=True)
    res_small = sim_small.raw_results()

    assert res_small.n == res_ref.n

    np.testing.assert_allclose(
        res_small.T_view,
        res_ref.T_view,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        res_small.Y_view,
        res_ref.Y_view,
        rtol=0.0,
        atol=1e-12,
    )
