# tests/steppers/ode/test_rk4_basic.py
"""
Integration tests: RK4 stepper using end-user API.

Tests verify:
- RK4 fixed-step provides 4th-order accuracy on dx/dt = -a*x
- Order-of-convergence scaling when halving dt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from dynlib import setup

# tests/steppers/ode/test_rk4_basic.py
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_rk4_decay_accuracy():
    """
    RK4 on dx/dt = -x should be very accurate with dt=0.1 over T=2.0.
    """
    sim = setup(DECAY_MODEL, stepper="rk4", jit=True)

    T = 2.0
    dt = 0.1
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()

    assert res.n > 0

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # RK4: expect very small error
    assert rel_error < 1e-5, f"RK4 error too large: {rel_error}"


def test_rk4_order_convergence():
    """
    Check 4th-order convergence:

    Error(dt=0.2) / Error(dt=0.1) ~ 2^4 = 16 (roughly).
    """
    T = 2.0
    x_analytic = np.exp(-T)

    # Coarse dt = 0.2
    sim_coarse = setup(DECAY_MODEL, stepper="rk4", jit=True)
    sim_coarse.run(T=T, dt=0.2, record=True)
    res_coarse = sim_coarse.raw_results()
    idx_c = int(np.argmin(np.abs(res_coarse.T_view - T)))
    x_c = res_coarse.Y_view[0, idx_c]
    err_c = abs(x_c - x_analytic)

    # Fine dt = 0.1
    sim_fine = setup(DECAY_MODEL, stepper="rk4", jit=True)
    sim_fine.run(T=T, dt=0.1, record=True)
    res_fine = sim_fine.raw_results()
    idx_f = int(np.argmin(np.abs(res_fine.T_view - T)))
    x_f = res_fine.Y_view[0, idx_f]
    err_f = abs(x_f - x_analytic)

    if err_f > 1e-14:  # avoid ratio blow-up
        ratio = err_c / err_f
        # Be generous with bounds; we just want "≈ 16"
        assert 10.0 < ratio < 25.0, f"Expected ~16x improvement, got {ratio}"
