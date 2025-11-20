# tests/steppers/ode/test_rk2_midpoint_basic.py
"""
Integration tests: RK2 (explicit midpoint) stepper using end-user API.

Tests verify:
- RK2 fixed-step provides 2nd-order accuracy on dx/dt = -a*x
- Order-of-convergence scaling when halving dt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from dynlib import setup

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_rk2_decay_accuracy():
    """
    RK2 on dx/dt = -x should be reasonably accurate with dt=0.1 over T=2.0.
    """
    sim = setup(DECAY_MODEL, stepper="rk2", jit=True)

    T = 2.0
    dt = 0.1
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()

    assert res.n > 0

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # RK2 (midpoint) is 2nd-order: expect small but not RK4-level error
    # This bound should be comfortably satisfied while still catching regressions.
    assert rel_error < 5e-3, f"RK2 error too large: {rel_error}"


def test_rk2_order_convergence():
    """
    Check 2nd-order convergence:

    Error(dt=0.2) / Error(dt=0.1) ~ 2^2 = 4 (roughly).
    """
    T = 2.0
    x_analytic = np.exp(-T)

    # Coarse dt = 0.2
    sim_coarse = setup(DECAY_MODEL, stepper="rk2", jit=True)
    sim_coarse.run(T=T, dt=0.2, record=True)
    res_coarse = sim_coarse.raw_results()
    idx_c = int(np.argmin(np.abs(res_coarse.T_view - T)))
    x_c = res_coarse.Y_view[0, idx_c]
    err_c = abs(x_c - x_analytic)

    # Fine dt = 0.1
    sim_fine = setup(DECAY_MODEL, stepper="rk2", jit=True)
    sim_fine.run(T=T, dt=0.1, record=True)
    res_fine = sim_fine.raw_results()
    idx_f = int(np.argmin(np.abs(res_fine.T_view - T)))
    x_f = res_fine.Y_view[0, idx_f]
    err_f = abs(x_f - x_analytic)

    if err_f > 1e-14:  # avoid ratio blow-up
        ratio = err_c / err_f
        # Be generous with bounds; we just want "≈ 4"
        assert 2.5 < ratio < 6.0, f"Expected ~4x improvement, got {ratio}"


def test_rk2_vs_euler_terminal_error():
    """
    For the same fixed dt, RK2 should be notably more accurate than Euler
    at T=2.0 on dx/dt = -x.
    """
    T = 2.0
    dt = 0.1
    x_analytic = np.exp(-T)

    # RK2
    sim_rk2 = setup(DECAY_MODEL, stepper="rk2", jit=True)
    sim_rk2.run(T=T, dt=dt, record=True)
    res_rk2 = sim_rk2.raw_results()
    idx_rk2 = int(np.argmin(np.abs(res_rk2.T_view - T)))
    x_rk2 = res_rk2.Y_view[0, idx_rk2]
    err_rk2 = abs(x_rk2 - x_analytic)

    # Euler
    sim_euler = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_euler.run(T=T, dt=dt, record=True)
    res_eul = sim_euler.raw_results()
    idx_eu = int(np.argmin(np.abs(res_eul.T_view - T)))
    x_eu = res_eul.Y_view[0, idx_eu]
    err_eu = abs(x_eu - x_analytic)

    # RK2 should beat Euler by a comfortable factor.
    if err_eu > 1e-12:  # avoid noisy ratios
        ratio = err_eu / err_rk2
        assert ratio > 5.0, (
            f"RK2 not significantly better than Euler: "
            f"ratio={ratio}, rk2={err_rk2}, euler={err_eu}"
        )
