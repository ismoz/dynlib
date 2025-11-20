# tests/steppers/ode/test_bdf2a_basic.py
"""
Integration tests: adaptive BDF2 using end-user API.

Tests verify:
- bdf2a produces reasonably accurate results on dx/dt = -x
- bdf2a outperforms Euler for the same nominal initial dt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from dynlib import setup

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_bdf2a_adaptive_accuracy():
    """
    Adaptive BDF2 (bdf2a) on dx/dt = -x with default tolerances.

    We expect reasonably small error over T=2.0.
    """
    # Non-jittable stepper
    sim = setup(DECAY_MODEL, stepper="bdf2a", jit=False)

    T = 2.0
    # dt is initial guess for adaptive method
    sim.run(T=T, dt=0.1, record=True)
    res = sim.raw_results()

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # BDF2 with a decent Newton + tolerances should easily satisfy this
    assert rel_error < 1e-3, f"bdf2a error too large: {rel_error}"


def test_bdf2a_vs_euler_terminal_error():
    """
    For the same nominal dt (initial dt for bdf2a, fixed dt for Euler),
    bdf2a should be notably more accurate at T=2.0.
    """
    T = 2.0
    dt = 0.1
    x_analytic = np.exp(-T)

    # Adaptive BDF2
    sim_bdf2a = setup(DECAY_MODEL, stepper="bdf2a", jit=False)
    sim_bdf2a.run(T=T, dt=dt, record=True)
    res_bdf2a = sim_bdf2a.raw_results()
    x_bdf2a = res_bdf2a.Y_view[0, -1]
    err_bdf2a = abs(x_bdf2a - x_analytic)

    # Euler (fixed-step, same nominal dt)
    sim_euler = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_euler.run(T=T, dt=dt, record=True)
    res_eul = sim_euler.raw_results()
    x_eu = res_eul.Y_view[0, -1]
    err_eu = abs(x_eu - x_analytic)

    # bdf2a should beat Euler by a comfortable margin
    if err_eu > 0:
        assert err_bdf2a < err_eu / 5.0, (
            f"bdf2a not significantly better than Euler: "
            f"bdf2a={err_bdf2a}, euler={err_eu}"
        )
