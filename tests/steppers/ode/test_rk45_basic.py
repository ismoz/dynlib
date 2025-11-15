# tests/steppers/ode/test_rk45_basic.py
"""
Integration tests: RK45 stepper using end-user API.

Tests verify:
- RK45 adaptive method produces accurate results on dx/dt = -a*x
- RK45 significantly outperforms Euler for the same nominal dt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from dynlib import setup

# tests/steppers/ode/test_rk45_basic.py
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_rk45_adaptive_accuracy():
    """
    RK45 adaptive method on dx/dt = -x with default tolerances.

    We expect reasonably small error over T=2.0.
    """
    sim = setup(DECAY_MODEL, stepper="rk45", jit=True)

    T = 2.0
    # dt is initial guess for adaptive; pick something reasonable
    sim.run(T=T, dt=0.1, record=True)
    res = sim.raw_results()

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # This bound should be comfortably satisfied with typical DP(5,4) tolerances
    assert rel_error < 1e-3, f"RK45 error too large: {rel_error}"


def test_rk45_vs_euler_terminal_error():
    """
    For the same nominal dt (initial dt for RK45, fixed dt for Euler),
    RK45 should be notably more accurate at T=2.0.
    """
    T = 2.0
    dt = 0.1
    x_analytic = np.exp(-T)

    # RK45
    sim_rk45 = setup(DECAY_MODEL, stepper="rk45", jit=True)
    sim_rk45.run(T=T, dt=dt, record=True)
    res_rk45 = sim_rk45.raw_results()
    x_rk = res_rk45.Y_view[0, -1]
    err_rk = abs(x_rk - x_analytic)

    # Euler
    sim_euler = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_euler.run(T=T, dt=dt, record=True)
    res_eul = sim_euler.raw_results()
    x_eu = res_eul.Y_view[0, -1]
    err_eu = abs(x_eu - x_analytic)

    # RK45 should beat Euler by at least ~10x
    if err_eu > 0:
        assert err_rk < err_eu / 10.0, f"RK45 not significantly better: rk45={err_rk}, euler={err_eu}"
