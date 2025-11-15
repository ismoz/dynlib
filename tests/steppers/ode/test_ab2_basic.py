# tests/steppers/ode/test_ab2_basic.py
"""
Integration tests: AB2 stepper using end-user API.

Tests verify:
- AB2 fixed-step method produces reasonable accuracy on dx/dt = -a*x
- AB2 is notably more accurate than Euler for the same dt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from dynlib import setup

# tests/steppers/ode/test_ab2_basic.py
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_ab2_fixed_step_accuracy():
    """
    AB2 fixed-step method on dx/dt = -x.

    With a modest dt, we expect reasonably small relative error over T=2.0.
    """
    T = 2.0
    dt = 0.05

    sim = setup(DECAY_MODEL, stepper="ab2", jit=True)
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # Second-order AB2 with dt=0.1 on dx/dt = -x should comfortably satisfy this.
    assert rel_error < 5e-3, f"AB2 error too large: {rel_error}"


def test_ab2_vs_euler_terminal_error():
    """
    For the same fixed dt, AB2 should be notably more accurate than Euler
    at T=2.0 on dx/dt = -x.
    """
    T = 2.0
    dt = 0.1
    x_analytic = np.exp(-T)

    # AB2
    sim_ab2 = setup(DECAY_MODEL, stepper="ab2", jit=True)
    sim_ab2.run(T=T, dt=dt, record=True)
    res_ab2 = sim_ab2.raw_results()
    x_ab2 = res_ab2.Y_view[0, -1]
    err_ab2 = abs(x_ab2 - x_analytic)

    # Euler
    sim_euler = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_euler.run(T=T, dt=dt, record=True)
    res_eul = sim_euler.raw_results()
    x_eu = res_eul.Y_view[0, -1]
    err_eu = abs(x_eu - x_analytic)

    # AB2 should beat Euler by a comfortable factor.
    if err_eu > 0:
        assert err_ab2 < err_eu / 5.0, f"AB2 not significantly better: ab2={err_ab2}, euler={err_eu}"
