# tests/steppers/ode/test_ab3_basic.py
"""
Integration tests: AB3 stepper using end-user API.

Tests verify:
- AB3 fixed-step method produces good accuracy on dx/dt = -a*x
- AB3 is notably more accurate than AB2 for the same dt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from dynlib import setup

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_ab3_fixed_step_accuracy():
    """
    AB3 fixed-step method on dx/dt = -x.

    With a modest dt, we expect small relative error over T=2.0.
    """
    T = 2.0
    dt = 0.1

    sim = setup(DECAY_MODEL, stepper="ab3", jit=True)
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # Third-order AB3 with dt=0.1 on dx/dt = -x should comfortably satisfy this.
    assert rel_error < 1e-3, f"AB3 error too large: {rel_error}"


def test_ab3_vs_ab2_terminal_error():
    """
    For the same fixed dt, AB3 should be notably more accurate than AB2
    at T=2.0 on dx/dt = -x.
    """
    T = 2.0
    dt = 0.1
    x_analytic = np.exp(-T)

    # AB3
    sim_ab3 = setup(DECAY_MODEL, stepper="ab3", jit=True)
    sim_ab3.run(T=T, dt=dt, record=True)
    res_ab3 = sim_ab3.raw_results()
    x_ab3 = res_ab3.Y_view[0, -1]
    err_ab3 = abs(x_ab3 - x_analytic)

    # AB2
    sim_ab2 = setup(DECAY_MODEL, stepper="ab2", jit=True)
    sim_ab2.run(T=T, dt=dt, record=True)
    res_ab2 = sim_ab2.raw_results()
    x_ab2 = res_ab2.Y_view[0, -1]
    err_ab2 = abs(x_ab2 - x_analytic)

    # AB3 should beat AB2 by a comfortable factor.
    if err_ab2 > 0:
        assert err_ab3 < err_ab2 / 10.0, (
            f"AB3 not significantly better than AB2: "
            f"ab3={err_ab3}, ab2={err_ab2}"
        )
