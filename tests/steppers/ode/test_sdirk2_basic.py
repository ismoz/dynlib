"""
Integration tests: SDIRK2 stepper using end-user API.

Tests verify:
- SDIRK2 fixed-step method produces reasonable accuracy on dx/dt = -a*x
- SDIRK2 is notably more accurate than Euler for the same dt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from dynlib import setup

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_sdirk2_fixed_step_accuracy():
    """
    SDIRK2 fixed-step method on dx/dt = -x.

    With a modest dt, we expect reasonably small relative error over T=2.0.
    """
    T = 2.0
    dt = 0.05

    sim = setup(DECAY_MODEL, stepper="sdirk2", jit=True)
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()

    assert res.n > 0

    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]

    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / x_analytic

    # Second-order BDF2 with dt=0.05 on dx/dt = -x should comfortably satisfy this.
    assert rel_error < 5e-3, f"SDIRK2 error too large: {rel_error}"


def test_sdirk2_vs_euler_terminal_error():
    """
    For the same fixed dt, SDIRK2 should be notably more accurate than Euler
    at T=2.0 on dx/dt = -x.
    """
    T = 2.0
    dt = 0.1
    x_analytic = np.exp(-T)

    # SDIRK2
    sim_sdirk2 = setup(DECAY_MODEL, stepper="sdirk2", jit=True)
    sim_sdirk2.run(T=T, dt=dt, record=True)
    res_sdirk2 = sim_sdirk2.raw_results()
    assert res_sdirk2.n > 0
    x_sdirk2 = res_sdirk2.Y_view[0, -1]
    err_sdirk2 = abs(x_sdirk2 - x_analytic)

    # Euler
    sim_euler = setup(DECAY_MODEL, stepper="euler", jit=True)
    sim_euler.run(T=T, dt=dt, record=True)
    res_eul = sim_euler.raw_results()
    assert res_eul.n > 0
    x_eu = res_eul.Y_view[0, -1]
    err_eu = abs(x_eu - x_analytic)

    # SDIRK2 should beat Euler by a comfortable factor.
    if err_eu > 0:
        assert err_sdirk2 < err_eu / 5.0, (
            f"SDIRK2 not significantly better: sdirk2={err_sdirk2}, euler={err_eu}"
        )
