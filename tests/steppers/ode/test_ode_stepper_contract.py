# tests/steppers/ode/test_ode_stepper_contract.py
"""
ODE stepper contract tests (cross-stepper):

- JIT on/off parity for all core ODE steppers
- Registration and metadata checks (order, time_control, aliases)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynlib import setup
from dynlib.steppers.registry import get_stepper, registry

# tests/steppers/ode/test_ode_stepper_contract.py
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


@pytest.mark.parametrize("stepper", ["euler", "rk4", "rk45", "ab2", "ab3", "bdf2_jit"])
def test_jit_on_off_parity(stepper: str):
    """
    Guardrail: JIT on/off must produce identical results for the same
    model + stepper + run arguments.
    """
    T = 2.0
    dt = 0.1

    # JIT = True
    sim_jit = setup(DECAY_MODEL, stepper=stepper, jit=True)
    sim_jit.run(T=T, dt=dt, record=True)
    res_jit = sim_jit.raw_results()

    # JIT = False
    sim_no = setup(DECAY_MODEL, stepper=stepper, jit=False)
    sim_no.run(T=T, dt=dt, record=True)
    res_no = sim_no.raw_results()

    assert res_jit.n == res_no.n

    np.testing.assert_allclose(
        res_jit.T_view,
        res_no.T_view,
        rtol=1e-14,
        atol=0.0,
        err_msg=f"Time mismatch for stepper {stepper}",
    )
    np.testing.assert_allclose(
        res_jit.Y_view,
        res_no.Y_view,
        rtol=1e-14,
        atol=0.0,
        err_msg=f"State mismatch for stepper {stepper}",
    )


def test_stepper_registry_and_meta():
    """
    Basic contract for core ODE steppers:
      - registered names exist
      - meta.order / embedded_order / time_control are as expected
    """
    reg = registry()

    # Presence in registry
    for name in ("euler", "rk4", "rk45"):
        assert name in reg, f"Stepper '{name}' missing from registry()"

    euler = get_stepper("euler")
    assert euler.meta.name == "euler"
    assert euler.meta.order == 1
    assert getattr(euler.meta, "time_control", None) == "fixed"

    rk4 = get_stepper("rk4")
    assert rk4.meta.name == "rk4"
    assert rk4.meta.order == 4
    assert getattr(rk4.meta, "time_control", None) == "fixed"

    rk45 = get_stepper("rk45")
    assert rk45.meta.name == "rk45"
    assert rk45.meta.order == 5
    # Dormand–Prince(5,4)
    assert getattr(rk45.meta, "embedded_order", None) == 4
    assert getattr(rk45.meta, "time_control", None) == "adaptive"

    ab2 = get_stepper("ab2")
    assert ab2.meta.name == "ab2"
    # Second-order Adams–Bashforth
    assert ab2.meta.order == 2
    assert getattr(ab2.meta, "time_control", None) == "fixed"

    ab3 = get_stepper("ab3")
    assert ab3.meta.name == "ab3"
    # Third-order Adams–Bashforth
    assert ab3.meta.order == 3
    assert getattr(ab3.meta, "time_control", None) == "fixed"

    bdf2_jit = get_stepper("bdf2_jit")
    assert bdf2_jit.meta.name == "bdf2_jit"
    # Second-order BDF
    assert bdf2_jit.meta.order == 2
    assert getattr(bdf2_jit.meta, "time_control", None) == "fixed"
    # Jacobian capability: internal numeric J, no external Jacobian API
    assert bdf2_jit.meta.caps.jacobian == "internal"