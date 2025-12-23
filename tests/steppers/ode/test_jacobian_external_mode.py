"""
Integration tests: Accuracy of external Jacobian mode for implicit steppers.

Tests verify that steppers (bdf2, bdf2a, tr-bdf2a, sdirk2) using DSL-provided
Jacobian functions (jacobian_mode="external") produce accurate results.

All tests use models with explicitly declared Jacobian matrices to test
the external mode functionality.
"""
from __future__ import annotations
import pytest
import numpy as np

from dynlib import setup


# Models with declared Jacobians
EXPDECAY_MODEL = "builtin://ode/expdecay"  # dx/dt = -a*x, simple 1D
VANDERPOL_MODEL = "builtin://ode/vanderpol"  # 2D nonlinear oscillator


def test_bdf2_external_jacobian_accuracy():
    """
    BDF2 with external Jacobian on expdecay model.
    
    Tests that external Jacobian mode produces accurate results
    for the simple decay equation dx/dt = -a*x.
    """
    sim = setup(EXPDECAY_MODEL, stepper="bdf2", jit=True)
    sim.config(jacobian_mode="external")
    
    T = 2.0
    dt = 0.05
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()
    
    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]
    
    # Analytic solution: x(t) = x0 * exp(-a*t)
    x0 = 1.0  # initial condition from expdecay.toml
    a = 1.0   # parameter from expdecay.toml
    x_analytic = x0 * np.exp(-a * t_final)
    
    rel_error = abs(x_final - x_analytic) / abs(x_analytic)
    assert rel_error < 1e-3, f"BDF2 external Jacobian error too large: {rel_error}"


def test_bdf2a_external_jacobian_accuracy():
    """
    Adaptive BDF2 with external Jacobian on expdecay model.
    
    Tests that external Jacobian mode with adaptive stepping
    produces accurate results.
    """
    sim = setup(EXPDECAY_MODEL, stepper="bdf2a", jit=False)
    sim.config(jacobian_mode="external")
    
    T = 2.0
    sim.run(T=T, dt=0.1, record=True)
    res = sim.raw_results()
    
    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]
    
    x0 = 1.0
    a = 1.0
    x_analytic = x0 * np.exp(-a * t_final)
    
    rel_error = abs(x_final - x_analytic) / abs(x_analytic)
    assert rel_error < 1e-4, f"BDF2A external Jacobian error too large: {rel_error}"


def test_tr_bdf2a_external_jacobian_accuracy():
    """
    TR-BDF2A with external Jacobian on expdecay model.
    
    Tests that external Jacobian mode works correctly for the
    combined trapezoidal rule + BDF2 adaptive method.
    """
    sim = setup(EXPDECAY_MODEL, stepper="tr-bdf2a", jit=False)
    sim.config(jacobian_mode="external")
    
    T = 2.0
    sim.run(T=T, dt=0.1, record=True)
    res = sim.raw_results()
    
    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]
    
    x0 = 1.0
    a = 1.0
    x_analytic = x0 * np.exp(-a * t_final)
    
    rel_error = abs(x_final - x_analytic) / abs(x_analytic)
    assert rel_error < 1e-4, f"TR-BDF2A external Jacobian error too large: {rel_error}"


def test_sdirk2_external_jacobian_accuracy():
    """
    SDIRK2 with external Jacobian on expdecay model.
    
    Tests that external Jacobian mode works correctly for the
    SDIRK2 implicit Runge-Kutta method.
    """
    sim = setup(EXPDECAY_MODEL, stepper="sdirk2", jit=True)
    sim.config(jacobian_mode="external")
    
    T = 2.0
    dt = 0.05
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()
    
    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]
    
    x0 = 1.0
    a = 1.0
    x_analytic = x0 * np.exp(-a * t_final)
    
    rel_error = abs(x_final - x_analytic) / abs(x_analytic)
    assert rel_error < 1e-3, f"SDIRK2 external Jacobian error too large: {rel_error}"


def test_external_jacobian_on_vanderpol():
    """
    Test all implicit steppers with external Jacobian on Van der Pol oscillator.
    
    This is a more complex 2D nonlinear system that tests external Jacobian
    handling with a non-diagonal Jacobian matrix.
    """
    steppers = [
        ("bdf2", True, 0.01, 1e-3),
        ("bdf2a", False, 0.1, 1e-3),
        ("tr-bdf2a", False, 0.1, 1e-3),
        ("sdirk2", True, 0.01, 1e-3),
    ]
    
    for stepper_name, jit_enabled, dt, tol in steppers:
        sim = setup(VANDERPOL_MODEL, stepper=stepper_name, jit=jit_enabled)
        sim.config(jacobian_mode="external")
        
        T = 5.0
        sim.run(T=T, dt=dt, record=True)
        res = sim.raw_results()
        
        # Just check that simulation completes without errors
        # and produces reasonable results (no NaN/Inf)
        assert res.n > 0, f"{stepper_name}: No steps recorded"
        assert np.all(np.isfinite(res.Y_view)), f"{stepper_name}: Non-finite values detected"
        assert abs(res.T_view[-1] - T) < 0.01, f"{stepper_name}: Wrong final time"


def test_external_vs_internal_consistency():
    """
    Compare external and internal Jacobian modes for consistency.
    
    Both modes should produce similar results when solving the same problem.
    External mode (analytic) should generally be more accurate.
    """
    sim_internal = setup(EXPDECAY_MODEL, stepper="bdf2", jit=True)
    sim_internal.config(jacobian_mode="internal")
    
    sim_external = setup(EXPDECAY_MODEL, stepper="bdf2", jit=True)
    sim_external.config(jacobian_mode="external")
    
    T = 2.0
    dt = 0.05
    
    sim_internal.run(T=T, dt=dt, record=True)
    res_int = sim_internal.raw_results()
    
    sim_external.run(T=T, dt=dt, record=True)
    res_ext = sim_external.raw_results()
    
    # Both should reach the same final time
    assert abs(res_int.T_view[-1] - res_ext.T_view[-1]) < 1e-10
    
    # Final states should be close (external typically more accurate)
    x_int = res_int.Y_view[0, -1]
    x_ext = res_ext.Y_view[0, -1]
    x_analytic = np.exp(-res_ext.T_view[-1])
    
    # Both should be reasonably close to analytic solution
    assert abs(x_int - x_analytic) / abs(x_analytic) < 1e-2
    assert abs(x_ext - x_analytic) / abs(x_analytic) < 1e-2


@pytest.mark.parametrize("stepper", ["bdf2", "bdf2a", "tr-bdf2a", "sdirk2"])
def test_external_jacobian_parametric(stepper):
    """
    Parametric test for all implicit steppers with external Jacobian.
    
    Ensures all steppers can:
    1. Accept jacobian_mode="external" configuration
    2. Run successfully with DSL-provided Jacobian
    3. Produce finite results
    """
    jit_enabled = stepper in ["bdf2", "sdirk2"]
    
    sim = setup(EXPDECAY_MODEL, stepper=stepper, jit=jit_enabled)
    sim.config(jacobian_mode="external")
    
    T = 1.0
    dt = 0.05
    sim.run(T=T, dt=dt, record=True)
    res = sim.raw_results()
    
    assert res.n > 0
    assert np.all(np.isfinite(res.Y_view))
    
    # Check reasonable accuracy
    t_final = res.T_view[-1]
    x_final = res.Y_view[0, -1]
    x_analytic = np.exp(-t_final)
    rel_error = abs(x_final - x_analytic) / abs(x_analytic)
    
    assert rel_error < 5e-3, f"{stepper} external Jacobian accuracy check failed"
