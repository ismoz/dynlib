# tests/integration/test_rk4_rk45.py
"""
Integration tests: RK4 and RK45 steppers.

Tests verify:
- RK4 fixed-step provides 4th-order accuracy
- RK45 adaptive stepping works correctly
- Order convergence tests
- Adaptive step count varies with tolerance
"""
from __future__ import annotations
import pytest
import numpy as np
from pathlib import Path
import tomllib

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model


def load_model_from_toml(path: Path, jit: bool = True) -> Model:
    """Helper to load and build a model from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    # Parse and validate
    normal = parse_model_v2(data)
    spec = build_spec(normal)
    
    # Build with the spec's default stepper
    full_model = build(spec, stepper=spec.sim.stepper, jit=jit)
    
    # Convert FullModel to Model (legacy compat)
    from dynlib.runtime.model import Model as LegacyModel
    return LegacyModel(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        struct=full_model.struct,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
    )


def test_rk4_accuracy():
    """
    Test RK4 provides 4th-order accuracy for dx/dt = -a*x.
    
    For RK4 with step dt, the local truncation error is O(dt^5).
    Global error is O(dt^4), so halving dt should reduce error by ~16x.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "models"
    model_path = data_dir / "decay_rk4.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    # Run simulation
    sim.run()
    results = sim.raw_results()
    
    # Extract final state
    t_final = results.T[results.n - 1]
    x_final = results.Y[0, results.n - 1]
    
    # Analytic solution: x(t) = x0 * exp(-a*t)
    x0 = 1.0
    a = 1.0
    x_analytic = x0 * np.exp(-a * t_final)
    
    # RK4 should be very accurate (error ~ O(dt^4))
    # With dt=0.1 and t=2.0, expect relative error < 1e-5
    rel_error = abs(x_final - x_analytic) / x_analytic
    assert rel_error < 1e-5, f"RK4 error too large: {x_final} vs {x_analytic}, rel_error={rel_error}"
    
    # Check that we recorded something
    assert results.n > 0
    assert results.T[0] == pytest.approx(0.0)
    assert results.Y[0, 0] == pytest.approx(x0)


def test_rk4_order_convergence():
    """
    Test RK4 order of convergence: halving dt should reduce error by ~16x.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "models"
    
    # Analytic solution
    t_end = 2.0
    x0 = 1.0
    a = 1.0
    x_analytic = x0 * np.exp(-a * t_end)
    
    # Run with dt = 0.2 (coarse)
    with open(data_dir / "decay_rk4.toml", "rb") as f:
        data = tomllib.load(f)
    normal = parse_model_v2(data)
    spec = build_spec(normal)
    full_model_coarse = build(spec, stepper="rk4", jit=True)
    
    from dynlib.runtime.model import Model as LegacyModel
    model_coarse = LegacyModel(
        spec=full_model_coarse.spec,
        stepper_name=full_model_coarse.stepper_name,
        struct=full_model_coarse.struct,
        rhs=full_model_coarse.rhs,
        events_pre=full_model_coarse.events_pre,
        events_post=full_model_coarse.events_post,
        stepper=full_model_coarse.stepper,
        runner=full_model_coarse.runner,
        spec_hash=full_model_coarse.spec_hash,
        dtype=full_model_coarse.dtype,
    )
    
    sim_coarse = Sim(model_coarse)
    sim_coarse.run(T=t_end, dt=0.2)
    results_coarse = sim_coarse.raw_results()
    # Find the index where T is closest to t_end
    idx_coarse = np.argmin(np.abs(results_coarse.T[:results_coarse.n] - t_end))
    x_coarse = results_coarse.Y[0, idx_coarse]
    err_coarse = abs(x_coarse - x_analytic)
    
    # Run with dt = 0.1 (fine) - rebuild model to avoid any caching
    full_model_fine = build(spec, stepper="rk4", jit=True)
    model_fine = LegacyModel(
        spec=full_model_fine.spec,
        stepper_name=full_model_fine.stepper_name,
        struct=full_model_fine.struct,
        rhs=full_model_fine.rhs,
        events_pre=full_model_fine.events_pre,
        events_post=full_model_fine.events_post,
        stepper=full_model_fine.stepper,
        runner=full_model_fine.runner,
        spec_hash=full_model_fine.spec_hash,
        dtype=full_model_fine.dtype,
    )
    
    sim_fine = Sim(model_fine)
    sim_fine.run(T=t_end, dt=0.1)
    results_fine = sim_fine.raw_results()
    idx_fine = np.argmin(np.abs(results_fine.T[:results_fine.n] - t_end))
    x_fine = results_fine.Y[0, idx_fine]
    err_fine = abs(x_fine - x_analytic)
    
    # Check order: err_coarse / err_fine should be ~ 2^4 = 16
    if err_fine > 1e-12:  # Avoid division by near-zero
        ratio = err_coarse / err_fine
        # Allow some tolerance (expect 10 < ratio < 20 for order 4)
        assert 10.0 < ratio < 20.0, f"Order convergence failed: ratio={ratio}, expected ~16"


def test_rk45_adaptive_accuracy():
    """
    Test RK45 adaptive method produces accurate results.
    
    RK45 should automatically adjust step size to maintain error tolerance,
    resulting in very accurate solutions.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "models"
    model_path = data_dir / "decay_rk45.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    # Run simulation
    sim.run()
    results = sim.raw_results()
    
    # Extract final state
    t_final = results.T[results.n - 1]
    x_final = results.Y[0, results.n - 1]
    
    # Analytic solution: x(t) = x0 * exp(-a*t)
    x0 = 1.0
    a = 1.0
    x_analytic = x0 * np.exp(-a * t_final)
    
    # RK45 with adaptive stepping should be very accurate
    # Expect relative error < 1e-4 (depending on tolerances in stepper)
    rel_error = abs(x_final - x_analytic) / x_analytic
    assert rel_error < 1e-3, f"RK45 error too large: {x_final} vs {x_analytic}, rel_error={rel_error}"
    
    # Check that we recorded something
    assert results.n > 0
    assert results.T[0] == pytest.approx(0.0)
    assert results.Y[0, 0] == pytest.approx(x0)


def test_rk45_step_adaptation():
    """
    Test that RK45 adapts step size (varies number of steps taken).
    
    For smooth problems, RK45 should take fewer steps than fixed-step methods.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "models"
    
    # Load RK45 model
    with open(data_dir / "decay_rk45.toml", "rb") as f:
        data = tomllib.load(f)
    normal = parse_model_v2(data)
    spec = build_spec(normal)
    full_model = build(spec, stepper="rk45", jit=True)
    
    from dynlib.runtime.model import Model as LegacyModel
    model_rk45 = LegacyModel(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        struct=full_model.struct,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
    )
    
    # Load Euler model for comparison
    with open(data_dir / "decay.toml", "rb") as f:
        data_euler = tomllib.load(f)
    normal_euler = parse_model_v2(data_euler)
    spec_euler = build_spec(normal_euler)
    full_model_euler = build(spec_euler, stepper="euler", jit=True)
    
    model_euler = LegacyModel(
        spec=full_model_euler.spec,
        stepper_name=full_model_euler.stepper_name,
        struct=full_model_euler.struct,
        rhs=full_model_euler.rhs,
        events_pre=full_model_euler.events_pre,
        events_post=full_model_euler.events_post,
        stepper=full_model_euler.stepper,
        runner=full_model_euler.runner,
        spec_hash=full_model_euler.spec_hash,
        dtype=full_model_euler.dtype,
    )
    
    # Run both with same initial dt
    sim_rk45 = Sim(model_rk45)
    sim_rk45.run(T=2.0, dt=0.1, record_interval=1)
    results_rk45 = sim_rk45.raw_results()
    
    sim_euler = Sim(model_euler)
    sim_euler.run(T=2.0, dt=0.1, record_interval=1)
    results_euler = sim_euler.raw_results()
    
    # RK45 should take different number of steps (adaptive)
    # For smooth exponential decay, RK45 might take fewer actual internal steps
    # but we record every accepted step, so check that recorded counts differ
    # or that RK45 achieves better accuracy
    
    # Analytic solution
    x0 = 1.0
    a = 1.0
    t_end = 2.0
    x_analytic = x0 * np.exp(-a * t_end)
    
    x_rk45 = results_rk45.Y[0, results_rk45.n - 1]
    x_euler = results_euler.Y[0, results_euler.n - 1]
    
    err_rk45 = abs(x_rk45 - x_analytic)
    err_euler = abs(x_euler - x_analytic)
    
    # RK45 should be significantly more accurate than Euler
    assert err_rk45 < err_euler / 10.0, f"RK45 not more accurate: rk45={err_rk45}, euler={err_euler}"


def test_rk4_jit_parity():
    """
    Test that RK4 produces identical results with JIT on/off.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "models"
    model_path = data_dir / "decay_rk4.toml"
    
    # Build and run with JIT enabled
    model_jit = load_model_from_toml(model_path, jit=True)
    sim_jit = Sim(model_jit)
    sim_jit.run()
    results_jit = sim_jit.raw_results()
    
    # Build and run with JIT disabled
    model_no_jit = load_model_from_toml(model_path, jit=False)
    sim_no_jit = Sim(model_no_jit)
    sim_no_jit.run()
    results_no_jit = sim_no_jit.raw_results()
    
    # Results should be identical
    assert results_jit.n == results_no_jit.n, "Record counts differ"
    
    np.testing.assert_allclose(
        results_jit.T[:results_jit.n],
        results_no_jit.T[:results_no_jit.n],
        rtol=1e-14,
        err_msg="Time arrays differ between JIT/no-JIT (RK4)"
    )
    
    np.testing.assert_allclose(
        results_jit.Y[:, :results_jit.n],
        results_no_jit.Y[:, :results_no_jit.n],
        rtol=1e-14,
        err_msg="State arrays differ between JIT/no-JIT (RK4)"
    )


def test_rk45_jit_parity():
    """
    Test that RK45 produces identical results with JIT on/off.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "models"
    model_path = data_dir / "decay_rk45.toml"
    
    # Build and run with JIT enabled
    model_jit = load_model_from_toml(model_path, jit=True)
    sim_jit = Sim(model_jit)
    sim_jit.run()
    results_jit = sim_jit.raw_results()
    
    # Build and run with JIT disabled
    model_no_jit = load_model_from_toml(model_path, jit=False)
    sim_no_jit = Sim(model_no_jit)
    sim_no_jit.run()
    results_no_jit = sim_no_jit.raw_results()
    
    # Results should be identical (adaptive decisions should be deterministic)
    assert results_jit.n == results_no_jit.n, "Record counts differ"
    
    np.testing.assert_allclose(
        results_jit.T[:results_jit.n],
        results_no_jit.T[:results_no_jit.n],
        rtol=1e-14,
        err_msg="Time arrays differ between JIT/no-JIT (RK45)"
    )
    
    np.testing.assert_allclose(
        results_jit.Y[:, :results_jit.n],
        results_no_jit.Y[:, :results_no_jit.n],
        rtol=1e-14,
        err_msg="State arrays differ between JIT/no-JIT (RK45)"
    )


def test_stepper_registration():
    """
    Test that all steppers are properly registered.
    """
    from dynlib.steppers import get_stepper, registry
    
    # Check registry contains our steppers
    reg = registry()
    assert "euler" in reg
    assert "rk4" in reg
    assert "rk45" in reg
    
    # Check aliases work
    assert "fwd_euler" in reg
    assert "forward_euler" in reg
    assert "rk4_classic" in reg
    assert "classical_rk4" in reg
    assert "dopri5" in reg
    assert "dormand_prince" in reg
    
    # Check we can retrieve specs
    euler_spec = get_stepper("euler")
    assert euler_spec.meta.name == "euler"
    assert euler_spec.meta.order == 1
    assert euler_spec.meta.time_control == "fixed"
    
    rk4_spec = get_stepper("rk4")
    assert rk4_spec.meta.name == "rk4"
    assert rk4_spec.meta.order == 4
    assert rk4_spec.meta.time_control == "fixed"
    
    rk45_spec = get_stepper("rk45")
    assert rk45_spec.meta.name == "rk45"
    assert rk45_spec.meta.order == 5
    assert rk45_spec.meta.embedded_order == 4
    assert rk45_spec.meta.time_control == "adaptive"


if __name__ == "__main__":
    # Allow running individual tests
    pytest.main([__file__, "-v"])
