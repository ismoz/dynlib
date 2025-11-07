# tests/integration/test_euler_basic.py
"""
Integration tests: end-to-end Euler simulations.

Tests verify:
- Analytic solution matching for dx/dt = -a*x
- Event mutations work correctly
- Growth paths are triggered with small capacities
- JIT on/off parity (identical results)
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
    full_model = build(spec, stepper_name=spec.sim.stepper, jit=jit)
    
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
        model_dtype=full_model.model_dtype,
    )


def test_euler_decay_analytic():
    """
    Test dx/dt = -a*x with Euler matches analytic solution x(t) = x0*exp(-a*t).
    
    For Euler with step dt, the solution is: x_n = x_0 * (1 - a*dt)^n
    At t=n*dt, the analytic solution is: x(t) = x_0 * exp(-a*t)
    
    With small dt, Euler should be reasonably close.
    """
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay.toml"
    
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
    
    # Euler error is O(dt), so with dt=0.1 and t=2.0 (20 steps), expect some error
    # Euler: x_n = x0 * (1 - a*dt)^n = 1.0 * (0.9)^20 ≈ 0.1216
    # Analytic: x(2) = exp(-2) ≈ 0.1353
    # Relative error should be < 15%
    rel_error = abs(x_final - x_analytic) / x_analytic
    assert rel_error < 0.15, f"Euler too far from analytic: {x_final} vs {x_analytic}"
    
    # Check that we recorded something
    assert results.n > 0
    assert results.T[0] == pytest.approx(0.0)
    assert results.Y[0, 0] == pytest.approx(x0)


def test_euler_with_event():
    """
    Test that events can mutate state correctly.
    
    Model: dx/dt = -a*x with a reset event when x < threshold.
    The event should reset x to 1.0, causing the simulation to continue.
    """
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay_with_event.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    # Run simulation
    sim.run()
    results = sim.raw_results()
    
    # With the reset event, x should never stay below threshold for long
    # After reset, it jumps back to 1.0
    # Check that we have some resets (x values near 1.0 appearing after decay)
    
    x_vals = results.Y[0, :results.n]
    
    # Find where x jumps back up (reset events)
    # Look for increases in x (should not happen in pure decay)
    increases = 0
    for i in range(1, len(x_vals)):
        if x_vals[i] > x_vals[i-1] + 0.1:  # Significant increase
            increases += 1
    
    # Should have at least one reset event during the simulation
    assert increases >= 1, f"Expected at least one reset event, found {increases} increases"
    
    # Check that simulation completed
    assert results.n > 0


def test_euler_growth_triggered():
    """
    Test that buffer growth is triggered when cap_rec is small.
    
    Uses a tiny initial capacity to force growth during execution.
    """
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    # Run with very small capacity (should force growth)
    sim.run(cap_rec=4, record_every_step=1)
    results = sim.raw_results()
    
    # Should have recorded more than 4 points (since t_end=2.0, dt=0.1 → ~20 steps)
    assert results.n > 4, f"Expected growth to allow >4 records, got {results.n}"
    
    # Verify results are valid
    assert results.T[0] == pytest.approx(0.0)
    assert results.Y[0, 0] == pytest.approx(1.0)


def test_jit_on_off_parity():
    """
    Test that JIT on/off produces identical results (per guardrails).
    
    This is a critical test for the "optional JIT" requirement.
    """
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay.toml"
    
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
        err_msg="Time arrays differ between JIT/no-JIT"
    )
    
    np.testing.assert_allclose(
        results_jit.Y[:, :results_jit.n],
        results_no_jit.Y[:, :results_no_jit.n],
        rtol=1e-14,
        err_msg="State arrays differ between JIT/no-JIT"
    )


if __name__ == "__main__":
    # Allow running individual tests
    pytest.main([__file__, "-v"])
