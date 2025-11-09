#!/usr/bin/env python3
"""
Test script for runtime stepper configuration.

Tests that stepper parameters can be overridden at runtime via run() kwargs.
"""
import numpy as np
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model

def test_rk45_runtime_config():
    """Test RK45 with runtime parameter overrides."""
    
    # Simple decay model
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
stepper = "rk45"
record = true
atol = 1e-6
rtol = 1e-4
"""
    
    # Build with RK45
    full_model = build(f"inline: {model_toml}", jit=False)
    
    # Convert to Model for Sim
    model = Model(
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
    
    sim = Sim(model)
    
    # Run with default tolerances from model (atol=1e-6, rtol=1e-4)
    print("=" * 60)
    print("Test 1: Default tolerances from model_spec")
    print("=" * 60)
    sim.run()
    res1 = sim.raw_results()
    print(f"Steps with defaults: {res1.n}")
    print(f"Final x: {res1.Y[0, res1.n - 1]:.6f}")
    print(f"Expected (exact): {np.exp(-1.0):.6f}")
    print()
    
    # Run with tighter tolerances
    print("=" * 60)
    print("Test 2: Tighter tolerances (atol=1e-10, rtol=1e-8)")
    print("=" * 60)
    sim.run(atol=1e-10, rtol=1e-8)
    res2 = sim.raw_results()
    print(f"Steps with tight tolerances: {res2.n}")
    print(f"Final x: {res2.Y[0, res2.n - 1]:.10f}")
    print(f"Expected (exact): {np.exp(-1.0):.10f}")
    print(f"Error: {abs(res2.Y[0, res2.n - 1] - np.exp(-1.0)):.2e}")
    print()
    
    # Run with looser tolerances
    print("=" * 60)
    print("Test 3: Looser tolerances (atol=1e-3, rtol=1e-2)")
    print("=" * 60)
    sim.run(atol=1e-3, rtol=1e-2)
    res3 = sim.raw_results()
    print(f"Steps with loose tolerances: {res3.n}")
    print(f"Final x: {res3.Y[0, res3.n - 1]:.6f}")
    print(f"Expected (exact): {np.exp(-1.0):.6f}")
    print()
    
    # Verify that tighter tolerances → more steps
    assert res2.n > res1.n, "Tighter tolerances should require more steps"
    # Verify that looser tolerances → fewer steps
    assert res3.n < res1.n, "Looser tolerances should require fewer steps"
    
    # Run with max_tries override
    print("=" * 60)
    print("Test 4: max_tries=50, min_step=1e-15")
    print("=" * 60)
    sim.run(atol=1e-8, rtol=1e-6, max_tries=50, min_step=1e-15)
    res4 = sim.raw_results()
    print(f"Steps: {res4.n}")
    print(f"Final x: {res4.Y[0, res4.n - 1]:.8f}")
    print()
    
    print("✓ All RK45 runtime config tests passed!")


def test_euler_ignores_config():
    """Test that Euler ignores stepper config parameters with a warning."""
    
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
stepper = "euler"
record = true
"""
    
    full_model = build(f"inline: {model_toml}", jit=False)
    model = Model(
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
    sim = Sim(model)
    
    print("=" * 60)
    print("Test 5: Euler with stepper kwargs (should warn)")
    print("=" * 60)
    
    # This should issue a warning but still work
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim.run(atol=1e-10, rtol=1e-8)  # Euler doesn't use these
        
        # Check that a warning was issued
        assert len(w) == 1
        assert "does not accept runtime parameters" in str(w[0].message)
        print(f"✓ Warning issued: {w[0].message}")
    
    res = sim.raw_results()
    print(f"Steps: {res.n}")
    print(f"Final x: {res.Y[0, res.n - 1]:.6f}")
    print()
    
    print("✓ Euler config ignore test passed!")


def test_rk45_with_jit():
    """Test RK45 runtime config with JIT enabled."""
    
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 2.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 0.5
dt = 0.05
stepper = "rk45"
record = true
atol = 1e-8
rtol = 1e-5
"""
    
    # Build with JIT enabled
    full_model = build(f"inline: {model_toml}", jit=True)
    model = Model(
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
    sim = Sim(model)
    
    print("=" * 60)
    print("Test 6: RK45 with JIT and runtime config")
    print("=" * 60)
    
    # Run with default
    sim.run()
    res1 = sim.raw_results()
    print(f"Steps with defaults (JIT): {res1.n}")
    
    # Run with tighter tolerances
    sim.run(atol=1e-12, rtol=1e-10)
    res2 = sim.raw_results()
    print(f"Steps with tight tolerances (JIT): {res2.n}")
    
    assert res2.n > res1.n, "Tighter tolerances should require more steps (JIT)"
    
    print(f"Final x: {res2.Y[0, res2.n - 1]:.12f}")
    print(f"Expected (exact): {np.exp(-1.0):.12f}")
    print()
    
    print("✓ RK45 with JIT test passed!")


if __name__ == "__main__":
    test_rk45_runtime_config()
    test_euler_ignores_config()
    test_rk45_with_jit()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓✓✓")
    print("=" * 60)
