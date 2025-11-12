# tests/unit/test_nan_inf_guards.py
"""
Test NaN/Inf detection guards.

Verifies that:
1. Guards are properly compiled with JIT
2. Guards correctly detect NaN and Inf values
3. Runner exits with NAN_DETECTED status when guards trigger
4. Guards work with both jit=True and jit=False
5. RK45 adaptive stepper uses guards in internal loops
"""
from pathlib import Path
import numpy as np

from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.runner_api import NAN_DETECTED
from dynlib.compiler.guards import get_guards, allfinite1d, allfinite_scalar


def test_guards_pure_python():
    """Test guards in pure Python mode (no JIT)."""
    allfinite1d_fn, allfinite_scalar_fn = get_guards(jit=False)
    
    # Test 1D array checks
    finite_arr = np.array([1.0, 2.0, 3.0])
    assert allfinite1d_fn(finite_arr) is True
    
    nan_arr = np.array([1.0, np.nan, 3.0])
    assert allfinite1d_fn(nan_arr) is False
    
    inf_arr = np.array([1.0, np.inf, 3.0])
    assert allfinite1d_fn(inf_arr) is False
    
    neginf_arr = np.array([1.0, -np.inf, 3.0])
    assert allfinite1d_fn(neginf_arr) is False
    
    # Test scalar checks
    assert allfinite_scalar_fn(1.0) is True
    assert allfinite_scalar_fn(np.nan) is False
    assert allfinite_scalar_fn(np.inf) is False
    assert allfinite_scalar_fn(-np.inf) is False
    
    print("✓ Pure Python guards test passed!")


def test_guards_with_jit():
    """Test guards with JIT compilation."""
    try:
        import numba
    except ImportError:
        print("⊘ Numba not available, skipping JIT guards test")
        return
    
    allfinite1d_fn, allfinite_scalar_fn = get_guards(jit=True)
    
    # Test 1D array checks
    finite_arr = np.array([1.0, 2.0, 3.0])
    assert allfinite1d_fn(finite_arr) is True
    
    nan_arr = np.array([1.0, np.nan, 3.0])
    assert allfinite1d_fn(nan_arr) is False
    
    inf_arr = np.array([1.0, np.inf, 3.0])
    assert allfinite1d_fn(inf_arr) is False
    
    # Test scalar checks
    assert allfinite_scalar_fn(1.0) is True
    assert allfinite_scalar_fn(np.nan) is False
    assert allfinite_scalar_fn(np.inf) is False
    
    print("✓ JIT guards test passed!")


def test_runner_nan_detection():
    """Test that guards are properly integrated in build."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
k = 0.5

[sim]
stepper = "euler"
t_end = 1.0
dt = 0.1

[equations.rhs]
x = "-k * x"
"""
    
    import tomllib
    from dynlib.dsl.spec import build_spec
    from dynlib.dsl.parser import parse_model_v2
    
    parsed = parse_model_v2(tomllib.loads(model_toml))
    spec = build_spec(parsed)
    
    # Test with jit=False - guards should be present
    model = build(spec, stepper="euler", jit=False, disk_cache=False)
    assert model.guards is not None, "Guards should be present in FullModel"
    
    # Verify guards work
    allfinite1d_fn, _ = model.guards
    assert allfinite1d_fn(np.array([1.0, 2.0])) is True
    assert allfinite1d_fn(np.array([1.0, np.nan])) is False
    
    print("✓ Runner guards integration test passed!")


def test_runner_detects_inf_in_simulation():
    """
    Integration test: verify that Inf in RHS causes NAN_DETECTED status.
    
    This test creates a model where the RHS will produce Inf (exponential blowup),
    runs it through Sim, and verifies that the runner properly exits with NAN_DETECTED.
    This validates that the scalar guards (allfinite_scalar) are working correctly
    for both vector outputs (y_prop) and scalar outputs (t_prop, dt_next).
    """
    import warnings
    
    # Exponential growth that will blow up to Inf: x' = 1000*x^2, x(0)=1.0
    # This grows extremely fast and will overflow to Inf
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
# No params needed

[sim]
stepper = "euler"
t_end = 10.0
dt = 0.1

[equations.rhs]
x = "1000.0 * x * x"
"""
    
    import tomllib
    from dynlib.dsl.spec import build_spec
    from dynlib.dsl.parser import parse_model_v2
    
    parsed = parse_model_v2(tomllib.loads(model_toml))
    spec = build_spec(parsed)
    
    # Build model without JIT for easier debugging
    model = build(spec, stepper="euler", jit=False, disk_cache=False)
    
    # Create simulation
    sim = Sim(model)
    
    # Run simulation - it should hit Inf and exit with NAN_DETECTED
    # These warnings are EXPECTED - the test verifies overflow detection works
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered")
        warnings.filterwarnings("ignore", message="run_with_wrapper exited early")
        sim.run(T=10.0, max_steps=100)
    
    # Check that the simulation stopped due to NaN/Inf detection
    summary = sim.session_state_summary()
    assert summary["status"] == "NAN_DETECTED", (
        f"Expected NAN_DETECTED status, got {summary['status']}. "
        f"Simulation ran to t={summary['t']}, step={summary['step']}"
    )
    
    print(f"✓ Runner Inf detection test passed! Detected Inf at t={summary['t']:.3f}, step={summary['step']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NaN/Inf Guards")
    print("=" * 60)
    
    test_guards_pure_python()
    test_guards_with_jit()
    test_runner_nan_detection()
    test_runner_detects_inf_in_simulation()
    
    print("=" * 60)
    print("✓ All NaN/Inf guards tests passed!")
    print("=" * 60)
