"""
Test that map steppers work correctly with integer dtypes.

Regression test for issue where dt_next, t_prop, and err_est arrays
were incorrectly using the model dtype instead of float64, causing
truncation when dtype was int64.
"""
import numpy as np
import pytest
from dynlib import build, setup


def test_map_int64_dtype_basic():
    """Test that int64 dtype works without time truncation."""
    model = """
    inline:
    [model]
    type = "map"
    dtype = "int64"
    
    [states]
    n = 10
    
    [params]
    
    [equations.rhs]
    n = "n + 1"
    """
    
    sim = setup(model, stepper="map", jit=False)
    sim.run(N=10)
    
    res = sim.results()
    
    # Check state dtype
    assert res['n'].dtype == np.int64, f"Expected int64 but got {res['n'].dtype}"
    
    # Check time is float64 (not int64!)
    assert res.t.dtype == np.float64, f"Time should be float64 but got {res.t.dtype}"
    
    # Check time values are correct (not truncated to 0)
    expected_times = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    np.testing.assert_allclose(res.t, expected_times, rtol=1e-10)
    
    # Check state values
    expected_states = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int64)
    np.testing.assert_array_equal(res['n'], expected_states)
    
    print(f"✓ int64 dtype: time={res.t.dtype}, state={res['n'].dtype}")
    print(f"✓ Times: {res.t}")
    print(f"✓ States: {res['n']}")


def test_map_int32_dtype():
    """Test that int32 dtype also works."""
    model = """
    inline:
    [model]
    type = "map"
    dtype = "int32"
    
    [states]
    x = 100
    
    [params]
    
    [equations.rhs]
    x = "x - 1"
    """
    
    sim = setup(model, stepper="map", jit=False)
    sim.run(N=5)
    
    res = sim.results()
    
    # Check dtypes
    assert res['x'].dtype == np.int32
    assert res.t.dtype == np.float64
    
    # Check time monotonicity (no truncation)
    assert np.all(np.diff(res.t) > 0), "Time should be strictly increasing"
    
    print(f"✓ int32 dtype: time={res.t.dtype}, state={res['x'].dtype}")


def test_map_collatz_sequence():
    """Test Collatz conjecture sequence with int64."""
    model = """
    inline:
    [model]
    type = "map"
    dtype = "int64"
    name = "Collatz"
    
    [states]
    n = 27
    
    [params]
    
    [equations.rhs]
    n = "n//2 if n % 2 == 0 else 3*n + 1"
    """
    
    sim = setup(model, stepper="map", jit=False)
    sim.run(N=50)
    
    res = sim.results()
    
    # All values should be positive integers
    assert np.all(res['n'] > 0)
    assert res['n'].dtype == np.int64
    
    # Time should be properly spaced
    dt_values = np.diff(res.t)
    expected_dt = 0.01  # default dt
    np.testing.assert_allclose(dt_values, expected_dt, rtol=1e-10)
    
    print(f"✓ Collatz sequence (n=27): {res['n'][:10]}...")
    print(f"✓ Time progression: {res.t[:5]}...")


if __name__ == "__main__":
    test_map_int64_dtype_basic()
    test_map_int32_dtype()
    test_map_collatz_sequence()
    print("\n✅ All int dtype tests PASSED")
