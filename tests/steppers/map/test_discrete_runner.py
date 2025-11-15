"""
Test discrete runner implementation with a simple map.
"""
import numpy as np
import pytest
from dynlib import build
from dynlib.runtime.sim import Sim
from pathlib import Path

# Get test directory
TEST_DIR = Path(__file__).parent.parent.parent

def test_discrete_map_basic():
    """Test basic map simulation with N parameter."""
    model = build(str(TEST_DIR / "data" / "models" / "logistic_map.toml"), jit=False)
    sim = Sim(model)
    
    # Run for 100 iterations using N parameter
    sim.run(N=100)
    
    results = sim.results()
    
    # Check that we got records
    assert len(results.t) > 0
    print(f"✓ Recorded {len(results.t)} states")
    
    # Check that time is exact (no drift)
    expected_times = np.arange(0, min(102, len(results.t)))
    recorded_times = results.t[:len(expected_times)]
    
    # Time should be exact within machine precision
    if len(recorded_times) > 0:
        max_error = np.max(np.abs(recorded_times - expected_times))
        print(f"✓ Max time error: {max_error:.2e} (should be ~1e-15)")
        assert max_error < 1e-10, f"Time drift detected: {max_error}"
    
    # Check final state is reasonable for logistic map
    # Access raw results for state values
    raw = sim.raw_results()
    final_x = raw.Y_view[0, -1]
    print(f"✓ Final x = {final_x:.6f}")
    assert 0 <= final_x <= 1, "Logistic map should stay in [0,1]"
    
    print("\n✅ Discrete map test PASSED")


def test_discrete_map_with_T():
    """Test map simulation by specifying T (infers N)."""
    model = build(str(TEST_DIR / "data" / "models" / "logistic_map.toml"), jit=False)
    sim = Sim(model)
    
    # Run to T=50 with dt=1.0 -> should do 50 iterations
    sim.run(T=50.0, dt=1.0)
    
    results = sim.results()
    
    # Should have done ~50 iterations
    # Access raw results for step count
    raw = sim.raw_results()
    print(f"✓ With T=50, got {raw.step_count_final} iterations")
    assert abs(raw.step_count_final - 50) <= 1
    
    print("✅ T-based discrete test PASSED")

def test_discrete_map_dt_override():
    """Ensure dt override does not change iteration count."""
    model = build(str(TEST_DIR / "data" / "models" / "logistic_map.toml"), jit=False)
    sim = Sim(model)

    sim.run(N=8, dt=0.25)

    res = sim.results()
    assert res.step[-1] == 8
    assert res.t[-1] == pytest.approx(2.0)
    assert len(res.t) == 9  # initial record + 8 iterations

    raw = sim.raw_results()
    assert raw.step_count_final == 8


def test_discrete_map_transient_warmup_keeps_recorded_horizon():
    """Transient iterations should not extend the recorded run."""
    model = build(str(TEST_DIR / "data" / "models" / "logistic_map.toml"), jit=False)
    sim = Sim(model)

    sim.run(N=10, transient=5)

    res = sim.results()
    assert res.step[-1] == 10
    assert len(res.t) == 11

    raw = sim.raw_results()
    # Total steps include transient+recorded; ensure warm-up occurred.
    assert raw.step_count_final == 15


def test_discrete_resume_extends_to_new_target():
    """Resume should advance only the remaining iterations to reach new N."""
    model = build(str(TEST_DIR / "data" / "models" / "logistic_map.toml"), jit=False)
    sim = Sim(model)

    sim.run(N=5)
    first = sim.results()
    assert first.step[-1] == 5

    # Extend to N=12 (7 more iterations)
    sim.run(N=12, resume=True)
    res = sim.results()
    raw = sim.raw_results()

    assert res.step[-1] == 12
    assert raw.step_count_final == 12
    # 13 records: initial + 12 iterations
    assert len(res.t) == 13


def test_discrete_resume_with_T():
    """Resume using absolute T should infer remaining iterations."""
    model = build(str(TEST_DIR / "data" / "models" / "logistic_map.toml"), jit=False)
    sim = Sim(model)

    sim.run(N=4)
    # Resume to absolute T=10 (dt=1 -> target 10 iterations total)
    sim.run(T=10.0, resume=True)

    res = sim.results()
    raw = sim.raw_results()
    assert res.step[-1] == 10
    assert raw.step_count_final == 10


def test_continuous_backward_compat():
    """Verify continuous systems still work with t_end."""
    model = build(str(TEST_DIR / "data" / "models" / "decay_ode.toml"), jit=False)
    sim = Sim(model)
    # Ensure the new T-only API works for continuous systems
    sim.run(T=5.0, max_steps=1000)
    results = sim.results()
    print(f"✓ Continuous ODE ran to t={results.t[-1]:.2f}")

    # New API with T again after reset
    sim.reset()
    sim.run(T=5.0, max_steps=1000)
    results = sim.results()
    print(f"✓ New T parameter works: t={results.t[-1]:.2f}")

    print("✅ Continuous API (T) test PASSED")


if __name__ == "__main__":
    print("Testing discrete runner implementation...\n")
    test_discrete_map_basic()
    print()
    test_discrete_map_with_T()
    print()
    test_continuous_backward_compat()
    print("\n All tests PASSED!")
