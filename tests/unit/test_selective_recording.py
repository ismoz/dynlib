"""Tests for selective variable recording (record_vars parameter)."""
import numpy as np
import pytest

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build, FullModel
from dynlib.runtime.sim import Sim

# Simple test model with multiple states and aux variables
TEST_MODEL_ODE = """
[model]
type = "ode"
label = "Test ODE"
dtype = "float64"

[sim]
t0 = 0.0
t_end = 10.0
dt = 0.1

[states]
x = 1.0
y = 2.0
z = 3.0

[params]
a = 1.0
b = 2.0

[aux]
sum_xy = "x + y"
product_yz = "y * z"
energy = "0.5 * (x^2 + y^2 + z^2)"

[equations.rhs]
x = "a * y"
y = "-b * x"
z = "x - y"
"""

TEST_MODEL_MAP = """
[model]
type = "map"
label = "Test Map"
dtype = "float64"

[sim]
t0 = 0.0
dt = 1.0
max_steps = 100

[states]
x = 0.1
y = 0.2

[params]
r = 3.5

[aux]
xy = "x * y"
x2 = "x^2"

[equations.next]
x = "r * x * (1 - x)"
y = "0.5 * y + 0.1 * x"
"""


def _build_model(toml_string: str, jit: bool = False) -> FullModel:
    """Helper to build a model from TOML string."""
    import tomllib
    data = tomllib.loads(toml_string)
    spec = build_spec(parse_model_v2(data))
    return build(spec, stepper=spec.sim.stepper, jit=jit)


class TestSelectiveRecording:
    """Test selective variable recording functionality."""
    
    def test_default_behavior_all_states(self):
        """Test default behavior: record all states, no aux (backward compatible)."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_interval=1)
        
        res = sim.raw_results()
        
        # Should record all 3 states
        assert res.Y.shape[0] == 3, "Should record all states by default"
        assert res.state_names == ["x", "y", "z"]
        
        # Should not record aux by default
        assert res.AUX is None
        assert res.aux_names == []
    
    def test_record_specific_states(self):
        """Test recording only specific states."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_interval=1, record_vars=["x", "z"])
        
        res = sim.raw_results()
        
        # Should record only x and z
        assert res.Y.shape[0] == 2
        assert res.state_names == ["x", "z"]
        assert res.AUX is None
        assert res.aux_names == []
        
        # Check accessor methods
        x_data = res.get_var("x")
        z_data = res.get_var("z")
        assert x_data.shape == (res.n,)
        assert z_data.shape == (res.n,)
        
        # Should not be able to access unrecorded state
        with pytest.raises(KeyError, match="Unknown variable: 'y'"):
            res.get_var("y")
    
    def test_record_aux_only(self):
        """Test recording only aux variables."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_interval=1, record_vars=["aux.energy", "aux.sum_xy"])
        
        res = sim.raw_results()
        
        # Should have no states
        assert res.Y.shape[0] == 0
        assert res.state_names == []
        
        # Should have aux
        assert res.AUX is not None
        assert res.AUX.shape[0] == 2
        assert res.aux_names == ["energy", "sum_xy"]
        
        # Check accessor methods
        energy = res.get_var("aux.energy")
        sum_xy = res.get_var("aux.sum_xy")
        assert energy.shape == (res.n,)
        assert sum_xy.shape == (res.n,)
        
        # Should not be able to access states
        with pytest.raises(KeyError, match="Unknown variable: 'x'"):
            res.get_var("x")
    
    def test_record_mixed_states_and_aux(self):
        """Test recording mix of states and aux variables."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_interval=1, 
                record_vars=["x", "aux.energy", "z", "aux.product_yz"])
        
        res = sim.raw_results()
        
        # Should have 2 states
        assert res.Y.shape[0] == 2
        assert res.state_names == ["x", "z"]
        
        # Should have 2 aux
        assert res.AUX is not None
        assert res.AUX.shape[0] == 2
        assert res.aux_names == ["energy", "product_yz"]
        
        # Check all can be accessed
        assert res["x"].shape == (res.n,)
        assert res["z"].shape == (res.n,)
        assert res["aux.energy"].shape == (res.n,)
        assert res["aux.product_yz"].shape == (res.n,)
    
    def test_record_nothing(self):
        """Test recording nothing (only time, step, flags)."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_interval=1, record_vars=[])
        
        res = sim.raw_results()
        
        # Should have no states or aux
        assert res.Y.shape[0] == 0
        assert res.state_names == []
        assert res.AUX is None or res.AUX.shape[0] == 0
        assert res.aux_names == []
        
        # But should still have time data
        assert res.n > 0
        assert res.T.shape[0] > 0
        assert res.STEP.shape[0] > 0
    
    def test_invalid_state_name(self):
        """Test error on unknown state variable."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        
        with pytest.raises(ValueError, match="Unknown variable: 'w'"):
            sim.run(T=1.0, record=True, record_vars=["x", "w"])
    
    def test_invalid_aux_name(self):
        """Test error on unknown aux variable."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        
        with pytest.raises(ValueError, match="Unknown aux variable: invalid"):
            sim.run(T=1.0, record=True, record_vars=["x", "aux.invalid"])
    
    def test_getitem_accessor(self):
        """Test __getitem__ shorthand accessor."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_vars=["x", "aux.energy"])
        
        res = sim.raw_results()
        
        # Test __getitem__
        x_data = res["x"]
        energy_data = res["aux.energy"]
        
        assert x_data.shape == (res.n,)
        assert energy_data.shape == (res.n,)
        
        # Should be same as get_var
        np.testing.assert_array_equal(x_data, res.get_var("x"))
        np.testing.assert_array_equal(energy_data, res.get_var("aux.energy"))
    
    def test_discrete_system_selective_recording(self):
        """Test selective recording works with discrete systems (maps)."""
        model = _build_model(TEST_MODEL_MAP)
        sim = Sim(model)
        sim.run(N=10, record=True, record_vars=["x", "aux.xy"])
        
        res = sim.raw_results()
        
        # Should have 1 state and 1 aux
        assert res.Y.shape[0] == 1
        assert res.state_names == ["x"]
        assert res.AUX is not None
        assert res.AUX.shape[0] == 1
        assert res.aux_names == ["xy"]
        
        # Check data
        assert res["x"].shape == (res.n,)
        assert res["aux.xy"].shape == (res.n,)
    
    def test_resume_with_selective_recording(self):
        """Test that resume enforces consistent record_vars."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        
        # First run: record x and energy
        sim.run(T=1.0, record=True, record_vars=["x", "aux.energy"])
        res1 = sim.raw_results()
        n1 = res1.n
        
        # Resume without specifying record_vars: should use same selection
        sim.run(T=2.0, record=True, resume=True)
        res2 = sim.raw_results()
        
        # Should have more records with same selection
        assert res2.n > n1
        assert res2.state_names == ["x"]
        assert res2.aux_names == ["energy"]
        
    def test_resume_with_different_record_vars_raises_error(self):
        """Test that changing record_vars during resume raises an error."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        
        # First run: record x
        sim.run(T=1.0, record=True, record_vars=["x"])
        
        # Resume with different record_vars: should raise error
        with pytest.raises(ValueError, match="record_vars cannot be changed during resume"):
            sim.run(T=2.0, record=True, record_vars=["y"], resume=True)
    
    def test_reset_allows_new_record_vars(self):
        """Test that reset() allows changing record_vars."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        
        # First run: record x
        sim.run(T=1.0, record=True, record_vars=["x"])
        res1 = sim.raw_results()
        assert res1.state_names == ["x"]
        
        # Reset and run with different record_vars
        sim.reset()
        sim.run(T=1.0, record=True, record_vars=["y", "aux.energy"])
        res2 = sim.raw_results()
        
        # Should have new selection
        assert res2.state_names == ["y"]
        assert res2.aux_names == ["energy"]
    
    def test_buffer_growth_with_selective_recording(self):
        """Test that buffer growth works correctly with selective recording."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        
        # Force small buffer to trigger growth
        sim.run(T=10.0, record=True, cap_rec=2, 
                record_vars=["x", "y", "aux.energy"])
        
        res = sim.raw_results()
        
        # Should have recorded many points despite small initial buffer
        assert res.n > 10
        assert res.Y.shape[0] == 2
        assert res.AUX.shape[0] == 1
        
        # Data should be correct
        assert res.state_names == ["x", "y"]
        assert res.aux_names == ["energy"]
    
    def test_to_pandas_with_selective_recording(self):
        """Test to_pandas() works with selective recording."""
        pytest.importorskip("pandas")
        
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_vars=["x", "aux.energy"])
        
        res = sim.raw_results()
        df = res.to_pandas()
        
        # Should have time columns
        assert "t" in df.columns
        assert "step" in df.columns
        
        # Should have selected state
        assert "x" in df.columns
        
        # Should have selected aux
        assert "aux.energy" in df.columns
        
        # Should not have unselected variables
        assert "y" not in df.columns
        assert "z" not in df.columns
    
    def test_record_vars_none_vs_empty_list(self):
        """Test distinction between None (default) and [] (nothing)."""
        model = _build_model(TEST_MODEL_ODE)
        
        # None: default behavior (all states)
        sim1 = Sim(model)
        sim1.run(T=1.0, record=True, record_vars=None)
        res1 = sim1.raw_results()
        assert res1.Y.shape[0] == 3  # all states
        assert res1.AUX is None  # no aux
        
        # []: record nothing
        sim2 = Sim(model)
        sim2.run(T=1.0, record=True, record_vars=[])
        res2 = sim2.raw_results()
        assert res2.Y.shape[0] == 0  # no states
        assert res2.AUX is None or res2.AUX.shape[0] == 0  # no aux
    
    def test_y_view_shape_with_selective_recording(self):
        """Test Y_view property has correct shape."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        sim.run(T=1.0, record=True, record_vars=["x"])
        
        res = sim.raw_results()
        y_view = res.Y_view
        
        # Y is (n_rec_states, cap_rec), Y_view is (n_rec_states, n)
        assert y_view.shape == (1, res.n)
        
        # Access by name should work
        x_data = res["x"]
        np.testing.assert_array_equal(x_data, y_view[0, :])
    
    def test_aux_auto_detection_without_prefix(self):
        """Test that aux variables can be specified without 'aux.' prefix."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        # Use aux names without prefix - should auto-detect
        sim.run(T=1.0, record=True, record_interval=1, record_vars=["energy", "sum_xy"])
        
        res = sim.raw_results()
        
        # Should have no states
        assert res.Y.shape[0] == 0
        assert res.state_names == []
        
        # Should have aux
        assert res.AUX is not None
        assert res.AUX.shape[0] == 2
        assert res.aux_names == ["energy", "sum_xy"]
        
        # Check accessor methods work
        energy = res.get_var("aux.energy")
        sum_xy = res.get_var("aux.sum_xy")
        assert energy.shape == (res.n,)
        assert sum_xy.shape == (res.n,)
    
    def test_mixed_auto_detect_and_explicit_prefix(self):
        """Test mixing auto-detected and explicitly prefixed aux variables."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        # Mix: state, auto-detected aux, explicit aux prefix
        sim.run(T=1.0, record=True, record_interval=1, 
                record_vars=["x", "energy", "aux.product_yz"])
        
        res = sim.raw_results()
        
        # Should have 1 state
        assert res.Y.shape[0] == 1
        assert res.state_names == ["x"]
        
        # Should have 2 aux
        assert res.AUX is not None
        assert res.AUX.shape[0] == 2
        assert res.aux_names == ["energy", "product_yz"]
    
    def test_state_priority_over_aux_same_name(self):
        """Test that states take priority when name collision exists."""
        # This is a theoretical test - in practice, name collisions are prevented
        # But the logic checks states first, then aux
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        # "x" is a state, should be detected as state not aux
        sim.run(T=1.0, record=True, record_vars=["x"])
        
        res = sim.raw_results()
        
        # Should be recorded as state
        assert res.Y.shape[0] == 1
        assert res.state_names == ["x"]
        assert res.AUX is None or res.AUX.shape[0] == 0
    
    def test_unknown_variable_helpful_error(self):
        """Test error message lists available variables."""
        model = _build_model(TEST_MODEL_ODE)
        sim = Sim(model)
        
        with pytest.raises(ValueError, match="Unknown variable: 'nonexistent'"):
            sim.run(T=1.0, record=True, record_vars=["nonexistent"])
        
        # Error should list available states and aux
        try:
            sim.run(T=1.0, record=True, record_vars=["bad_name"])
        except ValueError as e:
            error_msg = str(e)
            assert "Available states:" in error_msg
            assert "Available aux:" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
