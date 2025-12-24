"""
Tests for Sim.assign() method and updated _select_seed() semantics.
"""
import pytest
import numpy as np
import tomllib
from pathlib import Path
from dynlib.runtime.sim import Sim
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build, FullModel


def _create_test_model() -> FullModel:
    """Create a simple Izhikevich-like model for testing."""
    toml_str = """
[model]
type = "ode"
stepper = "rk4"

[sim]
t0 = 0.0
t_end = 10.0
dt = 0.1
record = true
stepper = "rk4"

[states]
v = -65.0
u = -13.0

[params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0
I = 10.0

[equations.rhs]
v = "0.04 * v^2 + 5.0 * v + 140.0 - u + I"
u = "a * (b * v - u)"
"""
    data = tomllib.loads(toml_str)
    spec = build_spec(parse_model_v2(data))
    return build(spec, stepper=spec.sim.stepper, jit=True)


@pytest.fixture
def simple_sim():
    """Create a simple simulation for testing."""
    model = _create_test_model()
    return Sim(model)


class TestAssignBasics:
    """Test basic assign() functionality."""

    def test_assign_state_by_mapping(self, simple_sim):
        """Test assigning state values using a dict."""
        simple_sim.assign({"v": -70.0})
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-70.0)

    def test_assign_state_by_kwargs(self, simple_sim):
        """Test assigning state values using kwargs."""
        simple_sim.assign(v=-70.0, u=-15.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-70.0)
        assert simple_sim._session_state.y_curr[1] == pytest.approx(-15.0)

    def test_assign_param_by_mapping(self, simple_sim):
        """Test assigning parameter values using a dict."""
        simple_sim.assign({"I": 15.0})
        # I is the 5th parameter (index 4)
        param_names = list(simple_sim.model.spec.params)
        I_idx = param_names.index("I")
        assert simple_sim._session_state.params_curr[I_idx] == pytest.approx(15.0)

    def test_assign_param_by_kwargs(self, simple_sim):
        """Test assigning parameter values using kwargs."""
        simple_sim.assign(a=0.03, I=12.0)
        param_names = list(simple_sim.model.spec.params)
        a_idx = param_names.index("a")
        I_idx = param_names.index("I")
        assert simple_sim._session_state.params_curr[a_idx] == pytest.approx(0.03)
        assert simple_sim._session_state.params_curr[I_idx] == pytest.approx(12.0)

    def test_assign_mixed_states_and_params(self, simple_sim):
        """Test assigning both states and params in one call."""
        simple_sim.assign({"v": -70.0, "I": 15.0})
        state_names = list(simple_sim.model.spec.states)
        param_names = list(simple_sim.model.spec.params)
        v_idx = state_names.index("v")
        I_idx = param_names.index("I")
        assert simple_sim._session_state.y_curr[v_idx] == pytest.approx(-70.0)
        assert simple_sim._session_state.params_curr[I_idx] == pytest.approx(15.0)

    def test_assign_kwargs_override_mapping(self, simple_sim):
        """Test that kwargs override mapping for the same key."""
        simple_sim.assign({"v": -70.0}, v=-75.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-75.0)

    def test_assign_empty_does_nothing(self, simple_sim):
        """Test that assign() with no arguments does nothing."""
        orig_v = simple_sim._session_state.y_curr[0]
        simple_sim.assign()
        assert simple_sim._session_state.y_curr[0] == pytest.approx(orig_v)

    def test_assign_mapping_only(self, simple_sim):
        """Test assign with mapping=None and only kwargs."""
        simple_sim.assign(None, v=-80.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-80.0)


class TestAssignErrors:
    """Test error handling in assign()."""

    def test_assign_unknown_name_raises(self, simple_sim):
        """Test that unknown names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown state/param name"):
            simple_sim.assign({"unknown_var": 10.0})

    def test_assign_unknown_name_with_suggestion(self, simple_sim):
        """Test that unknown names get suggestions."""
        # 'Icur' is too far from 'I' (3 edits), but 'Ix' should work
        with pytest.raises(ValueError, match="did you mean"):
            simple_sim.assign({"Ix": 10.0})  # Should suggest 'I'

    def test_assign_invalid_mapping_type(self, simple_sim):
        """Test that non-Mapping type raises TypeError."""
        with pytest.raises(TypeError, match="must be a Mapping"):
            simple_sim.assign([("v", -70.0)])  # list of tuples, not a Mapping


class TestAssignClearHistory:
    """Test clear_history flag behavior."""

    def test_clear_history_false_keeps_results(self, simple_sim):
        """Test that clear_history=False preserves results."""
        simple_sim.run(T=1.0)
        assert simple_sim._raw_results is not None
        segments_before = len(simple_sim._segments)

        simple_sim.assign(v=-70.0, clear_history=False)

        # Results should still be there
        assert simple_sim._raw_results is not None
        assert len(simple_sim._segments) == segments_before

    def test_clear_history_true_clears_results(self, simple_sim):
        """Test that clear_history=True clears results but not session state."""
        simple_sim.run(T=1.0)
        assert simple_sim._raw_results is not None

        # Record current session state values
        t_before = simple_sim._session_state.t_curr
        step_before = simple_sim._session_state.step_count
        dt_before = simple_sim._session_state.dt_curr

        simple_sim.assign(v=-70.0, clear_history=True)

        # Results should be cleared
        assert simple_sim._raw_results is None
        assert simple_sim._results_view is None
        assert len(simple_sim._segments) == 0
        assert simple_sim._pending_run_tag is None
        assert simple_sim._pending_run_cfg_hash is None

        # Session state should be unchanged (except for the assigned value)
        assert simple_sim._session_state.t_curr == pytest.approx(t_before)
        assert simple_sim._session_state.step_count == step_before
        assert simple_sim._session_state.dt_curr == pytest.approx(dt_before)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-70.0)


class TestAssignWithRun:
    """Test interaction between assign() and run()."""

    def test_assign_affects_next_run_resume_false(self, simple_sim):
        """Test that assign() affects the next run(resume=False)."""
        # Initial run
        simple_sim.run(T=1.0)
        
        # Assign new values
        simple_sim.assign(v=-80.0, I=15.0)
        
        # Run again with resume=False (should use assigned values)
        simple_sim.run(T=2.0, resume=False)
        
        # Check that the run started from the assigned values
        results = simple_sim.results()
        # First recorded point should have v close to -80.0 (will evolve slightly)
        # This is a bit tricky to test precisely, but we can check that it's different
        # from the original IC (-65.0)
        v_trace = results["v"]
        assert v_trace[0] != pytest.approx(-65.0)

    def test_assign_affects_next_run_resume_true(self, simple_sim):
        """Test that assign() affects the next run(resume=True)."""
        # Initial run
        simple_sim.run(T=1.0)
        
        # Record where we ended
        final_t = simple_sim._session_state.t_curr
        
        # Assign new state value
        simple_sim.assign(v=-100.0)
        
        # Resume should continue from current time but use updated state
        simple_sim.run(T=2.0, resume=True)
        
        # Check that time continued (didn't restart)
        assert simple_sim._session_state.t_curr > final_t

    def test_explicit_ic_overrides_assign(self, simple_sim):
        """Test that explicit ic= argument overrides assign()."""
        # Assign a value
        simple_sim.assign(v=-80.0)
        
        # Run with explicit ic (should override assign)
        ic = np.array([-90.0, -20.0], dtype=simple_sim._dtype)
        simple_sim.run(T=1.0, ic=ic)
        
        # First point should be from ic, not from assign
        results = simple_sim.results()
        v_trace = results["v"]
        assert v_trace[0] == pytest.approx(-90.0)

    def test_explicit_params_overrides_assign(self, simple_sim):
        """Test that explicit params= argument overrides assign()."""
        # Assign a parameter value
        simple_sim.assign(I=20.0)
        
        # Run with explicit params (should override assign)
        # Create params array with I=25.0
        param_names = list(simple_sim.model.spec.params)
        params = np.array(simple_sim.model.spec.param_vals, dtype=simple_sim._dtype)
        I_idx = param_names.index("I")
        params[I_idx] = 25.0
        
        simple_sim.run(T=1.0, params=params)
        
        # The run should have used params[I]=25.0, not the assigned 20.0
        # This is harder to verify directly, but we can check session state
        assert simple_sim._session_state.params_curr[I_idx] == pytest.approx(25.0)


class TestAssignDoesNotModifySnapshots:
    """Test that assign() does not modify snapshots."""

    def test_assign_does_not_modify_initial_snapshot(self, simple_sim):
        """Test that assign() doesn't change the initial snapshot."""
        # Run once to ensure initial snapshot is created
        simple_sim.run(T=0.1)
        
        # Get initial snapshot state
        initial_snap = simple_sim._snapshots["initial"]
        orig_v = initial_snap.state.y_curr[0]
        
        # Assign new value
        simple_sim.assign(v=-80.0)
        
        # Initial snapshot should be unchanged
        assert initial_snap.state.y_curr[0] == pytest.approx(orig_v)

    def test_assign_does_not_modify_custom_snapshot(self, simple_sim):
        """Test that assign() doesn't change custom snapshots."""
        simple_sim.run(T=0.1)
        simple_sim.create_snapshot("test", "Test snapshot")
        
        # Get snapshot state
        test_snap = simple_sim._snapshots["test"]
        orig_v = test_snap.state.y_curr[0]
        
        # Assign new value
        simple_sim.assign(v=-80.0)
        
        # Snapshot should be unchanged
        assert test_snap.state.y_curr[0] == pytest.approx(orig_v)


class TestAssignIntegration:
    """Integration tests combining assign() with other Sim operations."""

    def test_assign_reset_then_run(self, simple_sim):
        """Test assign(), then reset(), then run()."""
        # First run to create initial snapshot with default ICs
        simple_sim.run(T=0.1)
        
        # Now assign a new value
        simple_sim.assign(v=-90.0)
        
        # Reset should go back to initial snapshot (which has the original IC)
        simple_sim.reset("initial")
        
        # Should be back to original IC, not -90.0
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-65.0)

    def test_assign_multiple_times(self, simple_sim):
        """Test calling assign() multiple times."""
        simple_sim.assign(v=-70.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-70.0)
        
        simple_sim.assign(v=-80.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-80.0)
        
        simple_sim.assign(u=-25.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-80.0)
        assert simple_sim._session_state.y_curr[1] == pytest.approx(-25.0)

    def test_assign_then_apply_preset(self, simple_sim):
        """Test that apply_preset() can override assign()."""
        # First assign
        simple_sim.assign(I=20.0)
        
        # Then apply a preset (if there is one)
        # Since we don't have a preset in SIMPLE_MODEL, we'll skip this
        # or manually add one via the internal API
        from dynlib.runtime.sim import _PresetData
        simple_sim._presets["test"] = _PresetData(
            name="test",
            params={"I": 25.0},
            states=None,
            source="inline",
        )
        simple_sim.apply_preset("test")
        
        param_names = list(simple_sim.model.spec.params)
        I_idx = param_names.index("I")
        assert simple_sim._session_state.params_curr[I_idx] == pytest.approx(25.0)
    """Integration tests combining assign() with other Sim operations."""

    def test_assign_reset_then_run(self, simple_sim):
        """Test assign(), then reset(), then run()."""
        # First run to create initial snapshot with default ICs
        simple_sim.run(T=0.1)
        
        # Now assign a new value
        simple_sim.assign(v=-90.0)
        
        # Reset should go back to initial snapshot (which has the original IC)
        simple_sim.reset("initial")
        
        # Should be back to original IC, not -90.0
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-65.0)

    def test_assign_multiple_times(self, simple_sim):
        """Test calling assign() multiple times."""
        simple_sim.assign(v=-70.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-70.0)
        
        simple_sim.assign(v=-80.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-80.0)
        
        simple_sim.assign(u=-25.0)
        assert simple_sim._session_state.y_curr[0] == pytest.approx(-80.0)
        assert simple_sim._session_state.y_curr[1] == pytest.approx(-25.0)

    def test_assign_then_apply_preset(self, simple_sim):
        """Test that apply_preset() can override assign()."""
        # First assign
        simple_sim.assign(I=20.0)
        
        # Then apply a preset (if there is one)
        # Since we don't have a preset in SIMPLE_MODEL, we'll skip this
        # or manually add one via the internal API
        from dynlib.runtime.sim import _PresetData
        simple_sim._presets["test"] = _PresetData(
            name="test",
            params={"I": 25.0},
            states=None,
            source="inline",
        )
        simple_sim.apply_preset("test")
        
        param_names = list(simple_sim.model.spec.params)
        I_idx = param_names.index("I")
        assert simple_sim._session_state.params_curr[I_idx] == pytest.approx(25.0)
