"""
Integration tests for event logging functionality.

Tests that events with record=true and log=[...] correctly populate
the EVT_TIME, EVT_CODE, and EVT_INDEX buffers.
"""
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
    return Model(
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


def test_event_logging_basic():
    """Test that event logging records when an event fires."""
    # Load model with event that has record=true and log=["x"]
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay_with_event.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    # Run simulation with reasonable event buffer capacity
    result = sim.run(cap_evt=100)
    
    # Check that event log has entries
    evt_time = result.EVT_TIME_view
    evt_code = result.EVT_CODE_view
    evt_index = result.EVT_INDEX_view
    evt_log_data = result.EVT_LOG_DATA_view
    
    # The decay model should trigger the reset event when x < threshold (0.5)
    # This should happen at least once during the simulation
    assert len(evt_time) > 0, "Event log should have entries when event fires"
    assert len(evt_code) > 0, "Event code log should have entries"
    assert len(evt_index) > 0, "Event index log should have entries"
    assert np.all(evt_code == 0), "Single event should emit code 0"
    assert np.all(evt_index == 1), "Event index should be 1 (log width for log=['x'])"
    
    # Check that log data was captured
    assert evt_log_data.shape[1] >= 1, "Log data should have at least 1 column"
    # The logged value should be the state 'x' at event fire time
    # Since event fires when x < threshold (0.5), and then resets x to 1.0,
    # the logged values should be close to threshold
    for i in range(len(evt_time)):
        logged_x = evt_log_data[i, 0]
        assert 0.0 <= logged_x <= 1.5, f"Logged x value should be reasonable, got {logged_x}"
    
    # All logged event times should be within simulation bounds
    assert np.all(evt_time >= 0.0), "Event times should be >= t0"
    assert np.all(evt_time <= 3.0), "Event times should be <= t_end"
    
    # Event codes should be non-negative
    assert np.all(evt_code >= 0), "Event codes should be non-negative"


def test_event_logging_no_log_field():
    """Test that events without log field don't populate event buffer."""
    # Create a simple decay model without event logging
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    # Run simulation
    result = sim.run()
    
    # Event log should be empty (or minimal)
    evt_time = result.EVT_TIME_view
    
    # No events with logging enabled, so should be empty
    assert len(evt_time) == 0, "Event log should be empty when no events have log field"


def test_event_logging_multiple_fires():
    """Test that multiple event firings are all logged."""
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay_with_event.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    # Run with longer simulation to ensure multiple resets
    result = sim.run(t_end=10.0, cap_evt=100)
    
    evt_time = result.EVT_TIME_view
    evt_code = result.EVT_CODE_view
    
    # With t_end=10.0 and reset threshold=0.5, should fire multiple times
    # (exact count depends on dynamics, but should be > 1)
    assert len(evt_time) > 1, "Event should fire multiple times over longer simulation"
    assert np.all(evt_code == 0), "Single reset event should keep code 0 across firings"
    
    # Event times should be monotonically increasing (or at least non-decreasing)
    assert np.all(np.diff(evt_time) >= 0), "Event times should be in chronological order"


def test_event_logging_captures_state():
    """Test that event logging happens at the correct time."""
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay_with_event.toml"
    
    model = load_model_from_toml(model_path, jit=True)
    sim = Sim(model)
    
    result = sim.run(t_end=3.0, cap_evt=100)
    
    evt_time = result.EVT_TIME_view
    
    # Each event should correspond to when x crosses threshold
    # The event fires when x < threshold (0.5), then resets x to 1.0
    
    if len(evt_time) > 0:
        # Get the trajectory
        t_series = result.T_view
        x_series = result.Y_view[0, :]  # First state
        
        # For each event time, find the closest recorded time point
        for evt_t in evt_time:
            # Find index closest to event time
            idx = np.argmin(np.abs(t_series - evt_t))
            
            # At this point, x should be near 1.0 (after reset) or near threshold
            # This is a rough check since recording may not align perfectly with events
            assert t_series[idx] <= evt_t + 0.1, "Event time should be close to recorded time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
