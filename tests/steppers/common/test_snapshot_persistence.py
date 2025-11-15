# tests/integration/test_snapshot_persistence.py
"""Integration tests for snapshot export/import functionality."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from dynlib import setup


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DECAY_MODEL = str(DATA_DIR / "decay.toml")


def test_export_import_current_state(tmp_path):
    """Test round-trip export/import of current session state."""
    sim = setup(DECAY_MODEL)
    
    # Run simulation to some intermediate state
    sim.run(T=0.5)
    original_summary = sim.session_state_summary()
    original_y = sim._session_state.y_curr.copy()
    original_params = sim._session_state.params_curr.copy()
    
    # Export current state
    snap_path = tmp_path / "current_state.npz"
    sim.export_snapshot(snap_path, source="current")
    
    # Modify state locally
    sim.run(T=1.0, resume=True)
    modified_summary = sim.session_state_summary()
    assert modified_summary["t"] != original_summary["t"], "State should have changed"
    
    # Import and verify restoration
    sim.import_snapshot(snap_path)
    restored_summary = sim.session_state_summary()
    
    assert restored_summary["t"] == pytest.approx(original_summary["t"])
    assert restored_summary["step"] == original_summary["step"]
    assert restored_summary["dt"] == pytest.approx(original_summary["dt"])
    np.testing.assert_array_equal(sim._session_state.y_curr, original_y)
    np.testing.assert_array_equal(sim._session_state.params_curr, original_params)
    
    # Verify results/history cleared
    with pytest.raises(RuntimeError, match="No simulation results available"):
        sim.raw_results()


def test_export_import_named_snapshot(tmp_path):
    """Test round-trip export/import of named in-memory snapshot."""
    sim = setup(DECAY_MODEL)
    
    # Run to intermediate state and create named snapshot
    sim.run(T=0.3)
    sim.create_snapshot("mid", "Intermediate checkpoint")
    original_summary = sim.session_state_summary()
    
    # Continue simulation
    sim.run(T=1.0, resume=True)
    
    # Export the named snapshot
    snap_path = tmp_path / "mid_snapshot.npz"
    sim.export_snapshot(snap_path, source="snapshot", name="mid")
    
    # Change state further
    sim.run(T=2.0, resume=True)
    
    # Import named snapshot
    sim.import_snapshot(snap_path)
    restored_summary = sim.session_state_summary()
    
    assert restored_summary["t"] == pytest.approx(original_summary["t"])
    assert restored_summary["step"] == original_summary["step"]


def test_inspect_snapshot(tmp_path):
    """Test snapshot inspection without modifying sim state."""
    sim = setup(DECAY_MODEL)
    sim.run(T=0.4)
    
    snap_path = tmp_path / "inspect_test.npz"
    sim.export_snapshot(snap_path, source="current")
    
    # Inspect should not modify state
    original_summary = sim.session_state_summary()
    meta = sim.inspect_snapshot(snap_path)
    new_summary = sim.session_state_summary()
    
    assert new_summary == original_summary, "inspect_snapshot should not modify sim state"
    
    # Verify metadata content
    assert meta["schema"] == "dynlib-snapshot-v1"
    assert meta["name"] == "current"
    assert meta["t_curr"] == pytest.approx(original_summary["t"])
    assert "created_at" in meta
    assert "pins" in meta
    assert "n_state" in meta
    assert "n_params" in meta


def test_pin_mismatch_rejection():
    """Test that snapshots with mismatched pins are rejected."""
    sim1 = setup(DECAY_MODEL, stepper="euler")
    sim1.run(T=0.2)
    
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        snap_path = Path(tmp.name)
    
    try:
        # Export from first sim
        sim1.export_snapshot(snap_path, source="current")
        
        # Try to import into sim with different stepper pins
        sim2 = setup(DECAY_MODEL, stepper="rk4")
        
        with pytest.raises(RuntimeError, match="Snapshot incompatible"):
            sim2.import_snapshot(snap_path)
            
    finally:
        snap_path.unlink(missing_ok=True)


def test_invalid_export_arguments():
    """Test validation of export_snapshot arguments."""
    sim = setup(DECAY_MODEL)
    
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        snap_path = Path(tmp.name)
    
    try:
        # Missing name when source="snapshot"
        with pytest.raises(ValueError, match="name is required when source='snapshot'"):
            sim.export_snapshot(snap_path, source="snapshot")
        
        # Name provided when source="current"
        with pytest.raises(ValueError, match="name should not be provided when source='current'"):
            sim.export_snapshot(snap_path, source="current", name="test")
            
    finally:
        snap_path.unlink(missing_ok=True)


def test_malformed_file_handling(tmp_path):
    """Test handling of corrupted or malformed snapshot files."""
    sim = setup(DECAY_MODEL)
    
    # Test missing file
    nonexistent = tmp_path / "nonexistent.npz"
    with pytest.raises(ValueError, match="Cannot read snapshot file"):
        sim.import_snapshot(nonexistent)
    
    # Test file without meta.json
    no_meta = tmp_path / "no_meta.npz"
    np.savez(no_meta, y=np.array([1.0]), params=np.array([0.5]))
    
    with pytest.raises(ValueError, match="Missing 'meta.json'"):
        sim.import_snapshot(no_meta)
    
    # Test file with invalid JSON
    bad_json = tmp_path / "bad_json.npz"
    meta_bytes = np.frombuffer(b"invalid json", dtype=np.uint8)
    np.savez(bad_json, **{"meta.json": meta_bytes, "y": np.array([1.0]), "params": np.array([0.5])})
    
    with pytest.raises(ValueError, match="Invalid JSON in meta.json"):
        sim.import_snapshot(bad_json)


def test_wrong_schema_rejection(tmp_path):
    """Test rejection of files with wrong schema version."""
    sim = setup(DECAY_MODEL)
    
    # Create file with wrong schema
    wrong_schema = tmp_path / "wrong_schema.npz"
    meta = {"schema": "dynlib-snapshot-v999"}  # Wrong version
    meta_json = json.dumps(meta)
    meta_bytes = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)
    
    np.savez(wrong_schema, **{
        "meta.json": meta_bytes,
        "y": np.array([1.0]),
        "params": np.array([0.5])
    })
    
    with pytest.raises(ValueError, match="Unsupported schema"):
        sim.import_snapshot(wrong_schema)


def test_shape_mismatch_rejection(tmp_path):
    """Test rejection of files with wrong array shapes."""
    sim = setup(DECAY_MODEL)
    sim.run(T=0.1)
    
    # Export valid snapshot first
    valid_path = tmp_path / "valid.npz"
    sim.export_snapshot(valid_path, source="current")
    
    # Load and modify to have wrong shapes
    with np.load(valid_path, allow_pickle=False) as npz_file:
        meta_bytes = npz_file["meta.json"]
        y_original = npz_file["y"]
        params_original = npz_file["params"]
    
    # Test wrong y shape
    wrong_y_path = tmp_path / "wrong_y.npz"
    np.savez(wrong_y_path, **{
        "meta.json": meta_bytes,
        "y": np.array([1.0, 2.0]),  # Wrong shape
        "params": params_original
    })
    
    with pytest.raises(ValueError, match="State vector shape mismatch"):
        sim.import_snapshot(wrong_y_path)
    
    # Test wrong params shape
    wrong_params_path = tmp_path / "wrong_params.npz"
    np.savez(wrong_params_path, **{
        "meta.json": meta_bytes,
        "y": y_original,
        "params": np.array([1.0, 2.0, 3.0])  # Wrong shape
    })
    
    with pytest.raises(ValueError, match="Parameters shape mismatch"):
        sim.import_snapshot(wrong_params_path)


def test_atomic_write_behavior(tmp_path):
    """Test that export overwrites existing files atomically."""
    sim = setup(DECAY_MODEL)
    sim.run(T=0.2)
    
    snap_path = tmp_path / "atomic_test.npz"
    
    # Create initial file
    snap_path.write_text("dummy content")
    assert snap_path.exists()
    
    # Export should overwrite atomically
    sim.export_snapshot(snap_path, source="current")
    
    # File should now be valid snapshot
    meta = sim.inspect_snapshot(snap_path)
    assert meta["schema"] == "dynlib-snapshot-v1"


def test_workspace_persistence(tmp_path):
    """Test that stepper workspace is correctly persisted and restored."""
    # Note: This test depends on having a stepper that uses non-empty workspace
    # For now, we'll test the basic structure even if workspace is empty
    sim = setup(DECAY_MODEL)
    sim.run(T=0.3)
    
    snap_path = tmp_path / "workspace_test.npz"
    sim.export_snapshot(snap_path, source="current")
    
    # Check that we can round-trip even with workspace (empty or not)
    original_ws = sim._session_state.stepper_ws.copy()
    sim.run(T=1.0, resume=True)  # Change state
    
    sim.import_snapshot(snap_path)
    restored_ws = sim._session_state.stepper_ws

    # Verify workspace structure is preserved
    assert set(restored_ws.keys()) == set(original_ws.keys())
    for key in original_ws:
        np.testing.assert_array_equal(restored_ws[key], original_ws[key])


def test_export_snapshot_preserves_metadata(tmp_path):
    """Exporting stored snapshots must keep their original time shift / nominal dt."""
    sim = setup(DECAY_MODEL)

    # Create a snapshot after running with a transient warm-up to get a time shift.
    sim.run(T=1.0, transient=0.4)
    sim.create_snapshot("transient", "after transient run")
    stored_snapshot = sim._snapshots["transient"]
    stored_time_shift = stored_snapshot.time_shift
    stored_nominal_dt = stored_snapshot.nominal_dt

    # Run again with a different dt so the sim's current nominal dt/time shift change.
    sim.run(T=0.5, dt=0.002)
    assert sim._nominal_dt == pytest.approx(0.002)
    assert sim._time_shift == 0.0  # Non-transient run resets the shift

    # Export the previously stored snapshot and ensure metadata matches the stored values.
    snap_path = tmp_path / "transient_snapshot.npz"
    sim.export_snapshot(snap_path, source="snapshot", name="transient")
    meta = sim.inspect_snapshot(snap_path)

    assert meta["time_shift"] == pytest.approx(stored_time_shift)
    assert meta["nominal_dt"] == pytest.approx(stored_nominal_dt)
