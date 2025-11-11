# tests/unit/test_export_sources.py
"""Unit tests for model source export functionality."""

from pathlib import Path
import tempfile
import pytest

from dynlib import build
from dynlib.compiler.build import export_model_sources


def test_export_sources_creates_files():
    """Test that export_sources creates the expected files."""
    model_path = Path(__file__).parent.parent / "data" / "models" / "decay.toml"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build model
        model = build(str(model_path), stepper="euler", jit=True, disk_cache=False)
        
        # Export sources
        export_dir = Path(tmpdir) / "exported"
        files = export_model_sources(model, export_dir)
        
        # Check that files were created
        assert "rhs" in files
        assert "events_pre" in files
        assert "events_post" in files
        assert "stepper" in files
        assert "info" in files
        
        # Check files exist
        assert files["rhs"].exists()
        assert files["events_pre"].exists()
        assert files["events_post"].exists()
        assert files["stepper"].exists()
        assert files["info"].exists()


def test_export_sources_content_validity():
    """Test that exported sources contain valid Python code."""
    model_path = Path(__file__).parent.parent / "data" / "models" / "decay.toml"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build(str(model_path), stepper="euler", jit=True, disk_cache=False)
        files = export_model_sources(model, tmpdir)
        
        # Check RHS contains expected content
        rhs_content = files["rhs"].read_text()
        assert "def rhs" in rhs_content
        assert "dy_out" in rhs_content
        assert "params" in rhs_content
        
        # Check stepper contains expected content
        stepper_content = files["stepper"].read_text()
        assert "def euler_stepper" in stepper_content
        assert "y_prop" in stepper_content
        
        # Check info file
        info_content = files["info"].read_text()
        assert "Model Information" in info_content
        assert "Spec Hash:" in info_content
        assert "States:" in info_content


def test_sources_available_in_model():
    """Test that source code is stored in the model object."""
    model_path = Path(__file__).parent.parent / "data" / "models" / "decay.toml"
    
    # Test with disk_cache=False
    model = build(str(model_path), stepper="euler", jit=True, disk_cache=False)
    
    assert model.rhs_source is not None
    assert model.events_pre_source is not None
    assert model.events_post_source is not None
    assert model.stepper_source is not None
    
    assert "def rhs" in model.rhs_source
    assert "def events_pre" in model.events_pre_source
    assert "def events_post" in model.events_post_source
    assert "def euler_stepper" in model.stepper_source


def test_sources_available_with_disk_cache():
    """Test that sources are available even with disk_cache=True."""
    model_path = Path(__file__).parent.parent / "data" / "models" / "decay.toml"
    
    # Test with disk_cache=True
    model = build(str(model_path), stepper="euler", jit=True, disk_cache=True)
    
    assert model.rhs_source is not None
    assert model.events_pre_source is not None
    assert model.events_post_source is not None
    assert model.stepper_source is not None


def test_sources_available_different_steppers():
    """Test that sources are available for different steppers."""
    model_path = Path(__file__).parent.parent / "data" / "models" / "decay.toml"
    
    for stepper_name in ["euler", "rk4"]:
        model = build(str(model_path), stepper=stepper_name, jit=True, disk_cache=False)
        
        assert model.rhs_source is not None
        assert model.stepper_source is not None
        assert f"def {stepper_name}_stepper" in model.stepper_source


def test_export_creates_directory():
    """Test that export creates the output directory if it doesn't exist."""
    model_path = Path(__file__).parent.parent / "data" / "models" / "decay.toml"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build(str(model_path), stepper="euler", jit=True, disk_cache=False)
        
        # Export to nested directory that doesn't exist
        export_dir = Path(tmpdir) / "nested" / "output" / "dir"
        assert not export_dir.exists()
        
        files = export_model_sources(model, export_dir)
        
        # Directory should now exist
        assert export_dir.exists()
        assert export_dir.is_dir()
        assert len(list(export_dir.iterdir())) > 0


def test_export_with_events():
    """Test export with a model that has events."""
    model_path = Path(__file__).parent.parent / "data" / "models" / "decay_with_event.toml"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build(str(model_path), stepper="euler", jit=True, disk_cache=False)
        files = export_model_sources(model, tmpdir)
        
        # Check info file mentions events
        info_content = files["info"].read_text()
        assert "Events" in info_content
