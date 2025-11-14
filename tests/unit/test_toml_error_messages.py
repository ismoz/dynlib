"""Test improved TOML parsing error messages."""
import pytest
from dynlib.compiler.build import build
from dynlib.errors import ModelLoadError


def test_inline_model_division_error():
    """Test that division in parameter values shows helpful error."""
    bad_inline = """
inline:
[model]
type="ode"

[states]
x=1.0

[params]
a=1/2

[equations.rhs]
x="-a*x"

[sim]
stepper="euler"
"""
    
    with pytest.raises(ModelLoadError) as exc_info:
        build(bad_inline, jit=False)
    
    error_msg = str(exc_info.value)
    
    # Check that error shows context
    assert "a=1/2" in error_msg
    assert ">>>" in error_msg  # Shows line marker
    assert "^" in error_msg  # Shows column pointer
    
    # Check that hint is included
    assert "Hint:" in error_msg
    assert "Division (/) in numeric values" in error_msg
    assert "beta=8/3" in error_msg  # Example in hint


def test_inline_model_missing_bracket():
    """Test that missing bracket shows helpful error."""
    bad_inline = """
inline:
[model
type="ode"
"""
    
    with pytest.raises(ModelLoadError) as exc_info:
        build(bad_inline, jit=False)
    
    error_msg = str(exc_info.value)
    
    # Check that error shows the problematic line
    assert "[model" in error_msg
    assert ">>>" in error_msg
    assert "^" in error_msg


def test_inline_model_unclosed_string():
    """Test that unclosed string shows helpful error."""
    bad_inline = """
inline:
[model]
type="ode

[states]
x=1.0
"""
    
    with pytest.raises(ModelLoadError) as exc_info:
        build(bad_inline, jit=False)
    
    error_msg = str(exc_info.value)
    
    # Check that error shows context
    assert 'type="ode' in error_msg
    assert ">>>" in error_msg
    assert "^" in error_msg


def test_file_model_division_error(tmp_path):
    """Test that division error in file shows helpful context."""
    model_file = tmp_path / "bad_model.toml"
    model_file.write_text("""
[model]
type="ode"

[states]
x=1.0

[params]
beta=8/3

[equations.rhs]
x="-beta*x"

[sim]
stepper="euler"
""")
    
    with pytest.raises(ModelLoadError) as exc_info:
        build(str(model_file), jit=False)
    
    error_msg = str(exc_info.value)
    
    # Check that error shows the file path
    assert str(model_file) in error_msg
    
    # Check that error shows context
    assert "beta=8/3" in error_msg
    assert ">>>" in error_msg
    assert "^" in error_msg
    
    # Check that hint is included
    assert "Hint:" in error_msg


def test_error_line_numbers_are_accurate():
    """Test that line numbers in error messages are accurate."""
    bad_inline = """
inline:
[model]
type="ode"

[states]
x=1.0

[params]
good=1.0
bad=8/3

[equations.rhs]
x="-bad*x"
"""
    
    with pytest.raises(ModelLoadError) as exc_info:
        build(bad_inline, jit=False)
    
    error_msg = str(exc_info.value)
    
    # The error should point to the line with "bad=8/3"
    # and show it with >>>
    assert "bad=8/3" in error_msg
    assert ">>> " in error_msg
    
    # Should also show surrounding context
    assert "good=1.0" in error_msg
