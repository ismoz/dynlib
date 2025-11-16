# tests/unit/test_equations_block_form.py
"""
Unit tests for block-form equations ([equations].expr).
Tests parsing, validation, and codegen.
"""
import pytest
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build
from dynlib.runtime.workspace import make_runtime_workspace
from dynlib.errors import ModelLoadError


def test_block_form_only():
    """Test model with only block-form equations."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 2.0, "b": 0.5},
        "equations": {
            "rhs": {},
            "expr": """
                dx = -a*x
                dy = x - b*y
            """
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    model = build(spec, stepper="euler")
    
    assert spec.equations_block is not None
    assert "dx = -a*x" in spec.equations_block or "dx=-a*x" in spec.equations_block.replace(" ", "")
    
    # Compile and verify RHS works
    import numpy as np
    y_vec = np.array([1.0, 0.5], dtype=np.float64)
    dy_out = np.zeros(2, dtype=np.float64)
    params = np.array([2.0, 0.5], dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    
    model.rhs(0.0, y_vec, dy_out, params, runtime_ws)
    
    # dx = -a*x = -2.0*1.0 = -2.0
    # dy = x - b*y = 1.0 - 0.5*0.5 = 0.75
    assert abs(dy_out[0] - (-2.0)) < 1e-10
    assert abs(dy_out[1] - 0.75) < 1e-10


def test_block_form_d_paren_syntax():
    """Test d(x) = ... syntax."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 2.0},
        "equations": {
            "expr": """
                d(x) = -a*x
                d(y) = x
            """
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    model = build(spec, stepper="euler")
    
    import numpy as np
    y_vec = np.array([1.0, 0.5], dtype=np.float64)
    dy_out = np.zeros(2, dtype=np.float64)
    params = np.array([2.0], dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    
    model.rhs(0.0, y_vec, dy_out, params, runtime_ws)
    
    # d(x) = -a*x = -2.0*1.0 = -2.0
    # d(y) = x = 1.0
    assert abs(dy_out[0] - (-2.0)) < 1e-10
    assert abs(dy_out[1] - 1.0) < 1e-10


def test_mixed_form_allowed():
    """Test that mixed form (rhs + block) is allowed if no overlap."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 2.0, "b": 0.5},
        "equations": {
            "rhs": {"x": "-a*x"},
            "expr": "dy = x - b*y"
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)  # Should not raise
    
    # No need to build or call RHS, just check parsing


def test_duplicate_targets_rejected():
    """Test that defining a state in both rhs and block raises error."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 2.0},
        "equations": {
            "rhs": {"x": "-a*x"},
            "expr": "dx = -2*a*x"  # x defined twice
        },
    }
    
    # This error is caught at parse time by validate_name_collisions in schema.py
    with pytest.raises(ModelLoadError) as exc:
        normal = parse_model_v2(doc)
    
    assert "duplicate" in str(exc.value).lower()
    assert "x" in str(exc.value)


def test_invalid_block_lhs_rejected():
    """Test that invalid LHS in block form raises clear error (e.g., typo in state name)."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 2.0},
        "equations": {
            "expr": "dz = -a*x"  # z is not a declared state
        },
    }
    
    # Error caught during validation - unknown state 'z' in derivative
    with pytest.raises(ModelLoadError) as exc:
        normal = parse_model_v2(doc)
        build_spec(normal)
    
    assert "unknown" in str(exc.value).lower() or "state" in str(exc.value).lower()


def test_unknown_state_in_block_rejected():
    """Test that referencing unknown state in block raises error."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 2.0},
        "equations": {
            "expr": "dy = x"  # y not declared
        },
    }
    
    # This error is caught at parse time by validate_name_collisions in schema.py
    with pytest.raises(ModelLoadError) as exc:
        normal = parse_model_v2(doc)
    
    assert "unknown" in str(exc.value).lower() or "must be declared" in str(exc.value).lower()
    assert "y" in str(exc.value)


def test_block_form_with_aux():
    """Test block form can use aux variables."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 2.0},
        "aux": {"E": "0.5*a*x**2"},
        "equations": {
            "expr": """
                dx = -a*x
                dy = E
            """
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    model = build(spec, stepper="euler")
    
    import numpy as np
    y_vec = np.array([1.0, 0.5], dtype=np.float64)
    dy_out = np.zeros(2, dtype=np.float64)
    params = np.array([2.0], dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    
    model.rhs(0.0, y_vec, dy_out, params, runtime_ws)
    
    # dx = -a*x = -2.0*1.0 = -2.0
    # dy = E = 0.5*a*x**2 = 0.5*2.0*1.0 = 1.0
    assert abs(dy_out[0] - (-2.0)) < 1e-10
    assert abs(dy_out[1] - 1.0) < 1e-10


def test_block_form_with_functions():
    """Test block form can use user-defined functions."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 2.0},
        "functions": {
            "decay": {"args": ["u", "k"], "expr": "-k*u"}
        },
        "equations": {
            "expr": "dx = decay(x, a)"
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    model = build(spec, stepper="euler")
    
    import numpy as np
    y_vec = np.array([1.0], dtype=np.float64)
    dy_out = np.zeros(1, dtype=np.float64)
    params = np.array([2.0], dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    
    model.rhs(0.0, y_vec, dy_out, params, runtime_ws)
    
    # dx = decay(x, a) = -a*x = -2.0*1.0 = -2.0
    assert abs(dy_out[0] - (-2.0)) < 1e-10


def test_empty_lines_in_block_ignored():
    """Test that empty lines and whitespace are handled correctly."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 2.0},
        "equations": {
            "expr": """
            
                dx = -a*x
                
                dy = x
                
            """
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    model = build(spec, stepper="euler")
    
    import numpy as np
    y_vec = np.array([1.0, 0.5], dtype=np.float64)
    dy_out = np.zeros(2, dtype=np.float64)
    params = np.array([2.0], dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    
    model.rhs(0.0, y_vec, dy_out, params, runtime_ws)
    
    assert abs(dy_out[0] - (-2.0)) < 1e-10
    assert abs(dy_out[1] - 1.0) < 1e-10
def test_missing_equals_rejected():
    """Test that lines without '=' are rejected."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 2.0},
        "equations": {
            "expr": "dx -a*x"  # Missing =
        },
    }
    normal = parse_model_v2(doc)
    
    with pytest.raises(ModelLoadError) as exc:
        build_spec(normal)
    
    assert "must contain '='" in str(exc.value)


def test_map_model_rejects_derivative_notation():
    """Test that map models reject derivative notation."""
    doc = {
        "model": {"type": "map"},
        "states": {"x": 1.0},
        "params": {"a": 2.0},
        "equations": {
            "expr": "dx = -a*x"  # Should be x = ... for map
        },
    }
    
    with pytest.raises(ModelLoadError) as exc:
        normal = parse_model_v2(doc)
        build_spec(normal)  # Error happens during validation
    
    assert "derivative notation" in str(exc.value).lower()
    assert "map" in str(exc.value).lower()


def test_ode_model_allows_both_notations():
    """Test that ODE models accept both derivative and direct assignment notation."""
    # Test derivative notation
    doc1 = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 2.0},
        "equations": {
            "expr": "dx = -a*x"
        },
    }
    normal1 = parse_model_v2(doc1)
    spec1 = build_spec(normal1)
    assert spec1.kind == "ode"
    
    # Test direct assignment - also valid for ODE!
    doc2 = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 2.0},
        "equations": {
            "expr": "x = -a*x"
        },
    }
    normal2 = parse_model_v2(doc2)
    spec2 = build_spec(normal2)
    assert spec2.kind == "ode"


def test_map_model_with_direct_assignment():
    """Test that map models work with direct assignment."""
    doc = {
        "model": {"type": "map"},
        "states": {"x": 1.0},
        "params": {"a": 0.9},
        "equations": {
            "expr": "x = a*x"
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    model = build(spec, stepper="map")
    
    import numpy as np
    y_vec = np.array([1.0], dtype=np.float64)
    dy_out = np.zeros(1, dtype=np.float64)
    params = np.array([0.9], dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    
    model.rhs(0.0, y_vec, dy_out, params, runtime_ws)
    
    # For map: next state = a*x = 0.9*1.0 = 0.9
    assert abs(dy_out[0] - 0.9) < 1e-10


def test_map_model_with_variable_starting_with_d():
    """Test that map models accept variables like 'delta' without treating them as derivatives.
    
    This is a critical edge case: without regex validation, 'delta' could be misparsed
    as 'd(elta)' or 'delta' as 'd' + 'elta'.
    """
    doc = {
        "model": {"type": "map"},
        "states": {"delta": 0.1, "data": 1.0},
        "params": {"rate": 0.95},
        "equations": {
            "expr": """
                delta = delta * rate
                data = data + delta
            """
        },
    }
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    model = build(spec, stepper="map")
    
    import numpy as np
    y_vec = np.array([0.1, 1.0], dtype=np.float64)
    dy_out = np.zeros(2, dtype=np.float64)
    params = np.array([0.95], dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    
    model.rhs(0.0, y_vec, dy_out, params, runtime_ws)
    
    # delta = delta * rate = 0.1 * 0.95 = 0.095
    # data = data + delta = 1.0 + 0.1 = 1.1
    assert abs(dy_out[0] - 0.095) < 1e-10
    assert abs(dy_out[1] - 1.1) < 1e-10
