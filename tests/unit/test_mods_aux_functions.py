# tests/unit/test_mods_aux_functions.py
"""Test mod verbs for aux and functions."""
import pytest

from dynlib.errors import ModelLoadError
from dynlib.dsl.parser import parse_model_v2
from dynlib.compiler.mods import ModSpec, apply_mods_v2


def base_normal_with_aux_funcs():
    """Normal dict with aux and functions."""
    doc = {
        "model": {"type": "ode", "dtype": "float64"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 1.0, "b": 2.0},
        "equations": {"rhs": {"x": "-a*x", "y": "x - b*y"}, "expr": None},
        "aux": {"E": "0.5*a*x^2", "power": "a*x"},
        "functions": {
            "sat": {"args": ["u", "c"], "expr": "u/(1+abs(u)^c)"},
            "step": {"args": ["x"], "expr": "x if x > 0 else 0"},
        },
        "events": {},  # Empty dict, not list
        "sim": {},
    }
    return parse_model_v2(doc)


def test_add_aux():
    """Test adding new aux variables."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="add_aux",
        add={"aux": {"new_aux": "x + y", "another": "2*x"}}
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert "new_aux" in out["aux"]
    assert out["aux"]["new_aux"] == "x + y"
    assert "another" in out["aux"]
    assert out["aux"]["another"] == "2*x"
    # Original aux preserved
    assert out["aux"]["E"] == "0.5*a*x^2"
    assert out["aux"]["power"] == "a*x"


def test_add_aux_duplicate_raises():
    """Test that adding duplicate aux raises error."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="bad_add",
        add={"aux": {"E": "new_expr"}}  # E already exists
    )
    
    with pytest.raises(ModelLoadError, match="add.aux.E: aux already exists"):
        apply_mods_v2(normal, [mod])


def test_replace_aux():
    """Test replacing existing aux variables."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="replace_aux",
        replace={"aux": {"E": "1.5*a*x^2", "power": "b*y"}}
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert out["aux"]["E"] == "1.5*a*x^2"
    assert out["aux"]["power"] == "b*y"


def test_replace_aux_nonexistent_raises():
    """Test that replacing nonexistent aux raises error."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="bad_replace",
        replace={"aux": {"nonexistent": "x"}}
    )
    
    with pytest.raises(ModelLoadError, match="replace.aux.nonexistent: aux does not exist"):
        apply_mods_v2(normal, [mod])


def test_remove_aux():
    """Test removing aux variables."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="remove_aux",
        remove={"aux": {"names": ["E", "power"]}}
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert "E" not in out["aux"]
    assert "power" not in out["aux"]


def test_remove_aux_nonexistent_silent():
    """Test that removing nonexistent aux is silent (no error)."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="remove_aux",
        remove={"aux": {"names": ["nonexistent", "E"]}}
    )
    
    out = apply_mods_v2(normal, [mod])
    
    # E should be removed, nonexistent silently ignored
    assert "E" not in out["aux"]
    assert "power" in out["aux"]  # not removed


def test_set_aux_upsert():
    """Test set.aux can create or update."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="set_aux",
        set={"aux": {"E": "modified", "new_one": "x*y"}}
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert out["aux"]["E"] == "modified"  # updated
    assert out["aux"]["new_one"] == "x*y"  # created
    assert out["aux"]["power"] == "a*x"  # unchanged


def test_add_functions():
    """Test adding new functions."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="add_funcs",
        add={
            "functions": {
                "sigmoid": {"args": ["x"], "expr": "1/(1+exp(-x))"},
                "square": {"args": ["u", "v"], "expr": "u^2 + v^2"},
            }
        }
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert "sigmoid" in out["functions"]
    assert out["functions"]["sigmoid"]["args"] == ["x"]
    assert out["functions"]["sigmoid"]["expr"] == "1/(1+exp(-x))"
    assert "square" in out["functions"]
    # Original functions preserved
    assert "sat" in out["functions"]
    assert "step" in out["functions"]


def test_add_functions_duplicate_raises():
    """Test that adding duplicate function raises error."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="bad_add",
        add={"functions": {"sat": {"args": ["x"], "expr": "x"}}}  # sat already exists
    )
    
    with pytest.raises(ModelLoadError, match="add.functions.sat: function already exists"):
        apply_mods_v2(normal, [mod])


def test_replace_functions():
    """Test replacing existing functions."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="replace_funcs",
        replace={
            "functions": {
                "sat": {"args": ["u", "c", "scale"], "expr": "scale*u/(1+abs(u)^c)"},
                "step": {"args": ["x", "threshold"], "expr": "x if x > threshold else 0"},
            }
        }
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert out["functions"]["sat"]["args"] == ["u", "c", "scale"]
    assert "scale" in out["functions"]["sat"]["expr"]
    assert out["functions"]["step"]["args"] == ["x", "threshold"]


def test_replace_functions_nonexistent_raises():
    """Test that replacing nonexistent function raises error."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="bad_replace",
        replace={"functions": {"nonexistent": {"args": ["x"], "expr": "x"}}}
    )
    
    with pytest.raises(ModelLoadError, match="replace.functions.nonexistent: function does not exist"):
        apply_mods_v2(normal, [mod])


def test_remove_functions():
    """Test removing functions."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="remove_funcs",
        remove={"functions": {"names": ["sat", "step"]}}
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert "sat" not in out["functions"]
    assert "step" not in out["functions"]


def test_remove_functions_nonexistent_silent():
    """Test that removing nonexistent function is silent (no error)."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="remove_funcs",
        remove={"functions": {"names": ["nonexistent", "sat"]}}
    )
    
    out = apply_mods_v2(normal, [mod])
    
    # sat should be removed, nonexistent silently ignored
    assert "sat" not in out["functions"]
    assert "step" in out["functions"]  # not removed


def test_set_functions_upsert():
    """Test set.functions can create or update."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="set_funcs",
        set={
            "functions": {
                "sat": {"args": ["x"], "expr": "modified"},  # update
                "new_func": {"args": ["a", "b"], "expr": "a+b"},  # create
            }
        }
    )
    
    out = apply_mods_v2(normal, [mod])
    
    assert out["functions"]["sat"]["args"] == ["x"]  # updated
    assert out["functions"]["sat"]["expr"] == "modified"
    assert out["functions"]["new_func"]["args"] == ["a", "b"]  # created
    assert out["functions"]["step"]["args"] == ["x"]  # unchanged


def test_verb_order_remove_then_add():
    """Test verb order: remove → replace → add → set."""
    normal = base_normal_with_aux_funcs()
    
    # Single mod with multiple verbs
    mod = ModSpec(
        name="combo",
        remove={"aux": {"names": ["E"]}},
        add={"aux": {"E": "new_E_expr"}},  # Re-add after remove
    )
    
    out = apply_mods_v2(normal, [mod])
    
    # E should be re-added with new expr
    assert out["aux"]["E"] == "new_E_expr"


def test_multiple_mods_aux_and_functions():
    """Test multiple mods applied in sequence."""
    normal = base_normal_with_aux_funcs()
    
    mods = [
        ModSpec(name="m1", add={"aux": {"temp": "x+1"}}),
        ModSpec(name="m2", replace={"aux": {"E": "modified_E"}}),
        ModSpec(name="m3", add={"functions": {"relu": {"args": ["x"], "expr": "max(0,x)"}}}),
        ModSpec(name="m4", remove={"functions": {"names": ["step"]}}),
    ]
    
    out = apply_mods_v2(normal, mods)
    
    assert "temp" in out["aux"]
    assert out["aux"]["E"] == "modified_E"
    assert "relu" in out["functions"]
    assert "step" not in out["functions"]
    assert "sat" in out["functions"]  # unchanged


def test_invalid_aux_value_type():
    """Test that non-string aux values are rejected."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="bad_aux",
        add={"aux": {"bad": 123}}  # Should be string
    )
    
    with pytest.raises(ModelLoadError, match="add.aux.bad: value must be a string expression"):
        apply_mods_v2(normal, [mod])


def test_invalid_function_args():
    """Test that invalid function args are rejected."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="bad_func",
        add={"functions": {"bad": {"args": "not_a_list", "expr": "x"}}}
    )
    
    with pytest.raises(ModelLoadError, match="functions.bad.args must be list of strings"):
        apply_mods_v2(normal, [mod])


def test_invalid_function_expr():
    """Test that invalid function expr is rejected."""
    normal = base_normal_with_aux_funcs()
    
    mod = ModSpec(
        name="bad_func",
        add={"functions": {"bad": {"args": ["x"], "expr": 123}}}  # Should be string
    )
    
    with pytest.raises(ModelLoadError, match="functions.bad.expr must be a string"):
        apply_mods_v2(normal, [mod])
