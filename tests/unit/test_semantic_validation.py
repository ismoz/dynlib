"""
Integration tests for semantic validation enforcement.

Tests that build_spec properly validates models and rejects invalid ones with clear errors.
"""
import pytest
import tomllib
from dynlib.errors import ModelLoadError
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec


def _parse_toml(toml_str: str):
    """Helper to parse TOML string, parse the model, and build the spec (which triggers validation)."""
    doc = tomllib.loads(toml_str)
    normal = parse_model_v2(doc)
    return build_spec(normal)


def test_identifier_conflicts_across_sections_rejected():
    """Names must not be reused across params/aux/states/functions."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    phi = 0.0
    
    [params]
    V = 0.0
    
    [equations.rhs]
    phi = "V"
    
    [aux]
    V = "2*phi"
    """

    with pytest.raises(ModelLoadError, match=r"V \(params, aux\)"):
        _parse_toml(toml_str)


def test_cyclic_aux_detected():
    """Cyclic aux dependencies should fail at build time."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [aux]
    A = "B"
    B = "A"
    """
    
    with pytest.raises(ModelLoadError, match="Cyclic dependency detected involving"):
        _parse_toml(toml_str)


def test_cyclic_aux_indirect():
    """Cyclic aux dependencies through multiple steps should fail."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [aux]
    A = "B + 1"
    B = "C + 2"
    C = "A + 3"
    """
    
    with pytest.raises(ModelLoadError, match="Cyclic dependency detected involving"):
        _parse_toml(toml_str)


def test_cyclic_functions_detected():
    """Cyclic function dependencies should fail at build time."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f1]
    args = ["y"]
    expr = "f2(y) + 1"
    
    [functions.f2]
    args = ["z"]
    expr = "f1(z) + 1"
    """
    
    with pytest.raises(ModelLoadError, match="Cyclic dependency detected involving"):
        _parse_toml(toml_str)


def test_mixed_aux_function_cycle():
    """Cycle between aux and function should fail."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [aux]
    A = "f(x)"
    
    [functions.f]
    args = ["y"]
    expr = "A + y"
    """
    
    with pytest.raises(ModelLoadError, match="Cyclic dependency detected involving"):
        _parse_toml(toml_str)


def test_illegal_event_mutation_aux():
    """Events should not be allowed to mutate aux variables."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [aux]
    my_aux = "x * 2"
    
    [events.reset]
    phase = "post"
    cond = "x < 0.1"
    action.my_aux = "1.0"
    """
    
    with pytest.raises(ModelLoadError, match="may mutate only states/params; illegal"):
        _parse_toml(toml_str)


def test_illegal_event_mutation_function():
    """Events should not be allowed to mutate functions."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f]
    args = ["y"]
    expr = "y * 2"
    
    [events.reset]
    phase = "post"
    cond = "x < 0.1"
    action.f = "1.0"
    """
    
    with pytest.raises(ModelLoadError, match="may mutate only states/params; illegal"):
        _parse_toml(toml_str)


def test_invalid_function_args_not_list():
    """Function args must be a list."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f]
    args = "y"
    expr = "y * 2"
    """
    
    with pytest.raises(ModelLoadError, match="args must be a list of"):
        _parse_toml(toml_str)


def test_invalid_function_args_duplicate():
    """Function args must be unique."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f]
    args = ["y", "z", "y"]
    expr = "y + z"
    """
    
    with pytest.raises(ModelLoadError, match="args must be unique"):
        _parse_toml(toml_str)


def test_invalid_function_args_not_identifiers():
    """Function args must be valid identifiers."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f]
    args = ["y", "2bad"]
    expr = "y * 2"
    """
    
    with pytest.raises(ModelLoadError, match="args must be a list of identifiers"):
        _parse_toml(toml_str)


def test_invalid_function_expr_not_string():
    """Function expr must be a string."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f]
    args = ["y"]
    expr = 123
    """
    
    with pytest.raises(ModelLoadError, match="expr must be a string"):
        _parse_toml(toml_str)


def test_valid_aux_dependencies_allowed():
    """Valid aux dependencies should work fine."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    y = 2.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x + A"
    y = "B"
    
    [aux]
    A = "x + y"
    B = "A * 2"
    C = "B + A"
    """
    
    # Should not raise
    spec = _parse_toml(toml_str)
    assert "A" in spec.aux
    assert "B" in spec.aux
    assert "C" in spec.aux


def test_valid_function_dependencies_allowed():
    """Valid function dependencies should work fine."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f1]
    args = ["y"]
    expr = "y * 2"
    
    [functions.f2]
    args = ["z"]
    expr = "f1(z) + 1"
    
    [functions.f3]
    args = ["w"]
    expr = "f1(w) + f2(w)"
    """
    
    # Should not raise
    spec = _parse_toml(toml_str)
    assert "f1" in spec.functions
    assert "f2" in spec.functions
    assert "f3" in spec.functions


def test_valid_event_mutations_allowed():
    """Events can mutate states and params."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    y = 2.0
    
    [params]
    a = 0.1
    b = 0.2
    
    [equations.rhs]
    x = "-a*x"
    y = "-b*y"
    
    [events.reset]
    phase = "post"
    cond = "x < 0.1"
    action.x = "1.0"
    action.y = "2.0"
    action.a = "a * 1.1"
    """
    
    # Should not raise
    spec = _parse_toml(toml_str)
    assert len(spec.events) == 1


def test_valid_aux_with_functions():
    """Aux can use functions (acyclic)."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x + A"
    
    [aux]
    A = "f(x)"
    B = "g(A)"
    
    [functions.f]
    args = ["y"]
    expr = "y * 2"
    
    [functions.g]
    args = ["z"]
    expr = "z + 1"
    """
    
    # Should not raise
    spec = _parse_toml(toml_str)
    assert "A" in spec.aux
    assert "B" in spec.aux


def test_self_referential_aux():
    """Aux variable cannot reference itself."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [aux]
    A = "A + 1"
    """
    
    with pytest.raises(ModelLoadError, match="Cyclic dependency detected involving"):
        _parse_toml(toml_str)


def test_self_referential_function():
    """Function cannot call itself (direct recursion)."""
    toml_str = """
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 0.1
    
    [equations.rhs]
    x = "-a*x"
    
    [functions.f]
    args = ["n"]
    expr = "f(n-1) if n > 0 else 1"
    """
    
    with pytest.raises(ModelLoadError, match="Cyclic dependency detected involving"):
        _parse_toml(toml_str)
