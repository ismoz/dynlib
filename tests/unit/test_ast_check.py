# tests/unit/test_ast_check.py
import pytest

from dynlib.errors import ModelLoadError
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.astcheck import (
    collect_names,
    validate_expr_acyclic,
    validate_equation_targets,
    validate_event_legality,
    validate_dtype_rules,
    validate_functions_signature,
)

def base_doc():
    return {
        "model": {"type": "ode", "dtype": "float64"},
        "states": {"x": 1.0, "u": 0.0},
        "params": {"a": 1.0, "b": 2.0},
        "equations": {"rhs": {"x": "-a*x", "u": "x - b*u"}},
        "aux": {},
        "functions": {},
        "events": {},
        "sim": {},
    }

def test_collect_names_basic():
    n = parse_model_v2(base_doc())
    names = collect_names(n)
    assert names["states"] == {"x", "u"}
    assert names["params"] == {"a", "b"}
    assert names["aux"] == set()
    assert names["functions"] == set()
    assert names["events"] == set()

def test_validate_expr_acyclic_ok_with_aux_and_functions():
    d = base_doc()
    d["aux"] = {"z": "x + f1(u)", "w": "z + 1"}
    d["functions"] = {"f1": {"args": ["q"], "expr": "q + 1"}}
    n = parse_model_v2(d)
    validate_functions_signature(n)
    validate_expr_acyclic(n)  # no cycles

def test_validate_expr_acyclic_cycle_in_aux():
    d = base_doc()
    d["aux"] = {"a": "b + 1", "b": "a + 1"}
    n = parse_model_v2(d)
    with pytest.raises(ModelLoadError):
        validate_expr_acyclic(n)

def test_validate_expr_acyclic_cycle_in_functions():
    d = base_doc()
    d["functions"] = {
        "f": {"args": ["x"], "expr": "g(x)"},
        "g": {"args": ["y"], "expr": "f(y)"},
    }
    n = parse_model_v2(d)
    with pytest.raises(ModelLoadError):
        validate_expr_acyclic(n)

def test_validate_equation_targets_unknown_and_duplicate():
    d = base_doc()
    # unknown
    d["equations"] = {"rhs": {"x": "-x"}, "expr": "z = 0"}
    with pytest.raises(ModelLoadError):
        parse_model_v2(d)
    # duplicate across rhs/block
    d = base_doc()
    d["equations"] = {"rhs": {"x": "-x"}, "expr": "x = -x"}
    with pytest.raises(ModelLoadError):
        parse_model_v2(d)

def test_validate_event_legality_only_states_or_params():
    d = base_doc()
    d["events"] = {
        "ok": {"phase": "post", "cond": "1", "action.x": "0"},
        "bad": {"phase": "post", "cond": "1", "action.zz": "0"},
    }
    n = parse_model_v2(d)
    # first passes, second fails
    with pytest.raises(ModelLoadError):
        validate_event_legality(n)

def test_validate_dtype_rules_ode_requires_float():
    d = base_doc()
    d["model"]["dtype"] = "int32"
    with pytest.raises(ModelLoadError):
        parse_model_v2(d)

def test_validate_functions_signature_rejects_non_ident_or_duplicate_args():
    d = base_doc()
    d["functions"] = {"f": {"args": ["x", "x"], "expr": "x"}}
    with pytest.raises(ModelLoadError):
        validate_functions_signature(parse_model_v2(d))
    d = base_doc()
    d["functions"] = {"f": {"args": ["not ident!"], "expr": "1"}}
    with pytest.raises(ModelLoadError):
        validate_functions_signature(parse_model_v2(d))
