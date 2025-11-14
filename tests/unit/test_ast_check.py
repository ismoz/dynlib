# tests/unit/test_ast_check.py
import pytest

from dynlib.errors import ModelLoadError
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.astcheck import (
    collect_names,
    collect_lag_requests,
    validate_expr_acyclic,
    validate_event_legality,
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
    """
    Equation target validation (unknown states, duplicates) is now done
    in schema.py:validate_name_collisions() during parse_model_v2().
    This test verifies that the parser catches these errors.
    """
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
    """
    Dtype validation is now done in schema.py:validate_model_header()
    during parse_model_v2(). This test verifies the parser catches it.
    """
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


def test_collect_lag_requests_accumulates_max_depths_across_sections():
    d = base_doc()
    d["equations"]["rhs"]["x"] = "lag_x(2) + lag_u()"
    d["aux"] = {"z": "lag_u(5) - lag_x()"}
    d["events"] = {
        "kick": {
            "phase": "post",
            "cond": "lag_x(4) > 0",
            "action.x": "x",
        }
    }
    n = parse_model_v2(d)
    lags = collect_lag_requests(n)
    assert lags == {"x": 4, "u": 5}


def test_collect_lag_requests_rejects_non_state_targets():
    d = base_doc()
    d["equations"]["rhs"]["x"] = "lag_a(2)"
    n = parse_model_v2(d)
    with pytest.raises(ModelLoadError, match="not a declared state"):
        collect_lag_requests(n)


def test_collect_lag_requests_detects_macro_usage():
    d = base_doc()
    d["events"] = {
        "edge": {
            "phase": "post",
            "cond": "cross_up(x, 0.0)",
            "action.x": "x",
        }
    }
    n = parse_model_v2(d)
    assert collect_lag_requests(n) == {"x": 1}


def test_collect_lag_requests_macro_requires_state():
    d = base_doc()
    d["events"] = {
        "edge": {
            "phase": "post",
            "cond": "cross_up(a, 0.0)",
            "action.x": "x",
        }
    }
    n = parse_model_v2(d)
    with pytest.raises(ModelLoadError, match="declared state"):
        collect_lag_requests(n)
