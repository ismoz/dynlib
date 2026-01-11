# tests/unit/test_dsl_spec.py
import math

import numpy as np
import pytest

from dynlib.errors import ModelLoadError
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec, compute_spec_hash
from dynlib.compiler.codegen.emitter import emit_rhs_and_events

def minimal_doc():
    return {
        "model": {"type": "ode", "dtype": "double", "label": "L"},  # alias should canon to float64
        "states": {"x": 1.0, "u": 0.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a*x", "u": "x - u"}, "expr": None},
        "aux": {"z": "x+u"},
        "functions": {"f": {"args": ["q"], "expr": "q+1"}},
        "events": {"tick": {"phase": "post", "cond": "1", "action": "u = 0", "log": ["t", "x"]}},
        "sim": {"t0": 0.0, "t_end": 2.0, "dt": 0.1, "stepper": "rk4", "record": False},
    }

def test_build_spec_and_hash_determinism_and_dtype_alias():
    n = parse_model_v2(minimal_doc())
    spec = build_spec(n)

    assert spec.kind == "ode"
    assert spec.dtype == "float64"          # canon from "double"
    assert spec.states == ("x", "u")
    assert spec.state_ic == (1.0, 0.0)
    assert spec.params == ("a",)
    assert spec.param_vals == (1.0,)
    assert spec.constants == ()
    assert spec.constant_vals == ()
    assert spec.equations_rhs == {"x": "-a*x", "u": "x - u"}
    assert spec.equations_block is None
    assert spec.inverse_rhs is None
    assert spec.inverse_block is None
    assert spec.aux == {"z": "x+u"}
    assert "f" in spec.functions and spec.functions["f"] == (["q"], "q+1")
    assert len(spec.events) == 1 and spec.events[0].name == "tick"
    assert spec.sim.t0 == 0.0 and spec.sim.t_end == 2.0 and spec.sim.dt == 0.1
    assert spec.sim.stepper == "rk4" and spec.sim.record is False

    h1 = compute_spec_hash(spec)
    h2 = compute_spec_hash(spec)
    assert h1 == h2                       # deterministic

    # Changing a single value must change the hash
    n2 = parse_model_v2(minimal_doc())
    n2["params"]["a"] = 2.0
    spec2 = build_spec(n2)
    assert compute_spec_hash(spec2) != h1


def test_numeric_strings_for_states_and_params():
    n = minimal_doc()
    n["states"]["x"] = "1/2"
    n["params"]["a"] = "8/4"
    parsed = parse_model_v2(n)
    spec = build_spec(parsed)
    assert spec.state_ic[0] == 0.5
    assert spec.param_vals == (2.0,)


def test_aux_name_reserved():
    n = minimal_doc()
    n["aux"] = {"t": "x + 1", "pi": "1"}
    with pytest.raises(ModelLoadError, match=r"reserved"):
        build_spec(parse_model_v2(n))


def test_invalid_numeric_string_raises():
    n = minimal_doc()
    n["params"]["a"] = "foo"
    with pytest.raises(ModelLoadError):
        parse_model_v2(n)


def test_sim_unknown_keys_preserved_and_hash_changes():
    n = minimal_doc()
    n["sim"]["safety"] = 0.9
    n_parsed = parse_model_v2(n)
    spec = build_spec(n_parsed)
    assert hasattr(spec.sim, "safety")
    assert spec.sim.safety == pytest.approx(0.9)

    n_alt = minimal_doc()
    n_alt["sim"]["safety"] = 0.95
    spec_alt = build_spec(parse_model_v2(n_alt))
    assert spec_alt.sim.safety == pytest.approx(0.95)
    assert compute_spec_hash(spec_alt) != compute_spec_hash(spec)


def test_builtin_constants_in_numeric_strings_and_dtype_cast():
    n = minimal_doc()
    n["model"]["dtype"] = "float32"
    n["states"]["x"] = "2*pi"
    n["params"]["a"] = "e/2"
    parsed = parse_model_v2(n)
    spec = build_spec(parsed)

    expected_state = np.dtype("float32").type(2 * math.pi).item()
    expected_param = np.dtype("float32").type(math.e / 2).item()
    assert spec.state_ic[0] == pytest.approx(expected_state, rel=1e-6)
    assert spec.param_vals[0] == pytest.approx(expected_param, rel=1e-6)


def test_reserved_constants_not_allowed_as_identifiers():
    n = minimal_doc()
    n["states"] = {"x": 1.0, "u": 0.0, "pi": 0.0}
    with pytest.raises(ModelLoadError, match="reserved"):
        build_spec(parse_model_v2(n))

    n = minimal_doc()
    n["params"] = {"a": 1.0, "pi": 1.0}
    with pytest.raises(ModelLoadError, match="reserved"):
        build_spec(parse_model_v2(n))

    n = minimal_doc()
    n["functions"] = {"pi": {"args": ["x"], "expr": "x"}}
    with pytest.raises(ModelLoadError, match="reserved"):
        build_spec(parse_model_v2(n))

def test_constants_may_reference_prior_constants():
    n = minimal_doc()
    n["constants"] = {"kb": 2.0, "kb2": "2*kb"}
    parsed = parse_model_v2(n)
    spec = build_spec(parsed)
    assert spec.constants == ("kb", "kb2")
    assert spec.constant_vals == (2.0, 4.0)


def test_constant_name_conflicts_raise():
    n = minimal_doc()
    n["constants"] = {"kb": 1.0}
    n["params"]["kb"] = 2.0
    with pytest.raises(ModelLoadError, match=r"kb is defined as a constant and cannot be reused as a parameter"):
        build_spec(parse_model_v2(n))

    n = minimal_doc()
    n["constants"] = {"kb": 1.0}
    n["states"]["kb"] = 0.0
    with pytest.raises(ModelLoadError, match=r"kb is defined as a constant and cannot be reused as a state"):
        build_spec(parse_model_v2(n))

    n = minimal_doc()
    n["constants"] = {"kb": 1.0}
    n["aux"]["kb"] = "1"
    with pytest.raises(ModelLoadError, match=r"kb is defined as a constant and cannot be reused as an auxiliary variable"):
        build_spec(parse_model_v2(n))


def test_user_constants_are_inlined_and_usable_in_numeric_strings():
    n = minimal_doc()
    n["constants"] = {"kb": 1.380649e-23, "qe": "2*e"}
    n["params"]["a"] = "kb * 2"
    n["aux"]["extra"] = "qe / kb"
    n["equations"]["rhs"]["x"] = "qe * x + kb"
    n["equations"]["rhs"]["u"] = "kb"

    parsed = parse_model_v2(n)
    spec = build_spec(parsed)
    compiled = emit_rhs_and_events(spec)

    assert spec.constants == ("kb", "qe")
    assert spec.constant_vals[0] == pytest.approx(1.380649e-23)
    assert spec.param_vals[0] == pytest.approx(2 * 1.380649e-23)
    assert "kb" not in compiled.rhs_source and "qe" not in compiled.rhs_source
    assert "kb" not in compiled.update_aux_source and "qe" not in compiled.update_aux_source
    assert "1.380649e-23" in compiled.rhs_source


def test_builtin_constants_are_inlined_in_codegen():
    n = minimal_doc()
    n["equations"]["rhs"]["x"] = "pi * x + e"
    n["aux"]["extra"] = "2 * pi"
    n["events"] = {
        "tick": {"phase": "post", "cond": "x > pi", "action.x": "e"}
    }
    parsed = parse_model_v2(n)
    spec = build_spec(parsed)
    compiled = emit_rhs_and_events(spec)

    for src in (compiled.rhs_source, compiled.update_aux_source, compiled.events_post_source):
        assert "pi" not in src
    assert "3.1415" in compiled.rhs_source
    assert "2.71828" in compiled.rhs_source


def test_state_order_is_authoritative_and_hashes_change():
    base = minimal_doc()
    parsed = parse_model_v2(base)
    spec = build_spec(parsed)

    reordered = minimal_doc()
    reordered["states"] = {"u": 0.0, "x": 1.0}
    parsed_reordered = parse_model_v2(reordered)
    spec_reordered = build_spec(parsed_reordered)

    assert spec.states == ("x", "u")
    assert spec_reordered.states == ("u", "x")
    assert compute_spec_hash(spec) != compute_spec_hash(spec_reordered)


def test_jacobian_matrix_roundtrip_and_validation():
    n = minimal_doc()
    n["equations"]["jacobian"] = {"expr": [["-a", "1"], ["x", "0"]]}
    spec = build_spec(parse_model_v2(n))
    assert spec.jacobian_exprs == (("-a", "1"), ("x", "0"))

    bad_shape = minimal_doc()
    bad_shape["equations"]["jacobian"] = {"expr": [["1"]]}
    with pytest.raises(ModelLoadError, match=r"must have 2 rows"):
        build_spec(parse_model_v2(bad_shape))

    bad_ident = minimal_doc()
    bad_ident["equations"]["jacobian"] = {"expr": [["unknown", "0"], ["0", "0"]]}
    with pytest.raises(ModelLoadError, match=r"Unknown identifier"):
        build_spec(parse_model_v2(bad_ident))
