import copy
import pytest

from dynlib.errors import ModelLoadError
from dynlib.dsl.schema import validate_model_header, validate_tables, validate_name_collisions
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.astcheck import validate_functions_signature


# -------- fixtures -----------------------------------------------------------

def minimal_doc(**over):
    doc = {
        "model": {"type": "ode", "label": "m"},
        "states": {"x": 1.0, "u": 0.0},
        "params": {"a": 1.0, "b": 2.0},
        "equations": {"rhs": {"x": "-a*x", "u": "x - b*u"}},
        "aux": {},
        "functions": {},
        "events": {},
        "sim": {"t0": 0.0, "t_end": 1.0, "dt": 1e-2, "stepper": "rk4", "record": True},
    }
    doc.update(over)
    return doc


# -------- schema.validate_model_header / validate_tables ---------------------

def test_validate_model_header_ok_and_default_dtype():
    d = minimal_doc()
    # remove dtype to trigger defaulting path
    del d["model"]["label"]
    validate_model_header(d)
    # ensure tables validation doesn’t raise
    validate_tables(d)

def test_validate_model_header_bad_type():
    d = minimal_doc()
    d["model"]["type"] = "weird"
    with pytest.raises(ModelLoadError):
        validate_model_header(d)

def test_validate_model_header_ode_requires_float_dtype():
    d = minimal_doc()
    d["model"]["dtype"] = "int32"
    with pytest.raises(ModelLoadError):
        validate_model_header(d)

def test_validate_tables_missing_states():
    d = minimal_doc()
    del d["states"]
    with pytest.raises(ModelLoadError):
        validate_tables(d)

def test_validate_tables_equations_shapes():
    d = minimal_doc()
    d["equations"] = "not a table"
    with pytest.raises(ModelLoadError):
        validate_tables(d)
    d = minimal_doc(equations={"rhs": "nope"})
    with pytest.raises(ModelLoadError):
        validate_tables(d)
    d = minimal_doc(equations={"rhs": {"x": "1"}, "expr": 123})
    with pytest.raises(ModelLoadError):
        validate_tables(d)

# -------- schema.validate_name_collisions ------------------------------------

def test_validate_name_collisions_duplicate_between_rhs_and_block():
    d = minimal_doc(equations={"rhs": {"x": "-x"}, "expr": "x = -x\nu = x"})
    with pytest.raises(ModelLoadError):
        validate_name_collisions(d)

def test_validate_name_collisions_unknown_targets():
    d = minimal_doc(equations={"rhs": {"z": "1"}, "expr": None})
    with pytest.raises(ModelLoadError):
        validate_name_collisions(d)

def test_validate_name_collisions_ok():
    d = minimal_doc(equations={"rhs": {"x": "-x"}, "expr": "u = x"})
    # both targets exist in [states], and no duplicates
    validate_name_collisions(d)

# -------- parser.parse_model_v2 ----------------------------------------------

def test_parse_model_v2_roundtrip_shapes_and_defaults():
    d = minimal_doc()
    # delete dtype to check defaulting to float64
    d["model"].pop("dtype", None)
    normal = parse_model_v2(d)

    assert normal["model"]["type"] == "ode"
    assert normal["model"]["dtype"] == "float64"  # default
    assert list(normal["states"].keys()) == ["x", "u"]                     # order preserved
    assert list(normal["params"].keys()) == ["a", "b"]                     # order preserved
    assert normal["equations"]["rhs"] == {"x": "-a*x", "u": "x - b*u"}     # copied
    assert normal["equations"]["expr"] is None
    assert normal["equations"]["jacobian"] is None
    assert normal["equations"]["inverse"] is None
    assert normal["events"] == []                                           # table→list normalized
    assert isinstance(normal["functions"], dict)
    assert isinstance(normal["aux"], dict)
    assert isinstance(normal["sim"], dict)

def test_parse_model_v2_functions_validation_and_normalization():
    d = minimal_doc(functions={"f1": {"args": ["x", "y"], "expr": "x+y"}})
    normal = parse_model_v2(d)
    assert normal["functions"]["f1"]["args"] == ["x", "y"]
    assert normal["functions"]["f1"]["expr"] == "x+y"

def test_parse_model_v2_functions_reject_bad_args():
    # Parser accepts list[str]; identifier/uniqueness rules are enforced by astcheck
    d = minimal_doc(functions={"bad": {"args": ["x", "not ident!"], "expr": "1"}})
    n = parse_model_v2(d)
    with pytest.raises(ModelLoadError):
        validate_functions_signature(n)

def test_parse_model_v2_functions_reject_duplicate_args_via_astcheck():
    d = minimal_doc(functions={"dup": {"args": ["x", "x"], "expr": "x"}})
    n = parse_model_v2(d)
    with pytest.raises(ModelLoadError):
        validate_functions_signature(n)

def test_parse_model_v2_events_keyed_and_block_forms():
    d1 = minimal_doc(events={
        "tick": {
            "phase": "post", "cond": "1",
            "action.dx": "1", "action.u": "0",
            "log": ["t", "x", "u"]
        }
    })
    n1 = parse_model_v2(d1)
    assert n1["events"][0]["name"] == "tick"
    assert n1["events"][0]["action_keyed"] == {"dx": "1", "u": "0"}
    assert n1["events"][0]["action_block"] is None
    assert n1["events"][0]["log"] == ["t", "x", "u"]

    d2 = minimal_doc(events={
        "tick": {
            "phase": "pre", "cond": "x>0",
            "action": "u = 0\nx = x",
        }
    })
    n2 = parse_model_v2(d2)
    assert n2["events"][0]["action_keyed"] is None
    assert "u = 0" in n2["events"][0]["action_block"]

def test_parse_model_v2_events_default_phase():
    d = minimal_doc(events={
        "tick": {
            "cond": "1",
            "action.dx": "1",
        }
    })
    n = parse_model_v2(d)
    assert n["events"][0]["phase"] == "post"

def test_parse_model_v2_events_validation_errors():
    # bad phase
    d = minimal_doc(events={"e": {"phase": "now", "cond": "1", "action": "x=0"}})
    with pytest.raises(ModelLoadError):
        parse_model_v2(d)
    # bad cond
    d = minimal_doc(events={"e": {"phase": "pre", "cond": 3.14, "action": "x=0"}})
    with pytest.raises(ModelLoadError):
        parse_model_v2(d)
    # bad log
    d = minimal_doc(events={"e": {"phase": "pre", "cond": "1", "action": "x=0", "log": [1, 2]}})
    with pytest.raises(ModelLoadError):
        parse_model_v2(d)
