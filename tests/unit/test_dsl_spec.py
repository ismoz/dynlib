# tests/unit/test_dsl_spec.py
import pytest

from dynlib.errors import ModelLoadError
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec, compute_spec_hash

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
    assert spec.equations_rhs == {"x": "-a*x", "u": "x - u"}
    assert spec.equations_block is None
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
