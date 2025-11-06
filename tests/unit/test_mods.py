# tests/unit/test_mods.py
import pytest

from dynlib.errors import ModelLoadError
from dynlib.dsl.parser import parse_model_v2
from dynlib.compiler.mods import ModSpec, apply_mods_v2

def base_normal():
    # normal = output of parse_model_v2
    doc = {
        "model": {"type": "ode", "dtype": "float64"},
        "states": {"x": 1.0, "u": 0.0},
        "params": {"a": 1.0, "b": 2.0},
        "equations": {"rhs": {"x": "-a*x", "u": "x - b*u"}, "expr": None},
        "aux": {},
        "functions": {},
        "events": {
            "tick": {"phase": "post", "cond": "1", "action": "x = x", "log": []}
        },
        "sim": {},
    }
    return parse_model_v2(doc)

def test_mods_remove_replace_add_set_and_ordering():
    normal = base_normal()

    mods = [
        # remove existing event
        ModSpec(name="rm", remove={"events": {"names": ["tick"]}}),
        # add two events
        ModSpec(name="add1", add={"events": {"e1": {"phase": "pre", "cond": "1", "action": "x = 0"}}}),
        ModSpec(name="add2", add={"events": {"e2": {"phase": "post", "cond": "1", "action.dx": "1"}}}),
        # replace e2
        ModSpec(name="repl", replace={"events": {"e2": {"phase": "post", "cond": "x>0", "action.dx": "2"}}}),
        # set values
        ModSpec(name="setv", set={"states": {"x": 5.0}, "params": {"a": 3.0}}),
    ]

    out = apply_mods_v2(normal, mods)

    # removed tick
    names = [e["name"] for e in out["events"]]
    assert "tick" not in names
    # added e1, e2, and e2 replaced
    assert set(names) == {"e1", "e2"}
    e2 = next(e for e in out["events"] if e["name"] == "e2")
    assert e2["phase"] == "post" and e2["cond"] == "x>0"
    assert e2["action_keyed"] == {"dx": "2"} and e2["action_block"] is None
    # set applied
    assert out["states"]["x"] == 5.0
    assert out["params"]["a"] == 3.0

def test_mods_replace_nonexistent_raises():
    normal = base_normal()
    with pytest.raises(ModelLoadError):
        apply_mods_v2(normal, [ModSpec(name="bad", replace={"events": {"nope": {"phase": "pre", "cond": "1", "action": "x=0"}}})])

def test_mods_add_duplicate_raises():
    normal = base_normal()
    mods = [
        ModSpec(name="add1", add={"events": {"e": {"phase": "pre", "cond": "1", "action": "x=0"}}}),
        ModSpec(name="add2", add={"events": {"e": {"phase": "post", "cond": "1", "action.dx": "1"}}}),
    ]
    with pytest.raises(ModelLoadError):
        apply_mods_v2(normal, mods)

def test_mods_group_exclusive_priority_and_stable_order():
    normal = base_normal()
    # Same group, both exclusive → choose lowest priority then name
    m1 = ModSpec(name="B", group="G", exclusive=True, priority=10, add={"events": {"eB": {"phase": "pre", "cond": "1", "action": "x=0"}}})
    m2 = ModSpec(name="A", group="G", exclusive=True, priority=5, add={"events": {"eA": {"phase": "pre", "cond": "1", "action": "x=0"}}})
    # Passthrough before grouped
    p0 = ModSpec(name="P0", add={"events": {"p": {"phase": "pre", "cond": "1", "action": "x=0"}}})
    out = apply_mods_v2(normal, [p0, m1, m2])

    names = [e["name"] for e in out["events"]]
    # p applied, and only m2 (priority 5) chosen from the group
    assert "p" in names and "eA" in names and "eB" not in names
