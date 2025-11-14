import pytest

pytest.importorskip("numba")

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build


def _model_doc(cond: str):
    return {
        "model": {"type": "ode", "dtype": "float64"},
        "states": {"x": 0.0},
        "params": {},
        "equations": {"rhs": {"x": "0.0"}},
        "aux": {},
        "functions": {},
        "events": {
            "watch": {
                "phase": "post",
                "cond": cond,
                "action.x": "x",
            }
        },
        "sim": {},
    }


@pytest.mark.parametrize(
    "cond,uses_lag",
    [
        ("cross_up(x, 0.1)", True),
        ("cross_down(x, 0.1)", True),
        ("cross_either(x, 0.1)", True),
        ("changed(x)", True),
        ("in_interval(x, -1.0, 1.0)", False),
        ("enters_interval(x, -1.0, 1.0)", True),
        ("leaves_interval(x, -1.0, 1.0)", True),
        ("increasing(x)", True),
        ("decreasing(x)", True),
    ],
)
def test_event_macros_build_with_jit(cond, uses_lag):
    doc = _model_doc(cond)
    spec = build_spec(parse_model_v2(doc))
    full = build(spec, stepper="euler", jit=True)
    assert full.events_post is not None
    assert full.uses_lag == uses_lag
