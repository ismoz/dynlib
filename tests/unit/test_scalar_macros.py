import numpy as np
import pytest

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build_callables


def test_scalar_macros_available_in_all_sections():
    doc = {
        "model": {"type": "ode", "dtype": "float64"},
        "states": {"x": -0.5},
        "params": {},
        "equations": {
            "rhs": {
                "x": "relu(x) + drive + gate - clip(t, -1.0, 0.5)",
            }
        },
        "aux": {
            "drive": "sign(x)",
            "gate": "heaviside(t - 0.25) - step(t - 0.75)",
        },
        "functions": {},
        "events": {
            "nudge": {
                "phase": "post",
                "cond": "approx(t, 0.0, 1e-9)",
                "action.x": "clip(x + relu(-x), -1.0, 1.0)",
            }
        },
        "sim": {},
    }

    spec = build_spec(parse_model_v2(doc))
    callables = build_callables(spec, stepper_name="euler", jit=False, dtype="float64")

    y = np.array([-0.5], dtype=np.float64)
    dy = np.zeros_like(y)
    params = np.zeros(0, dtype=np.float64)
    ss = np.zeros(0, dtype=np.float64)
    iw0 = np.zeros(0, dtype=np.int32)

    callables.rhs(0.0, y, dy, params, ss, iw0)
    assert dy[0] == pytest.approx(-1.0)

    scratch = np.zeros(1, dtype=np.float64)
    code, log_len = callables.events_post(0.0, y, params, scratch, ss, iw0)
    assert code == 0
    assert log_len == 0
    assert y[0] == pytest.approx(0.0)
