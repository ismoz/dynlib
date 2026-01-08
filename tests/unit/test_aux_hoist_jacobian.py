# tests/unit/test_aux_hoist_jacobian.py
import tomllib
import numpy as np
import pytest

from dynlib.compiler.codegen.emitter import emit_jacobian
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.runtime.workspace import make_runtime_workspace


MODEL_SRC = "inline:" + """
[model]
type = "ode"
dtype = "float64"

[states]
x = 0.0

[aux]
b = "a + 1"     # depends on a (declared later on purpose)
a = "x + 2"

[equations]
expr = "dx = b"

[equations.jacobian]
expr = [
    ["1.0"]
]
"""


def _build_spec():
    text = MODEL_SRC[len("inline:"):]
    doc = tomllib.loads(text)
    normal = parse_model_v2(doc)
    return build_spec(normal)


def test_jacobian_respects_aux_dependencies_when_hoisted():
    """
    Regression: aux hoisting must honor dependencies even when aux are declared
    out of order. Here b depends on a, but is declared first.
    """
    spec = _build_spec()
    compiled = emit_jacobian(spec)
    assert compiled is not None and compiled.jacobian is not None

    jac_fn = compiled.jacobian
    y = np.array([0.0], dtype=np.float64)
    params = np.zeros((0,), dtype=np.float64)
    J = np.zeros((1, 1), dtype=np.float64)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64, n_aux=len(spec.aux or {}))

    jac_fn(0.0, y, params, J, runtime_ws)
    # dx/dx = d/dx (b) = d/dx (x + 3) = 1
    assert J[0, 0] == pytest.approx(1.0)
