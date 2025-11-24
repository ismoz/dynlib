# src/tests/unit/test_sum_generator_lowering.py
import tomllib
import numpy as np
import pytest

from dynlib.compiler.build import build_callables
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.runtime.workspace import make_runtime_workspace


MODEL_SUM = "inline:" + """
[model]
type = "ode"
dtype = "float64"

[states]
x = 0.0

[params]
N = 4

[equations.rhs]
x = "sum(i*i for i in range(N))"
"""

MODEL_TERNARY = "inline:" + """
[model]
type = "ode"
dtype = "float64"

[states]
x = 0.0

[params]
N = 3

[equations.rhs]
x = "1.0 if t < 0 else sum(i for i in range(N))"
"""

MODEL_PROD = "inline:" + """
[model]
type = "ode"
dtype = "float64"

[states]
x = 0.0

[params]
N = 4

[equations.rhs]
x = "prod((i+1) for i in range(N))"
"""


def _build_spec(model_src: str):
    text = model_src[len("inline:") :]
    doc = tomllib.loads(text)
    normal = parse_model_v2(doc)
    return build_spec(normal)


def _workspace(dtype):
    return make_runtime_workspace(lag_state_info=None, dtype=dtype)


def test_sum_generator_rhs_matches_python_and_jit():
    spec = _build_spec(MODEL_SUM)
    cp_py = build_callables(spec, stepper_name="euler", jit=False, dtype="float64")
    cp_jit = build_callables(spec, stepper_name="euler", jit=True, dtype="float64")

    y = np.array([0.0], dtype=np.float64)
    params = np.array([4.0], dtype=np.float64)
    dy_py = np.zeros_like(y)
    dy_jit = np.zeros_like(y)
    runtime_ws = _workspace(np.float64)

    cp_py.rhs(0.0, y, dy_py, params, runtime_ws)
    cp_jit.rhs(0.0, y, dy_jit, params, runtime_ws)

    expected = sum(i * i for i in range(int(params[0])))
    assert dy_py[0] == pytest.approx(expected)
    assert dy_jit[0] == pytest.approx(expected)


def test_sum_generator_inside_ternary_branch():
    spec = _build_spec(MODEL_TERNARY)
    cp_py = build_callables(spec, stepper_name="euler", jit=False, dtype="float64")
    cp_jit = build_callables(spec, stepper_name="euler", jit=True, dtype="float64")

    y = np.array([0.0], dtype=np.float64)
    params = np.array([3.0], dtype=np.float64)
    runtime_ws = _workspace(np.float64)

    dy_neg_py = np.zeros_like(y)
    dy_neg_jit = np.zeros_like(y)
    cp_py.rhs(-1.0, y, dy_neg_py, params, runtime_ws)
    cp_jit.rhs(-1.0, y, dy_neg_jit, params, runtime_ws)
    assert dy_neg_py[0] == pytest.approx(1.0)
    assert dy_neg_jit[0] == pytest.approx(1.0)

    dy_pos_py = np.zeros_like(y)
    dy_pos_jit = np.zeros_like(y)
    cp_py.rhs(1.0, y, dy_pos_py, params, runtime_ws)
    cp_jit.rhs(1.0, y, dy_pos_jit, params, runtime_ws)
    expected = sum(i for i in range(int(params[0])))
    assert dy_pos_py[0] == pytest.approx(expected)
    assert dy_pos_jit[0] == pytest.approx(expected)


def test_prod_generator_rhs_matches_python_and_jit():
    spec = _build_spec(MODEL_PROD)
    cp_py = build_callables(spec, stepper_name="euler", jit=False, dtype="float64")
    cp_jit = build_callables(spec, stepper_name="euler", jit=True, dtype="float64")

    y = np.array([0.0], dtype=np.float64)
    params = np.array([4.0], dtype=np.float64)
    dy_py = np.zeros_like(y)
    dy_jit = np.zeros_like(y)
    runtime_ws = _workspace(np.float64)

    cp_py.rhs(0.0, y, dy_py, params, runtime_ws)
    cp_jit.rhs(0.0, y, dy_jit, params, runtime_ws)

    expected = 1.0
    for i in range(int(params[0])):
        expected *= (i + 1)
    assert dy_py[0] == pytest.approx(expected)
    assert dy_jit[0] == pytest.approx(expected)
