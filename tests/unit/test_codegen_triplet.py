# src/tests/unit/test_codegen_triplet.py
import os
import math
import tomllib
import numpy as np
import pytest

from dynlib.compiler.build import build_callables
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.runtime.workspace import make_runtime_workspace

MODEL_SRC = "inline:" + """
[model]
type = "ode"
dtype = "float64"

[states]
x = 1.0

[params]
a = 2.0

[equations.rhs]
x = "-a*x"

[events.bump]
phase = "post"
cond  = "t >= 0.0"
action.x = "x + 1.0"

[aux]
E = "0.5*a*x^2"
"""

def _build_spec():
    src = MODEL_SRC
    assert src.startswith("inline:")
    text = src[len("inline:"):]
    doc = tomllib.loads(text)
    # Normalize TOML → dict, then build ModelSpec for codegen
    normal = parse_model_v2(doc)      # dict
    spec = build_spec(normal)         # ModelSpec
    return spec

def test_rhs_eval_jit_parity():
    spec = _build_spec()
    cp0 = build_callables(spec, stepper_name="euler", jit=False, dtype="float64")
    cp1 = build_callables(spec, stepper_name="euler", jit=True,  dtype="float64")

    y = np.array([1.0], dtype=np.float64)
    p = np.array([2.0], dtype=np.float64)
    dy0 = np.zeros_like(y)
    dy1 = np.zeros_like(y)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)

    t = 0.0
    cp0.rhs(t, y, dy0, p, runtime_ws)
    cp1.rhs(t, y, dy1, p, runtime_ws)

    # dx/dt = -a*x = -2*1 = -2
    assert dy0.shape == (1,)
    assert dy1.shape == (1,)
    assert dy0[0] == pytest.approx(-2.0)
    assert dy1[0] == pytest.approx(-2.0)
    assert dy0[0] == pytest.approx(dy1[0])

def test_events_only_mutate_states_params_and_effect_is_visible():
    spec = _build_spec()
    cp = build_callables(spec, stepper_name="euler", jit=False, dtype="float64")

    y = np.array([1.0], dtype=np.float64)
    p = np.array([2.0], dtype=np.float64)
    scratch = np.zeros(1, dtype=np.float64)  # event log scratch buffer
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)

    # pre does nothing; post adds +1.0 to x
    pre_code, pre_log = cp.events_pre(0.0, y, p, scratch, runtime_ws)
    assert pre_code == -1  # no event fired
    assert y[0] == pytest.approx(1.0)
    post_code, post_log = cp.events_post(0.0, y, p, scratch, runtime_ws)
    assert post_code == 0  # event fired (always fires since cond is implicit True)
    assert post_log == 0  # no log items
    assert y[0] == pytest.approx(2.0)

def test_aux_is_recomputed_inside_rhs_every_call():
    spec = _build_spec()
    cp = build_callables(spec, stepper_name="euler", jit=False, dtype="float64")

    y = np.array([3.0], dtype=np.float64)
    p = np.array([2.0], dtype=np.float64)

    dy = np.zeros_like(y)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=np.float64)
    cp.rhs(0.0, y, dy, p, runtime_ws)
    assert dy[0] == pytest.approx(-6.0)

    # Change y; aux depends on x and must be recomputed
    y[0] = 4.0
    cp.rhs(0.0, y, dy, p, runtime_ws)
    assert dy[0] == pytest.approx(-8.0)
