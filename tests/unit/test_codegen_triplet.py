# src/tests/unit/test_codegen_triplet.py
import os
import math
import tomllib
import numpy as np
import pytest

from dynlib.dsl.parser import parse_model_v2
from dynlib.compiler.build import build_callables
from dynlib.dsl.spec import build_spec

# --- ensure a stepper is registered (we only need its StructSpec sizes for cache key) ---
# We do this here so Slice 3 can run without Slice 4's real euler stepper.
try:
    from dynlib.steppers.registry import register, get_stepper
    from dynlib.steppers.base import StructSpec
    _REG_OK = True
except Exception:
    _REG_OK = False

if _REG_OK:
    class _DummyEuler:
        class _Meta:
            name = "euler"
            kind = "ode"
            time_control = "fixed"
            scheme = "explicit"
            geometry = frozenset()
            family = "probe"
            order = 1
            embedded_order = None
            dense_output = False
            stiff_ok = False
            aliases = ()

        def __init__(self):
            self.meta = self._Meta()

        def struct_spec(self) -> StructSpec:
            # All zeros are fine for cache signature in Slice 3
            return StructSpec(
                sp_size=0, ss_size=0,
                sw0_size=0, sw1_size=0, sw2_size=0, sw3_size=0,
                iw0_size=0, bw0_size=0,
                use_history=False, use_f_history=False,
                dense_output=False, needs_jacobian=False,
                embedded_order=None, stiff_ok=False
            )

        def emit(self, rhs_fn, struct: StructSpec):
            # Not used in Slice 3 tests
            raise RuntimeError("emit() not used in Slice 3 tests")

    try:
        get_stepper("euler")
    except Exception:
        register(_DummyEuler())

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
    cp0 = build_callables(spec, stepper_name="euler", jit=False, model_dtype="float64")
    cp1 = build_callables(spec, stepper_name="euler", jit=True,  model_dtype="float64")

    y = np.array([1.0], dtype=np.float64)
    p = np.array([2.0], dtype=np.float64)
    dy0 = np.zeros_like(y)
    dy1 = np.zeros_like(y)

    t = 0.0
    cp0.rhs(t, y, dy0, p)
    cp1.rhs(t, y, dy1, p)

    # dx/dt = -a*x = -2*1 = -2
    assert dy0.shape == (1,)
    assert dy1.shape == (1,)
    assert dy0[0] == pytest.approx(-2.0)
    assert dy1[0] == pytest.approx(-2.0)
    assert dy0[0] == pytest.approx(dy1[0])

def test_events_only_mutate_states_params_and_effect_is_visible():
    spec = _build_spec()
    cp = build_callables(spec, stepper_name="euler", jit=False, model_dtype="float64")

    y = np.array([1.0], dtype=np.float64)
    p = np.array([2.0], dtype=np.float64)

    # pre does nothing; post adds +1.0 to x
    pre_code = cp.events_pre(0.0, y, p)
    assert pre_code == -1  # non-recording event -> sentinel
    assert y[0] == pytest.approx(1.0)
    post_code = cp.events_post(0.0, y, p)
    assert post_code == -1  # non-recording event -> sentinel
    assert y[0] == pytest.approx(2.0)

def test_aux_is_recomputed_inside_rhs_every_call():
    spec = _build_spec()
    cp = build_callables(spec, stepper_name="euler", jit=False, model_dtype="float64")

    y = np.array([3.0], dtype=np.float64)
    p = np.array([2.0], dtype=np.float64)

    dy = np.zeros_like(y)
    cp.rhs(0.0, y, dy, p)
    assert dy[0] == pytest.approx(-6.0)

    # Change y; aux depends on x and must be recomputed
    y[0] = 4.0
    cp.rhs(0.0, y, dy, p)
    assert dy[0] == pytest.approx(-8.0)
