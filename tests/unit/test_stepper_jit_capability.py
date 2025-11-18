# tests/unit/test_stepper_jit_capability.py
import importlib
import tomllib
import numpy as np
import pytest

from dynlib.compiler.build import build
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.errors import StepperJitCapabilityError
from dynlib.steppers.base import StepperMeta, StepperCaps

build_module = importlib.import_module("dynlib.compiler.build")


MODEL_SRC = """
[model]
type = "ode"
dtype = "float64"

[states]
x = 1.0

[params]
a = 2.0

[equations.rhs]
x = "-a * x"
"""


def _build_spec():
    doc = tomllib.loads(MODEL_SRC)
    normal = parse_model_v2(doc)
    return build_spec(normal)


class _DummyStepperSpec:
    def __init__(self):
        self.meta = StepperMeta(
            name="fake_no_jit",
            kind="ode",
            caps=StepperCaps(jit_capable=False),
        )

    def workspace_type(self):
        return None

    def make_workspace(self, n_state, dtype, model_spec=None):
        return None

    def config_spec(self):
        return None

    def default_config(self, model_spec=None):
        return None

    def pack_config(self, config) -> np.ndarray:
        return np.array([], dtype=np.float64)

    def emit(self, rhs_fn, model_spec=None):
        def _stepper(*args, **kwargs):
            return 0
        return _stepper


def test_build_rejects_stepper_that_cannot_be_jitted(monkeypatch):
    spec = _build_spec()
    dummy_stepper = _DummyStepperSpec()

    def _fake_get_stepper(name: str):
        assert name == dummy_stepper.meta.name
        return dummy_stepper

    monkeypatch.setattr(build_module, "get_stepper", _fake_get_stepper)

    with pytest.raises(StepperJitCapabilityError) as excinfo:
        build(spec, stepper=dummy_stepper.meta.name, jit=True, disk_cache=False)

    assert str(excinfo.value) == (
        "Stepper 'fake_no_jit' is not jit capable. "
        "Please choose another stepper or disable jit."
    )
