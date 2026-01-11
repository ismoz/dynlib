# tests/unit/test_inverse_equations.py
from __future__ import annotations

import numpy as np
import pytest

from dynlib.compiler.build import build
from dynlib.runtime.workspace import make_runtime_workspace


def _build_inverse_map_model(*, inverse_form: str, jit: bool):
    base = """
    [model]
    type = "map"

    [states]
    x = 1.0

    [params]
    a = 0.5

    [equations.rhs]
    x = "x + a"
    """
    if inverse_form == "block":
        inverse_section = """
        [equations.inverse]
        expr = \"\"\"
        x = x - a
        \"\"\"
        """
    else:
        inverse_section = """
        [equations.inverse.rhs]
        x = "x - a"
        """
    model_src = f"{base}{inverse_section}"

    return build(f"inline:{model_src}", stepper="map", jit=jit, disk_cache=False)


def _eval_inv(model, x_value: float, a_value: float) -> float:
    y = np.array([x_value], dtype=model.dtype)
    params = np.array([a_value], dtype=model.dtype)
    out = np.zeros_like(y)
    runtime_ws = make_runtime_workspace(lag_state_info=None, dtype=model.dtype, n_aux=len(model.spec.aux))
    model.inv_rhs(0.0, y, out, params, runtime_ws)
    return float(out[0])


@pytest.mark.parametrize("inverse_form", ["rhs", "block"])
def test_inverse_map_rhs_python(inverse_form: str):
    model = _build_inverse_map_model(inverse_form=inverse_form, jit=False)
    assert callable(model.inv_rhs)
    assert _eval_inv(model, 2.0, 0.5) == pytest.approx(1.5)


@pytest.mark.parametrize("inverse_form", ["rhs", "block"])
def test_inverse_map_rhs_jit(inverse_form: str):
    pytest.importorskip("numba")
    model = _build_inverse_map_model(inverse_form=inverse_form, jit=True)
    assert callable(model.inv_rhs)
    assert _eval_inv(model, 2.0, 0.5) == pytest.approx(1.5)
