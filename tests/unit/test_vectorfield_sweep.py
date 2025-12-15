import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import tomllib

from dynlib.plot.vectorfield import VectorFieldHandle, vectorfield_sweep
from dynlib.compiler.build import build
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec


def _linear_model():
    toml_str = """
[model]
type = "ode"
stepper = "rk4"

[sim]
t0 = 0.0
dt = 0.1

[states]
x = 0.0
y = 0.0
z = 1.0

[params]
a = 1.0
b = 2.0

[equations.rhs]
x = "a * x + z"
y = "b * y"
z = "0.0"
"""
    data = tomllib.loads(toml_str)
    spec = build_spec(parse_model_v2(data))
    return build(spec, jit=False, disk_cache=False)


def test_vectorfield_sweep_shared_norm_and_titles():
    model = _linear_model()
    res = vectorfield_sweep(
        model,
        param="a",
        values=[0.5, 1.0, 1.5],
        xlim=(-1, 1),
        ylim=(-1, 1),
        grid=(3, 2),
        speed_color=True,
        share_speed_norm=True,
        cols=2,
        facet_titles="a={value}",
        interactive=False,
    )
    assert len(res.handles) == 3
    assert all(isinstance(h, VectorFieldHandle) for h in res.handles)
    norms = {h.quiver.norm for h in res.handles if h.quiver is not None}
    assert len(norms) == 1
    assert res.colorbar is not None
    assert res.axes[0].get_title()
    plt.close(res.figure)


def test_vectorfield_sweep_accepts_custom_sweep_and_shared_axes():
    model = _linear_model()
    sweep = {
        "base": {"params": {"a": 1.0}},
        "offset": {"params": {"a": 2.0}, "fixed": {"z": 2.0}},
    }
    res = vectorfield_sweep(
        model,
        sweep=sweep,
        xlim=(-1, 1),
        ylim=(0, 1),
        grid=(3, 2),
        normalize=False,
        share_speed_norm=False,
        cols=2,
        sharex=True,
        sharey=True,
        interactive=False,
    )
    assert len(res.handles) == 2
    assert res.speed_norm is None
    assert res.axes[0].get_shared_x_axes().joined(res.axes[0], res.axes[1])
    h0, h1 = res.handles
    assert not np.allclose(h0.U, h1.U)
    plt.close(res.figure)
