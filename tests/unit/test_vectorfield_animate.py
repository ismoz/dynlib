import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tomllib

from dynlib.plot.vectorfield import VectorFieldAnimation, vectorfield_animate
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


def test_vectorfield_animate_simple_sweep():
    model = _linear_model()
    anim = vectorfield_animate(
        model,
        param="a",
        values=[0.5, 1.0, 1.5],
        xlim=(-1, 1),
        ylim=(0, 1),
        grid=(3, 2),
        speed_color=True,
        fps=5,
        interactive=False,
    )
    assert isinstance(anim, VectorFieldAnimation)
    assert anim.animation.save_count == 3
    anim.animation._func(anim.frames[-1])
    assert np.isclose(anim.handle.last_params.get("a"), 1.5)
    plt.close(anim.figure)


def test_vectorfield_animate_fixed_func():
    model = _linear_model()
    fixed_fn = lambda v, idx: {"z": 2.0 + idx}
    anim = vectorfield_animate(
        model,
        frames=3,
        fixed_func=fixed_fn,
        param="a",
        values=[1.0, 1.1, 1.2],
        interactive=False,
    )
    anim.animation._func(anim.frames[2])
    assert np.isclose(anim.handle.last_fixed.get("z"), 4.0)
    plt.close(anim.figure)


def test_vectorfield_animate_custom_functions():
    model = _linear_model()
    params_fn = lambda v, idx: {"a": 0.5 + 0.5 * idx}
    fixed_fn = lambda v, idx: {"z": 1.0 + idx}
    anim = vectorfield_animate(
        model,
        frames=4,
        params_func=params_fn,
        fixed_func=fixed_fn,
        interactive=False,
    )
    anim.animation._func(anim.frames[3])
    assert np.isclose(anim.handle.last_params.get("a"), 0.5 + 0.5 * 3)
    assert np.isclose(anim.handle.last_fixed.get("z"), 1.0 + 3)
    plt.close(anim.figure)


def test_vectorfield_animate_error_mixed_modes():
    model = _linear_model()
    with pytest.raises(ValueError):
        vectorfield_animate(model, param="a", values=[1, 2], params_func=lambda v, i: {"a": v})


def test_vectorfield_animate_error_no_frames_with_custom_func():
    model = _linear_model()
    with pytest.raises(ValueError):
        vectorfield_animate(model, params_func=lambda v, i: {"a": v})


def test_vectorfield_animate_error_no_animation_mode():
    model = _linear_model()
    with pytest.raises(ValueError):
        vectorfield_animate(model, frames=2, mode="none")


def test_vectorfield_animate_streamplot_mode():
    model = _linear_model()
    anim = vectorfield_animate(
        model,
        frames=2,
        mode="stream",
        stream_kwargs={"density": 1.0},
        interactive=False,
    )
    assert anim.handle.stream is not None
    plt.close(anim.figure)


def test_vectorfield_animate_with_duration():
    model = _linear_model()
    anim = vectorfield_animate(model, duration=1.0, fps=4, interactive=False)
    assert anim.animation.save_count == 4
    plt.close(anim.figure)


def test_vectorfield_animate_with_frames():
    model = _linear_model()
    anim = vectorfield_animate(model, frames=5, interactive=False)
    assert anim.animation.save_count == 5
    plt.close(anim.figure)


def test_vectorfield_animate_with_nullclines():
    model = _linear_model()
    anim = vectorfield_animate(model, frames=2, nullclines=True, interactive=False)
    assert anim.handle.nullclines_enabled
    anim.animation._func(anim.frames[1])
    assert anim.handle.nullclines_enabled
    plt.close(anim.figure)


def test_vectorfield_animate_with_title_func():
    model = _linear_model()
    anim = vectorfield_animate(
        model,
        frames=[0, 1],
        title_func=lambda v, idx: f"frame {idx}",
        interactive=False,
    )
    anim.animation._func(anim.frames[1])
    assert anim.ax.get_title() == "frame 1"
    plt.close(anim.figure)


def test_vectorfield_animation_properties():
    model = _linear_model()
    anim = vectorfield_animate(model, frames=3, fps=10, repeat=False, repeat_delay=500, interactive=False)
    assert anim.animation.save_count == 3
    assert anim.animation.repeat is False
    assert np.isclose(anim.animation._interval, 100.0)
    assert np.isclose(anim.animation.repeat_delay, 500.0)
    plt.close(anim.figure)
