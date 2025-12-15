from __future__ import annotations

"""Demonstration of plot.vectorfield_animate for a simple spiral system."""

from dynlib import build, plot


def _make_model():
    model_uri = """
inline:
[model]
type = "ode"
label = "spiral"

[sim]
t0 = 0.0
dt = 0.05

[states]
x = 0.0
y = 0.0

[params]
a = -0.4
b = 0.25

[equations.rhs]
x = "a * x - y"
y = "x + b * y"
"""
    return build(model_uri, jit=False, disk_cache=False)


def main() -> None:
    plot.theme.use("notebook")
    model = _make_model()

    values = [v for v in (-1.0, -0.4, 0.0, 0.6, 1.0, 1.4)]
    # You have to define anim (or any other name) even if you don't use it. 
    # Otherwise it gets garbage collected.
    anim = plot.vectorfield_animate(
        model,
        param="a",
        values=values,
        xlim=(-2.5, 2.5),
        ylim=(-2.5, 2.5),
        grid=(24, 24),
        normalize=True,
        speed_color=True,
        speed_cmap="plasma",
        title_func=lambda v, idx: f"Vector field: a={float(v):.2f}",
        nullclines=True,
        nullcline_style={"colors": ["#333333"], "linewidths": 1.0, "alpha": 0.6},
        interactive=False,
        fps=3,
        blit=False,
    )

    # Preview the animation in notebook/backends that support it, or save via anim.animation.save(...)
    plot.export.show()


if __name__ == "__main__":
    main()
