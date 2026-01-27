from __future__ import annotations

"""Demonstration of plot.vectorfield_sweep for a simple 2D system."""

from dynlib import build, plot


def _make_model():
    model_uri = """
inline:
[model]
type = "ode"
name = "spiral"

[sim]
t0 = 0.0
dt = 0.05

[states]
x = 0.0
y = 0.0

[params]
a = 0.5
b = -0.2

[equations.rhs]
x = "a * x - y"
y = "x + b * y"
"""
    return build(model_uri, jit=False, disk_cache=False)


def main() -> None:
    plot.theme.use("notebook")
    model = _make_model()

    plot.vectorfield_sweep(
        model,
        param="a",
        values=[-0.6, 0.0, 0.6, 1.2],
        xlim=(-2.5, 2.5),
        ylim=(-2.5, 2.5),
        grid=(22, 22),
        normalize=True,
        speed_color=True,
        speed_cmap="plasma",
        cols=2,
        facet_titles="a={value:.2f}",
        title="Vector field sweep over parameter 'a'",
        nullclines=True,
        nullcline_style={"colors": ["#333333"], "linewidths": 1.0, "alpha": 0.6},
        interactive=False,
        size=(8,6)
    )

    plot.export.show()


if __name__ == "__main__":
    main()
