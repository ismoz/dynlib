from __future__ import annotations

"""
Demonstration of the dynlib.plot.vectorfield helper.
"""

import numpy as np

from dynlib import build, plot


def _make_model():
    # Simple spiral system with tunable linear terms
    model_uri = """
inline:
[model]
type = "ode"
label = "spiral"

[sim]
t0 = 0.0
dt = 0.1

[states]
x = 0.0
y = 0.0

[params]
a = 0.8
b = 0.2

[equations.rhs]
x = "a * x - y"
y = "x + b * y"
"""
    return build(model_uri, jit=False, disk_cache=False)


def main() -> None:
    plot.theme.use("notebook")
    model = _make_model()

    ax = plot.fig.single(title="Vector field demo")

    handle = plot.vectorfield(
        model,
        ax=ax,
        xlim=(-2, 2),
        ylim=(-2, 2),
        grid=(25, 25),
        normalize=True,
        speed_color=True,
        speed_cmap="plasma",
        nullclines=True,
        nullcline_style={"colors": ["#333333"], "linewidths": 1.2, "alpha": 0.6},
    )

    # Update parameters and redraw to illustrate handle.update()
    handle.update(params={"a": 1.2, "b": -0.1})

    plot.export.show()


if __name__ == "__main__":
    main()
