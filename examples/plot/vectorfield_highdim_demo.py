from __future__ import annotations

"""
Demonstration of projecting a higher-dimensional vector field onto chosen 2D planes.

We use the 3D Lorenz system and visualize two slices:
- x/y plane with z fixed (and then updated to a new z to show handle.update)
- y/z plane with x fixed

Click on either panel to launch a short trajectory through that slice.
"""

from dynlib import build, plot


def _lorenz_model():
    model_uri = """
inline:
[model]
type = "ode"
label = "lorenz-3d"
stepper = "rk4"

[sim]
t0 = 0.0
dt = 0.01
t_end = 8.0

[states]
x = 1.0
y = 1.0
z = 1.0

[params]
sigma = 10.0
rho = 28.0
beta = 2.6666666666666665

[equations.rhs]
x = "sigma * (y - x)"
y = "x * (rho - z) - y"
z = "x * y - beta * z"
"""
    return build(model_uri, jit=False, disk_cache=False)


def main() -> None:
    plot.theme.use("notebook")
    model = _lorenz_model()

    ax = plot.fig.grid(rows=1, cols=2, size=(12, 5), sharex=False, sharey=False)

    handle_xy = plot.vectorfield(
        model,
        ax=ax[0, 0],
        vars=("x", "y"),
        fixed={"z": 5.0},
        xlim=(-20, 20),
        ylim=(-30, 30),
        grid=(25, 25),
        normalize=False,
        nullclines=False,
        T=6.0,
        dt=0.01,
        trajectory_style={"color": "C0"},
    )
    ax[0, 0].set_title("x-y slice with z fixed (click to trace)", fontsize=11)

    handle_yz = plot.vectorfield(
        model,
        ax=ax[0, 1],
        vars=("y", "z"),
        fixed={"x": 0.0},
        xlim=(-30, 30),
        ylim=(0, 50),
        grid=(25, 25),
        normalize=False,
        nullclines=False,
        T=6.0,
        dt=0.01,
        trajectory_style={"color": "C1"},
    )
    ax[0, 1].set_title("y-z slice with x fixed (click to trace)", fontsize=11)

    # Show that fixed values can be updated without rebuilding the figure.
    handle_xy.update(fixed={"z": 15.0}, redraw=True)
    handle_xy.ax.set_title("x-y slice with z updated to 15.0", fontsize=11)

    plot.export.show()


if __name__ == "__main__":
    main()
