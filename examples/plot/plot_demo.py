from __future__ import annotations
"""
Demonstration of various plotting functions from dynlib.plot module.
Only the ones not used in other examples are shown here.
"""

import numpy as np
from dynlib.plot import fig, series, phase, utils, export, theme


def main() -> None:
    theme.use("notebook")

    # ----- Time data -----
    t = np.linspace(0, 10, 101)
    y = np.sin(2 * np.pi * 0.5 * t)
    y2 = np.cos(2 * np.pi * 0.5 * t)
    # step-like signal
    y_step = np.floor(t) % 2
    # noisy distribution for histogram
    y_hist = y + 0.3 * np.random.randn(len(t))

    # ----- Discrete (map) data -----
    r = 3.7
    x0 = 0.2
    n_iter = 30
    xs = [x0]
    x = x0
    for k in range(n_iter):
        x = r * x * (1 - x)
        xs.append(x)
    ks = np.arange(len(xs))
    xs = np.array(xs)

    # Layout: 3 rows x 2 cols
    ax = fig.grid(rows=3, cols=2, size=(10, 12), sharex=False)

    # Row 0, Col 0: series.stem
    series.stem(
        x=t,
        y=y,
        ax=ax[0, 0],
        label="sin(t) stems",
        color="C0",
        xlabel="$t$",
        ylabel="$y$",
        title="series.stem: stem plot",
    )

    # Row 0, Col 1: series.step
    series.step(
        x=t,
        y=y_step,
        ax=ax[0, 1],
        label="step(t)",
        color="C1",
        xlabel="$t$",
        ylabel="$y$",
        title="series.step: step plot",
    )

    # Row 1, Col 0: utils.hist
    utils.hist(
        y=y_hist,
        bins=30,
        density=False,
        ax=ax[1, 0],
        color="C2",
        title="utils.hist: histogram",
        xlabel="$y$",
        ylabel="count",
    )

    # Row 1, Col 1: phase.xy with scatter style (for discrete maps)
    phase.xy(
        x=y,
        y=y2,
        style="scatter",
        ax=ax[1, 1],
        ms=6,
        color="C3",
        title="phase.xy: sin vs cos (scatter)",
        xlabel=r"$\sin$",
        ylabel=r"$\cos$",
    )

    # Row 2, Col 0: series with discrete/map style (stem-like effect)
    series.plot(
        x=ks,
        y=xs,
        style="map",
        ax=ax[2, 0],
        color="C4",
        title="series.plot: logistic iterations (map style)",
        xlabel="n",
        ylabel="$x_n$",
    )

    # Row 2, Col 1: series with mixed style (line + markers)
    series.plot(
        x=ks,
        y=xs,
        style="mixed",
        ax=ax[2, 1],
        color="C5",
        title="series.plot: logistic iterations (mixed style)",
        xlabel="n",
        ylabel="$x_n$",
    )

    # Tighten layout and show
    export.show()


if __name__ == "__main__":
    main()
