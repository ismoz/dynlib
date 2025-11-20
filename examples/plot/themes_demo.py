#!/usr/bin/env python3
"""
Demonstration of all theme presets available in dynlib.plot.theme.

This script creates a sample figure for each preset and saves it as an image.
"""

import numpy as np
from dynlib.plot import fig, series, theme, export, analysis


def create_sample_figure():
    """Create a sample figure with various plot elements."""
    # Generate sample data
    t = np.linspace(0, 10, 100)
    y1 = np.sin(t)
    y2 = np.cos(t)
    y3 = np.sin(t) * np.exp(-t / 10)

    # Create figure with subplots
    ax = fig.grid(rows=2, cols=2, size=(8, 6))

    # Line plot
    series.plot(x=t, y=y1, ax=ax[0, 0], label="sin(t)", xlabel="Time", ylabel="Amplitude", title="Line Plot")

    # Scatter plot
    series.plot(x=t[::5], y=y2[::5], ax=ax[0, 1], style="scatter", label="cos(t) samples", xlabel="Time", ylabel="Amplitude", title="Scatter Plot")

    # Multiple lines
    series.plot(x=t, y=y1, ax=ax[1, 0], label="sin(t)", color="C0")
    series.plot(x=t, y=y2, ax=ax[1, 0], label="cos(t)", color="C1")
    series.plot(x=t, y=y3, ax=ax[1, 0], label="damped sin(t)", color="C2")
    ax[1, 0].set_xlabel("Time")
    ax[1, 0].set_ylabel("Amplitude")
    ax[1, 0].set_title("Multiple Lines")
    ax[1, 0].legend()

    # Histogram
    data = np.random.normal(0, 1, 1000)
    analysis.hist(y=data, ax=ax[1, 1], bins=30, xlabel="Value", ylabel="Frequency", title="Histogram")

    return ax[0, 0].figure


def main():
    """Demonstrate each theme preset."""
    presets = ["notebook", "paper", "talk", "dark", "mono"]

    for preset in presets:
        print(f"Creating figure with '{preset}' preset...")

        # Apply theme
        theme.use(preset)

        # Create sample figure
        fig = create_sample_figure()

        # Save figure
        export.savefig(fig, f"theme_{preset}", fmts=("png",), dpi=150)

        print(f"Saved theme_{preset}.png")


if __name__ == "__main__":
    main()
