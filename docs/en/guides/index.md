# Guides

Dynlib’s guides go deeper than the quick-start material: each section untangles a core subsystem so you can customize models, steppers, analysis helpers, or the plotting stack without hunting for scattered notes.

## Guides at a glance

- [Command-line guide](cli/cli.md) — explains the `dynlib` (and `python -m dynlib.cli`) commands for model validation, stepper inspection, and cache management.
- [Modeling guide](modeling/index.md) — covers the TOML DSL, auxiliary helpers, mods, presets, and other authoring conveniences you need to keep specs readable and reusable.
- [Simulation guide](simulation/index.md) — surveys runtime concepts such as steppers, wrappers, snapshots, results, and configuration knobs that control the solver behavior.
- [Plotting guide](plotting/index.md) — surveys the Matplotlib-based helpers (`plot.series`, `plot.phase`, `plot.manifold`, decorations, exports, themes) tuned for dynamical systems.
- [Analysis guide](analysis/index.md) — explains the runtime observers, sweep utilities, basins, Lyapunov diagnostics, and manifold finders that turn results into scientific insights.
