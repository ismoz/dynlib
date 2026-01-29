# Modeling guide

This guide explains how to declare, extend, and tune dynlib models using the TOML DSL and the helper utilities that keep specs structured, readable, and reproducible. Start with the DSL overview, then explore the reusable components and the workflow helpers that tie a spec to the runtime.

## Model structure essentials

- [DSL basics](dsl-basics.md) — the canonical TOML template that lists every table you can write (`[model]`, `[states]`, `[params]`, `[constants]`, `[equations]`, `[aux]`, `[functions]`, `[events]`, `[sim]`, etc.).
- [Equations](equations.md) — compares the `rhs`, block, inverse, and Jacobian forms, explains which contexts accept each one, and outlines best practices for keeping your right-hand sides tidy.
- [Math & macros](math-and-macros.md) — catalogs the built-in math functions, scalar macros (`clip`, `approx`, `relu`, etc.), generator comprehensions, and event utilities available inside every expression.
- [Ternary `if`](ternary-if.md) — shows how the Python-style ternary expression streamlines small branches without pulling you into full `if`/`else` blocks.
- [Model registry](model-registry.md) — describes tag URIs (`builtin://`, custom tags, inline models), the `DYNLIB_CONFIG`/`DYN_MODEL_PATH` behavior, and the CLI helpers that validate or override registry paths.

## Reusable building blocks

- [Auxiliary variables](aux.md) — name derived expressions so you can share them between equations, events, or Jacobians without repeating the math.
- [DSL functions](functions.md) — define reusable functions with arguments, expression bodies, and clean callsites that keep the DSL declarative.
- [Events](events.md) — wire `cond`, `action`, and logging metadata to `pre`/`post` phases, use the event macros, and manage event logs without destabilizing fast-path runners.
- [Lagging](lagging.md) — enable `lag_<state>(k)` helpers, control the buffer depth, and understand how lagged states interact with ODEs, maps, and NumPy-friendly runtimes.
- [Inline models](inline-models.md) — embed a TOML snippet in a Python string so you can prototype models entirely inside tests or notebooks.

## Workflow helpers

- [Config file](config-file.md) — customize registry paths, cache roots, and plugin behavior through `~/.config/dynlib/config.toml` or the `DYNLIB_CONFIG` environment variable.
- [Mods](mods.md) — patch models dynamically with `remove`, `replace`, `add`, and `set` verbs so you can build variants, override parameters, or inject new events without cloning the base spec.
- [Presets](presets.md) — capture reusable state/parameter snapshots, load/save them from disk, and replay them via the simulation bank.
- [Simulation defaults](sim.md) — document the `[sim]` table, explain how it merges with `Sim.run` overrides, and highlight early-exit, recording, and tolerancing knobs.
