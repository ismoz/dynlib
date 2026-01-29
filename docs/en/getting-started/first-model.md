# Your First Model

This guide walks through defining a simple model both as a standalone TOML file and as an inline string that Dynlib can consume without ever touching disk. Refer to the Modeling guides for deeper DSL coverage (`dsl-basics.md` for the TOML structure, `inline-models.md` for the inline keyword, and `config-file.md` if you want to register the file in the model registry).

## 1. Create a TOML spec

Pick `type = "map"` or `type = "ode"` depending on the dynamics you need. At minimum supply `[model]`, `[states]`, and an equations table.

Save this as `first-model.toml` (or any path you prefer) inside your project:

```toml
[model]
type = "map"
name = "Simple Logistic Map"
dtype = "float64"

[params]
r = 3.9

[states]
x = 0.2

[equations.rhs]
x = "r * x * (1 - x)"

```

The `[dsl-basics](../guides/modeling/dsl-basics.md)` reference shows every table you can add (constants, aux, events, matrix-style Jacobians, etc.) and how Dynlib interprets expressions.

Run `dynlib model validate first-model.toml` (or `python -m dynlib.cli model validate ...`) to make sure the parser accepts your spec and the state ordering matches what you expect.

## 2. Inline models for quick experimentation

When you just want to prototype without saving a file, define the TOML string in place and point `setup()` (or any resolver) at an `inline:` URI. The inline document follows the same structure as the DSL file:

```python
from dynlib import setup

spec = '''
inline:
[model]
type = "map"
name = "Inline Logistic"
dtype = "float64"

[params]
r = 3.9

[states]
x = 0.3

[equations.rhs]
x = "r * x * (1 - x)"
'''

sim = setup(spec)
sim.run(N=30)
```

The inline helpers document lists the acceptable URI formats and the multiline `inline:` workflow (`docs/guides/modeling/inline-models.md`), so grab that page when you need macros, functions, or inline macros embedded in the string.

## 3. Minimal ODE reference

Need a continuous-time example? Switch `type = "ode"` and describe the time derivatives via `[equations.rhs]`. A tiny, self-contained spec is a good way to verify the DSL still parses when you add vectors, Jacobians, or helper functions later:

```toml
[model]
type = "ode"
name = "Simple Harmonic Oscillator"
dtype = "float64"

[params]
omega = 1.0

[states]
x = 1.0
v = 0.0

[equations.rhs]
x = "v"
v = "-omega ** 2 * x"
```

Feed this through `dynlib model validate harmonic.toml` or inline it like above, then run the below code to check the trajectories. Because the DSL treats states as a `dict`, you can sprinkle additional aux states or helpers in the same document without changing the core structure.

```python
sim = setup("path/to/harmonic.toml")
sim.run(T=30)
```

## 4. Next steps

Once the TOML file validates, point Dynlib at it via `setup("first-model.toml", ...)`, wire it into a runner, or register the directory using the config file so you can call it with a tag (`proj://first-model.toml`). For more options, see:

- `docs/guides/modeling/config-file.md` for customizing the tag map and cache roots.
- `docs/guides/modeling/sim.md` for runtime settings (steppers, recorders, persistence).
