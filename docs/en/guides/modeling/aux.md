# Auxiliary Variables

Auxiliary variables (`[aux]` in DSL files) let you name intermediate or derived expressions so you can reuse them throughout equations, events, and functions without repeating their logic. They are evaluated after the state updates for each step, and every expression is compiled to Python before the model runs.

## Syntax

```toml
[aux]
energy = "0.5 * mass * velocity^2"
gain = "baseline_gain * exp(-t / tau)"
# You can reference any previously defined aux as long as there is no cycle.
```

- Every value must be a string literal because it is parsed and type-checked as an expression. When the expression includes `^`, the compiler rewrites it to `**` for Python compatibility.
- Aux expressions can refer to states, parameters, time (`t`), other aux (no cycles), user-defined functions, math macros, and generator comprehensions. They cannot use event macros because they do not execute in event contexts.
- `t` is available just like in equations, so time-dependent aux are easy to write.

## Expression Context

- **States**: current values only (lag notation must reference states explicitly).
- **Parameters**: the numerical constants defined in `[params]`.
- **Auxiliary variables**: you may use another aux defined earlier in the file.
- **Built-in math functions & scalar macros**: everything from `dynlib`’s DSL library (`sin`, `cos`, `clip`, `approx`, generator comprehensions, etc.).
- **User-defined functions**: call them by name once they are declared in `[functions]` (no recursion).
- **Time (`t`)**: always present in expressions just like in equations.
- **Lag notation**: only available if the referenced symbol is a state; aux variables cannot be lagged directly.

## Example Usage

- Use aux to compute shared expressions such as energy, forces, or logging helpers so you do not repeat long calculations.
- Pair aux with events by referencing them in `cond`, `action`, or `log` lists rather than retyping the expressions.
- Aux can simplify Jacobian entries or generator comprehensions when the same subexpression appears in multiple equations.

## Interaction with Mods

Mods can manipulate auxiliary variables using the `remove`, `replace`, `add`, and `set` verbs.

- `mod.remove.aux` requires the aux to exist and simply drops it from the model.
- `mod.replace.aux` lets you change the definition while keeping the same name.
- `mod.add.aux` inserts a new auxiliary (errors if the name already exists).
- `mod.set.aux` upserts the expression (creates it if missing, updates if present).

These verbs respect the mod verb order (`remove → replace → add → set`), so you can remove an aux and add a new definition with the same name later in the same mod.

## Best Practices

1. **Prioritize readability**: Give aux names that describe the quantity (`kinetic_energy`, `normalized_voltage`) so they help rather than hide complexity.
2. **Avoid cycles**: Do not create mutual dependencies between aux variables; the compiler enforces a DAG.
3. **Keep expressions focused**: If you start requiring lagged values or differential behavior, consider promoting the quantity to a state instead of overloading aux logic.
4. **Document intent**: A short TOML comment next to the aux definition is enough to remind future readers why the derived quantity exists.
5. **Reuse judiciously**: Aux are great for repeated math, but do not over-index the model with trivial aliases that only obfuscate the math.
