# Equations

The `[equations]` table is where you describe how states change every step. It accepts several interchangeable sub-forms so you can pick the style that fits your model.

## Basic forms

- `[equations.rhs]` (per-state) – a TOML table of `state = "expr"` entries. Each expression must be a string so the DSL can parse macros before evaluating the right-hand side for that state.
- `[equations].expr` (block) – a single multi-line string where each line assigns to a state (`x = ...`) or, for ODE models, uses derivative notation (`dx = ...` or `d(x) = ...`). Use whichever style keeps your algebra tidy, but do not define the same state in both places (the loader enforces this).
- `[equations.inverse]` – available only for `map` models; it mirrors the main equation form and provides a callable inverse update. You may define `rhs` or `expr` inside this table, but not both for the same state.
- `[equations.jacobian]` – optional metadata containing a single `expr` key with a square list-of-list literal describing the dense Jacobian (each entry may be a string or numeric literal). This table is only used when you supply custom derivatives for the compiler (e.g., for stiff solvers or implicit steppers).

### Example

```toml
[equations.rhs]
x = "speed * cos(theta)"
theta = "speed * sin(theta)"

[equations.inverse]
expr = """
x = x - speed * cos(theta)
theta = theta - speed * sin(theta)
"""

[equations.jacobian]
expr = [
  ["0", "-speed * sin(theta)"],
  ["speed * cos(theta)", "0"]
]
```

## Expression context

Equation expressions share the same identifiers available elsewhere: states, parameters, constants, aux, functions, macros (`sin`, `clip`, `approx`, generator comprehensions) and `t`. ODE blocks also accept derivative targets (`dx`, `d(x)`), but map models must stick to `state = expr` assignments.

## Inverse equations

- The `inverse` table only exists for map models and supplies an `inv_rhs` callable used by inversion utilities and diagnostics.
- You can write it as a per-state table (`[equations.inverse.rhs]`) or a block string (`[equations.inverse].expr`), mirroring the primary equation form.
- Each state may appear only once across inverse forms; mixing `rhs` and `expr` for the same state raises an error.
- The inverse update must also resolve the same identifier sets as forward equations (states, params, aux, etc).

## Jacobian table

- `[equations.jacobian].expr` is a list of rows; the number of rows and columns must match the number of declared states (square matrix).
- Each matrix entry can be a string expression or a numeric literal (integers/floats). The compiler flattens this into the explicit Jacobian used for solver support.
- If you need a dense Jacobian but prefer to keep it organized, you may precompute shared expressions with aux variables and reference them inside the matrix entries.

## Validation hints

- The parser forbids unknown keys inside `[equations]`, `[equations.inverse]`, and `[equations.jacobian]` so typos are caught early.
- States can only be defined once across `[equations.rhs]` and `[equations].expr`, and similarly for the inverse table.
- Map models cannot use derivative notation (the loader explicitly rejects `d(x)` inside `[equations].expr` for maps).
- `[equations.jacobian].expr` must be provided as a list of rows; using the plural `exprs` or omitting the table will raise an error.

## Best practices

1. **Stick to one style per state** (either `rhs` or block) to avoid redundant logic.
2. **Use aux/functions** to factor complex right-hand sides so the equation tables stay readable.
3. **Document inverse tables** clearly—mention why they exist (e.g., for stepping backwards or diagnostics) since they execute outside the normal solver path.
4. **Only supply a Jacobian when necessary** (implicit solvers, stiffness); otherwise, let the compiler numerically estimate derivatives.
