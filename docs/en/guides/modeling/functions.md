# DSL Functions

User-defined functions let you encapsulate reusable logic once and call it from equations, aux, events, and other functions (no recursion). They keep expressions manageable and allow you to parameterize common computations.

## Syntax

```toml
[functions.sigmoid]
args = ["x", "gain", "offset"]
expr = "gain / (1 + exp(-x)) + offset"
```

- `args` is an array of parameter names; use only simple identifiers.
- `expr` is a string expression evaluated using the same macro-expanded DSL/ Python rules as other expressions (`^` → `**`, generator comprehensions compiled to loops, etc.).
- Functions do not declare a return type; the expression value is the return value.

## Context Inside Functions

- **Arguments:** Functions can use their own parameters as variables (`x`, `gain`).
- **Time (`t`):** Available only when the surrounding context provides time (e.g., calling from an equation or aux, not from a pure math helper).
- **States & parameters:** Referenced by name if they exist in the model.
- **Aux variables:** Can call aux defined in `[aux]`, but ensure you avoid dependency cycles.
- **Other user-defined functions:** Call them like normal, but prevent recursion.
- **Built-in math & macros:** Use `sin`, `cos`, `clip`, `approx`, generator comprehensions, etc.
- **Lag notation:** Allowed if the function is invoked where `lag_` access is valid (states only).
- **Event macros:** Not accessible inside functions directly; use them in event conditions instead.

## Calling Functions

- Reference functions by name just like built-ins: `sigmoid(x, gain, offset)`.
- You can pass expressions, states, aux, or literals as arguments.
- Functions can simplify repeated math, conditional logic, or complex transformations used across equations or events.
- Use helper functions to factor out tooling for Jacobians, logging expressions, or custom activation shapes.

## Mods & Functions

Function definitions can be modified via mods:

- `mod.remove.functions` deletes named functions (component must already exist).
- `mod.replace.functions.name` overwrites the body while keeping the identifier.
- `mod.add.functions.name` inserts a new function (fails if already present).
- `mod.set.functions.name` upserts the function definition (create or update).

These verbs obey the global rem/replace/add/set order, so you can remove or replace before adding another version.

## Best Practices

1. **Name helpers descriptively** (`functions.normalize_input`) so downstream equations clarify intent.
2. **Keep argument lists short**; overly many args suggest the function should operate on aux or state bundles instead.
3. **Avoid side effects**—functions should only return values and not mutate states or aux.
4. **Document assumptions** (e.g., expected ranges) in nearby comments or docs to keep integrators aware of constraints.
5. **Reuse judiciously**: don’t wrap trivial expressions unless they help readability or hide complex math.
