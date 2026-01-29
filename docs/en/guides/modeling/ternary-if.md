## Ternary `if` expressions

The DSL lets you keep short, two-way branches inline because every right-hand expression is parsed as a Python expression. That means you can write the familiar Python ternary form:

```dsl
<value when true> if <condition> else <value when false>
```

This form is ideal when the condition simply selects between two calculations without needing side effects or additional statements. Use a full `if`/`else` block when you need multiple assignments, logging, or other imperative steps before producing the final value.

### Examples from the repository

The mods guide itself shows a ternary expression inside an added helper function, using it to substitute one formula when `N == 0` and another when `N` is positive (see `docs/guides/modeling/mods.md:483-491`):

```toml
h = {args = ["phi","N"], expr="""
phi if N==0 else phi-sum(sign(phi+(2*j-1))+sign(phi-(2*j-1)) for j in range(1,N+1))
"""}
```

The unit tests also rely on a ternary branch inside an inline model to make the RHS depend on time; `tests/unit/test_sum_generator_lowering.py:27-110` defines the model with

```toml
[equations.rhs]
x = "1.0 if t < 0 else sum(i for i in range(N))"
```

and verifies both sides of the branch against the Python and JIT backends.

Those examples show how ternary expressions keep expressions concise while still feeding the compiler two distinct paths to choose from.

### Nested `if` expressions

You can nest ternary `if` expressions to handle multiple conditions in a single expression. This allows for more complex branching without resorting to full `if`/`else` blocks, as long as the logic remains purely functional.

#### Syntax

```dsl
<value1> if <condition1> else <value2> if <condition2> else <value3> if ... else <default_value>
```

Note that nesting can reduce readability, so use it sparingly and consider parentheses for clarity if needed.

#### Example

Suppose you need to select a value based on multiple thresholds. In a model equation, you might write:

```toml
result = "0 if x < 0 else 1 if x < 10 else 2"
```

This evaluates to:
- `0` if `x < 0`
- `1` if `0 <= x < 10`
- `2` otherwise