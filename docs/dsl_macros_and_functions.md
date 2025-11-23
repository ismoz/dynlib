# DSL Macros and Built-in Functions

This document describes all available Domain Specific Language (DSL) macros and built-in functions that can be used in dynlib model definitions.

## Built-in Math Functions

The following mathematical functions are available and map directly to Python's `math` module or built-in functions:

### Basic Functions
- `abs(x)` - Absolute value
- `min(x, y, ...)` - Minimum of arguments
- `max(x, y, ...)` - Maximum of arguments
- `round(x)` - Round to nearest integer

### Exponential and Logarithmic Functions
- `exp(x)` - Exponential function (e^x)
- `expm1(x)` - exp(x) - 1 (more accurate for small x)
- `log(x)` - Natural logarithm
- `log10(x)` - Base-10 logarithm
- `log2(x)` - Base-2 logarithm
- `log1p(x)` - log(1 + x) (more accurate for small x)
- `sqrt(x)` - Square root

### Trigonometric Functions
- `sin(x)` - Sine
- `cos(x)` - Cosine
- `tan(x)` - Tangent
- `asin(x)` - Inverse sine
- `acos(x)` - Inverse cosine
- `atan(x)` - Inverse tangent
- `atan2(y, x)` - Two-argument inverse tangent

### Hyperbolic Functions
- `sinh(x)` - Hyperbolic sine
- `cosh(x)` - Hyperbolic cosine
- `tanh(x)` - Hyperbolic tangent
- `asinh(x)` - Inverse hyperbolic sine
- `acosh(x)` - Inverse hyperbolic cosine
- `atanh(x)` - Inverse hyperbolic tangent

### Rounding Functions
- `floor(x)` - Floor (round down to integer)
- `ceil(x)` - Ceiling (round up to integer)
- `trunc(x)` - Truncate (remove fractional part)

### Special Functions
- `hypot(x, y)` - Euclidean distance (sqrt(x^2 + y^2))
- `copysign(x, y)` - Copy sign of y to magnitude of x
- `erf(x)` - Error function
- `erfc(x)` - Complementary error function

## Scalar Macros

Scalar macros are special functions that perform common mathematical operations:

- `sign(x)` - Sign function: returns -1 for negative, 0 for zero, 1 for positive
- `heaviside(x)` - Heaviside step function: returns 0 for x < 0, 1 for x >= 0
- `step(x)` - Same as heaviside (alias)
- `relu(x)` - Rectified Linear Unit: returns max(0, x)
- `clip(x, min, max)` - Clamp x to the range [min, max]
- `approx(x, y, tol)` - Check if |x - y| <= tol (returns boolean)

## Generator Comprehensions

The DSL supports generator comprehensions for efficient sum and product operations over ranges:

- `sum(expr for var in range(start, stop[, step]) [if condition])` - Sum of expressions over a range
- `prod(expr for var in range(start, stop[, step]) [if condition])` - Product of expressions over a range

These constructs are compiled into optimized for-loops. Only `range()` is supported as the iterator, and only a single generator is allowed. Conditional filters with `if` are supported.

Examples:
- `sum(i*i for i in range(10))` - Sum of squares from 0 to 9 (0+1+4+...+81)
- `prod((i+1) for i in range(1, 5))` - Product 2×3×4×5 = 120
- `sum(x[i] for i in range(N) if i % 2 == 0)` - Sum of even-indexed elements (assuming x is an array)

## Event Macros

Event macros are used in event conditions to detect state changes and transitions. These macros automatically use lagged state values for comparison:

- `cross_up(state, threshold)` - True when state crosses threshold from below to above
- `cross_down(state, threshold)` - True when state crosses threshold from above to below
- `cross_either(state, threshold)` - True when state crosses threshold in either direction
- `changed(state)` - True when state value changed from previous step
- `in_interval(state, lower, upper)` - True when state is currently in [lower, upper]
- `enters_interval(state, lower, upper)` - True when state enters [lower, upper] interval
- `leaves_interval(state, lower, upper)` - True when state leaves [lower, upper] interval
- `increasing(state)` - True when state is increasing (current > previous)
- `decreasing(state)` - True when state is decreasing (current < previous)

## Lag Notation

For models that use lagged state values, you can access previous values of states:

- `lag_state()` or `lag_state(1)` - Access the state value from 1 step ago
- `lag_state(k)` - Access the state value from k steps ago (k must be a positive integer)

Lag notation is only available for states that are detected to require lagging based on usage in equations or events.

## Special Variables

- `t` - Current time (available in all expressions)

## User-Defined Functions

You can define custom functions in the `[functions]` section:

```toml
[functions.my_func]
args = ["x", "y"]
expr = "x**2 + y**2"
```

These functions can then be called in equations, aux variables, and events like any built-in function.

## Expression Context

### Available in Equations
- State variables (by name)
- Parameters (by name)
- Time (`t`)
- Aux variables (by name)
- User-defined functions
- All built-in math functions, scalar macros, and generator comprehensions
- Lag notation (if applicable)

### Available in Aux Variables
- Same as equations, plus other aux variables (no cycles allowed)

### Available in Events
- State variables (by name)
- Parameters (by name)
- Time (`t`)
- Aux variables (by name)
- User-defined functions
- All built-in math functions, scalar macros, and generator comprehensions
- Event macros
- Lag notation (if applicable)

### Available in Functions
- Function arguments
- Time (`t`) - if the function is called in a context where time is available
- Other user-defined functions (no recursion allowed)
- All built-in math functions, scalar macros, and generator comprehensions
- Lag notation (if applicable and function is called in appropriate context)

## Notes

- All expressions are evaluated as Python code after macro expansion
- Mathematical operators `+`, `-`, `*`, `/`, `^` (exponentiation), and parentheses are supported
- Caret `^` is automatically converted to `**` for Python compatibility and `**` can be used directly
- Division by zero and other mathematical errors are caught during evaluation