# Integer Map Example: Collatz Sequence

## Overview

This example shows how to build a **map-type** simulation that uses an integer state variable. In contrast to the floating-point logistic map used elsewhere in the docs, this model keeps the Collatz iteration in the integer domain, letting you inspect integer arithmetic.

## Key Concepts

- **`map` stepper**: A discrete-time map that updates the state once per step with no notion of continuous time.
- **Integer dtype (`int64`)**: Ensures every iterated value remains precise so the Collatz sequence can be compared exactly to a pre-computed reference.
- **Assertion-based validation**: `numpy.testing.assert_array_equal` checks the full trajectory against the expected 1-4-2-1 cycle.
- **Series plotting**: Visualizes the integer trajectory to highlight when it converges to the 4-2-1 cycle.

## Collatz Map Definition

The inline Toml definition uses a single integer state `n` initialized to 27 and an `int64` right-hand side that branches on parity:

```toml
[model]
type = "map"
dtype = "int64"
name = "Collatz Conjecture"

[states]
n = 27

[equations.rhs]
n = "n//2 if n % 2 == 0 else 3*n + 1"
```

## Running the Simulation

The script instantiates the model with `setup(..., stepper="map")`, runs `len(expected) - 1` iterations, and reads the results table via `sim.results()`. The plot uses `series.plot` to show `n` vs. iteration and checks that the recorded values match the prepared `expected` array (the known sequence starting at 27 and terminating in the 1-4-2-1 loop). The example also prints the tail of the trajectory and the state dtype so you can confirm that the cycle and datatype are preserved.

## Plotting and Export

`theme.use("paper")` configures matplotlib for publication-ready styles, and `export.show()` displays the figure. The plot's axis labels (`iteration` and `n`) and title (`Collatz Conjecture`) make it easy to see how the iterates descend before settling into the familiar cycle.

## Reference Script

```python
--8<-- "examples/collatz.py"
```
for the full implementation, including the `sequence` validation and plotting logic. The script doubles as a regression test by asserting that every integer in the run matches the `expected` NumPy array.
