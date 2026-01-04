"""
Early Exit Demo
===============

This example demonstrates the early exit feature using the `stop` condition
with built-in DSL macros. The simulation will terminate early when a specified
condition is met.

Available macros for stop conditions:
- cross_up(state, threshold): detects upward threshold crossing
- cross_down(state, threshold): detects downward threshold crossing  
- in_interval(state, lower, upper): checks if state is in range
- enters_interval(state, lower, upper): detects entering a range
- leaves_interval(state, lower, upper): detects leaving a range
- increasing(state): detects if state is increasing
- decreasing(state): detects if state is decreasing
- changed(state): detects any change in state

Note: Event macros that use lagged values (like cross_up) require the
simulation to have at least one prior step, so they won't trigger at t0.
"""

from dynlib import setup


def get_exit_reason(res):
    """Return a user-friendly description of why the simulation ended."""
    if res.exited_early:
        return "Early exit (stop condition met)"
    elif res.ok:
        return "Completed normally (max steps reached)"
    else:
        return "Failed or interrupted"


def demo_simple_threshold():
    """Example 1: Stop using a simple threshold condition"""
    print("\n" + "=" * 70)
    print("Example 1: Simple Threshold (x > 0.8)")
    print("=" * 70)
    
    model = '''
inline:
[model]
type = "map"
label = "Logistic Map - Simple Threshold"

[states]
x = 0.1

[params]
r = 3.5

[equations.rhs]
x = "r * x * (1 - x)"

[sim]
t0 = 0.0
dt = 1.0
stepper = "map"
record = true
stop = "x > 0.8"
'''
    
    sim = setup(model, jit=False)
    sim.run(N=100)
    res = sim.results()
    
    print(f"Exit reason: {get_exit_reason(res)}")
    print(f"Steps executed: {res.step_count_final}")
    print(f"Final state: x = {res['x'][-1]:.6f}")
    print(f"Interpretation: Stopped when x first exceeded 0.8")


def demo_interval_check():
    """Example 2: Stop using in_interval macro"""
    print("\n" + "=" * 70)
    print("Example 2: Interval Check with in_interval()")
    print("=" * 70)
    
    model = '''
inline:
[model]
type = "map"
label = "Logistic Map - Interval Check"

[states]
x = 0.1

[params]
r = 3.5

[equations.rhs]
x = "r * x * (1 - x)"

[sim]
t0 = 0.0
dt = 1.0
stepper = "map"
record = true
stop = "in_interval(x, 0.82, 0.86)"
'''
    
    sim = setup(model, jit=False)
    sim.run(N=100)
    res = sim.results()
    
    print(f"Exit reason: {get_exit_reason(res)}")
    print(f"Steps executed: {res.step_count_final}")
    print(f"Final state: x = {res['x'][-1]:.6f}")
    print(f"Interpretation: Stopped when x entered the interval [0.82, 0.86]")


def demo_cross_up():
    """Example 3: Stop using cross_up macro (requires lag)"""
    print("\n" + "=" * 70)
    print("Example 3: Threshold Crossing with cross_up()")
    print("=" * 70)
    
    model = '''
inline:
[model]
type = "map"
label = "Logistic Map - Cross Up Detection"

[states]
x = 0.1

[params]
r = 3.5

[equations.rhs]
x = "r * x * (1 - x)"

[sim]
t0 = 0.0
dt = 1.0
stepper = "map"
record = true
stop = "cross_up(x, 0.8)"
'''
    
    sim = setup(model, jit=False)
    sim.run(N=100)
    res = sim.results()
    
    print(f"Exit reason: {get_exit_reason(res)}")
    print(f"Steps executed: {res.step_count_final}")
    print(f"Final state: x = {res['x'][-1]:.6f}")
    if res.n >= 2:
        print(f"Previous state: x = {res['x'][-2]:.6f}")
        print(f"Interpretation: Stopped when x crossed 0.8 from below")
        print(f"  (previous: {res['x'][-2]:.6f} <= 0.8, current: {res['x'][-1]:.6f} > 0.8)")


def demo_decreasing():
    """Example 4: Stop when state starts decreasing"""
    print("\n" + "=" * 70)
    print("Example 4: Detect Decreasing Trend with decreasing()")
    print("=" * 70)
    
    model = '''
inline:
[model]
type = "map"
label = "Logistic Map - Detect Decrease"

[states]
x = 0.1

[params]
r = 3.5

[equations.rhs]
x = "r * x * (1 - x)"

[sim]
t0 = 0.0
dt = 1.0
stepper = "map"
record = true
stop = "decreasing(x)"
'''
    
    sim = setup(model, jit=False)
    sim.run(N=100)
    res = sim.results()
    
    print(f"Exit reason: {get_exit_reason(res)}")
    print(f"Steps executed: {res.step_count_final}")
    print(f"Final state: x = {res['x'][-1]:.6f}")
    if res.n >= 2:
        print(f"Previous state: x = {res['x'][-2]:.6f}")
        print(f"Interpretation: Stopped when x started decreasing")
        print(f"  (previous: {res['x'][-2]:.6f}, current: {res['x'][-1]:.6f})")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EARLY EXIT FEATURE DEMONSTRATION")
    print("Using Built-in DSL Macros for Stop Conditions")
    print("=" * 70)
    
    demo_simple_threshold()
    demo_interval_check()
    demo_cross_up()
    demo_decreasing()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Use res.exited_early to check if stop condition was triggered")
    print("  • res.ok is True for both normal completion and early exit")
    print("  • Simple conditions (x > 0.8) work at any step")
    print("  • in_interval() checks current state without lag")
    print("  • cross_up(), decreasing(), etc. use lag and require prior steps")
    print("  • All conditions are evaluated after each simulation step")
    print("=" * 70 + "\n")
