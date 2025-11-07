# tests/integration/test_block_equations_sim.py
"""
Integration tests for block-form equations.
Verifies that simulations with block-form equations produce correct results.
"""
import pytest
import numpy as np


def _run_inline_model(model_toml: str):
    """Helper to run a model from inline TOML."""
    from dynlib.compiler.build import build
    from dynlib.runtime.sim import Sim
    from dynlib.runtime.model import Model
    
    full_model = build(model="inline: " + model_toml)
    
    # Convert FullModel to Model for Sim
    model = Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        struct=full_model.struct,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        model_dtype=full_model.model_dtype,
    )
    
    sim = Sim(model)
    sim.run()
    return sim.raw_results()


def test_block_form_exponential_decay():
    """Test exponential decay using block-form equations."""
    model_toml = """
[model]
type = "ode"
dtype = "float64"

[states]
x = 1.0

[params]
k = 0.5

[equations]
expr = "dx = -k*x"

[sim]
t0 = 0.0
t_end = 4.0
dt = 0.1
stepper = "rk4"
record = true
"""
    
    result = _run_inline_model(model_toml)
    
    # Analytical solution: x(t) = x0 * exp(-k*t)
    # At t=4.0: x = 1.0 * exp(-0.5*4.0) = exp(-2.0) ≈ 0.1353
    t_final = result.T[result.n - 1]
    x_final = result.Y[0, result.n - 1]
    
    expected = np.exp(-0.5 * t_final)
    
    assert abs(t_final - 4.0) < 0.01
    assert abs(x_final - expected) < 1e-3


def test_block_form_coupled_system():
    """Test coupled ODE system with block-form equations."""
    model_toml = """
[model]
type = "ode"
dtype = "float64"

[states]
x = 1.0
y = 0.0

[params]
a = 1.0
b = 2.0

[equations]
expr = '''
dx = -a*x + b*y
dy = a*x - b*y
'''

[sim]
t0 = 0.0
t_end = 2.0
dt = 0.01
stepper = "rk4"
record = true
"""
    
    result = _run_inline_model(model_toml)
    
    # Verify conservation: d(x+y)/dt = 0, so x+y should be constant
    Y = result.Y[:, :result.n]
    x_vals = Y[0, :]
    y_vals = Y[1, :]
    sum_vals = x_vals + y_vals
    
    # x(0) + y(0) = 1.0 + 0.0 = 1.0
    assert np.allclose(sum_vals, 1.0, atol=1e-6)


def test_block_form_with_d_paren_syntax():
    """Test d(x) = ... syntax in simulation."""
    model_toml = """
[model]
type = "ode"

[states]
x = 2.0
y = 1.0

[params]
k = 0.3

[equations]
expr = '''
d(x) = -k*x
d(y) = k*x
'''

[sim]
t0 = 0.0
t_end = 5.0
dt = 0.05
stepper = "rk4"
"""
    
    result = _run_inline_model(model_toml)
    
    # Conservation: d(x+y)/dt = 0
    Y = result.Y[:, :result.n]
    sum_vals = Y[0, :] + Y[1, :]
    
    # x(0) + y(0) = 2.0 + 1.0 = 3.0
    assert np.allclose(sum_vals, 3.0, atol=1e-6)
    
    # x should decrease, y should increase
    assert Y[0, -1] < Y[0, 0]
    assert Y[1, -1] > Y[1, 0]


def test_mixed_form_simulation():
    """Test simulation with mixed rhs + block equations."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0
y = 0.0
z = 0.5

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[equations]
expr = '''
dy = a*x
dz = -a*z
'''

[sim]
t0 = 0.0
t_end = 3.0
dt = 0.01
stepper = "rk4"
"""
    
    result = _run_inline_model(model_toml)
    
    # Verify execution completes without error
    assert result.T[result.n - 1] > 2.9
    
    # x and z should decay, y should grow
    Y = result.Y[:, :result.n]
    assert Y[0, -1] < Y[0, 0]  # x decays
    assert Y[1, -1] > Y[1, 0]  # y grows
    assert Y[2, -1] < Y[2, 0]  # z decays


def test_block_form_with_aux_in_simulation():
    """Test that aux variables work correctly in block-form equations during simulation."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0
v = 0.0

[params]
k = 2.0
m = 1.0

[aux]
F = "-k*x"
a = "F/m"

[equations]
expr = '''
dx = v
dv = a
'''

[sim]
t0 = 0.0
t_end = 6.28
dt = 0.01
stepper = "rk4"
"""
    
    result = _run_inline_model(model_toml)
    
    # This is a harmonic oscillator: d²x/dt² = -k*x/m
    # With k/m = 2, angular frequency ω = sqrt(2)
    # Period T = 2π/ω ≈ 4.44
    # At t ≈ 6.28 (slightly more than one period), should be close to initial state
    
    Y = result.Y[:, :result.n]
    x_final = Y[0, -1]
    
    # Should oscillate; check that it hasn't blown up
    assert abs(x_final) < 2.0  # Should stay bounded


def test_block_form_with_functions_in_simulation():
    """Test user-defined functions in block-form equations during simulation."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
k = 0.4

[functions.saturated_decay]
args = ["u", "rate"]
expr = "-rate*u/(1+abs(u))"

[equations]
expr = "dx = saturated_decay(x, k)"

[sim]
t0 = 0.0
t_end = 10.0
dt = 0.1
stepper = "rk4"
"""
    
    result = _run_inline_model(model_toml)
    
    # Should decay to near zero but more slowly than linear
    Y = result.Y[:, :result.n]
    x_final = Y[0, -1]
    
    assert x_final < 1.0  # Should decay
    assert x_final > 0.0  # Should not go negative


def test_block_form_different_steppers():
    """Test that block-form works with different steppers."""
    model_toml_template = """
[model]
type = "ode"

[states]
x = 1.0

[params]
k = 0.5

[equations]
expr = "dx = -k*x"

[sim]
t0 = 0.0
t_end = 2.0
dt = 0.1
stepper = "{stepper}"
"""
    
    for stepper in ["euler", "rk4"]:
        model_toml = model_toml_template.format(stepper=stepper)
        result = _run_inline_model(model_toml)
        
        # Should complete and decay
        assert result.T[result.n - 1] > 1.9
        assert result.Y[0, result.n - 1] < result.Y[0, 0]


def test_block_form_preserves_order():
    """Test that equations in block form are evaluated in correct order."""
    model_toml = """
[model]
type = "ode"

[states]
a = 1.0
b = 2.0
c = 3.0

[params]
k = 0.1

[equations]
expr = '''
da = -k*a
db = a - k*b
dc = b - k*c
'''

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.01
stepper = "rk4"
"""
    
    result = _run_inline_model(model_toml)
    
    # Should complete without error
    assert result.T[result.n - 1] > 0.99
    
    # All states should remain positive given the parameters
    Y = result.Y[:, :result.n]
    assert np.all(Y >= 0)
