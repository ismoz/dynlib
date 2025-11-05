"""Test inline model declaration formats."""
import pytest
from dynlib.compiler.build import build


def test_inline_same_line():
    """Test inline model with content on same line as 'inline:'."""
    uri = "inline: [model]\ntype='ode'\n[states]\nx=1.0\n[params]\na=1.0\n[equations.rhs]\nx='-a*x'\n[sim]\nt0=0.0\nt_end=1.0\ndt=0.1\nstepper='euler'"
    
    model = build(uri, jit=False)
    assert model.spec.kind == "ode"
    assert "x" in model.spec.states


def test_inline_separate_lines():
    """Test inline model with 'inline:' on separate line (cleaner format)."""
    uri = """
    inline:
    [model]
    type = "ode"
    
    [states]
    x = 1.0
    
    [params]
    a = 1.0
    
    [equations.rhs]
    x = "-a * x"
    
    [sim]
    t0 = 0.0
    t_end = 1.0
    dt = 0.1
    stepper = "euler"
    """
    
    model = build(uri, jit=False)
    assert model.spec.kind == "ode"
    assert "x" in model.spec.states


def test_inline_no_leading_whitespace():
    """Test inline model without leading whitespace."""
    uri = """inline:
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
stepper = "euler"
"""
    
    model = build(uri, jit=False)
    assert model.spec.kind == "ode"
    assert "x" in model.spec.states


def test_inline_with_indented_content():
    """Test inline model with consistently indented content."""
    uri = """
    inline:
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a * x"
        
        [sim]
        t0 = 0.0
        t_end = 1.0
        dt = 0.1
        stepper = "euler"
    """
    
    model = build(uri, jit=False)
    assert model.spec.kind == "ode"
    assert "x" in model.spec.states
