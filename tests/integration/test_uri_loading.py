# tests/integration/test_uri_loading.py
"""
Integration tests for URI-based model loading (Slice 6).

Tests end-to-end model building using different URI schemes:
- inline: models
- Absolute and relative paths
- TAG:// resolution
- Fragment-based mod selection
- Multiple mods application
- Error handling with helpful messages
"""
from __future__ import annotations
import pytest
from pathlib import Path
import tempfile
import os

from dynlib.compiler.build import build, load_model_from_uri
from dynlib.compiler.paths import PathConfig
from dynlib.runtime.sim import Sim
from dynlib.errors import ModelNotFoundError, ConfigError, ModelLoadError
from dynlib.dsl.spec import ModelSpec


# Helper to get state/param values from ModelSpec
def get_state_value(spec: ModelSpec, name: str) -> float | int:
    """Get state initial condition by name."""
    idx = spec.states.index(name)
    return spec.state_ic[idx]


def get_param_value(spec: ModelSpec, name: str) -> float | int:
    """Get parameter value by name."""
    idx = spec.params.index(name)
    return spec.param_vals[idx]


# ---- Inline models ----------------------------------------------------------

def test_build_from_inline_uri():
    """Build a model from an inline: URI."""
    inline_model = """
[model]
type = "ode"
dtype = "float64"

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
    
    uri = f"inline: {inline_model}"
    model = build(uri, jit=False)
    
    assert model.spec.kind == "ode"
    assert model.spec.dtype == "float64"
    assert "x" in model.spec.states
    idx_x = model.spec.states.index("x")
    assert model.spec.state_ic[idx_x] == 1.0


def test_sim_with_inline_model():
    """Run a simulation with an inline model."""
    inline_model = """
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
    
    uri = f"inline: {inline_model}"
    full_model = build(uri, jit=False)
    
    # Convert to legacy Model for Sim
    from dynlib.runtime.model import Model
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
    results = sim.run()
    
    assert results.n > 0
    assert results.T[0] == 0.0
    assert results.Y[0, 0] == 1.0
    # Final value should be less than initial (decay)
    assert results.Y[0, results.n - 1] < 1.0


# ---- File-based models ------------------------------------------------------

def test_build_from_absolute_path(tmp_path):
    """Build a model from an absolute file path."""
    model_file = tmp_path / "test_model.toml"
    model_file.write_text("""
[model]
type = "ode"

[states]
x = 2.0

[params]
a = 0.5

[equations.rhs]
x = "-a * x"

[sim]
stepper = "euler"
""")
    
    model = build(str(model_file), jit=False)
    
    assert get_state_value(model.spec, "x") == 2.0
    assert get_param_value(model.spec, "a") == 0.5


def test_build_from_relative_path(tmp_path, monkeypatch):
    """Build a model from a relative file path."""
    model_file = tmp_path / "relative_model.toml"
    model_file.write_text("""
[model]
type = "ode"

[states]
y = 3.0

[params]
b = 2.0

[equations.rhs]
y = "-b * y"

[sim]
stepper = "rk4"
""")
    
    # Change to tmp_path
    monkeypatch.chdir(tmp_path)
    
    model = build("relative_model.toml", jit=False)
    
    assert get_state_value(model.spec, "y") == 3.0
    assert model.stepper_name == "rk4"


def test_build_extensionless_finds_toml(tmp_path, monkeypatch):
    """Extensionless path resolves to .toml file."""
    model_file = tmp_path / "mymodel.toml"
    model_file.write_text("""
[model]
type = "ode"

[states]
z = 1.5

[params]
c = 1.0

[equations.rhs]
z = "-c * z"

[sim]
stepper = "euler"
""")
    
    monkeypatch.chdir(tmp_path)
    
    # Should find mymodel.toml
    model = build("mymodel", jit=False)
    
    assert get_state_value(model.spec, "z") == 1.5


# ---- TAG:// resolution ------------------------------------------------------

def test_build_from_tag_uri(tmp_path):
    """Build a model using TAG:// URI."""
    root = tmp_path / "models"
    root.mkdir()
    
    model_file = root / "tagged.toml"
    model_file.write_text("""
[model]
type = "ode"

[states]
w = 10.0

[params]
k = 0.1

[equations.rhs]
w = "-k * w"

[sim]
stepper = "rk4"
""")
    
    config = PathConfig(tags={"test": [str(root)]})
    
    model = build("test://tagged.toml", config=config, jit=False)
    
    assert get_state_value(model.spec, "w") == 10.0
    assert get_param_value(model.spec, "k") == 0.1


def test_tag_uri_first_root_wins(tmp_path):
    """When multiple roots have the same file, first wins."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    
    model1 = root1 / "shared.toml"
    model1.write_text("""
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"

[sim]
stepper = "euler"
""")
    
    model2 = root2 / "shared.toml"
    model2.write_text("""
[model]
type = "ode"

[states]
x = 2.0

[params]
a = 2.0

[equations.rhs]
x = "-a * x"

[sim]
stepper = "rk4"
""")
    
    config = PathConfig(tags={"proj": [str(root1), str(root2)]})
    
    model = build("proj://shared.toml", config=config, jit=False)
    
    # Should use root1
    assert get_state_value(model.spec, "x") == 1.0
    assert model.stepper_name == "euler"


# ---- Mods with URIs ---------------------------------------------------------

def test_load_model_with_embedded_mod(tmp_path):
    """Load a model and apply an embedded mod using fragment."""
    model_file = tmp_path / "model_with_mods.toml"
    model_file.write_text("""
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"

[sim]
stepper = "euler"

[mods.fast]
name = "fast"

[mods.fast.set.params]
a = 10.0
""")
    
    spec = load_model_from_uri(f"{model_file}#mod=fast")
    
    # Mod should have changed a to 10.0
    assert get_param_value(spec, "a") == 10.0


def test_load_model_with_external_mod(tmp_path):
    """Load a model and apply an external mod file."""
    base_model = tmp_path / "base.toml"
    base_model.write_text("""
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"

[sim]
stepper = "euler"
""")
    
    mod_file = tmp_path / "speed_mod.toml"
    mod_file.write_text("""
[mod]
name = "speed_boost"

[mod.set.params]
a = 5.0
""")
    
    spec = load_model_from_uri(str(base_model), mods=[str(mod_file)])
    
    # Mod should have changed a to 5.0
    assert get_param_value(spec, "a") == 5.0


def test_load_model_with_multiple_mods(tmp_path):
    """Apply multiple mods in order."""
    base_model = tmp_path / "base.toml"
    base_model.write_text("""
[model]
type = "ode"

[states]
x = 1.0
y = 0.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"
y = "0"

[sim]
stepper = "euler"
""")
    
    mod1 = tmp_path / "mod1.toml"
    mod1.write_text("""
[mod]
name = "mod1"

[mod.set.states]
y = 5.0
""")
    
    mod2 = tmp_path / "mod2.toml"
    mod2.write_text("""
[mod]
name = "mod2"

[mod.set.params]
a = 2.0
""")
    
    spec = load_model_from_uri(str(base_model), mods=[str(mod1), str(mod2)])
    
    # Both mods should be applied
    assert get_state_value(spec, "y") == 5.0
    assert get_param_value(spec, "a") == 2.0


# ---- Error handling ---------------------------------------------------------

def test_model_not_found_lists_candidates(tmp_path):
    """ModelNotFoundError lists searched locations."""
    with pytest.raises(ModelNotFoundError) as exc_info:
        build(str(tmp_path / "nonexistent.toml"), jit=False)
    
    assert "nonexistent.toml" in str(exc_info.value)
    assert len(exc_info.value.candidates) > 0


def test_unknown_tag_helpful_error():
    """Unknown TAG raises ConfigError with helpful message."""
    config = PathConfig(tags={"known": ["/some/path"]})
    
    with pytest.raises(ConfigError) as exc_info:
        build("unknown://model.toml", config=config, jit=False)
    
    assert "unknown" in str(exc_info.value)
    assert "known" in str(exc_info.value)  # Lists known tags


def test_mod_not_found_in_file(tmp_path):
    """Selecting non-existent mod from file raises error."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("""
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"

[sim]
stepper = "euler"

[mods.existing]
name = "existing"
""")
    
    with pytest.raises(ModelLoadError) as exc_info:
        load_model_from_uri(f"{model_file}#mod=nonexistent")
    
    assert "nonexistent" in str(exc_info.value)
    assert "existing" in str(exc_info.value)  # Shows available mods


def test_malformed_inline_model():
    """Malformed inline TOML raises ModelLoadError."""
    bad_toml = "inline: [model\ntype = 'ode'"  # Missing ]
    
    with pytest.raises(ModelLoadError, match="Failed to parse inline model"):
        build(bad_toml, jit=False)


# ---- Backward compatibility -------------------------------------------------

def test_build_with_modelspec_still_works(tmp_path):
    """build() still accepts ModelSpec directly (backward compat)."""
    model_file = tmp_path / "test.toml"
    model_file.write_text("""
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"

[sim]
stepper = "euler"
""")
    
    # Load spec manually
    spec = load_model_from_uri(str(model_file))
    
    # Pass spec directly to build
    model = build(spec, jit=False)
    
    assert model.spec == spec
    assert model.stepper_name == "euler"


# ---- Real integration with existing models ----------------------------------

def test_load_existing_decay_model():
    """Load and run one of the existing test models using URI."""
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "decay.toml"
    
    if not model_path.exists():
        pytest.skip("Test data not available")
    
    full_model = build(str(model_path), jit=False)
    
    from dynlib.runtime.model import Model
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
    results = sim.run()
    
    assert results.n > 0
    assert results.T[0] == 0.0
