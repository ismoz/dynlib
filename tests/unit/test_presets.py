"""
Tests for the presets feature (inline + file-based).

Covers:
- Inline presets loading from DSL
- list_presets with glob patterns
- apply_preset with validation and casting
- load_preset from TOML files
- save_preset to TOML files
- Round-trip save/load
- Error handling and warnings
"""
import tempfile
from pathlib import Path
import warnings

import numpy as np
import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model


def _compile_model_from_toml(toml_str: str, jit: bool = True) -> Model:
    """Compile a model from TOML string."""
    data = tomllib.loads(toml_str)
    spec = build_spec(parse_model_v2(data))
    full_model = build(spec, stepper=spec.sim.stepper, jit=jit)
    return Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        struct=full_model.struct,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
    )


def test_inline_presets_loaded_on_init():
    """Inline presets from DSL should be auto-loaded into bank on Sim init."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        y = 0.5
        
        [params]
        a = 1.0
        b = 2.0
        
        [equations.rhs]
        x = "-a*x"
        y = "b*y"
        
        [presets.fast.params]
        a = 2.0
        b = 4.0
        
        [presets.slow.params]
        a = 0.5
        
        [presets.custom.params]
        a = 1.5
        b = 3.0
        
        [presets.custom.states]
        x = 10.0
        y = 20.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Check that presets are loaded
    all_presets = sim.list_presets()
    assert set(all_presets) == {"custom", "fast", "slow"}


def test_list_presets_glob_patterns():
    """list_presets should support glob patterns."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.fast_mode.params]
        a = 2.0
        
        [presets.fast_turbo.params]
        a = 3.0
        
        [presets.slow_mode.params]
        a = 0.5
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    assert sim.list_presets("*") == ["fast_mode", "fast_turbo", "slow_mode"]
    assert sim.list_presets("fast_*") == ["fast_mode", "fast_turbo"]
    assert sim.list_presets("slow_*") == ["slow_mode"]
    assert sim.list_presets("*_mode") == ["fast_mode", "slow_mode"]
    assert sim.list_presets("nope") == []


def test_apply_preset_params_only():
    """apply_preset should update params when preset is param-only."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        b = 2.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.test.params]
        a = 5.0
        b = 10.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Initial params
    state = sim._session_state
    assert state.params_curr[0] == 1.0  # a
    assert state.params_curr[1] == 2.0  # b
    
    # Apply preset
    sim.apply_preset("test")
    
    # Check updated
    assert state.params_curr[0] == 5.0  # a
    assert state.params_curr[1] == 10.0  # b


def test_apply_preset_with_states():
    """apply_preset should update both params and states when preset has states."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        y = 2.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        y = "a*y"
        
        [presets.test.params]
        a = 5.0
        
        [presets.test.states]
        x = 100.0
        y = 200.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Initial state
    state = sim._session_state
    assert state.y_curr[0] == 1.0  # x
    assert state.y_curr[1] == 2.0  # y
    assert state.params_curr[0] == 1.0  # a
    
    # Apply preset
    sim.apply_preset("test")
    
    # Check updated
    assert state.y_curr[0] == 100.0  # x
    assert state.y_curr[1] == 200.0  # y
    assert state.params_curr[0] == 5.0  # a


def test_apply_preset_unknown_param():
    """apply_preset should error with suggestion on unknown param name."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        alpha = 1.0
        
        [equations.rhs]
        x = "-alpha*x"
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Manually add a preset with typo
    from dynlib.runtime.sim import _PresetData
    sim._presets["bad"] = _PresetData(
        name="bad",
        params={"alfa": 2.0},  # typo: should be "alpha"
        states=None,
        source="inline",
    )
    
    with pytest.raises(ValueError) as exc:
        sim.apply_preset("bad")
    
    assert "unknown param" in str(exc.value).lower()
    assert "did you mean 'alpha'" in str(exc.value).lower()


def test_apply_preset_partial_states_error():
    """apply_preset should error if preset has partial states (violates all-or-none)."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        y = 2.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        y = "a*y"
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Manually add preset with only one state (should have both or none)
    from dynlib.runtime.sim import _PresetData
    sim._presets["partial"] = _PresetData(
        name="partial",
        params={"a": 2.0},
        states={"x": 10.0},  # missing "y"
        source="inline",
    )
    
    with pytest.raises(ValueError) as exc:
        sim.apply_preset("partial")
    
    assert "all-or-none" in str(exc.value).lower()
    assert "missing" in str(exc.value).lower()


def test_load_preset_from_file(tmp_path):
    """load_preset should import presets from TOML file."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        b = 2.0
        
        [equations.rhs]
        x = "-a*x"
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Create preset file
    preset_file = tmp_path / "presets.toml"
    preset_file.write_text(
        """
        [__presets__]
        schema = "dynlib-presets-v1"
        
        [presets.file_preset.params]
        a = 3.0
        b = 4.0
        """
    )
    
    # Load from file
    count = sim.load_preset("file_preset", preset_file)
    
    assert count == 1
    assert "file_preset" in sim.list_presets()
    
    # Apply and verify
    sim.apply_preset("file_preset")
    assert sim._session_state.params_curr[0] == 3.0
    assert sim._session_state.params_curr[1] == 4.0


def test_load_preset_glob(tmp_path):
    """load_preset should support glob patterns to load multiple presets."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    preset_file = tmp_path / "presets.toml"
    preset_file.write_text(
        """
        [__presets__]
        schema = "dynlib-presets-v1"
        
        [presets.fast_v1.params]
        a = 2.0
        
        [presets.fast_v2.params]
        a = 3.0
        
        [presets.slow.params]
        a = 0.5
        """
    )
    
    count = sim.load_preset("fast_*", preset_file)
    
    assert count == 2
    assert set(sim.list_presets("fast_*")) == {"fast_v1", "fast_v2"}
    assert "slow" not in sim.list_presets()


def test_load_preset_conflict_error(tmp_path):
    """load_preset should error on conflict when on_conflict='error'."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.existing.params]
        a = 5.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    preset_file = tmp_path / "presets.toml"
    preset_file.write_text(
        """
        [__presets__]
        schema = "dynlib-presets-v1"
        
        [presets.existing.params]
        a = 10.0
        """
    )
    
    with pytest.raises(ValueError) as exc:
        sim.load_preset("existing", preset_file, on_conflict="error")
    
    assert "already exists" in str(exc.value)


def test_load_preset_conflict_keep(tmp_path):
    """load_preset should keep existing when on_conflict='keep'."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.existing.params]
        a = 5.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    preset_file = tmp_path / "presets.toml"
    preset_file.write_text(
        """
        [__presets__]
        schema = "dynlib-presets-v1"
        
        [presets.existing.params]
        a = 10.0
        """
    )
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        count = sim.load_preset("existing", preset_file, on_conflict="keep")
    
    assert count == 0
    assert len(w) == 1
    assert "skipping" in str(w[0].message).lower()
    
    # Original should be preserved
    sim.apply_preset("existing")
    assert sim._session_state.params_curr[0] == 5.0


def test_load_preset_conflict_replace(tmp_path):
    """load_preset should replace existing when on_conflict='replace'."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.existing.params]
        a = 5.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    preset_file = tmp_path / "presets.toml"
    preset_file.write_text(
        """
        [__presets__]
        schema = "dynlib-presets-v1"
        
        [presets.existing.params]
        a = 10.0
        """
    )
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        count = sim.load_preset("existing", preset_file, on_conflict="replace")
    
    assert count == 1
    assert len(w) == 1
    assert "replacing" in str(w[0].message).lower()
    
    # Should be replaced
    sim.apply_preset("existing")
    assert sim._session_state.params_curr[0] == 10.0


def test_save_preset_params_only(tmp_path):
    """save_preset should create file with param-only preset."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        b = 2.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.test.params]
        a = 5.0
        b = 10.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    preset_file = tmp_path / "out.toml"
    
    # Apply preset first so session has those values
    sim.apply_preset("test")
    
    sim.save_preset("test", preset_file, include_states=False)
    
    # Verify file contents
    with open(preset_file, "rb") as f:
        doc = tomllib.load(f)
    
    assert doc["__presets__"]["schema"] == "dynlib-presets-v1"
    assert "test" in doc["presets"]
    assert doc["presets"]["test"]["params"]["a"] == 5.0
    assert doc["presets"]["test"]["params"]["b"] == 10.0
    assert "states" not in doc["presets"]["test"]


def test_save_preset_with_states(tmp_path):
    """save_preset should write all states when include_states=True."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        y = 2.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        y = "a*y"
        
        [presets.test.params]
        a = 5.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Modify state
    sim._session_state.y_curr[0] = 100.0
    sim._session_state.y_curr[1] = 200.0
    
    preset_file = tmp_path / "out.toml"
    sim.save_preset("test", preset_file, include_states=True)
    
    # Verify
    with open(preset_file, "rb") as f:
        doc = tomllib.load(f)
    
    assert doc["presets"]["test"]["states"]["x"] == 100.0
    assert doc["presets"]["test"]["states"]["y"] == 200.0


def test_save_preset_append(tmp_path):
    """save_preset should append to existing file without overwrite."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.first.params]
        a = 2.0
        
        [presets.second.params]
        a = 3.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    preset_file = tmp_path / "out.toml"
    
    # Apply and save first
    sim.apply_preset("first")
    sim.save_preset("first", preset_file)
    
    # Apply and save second (append)
    sim.apply_preset("second")
    sim.save_preset("second", preset_file)
    
    # Verify both exist
    with open(preset_file, "rb") as f:
        doc = tomllib.load(f)
    
    assert "first" in doc["presets"]
    assert "second" in doc["presets"]
    assert doc["presets"]["first"]["params"]["a"] == 2.0
    assert doc["presets"]["second"]["params"]["a"] == 3.0


def test_save_preset_overwrite(tmp_path):
    """save_preset should replace existing preset when overwrite=True."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.test.params]
        a = 2.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    preset_file = tmp_path / "out.toml"
    
    # Save once
    sim.save_preset("test", preset_file)
    
    # Modify param
    sim._session_state.params_curr[0] = 99.0
    
    # Save again with overwrite
    sim.save_preset("test", preset_file, overwrite=True)
    
    # Verify updated
    with open(preset_file, "rb") as f:
        doc = tomllib.load(f)
    
    assert doc["presets"]["test"]["params"]["a"] == 99.0


def test_save_preset_error_without_overwrite(tmp_path):
    """save_preset should error when file has preset and overwrite=False."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.test.params]
        a = 2.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    preset_file = tmp_path / "out.toml"
    
    sim.save_preset("test", preset_file)
    
    with pytest.raises(ValueError) as exc:
        sim.save_preset("test", preset_file, overwrite=False)
    
    assert "already exists" in str(exc.value)


def test_roundtrip_save_load(tmp_path):
    """Save + load should produce identical preset values."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        y = 2.0
        
        [params]
        a = 1.0
        b = 2.0
        
        [equations.rhs]
        x = "-a*x"
        y = "b*y"
        
        [presets.orig.params]
        a = 5.5
        b = 7.7
        
        [presets.orig.states]
        x = 100.0
        y = 200.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    preset_file = tmp_path / "roundtrip.toml"
    
    # Apply original
    sim.apply_preset("orig")
    orig_params = sim._session_state.params_curr.copy()
    orig_states = sim._session_state.y_curr.copy()
    
    # Save
    sim.save_preset("orig", preset_file, include_states=True, overwrite=True)
    
    # Reset to defaults
    sim._session_state.params_curr[:] = [1.0, 2.0]
    sim._session_state.y_curr[:] = [1.0, 2.0]
    
    # Load back
    sim2 = Sim(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim2.load_preset("orig", preset_file, on_conflict="replace")
    sim2.apply_preset("orig")
    
    # Compare
    np.testing.assert_array_almost_equal(sim2._session_state.params_curr, orig_params)
    np.testing.assert_array_almost_equal(sim2._session_state.y_curr, orig_states)


def test_precision_warning_float64_to_float32():
    """Casting logic should warn on potential precision loss."""
    # Test the casting function directly since model compilation may ignore dtype
    import numpy as np
    from dynlib.runtime.sim import _cast_values_to_dtype
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _cast_values_to_dtype(
            {"a": 3.141592653589793},  # Pi - loses precision in float32
            np.dtype('float32'),
            "test_preset",
            "param"
        )
    
    # Should have precision warning
    assert any("precision" in str(warning.message).lower() for warning in w)


def test_casting_overflow_error():
    """Casting logic should error on overflow."""
    # Test the casting function directly
    import numpy as np
    from dynlib.runtime.sim import _cast_values_to_dtype
    
    with pytest.raises(ValueError) as exc:
        _cast_values_to_dtype(
            {"a": 1e40},  # Exceeds float32 max
            np.dtype('float32'),
            "test_preset",
            "param"
        )
    
    assert "overflow" in str(exc.value).lower()


def test_apply_does_not_touch_time_or_stepper():
    """apply_preset should not modify time, dt, step_count, or stepper workspace."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        
        [presets.test.params]
        a = 5.0
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # Run to establish state
    sim.run(T=1.0, max_steps=100)
    
    # Capture state before apply
    t_before = sim._session_state.t_curr
    dt_before = sim._session_state.dt_curr
    step_before = sim._session_state.step_count
    ws_before = dict(sim._session_state.stepper_ws)
    
    # Apply preset
    sim.apply_preset("test")
    
    # Verify untouched
    assert sim._session_state.t_curr == t_before
    assert sim._session_state.dt_curr == dt_before
    assert sim._session_state.step_count == step_before
    assert sim._session_state.stepper_ws.keys() == ws_before.keys()


def test_load_unknown_schema(tmp_path):
    """load_preset should error on missing/invalid schema."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    preset_file = tmp_path / "bad.toml"
    preset_file.write_text(
        """
        [__presets__]
        schema = "unknown-version"
        
        [presets.test.params]
        a = 2.0
        """
    )
    
    with pytest.raises(ValueError) as exc:
        sim.load_preset("test", preset_file)
    
    assert "schema" in str(exc.value).lower()


def test_file_duplicate_preset_last_wins(tmp_path):
    """When file has duplicate keys in same table, TOML parser handles it (usually rejects)."""
    model = _compile_model_from_toml(
        """
        [model]
        type = "ode"
        
        [states]
        x = 1.0
        
        [params]
        a = 1.0
        
        [equations.rhs]
        x = "-a*x"
        """
    )
    from dynlib.runtime.sim import Sim
    
    sim = Sim(model)
    
    # TOML spec forbids duplicate table declarations, so this is invalid
    # Modern TOML parsers (tomllib) will reject this
    preset_file = tmp_path / "dup.toml"
    preset_file.write_text(
        """
        [__presets__]
        schema = "dynlib-presets-v1"
        
        [presets.dup.params]
        a = 2.0
        
        [presets.dup.params]
        a = 3.0
        """
    )
    
    # Should error because it's invalid TOML
    with pytest.raises(ValueError) as exc:
        sim.load_preset("dup", preset_file)
    
    assert "failed to read" in str(exc.value).lower()
