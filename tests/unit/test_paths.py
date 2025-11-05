# tests/unit/test_paths.py
"""
Unit tests for path resolution and configuration.

Tests cover:
- Config file loading from platform-specific paths
- Environment variable overrides (DYNLIB_CONFIG, DYN_MODEL_PATH)
- TAG:// URI resolution with search order
- inline: parsing
- Absolute and relative path handling
- Fragment extraction (#mod=...)
- Security: path traversal prevention
- Error message quality
"""
from __future__ import annotations
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile

from dynlib.compiler.paths import (
    PathConfig,
    load_config,
    resolve_uri,
    parse_uri,
    _get_config_path,
    _parse_env_model_path,
)
from dynlib.errors import (
    ModelNotFoundError,
    ConfigError,
    PathTraversalError,
    AmbiguousModelError,
)


# ---- parse_uri tests --------------------------------------------------------

def test_parse_uri_no_fragment():
    """URI without fragment returns (uri, None)."""
    base, frag = parse_uri("model.toml")
    assert base == "model.toml"
    assert frag is None


def test_parse_uri_with_fragment():
    """URI with fragment splits correctly."""
    base, frag = parse_uri("model.toml#mod=drive")
    assert base == "model.toml"
    assert frag == "mod=drive"


def test_parse_uri_inline_no_split():
    """inline: URIs don't split on # even if present in content."""
    uri = "inline: [model]\ntype='ode'\n# comment"
    base, frag = parse_uri(uri)
    assert base == uri
    assert frag is None


def test_parse_uri_tag_with_fragment():
    """TAG:// URI with fragment."""
    base, frag = parse_uri("proj://decay.toml#mod=fast")
    assert base == "proj://decay.toml"
    assert frag == "mod=fast"


# ---- _get_config_path tests -------------------------------------------------

def test_get_config_path_env_override(monkeypatch):
    """DYNLIB_CONFIG env var overrides default path."""
    custom_path = "/custom/config.toml"
    monkeypatch.setenv("DYNLIB_CONFIG", custom_path)
    
    result = _get_config_path()
    assert str(result) == str(Path(custom_path).expanduser().resolve())


def test_get_config_path_linux(monkeypatch):
    """Linux uses XDG_CONFIG_HOME or ~/.config."""
    monkeypatch.delenv("DYNLIB_CONFIG", raising=False)
    
    # Test with XDG_CONFIG_HOME set
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/home/user/.config")
    
    result = _get_config_path()
    expected = Path("/home/user/.config/dynlib/config.toml").resolve()
    assert result == expected


def test_get_config_path_linux_no_xdg(monkeypatch):
    """Linux without XDG_CONFIG_HOME uses ~/.config."""
    monkeypatch.delenv("DYNLIB_CONFIG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    
    result = _get_config_path()
    expected = Path.home() / ".config" / "dynlib" / "config.toml"
    assert result == expected.resolve()


def test_get_config_path_macos(monkeypatch):
    """macOS uses ~/Library/Application Support."""
    monkeypatch.delenv("DYNLIB_CONFIG", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    
    result = _get_config_path()
    expected = Path.home() / "Library" / "Application Support" / "dynlib" / "config.toml"
    assert result == expected.resolve()


def test_get_config_path_windows(monkeypatch):
    """Windows uses %APPDATA%."""
    monkeypatch.delenv("DYNLIB_CONFIG", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", "C:\\Users\\test\\AppData\\Roaming")
    
    result = _get_config_path()
    # Just check it contains the right components (cross-platform)
    result_str = str(result)
    assert "dynlib" in result_str
    assert "config.toml" in result_str


# ---- _parse_env_model_path tests --------------------------------------------

def test_parse_env_model_path_empty(monkeypatch):
    """Empty DYN_MODEL_PATH returns empty dict."""
    monkeypatch.delenv("DYN_MODEL_PATH", raising=False)
    result = _parse_env_model_path()
    assert result == {}


def test_parse_env_model_path_unix(monkeypatch):
    """Unix format: TAG1=/p1,/p2:TAG2=/p3."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DYN_MODEL_PATH", "proj=/home/models,/opt/models:user=/home/user/models")
    
    result = _parse_env_model_path()
    assert result == {
        "proj": ["/home/models", "/opt/models"],
        "user": ["/home/user/models"],
    }


def test_parse_env_model_path_windows(monkeypatch):
    """Windows format: TAG1=C:\\p1,C:\\p2;TAG2=C:\\p3."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("DYN_MODEL_PATH", "proj=C:\\models,D:\\models;user=C:\\Users\\me\\models")
    
    result = _parse_env_model_path()
    assert result == {
        "proj": ["C:\\models", "D:\\models"],
        "user": ["C:\\Users\\me\\models"],
    }


def test_parse_env_model_path_invalid_no_equals(monkeypatch):
    """Invalid format without '=' raises ConfigError."""
    monkeypatch.setenv("DYN_MODEL_PATH", "proj/home/models")
    
    with pytest.raises(ConfigError, match="missing '='"):
        _parse_env_model_path()


def test_parse_env_model_path_invalid_empty_tag(monkeypatch):
    """Empty tag raises ConfigError."""
    monkeypatch.setenv("DYN_MODEL_PATH", "=/home/models")
    
    with pytest.raises(ConfigError, match="Empty tag"):
        _parse_env_model_path()


# ---- load_config tests ------------------------------------------------------

def test_load_config_no_file(tmp_path, monkeypatch):
    """Config loads successfully even if file doesn't exist (empty config)."""
    fake_config = tmp_path / "nonexistent.toml"
    monkeypatch.setenv("DYNLIB_CONFIG", str(fake_config))
    
    config = load_config()
    assert config.tags == {}


def test_load_config_valid_file(tmp_path, monkeypatch):
    """Valid config file is loaded correctly."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[paths]
proj = ["/home/models", "/opt/models"]
user = "/home/user/models"
""")
    monkeypatch.setenv("DYNLIB_CONFIG", str(config_file))
    
    config = load_config()
    assert config.tags == {
        "proj": ["/home/models", "/opt/models"],
        "user": ["/home/user/models"],
    }


def test_load_config_env_prepends(tmp_path, monkeypatch):
    """DYN_MODEL_PATH entries are prepended to config file tags."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[paths]
proj = ["/home/models"]
""")
    monkeypatch.setenv("DYNLIB_CONFIG", str(config_file))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DYN_MODEL_PATH", "proj=/env/models")
    
    config = load_config()
    # Env path should come first
    assert config.tags["proj"] == ["/env/models", "/home/models"]


def test_load_config_malformed_toml(tmp_path, monkeypatch):
    """Malformed TOML raises ConfigError."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("invalid toml [[[")
    monkeypatch.setenv("DYNLIB_CONFIG", str(config_file))
    
    with pytest.raises(ConfigError, match="Failed to load config"):
        load_config()


def test_load_config_invalid_paths_not_table(tmp_path, monkeypatch):
    """[paths] must be a table."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('[paths]\n"invalid"')
    monkeypatch.setenv("DYNLIB_CONFIG", str(config_file))
    
    # This will parse but paths won't be a dict - should handle gracefully
    # Actually tomllib will fail on this, so it's a parse error
    with pytest.raises(ConfigError):
        load_config()


def test_load_config_invalid_paths_value(tmp_path, monkeypatch):
    """[paths].tag must be string or list of strings."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('[paths]\nproj = 123')
    monkeypatch.setenv("DYNLIB_CONFIG", str(config_file))
    
    with pytest.raises(ConfigError, match="must be a string or list"):
        load_config()


# ---- resolve_uri tests ------------------------------------------------------

def test_resolve_uri_inline():
    """inline: URI returns content as-is."""
    uri = "inline: [model]\ntype='ode'"
    content, frag = resolve_uri(uri)
    assert content == "[model]\ntype='ode'"
    assert frag is None


def test_resolve_uri_inline_with_fragment():
    """inline: with fragment (though unusual)."""
    uri = "inline: [model]\ntype='ode'#mod=drive"
    content, frag = resolve_uri(uri)
    assert content == "[model]\ntype='ode'#mod=drive"
    assert frag is None  # inline: doesn't split on #


def test_resolve_uri_absolute_exists(tmp_path):
    """Absolute path to existing file."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("[model]\ntype='ode'")
    
    resolved, frag = resolve_uri(str(model_file))
    assert resolved == str(model_file.resolve())
    assert frag is None


def test_resolve_uri_absolute_not_found(tmp_path):
    """Absolute path to non-existent file raises ModelNotFoundError."""
    model_file = tmp_path / "nonexistent.toml"
    
    with pytest.raises(ModelNotFoundError) as exc_info:
        resolve_uri(str(model_file))
    
    assert str(model_file.resolve()) in exc_info.value.candidates


def test_resolve_uri_relative_exists(tmp_path, monkeypatch):
    """Relative path resolved from cwd."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("[model]\ntype='ode'")
    monkeypatch.chdir(tmp_path)
    
    resolved, frag = resolve_uri("model.toml")
    assert resolved == str(model_file.resolve())
    assert frag is None


def test_resolve_uri_extensionless_adds_toml(tmp_path, monkeypatch):
    """Extensionless path tries .toml extension."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("[model]\ntype='ode'")
    monkeypatch.chdir(tmp_path)
    
    resolved, frag = resolve_uri("model")
    assert resolved == str(model_file.resolve())


def test_resolve_uri_extensionless_ambiguous(tmp_path, monkeypatch):
    """Ambiguous extensionless match raises AmbiguousModelError."""
    (tmp_path / "model").write_text("plain")
    (tmp_path / "model.toml").write_text("[model]")
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(AmbiguousModelError) as exc_info:
        resolve_uri("model")
    
    assert len(exc_info.value.matches) == 2


def test_resolve_uri_with_fragment(tmp_path):
    """Fragment is extracted correctly."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("[model]\ntype='ode'")
    
    resolved, frag = resolve_uri(f"{model_file}#mod=drive")
    assert resolved == str(model_file.resolve())
    assert frag == "mod=drive"


def test_resolve_uri_tag_unknown_raises(tmp_path, monkeypatch):
    """Unknown TAG raises ConfigError."""
    config = PathConfig(tags={})
    
    with pytest.raises(ConfigError, match="Unknown tag 'unknown'"):
        resolve_uri("unknown://model.toml", config=config)


def test_resolve_uri_tag_found(tmp_path):
    """TAG:// resolves to first matching root."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    
    model_file = root2 / "model.toml"
    model_file.write_text("[model]\ntype='ode'")
    
    config = PathConfig(tags={"proj": [str(root1), str(root2)]})
    
    resolved, frag = resolve_uri("proj://model.toml", config=config)
    assert resolved == str(model_file.resolve())


def test_resolve_uri_tag_first_match_wins(tmp_path):
    """First root with matching file wins."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    
    model1 = root1 / "model.toml"
    model2 = root2 / "model.toml"
    model1.write_text("content1")
    model2.write_text("content2")
    
    config = PathConfig(tags={"proj": [str(root1), str(root2)]})
    
    resolved, frag = resolve_uri("proj://model.toml", config=config)
    assert resolved == str(model1.resolve())


def test_resolve_uri_tag_not_found_lists_candidates(tmp_path):
    """TAG:// not found lists all searched paths."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    
    config = PathConfig(tags={"proj": [str(root1), str(root2)]})
    
    with pytest.raises(ModelNotFoundError) as exc_info:
        resolve_uri("proj://missing.toml", config=config)
    
    assert "proj://missing.toml" in str(exc_info.value)
    # Should list all candidate paths
    assert len(exc_info.value.candidates) >= 2


def test_resolve_uri_tag_traversal_blocked(tmp_path):
    """TAG:// with path traversal is blocked."""
    root = tmp_path / "root"
    root.mkdir()
    
    config = PathConfig(tags={"proj": [str(root)]})
    
    with pytest.raises(PathTraversalError) as exc_info:
        resolve_uri("proj://../secret.toml", config=config)
    
    assert "proj://../secret.toml" in str(exc_info.value)


def test_resolve_uri_tag_nested_path(tmp_path):
    """TAG:// with nested relative path works."""
    root = tmp_path / "root"
    subdir = root / "subdir"
    subdir.mkdir(parents=True)
    
    model_file = subdir / "model.toml"
    model_file.write_text("[model]")
    
    config = PathConfig(tags={"proj": [str(root)]})
    
    resolved, frag = resolve_uri("proj://subdir/model.toml", config=config)
    assert resolved == str(model_file.resolve())


def test_resolve_uri_disable_extensionless(tmp_path, monkeypatch):
    """allow_extensionless=False doesn't try .toml extension."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("[model]")
    monkeypatch.chdir(tmp_path)
    
    # With extensionless disabled, "model" won't find "model.toml"
    with pytest.raises(ModelNotFoundError):
        resolve_uri("model", allow_extensionless=False)
    
    # But explicit extension still works
    resolved, _ = resolve_uri("model.toml", allow_extensionless=False)
    assert resolved == str(model_file.resolve())


def test_resolve_uri_expanduser(tmp_path, monkeypatch):
    """~ is expanded correctly."""
    # Create a model in a temp dir
    model_file = tmp_path / "model.toml"
    model_file.write_text("[model]")
    
    # Mock os.path.expanduser to return tmp_path for ~
    def mock_expanduser(path):
        if path.startswith("~"):
            return str(tmp_path) + path[1:]
        return path
    
    with patch("os.path.expanduser", side_effect=mock_expanduser):
        resolved, _ = resolve_uri("~/model.toml")
        assert resolved == str(model_file.resolve())


def test_resolve_uri_expandvars(tmp_path, monkeypatch):
    """Environment variables are expanded."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("[model]")
    
    monkeypatch.setenv("MY_DIR", str(tmp_path))
    
    resolved, _ = resolve_uri("$MY_DIR/model.toml")
    assert resolved == str(model_file.resolve())
