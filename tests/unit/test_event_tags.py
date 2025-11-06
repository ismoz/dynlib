"""
Unit tests for event tagging functionality.

Tests cover:
- Tag parsing from TOML
- Tag normalization (deduplication, sorting)
- Tag validation (format, duplicates)
- Tag index building
"""
import pytest
from pathlib import Path
import tomllib

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.errors import ModelLoadError


def test_event_tags_basic_parsing():
    """Test that tags are parsed correctly from TOML."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["important", "reset"],
            }
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    # Check that event has tags
    assert len(spec.events) == 1
    event = spec.events[0]
    assert event.name == "reset"
    
    # Tags should be normalized: sorted and deduped
    assert event.tags == ("important", "reset")


def test_event_tags_normalization():
    """Test that tags are normalized (deduplicated and sorted)."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["zebra", "alpha", "alpha", "beta"],  # duplicates and unsorted
            }
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    event = spec.events[0]
    # Should be sorted and deduplicated
    assert event.tags == ("alpha", "beta", "zebra")


def test_event_tags_empty():
    """Test that empty tags list is handled correctly."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": [],
            }
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    event = spec.events[0]
    assert event.tags == ()


def test_event_tags_absent():
    """Test that missing tags field defaults to empty tuple."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                # No tags field
            }
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    event = spec.events[0]
    assert event.tags == ()


def test_event_tags_validation_invalid_format():
    """Test that invalid tag formats are rejected."""
    # Tag starting with digit
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["123invalid"],
            }
        },
    }
    
    normal = parse_model_v2(doc)
    with pytest.raises(ModelLoadError, match="tag '123invalid' is invalid"):
        build_spec(normal)


def test_event_tags_validation_special_chars():
    """Test that tags with invalid special characters are rejected."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["tag@invalid"],
            }
        },
    }
    
    normal = parse_model_v2(doc)
    with pytest.raises(ModelLoadError, match="tag 'tag@invalid' is invalid"):
        build_spec(normal)


def test_event_tags_validation_valid_formats():
    """Test that valid tag formats are accepted."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["valid_tag", "also-valid", "_underscore", "CamelCase123"],
            }
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    event = spec.events[0]
    # Should be sorted
    assert "CamelCase123" in event.tags
    assert "also-valid" in event.tags
    assert "valid_tag" in event.tags
    assert "_underscore" in event.tags


def test_event_tags_validation_empty_tag():
    """Test that empty tags are rejected."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": [""],
            }
        },
    }
    
    normal = parse_model_v2(doc)
    with pytest.raises(ModelLoadError, match="tag cannot be empty"):
        build_spec(normal)


def test_event_tags_validation_non_string():
    """Test that non-string tags are rejected during parsing."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["valid", 123],  # int instead of string
            }
        },
    }
    
    with pytest.raises(ModelLoadError, match=r"tags must be a list of strings"):
        parse_model_v2(doc)


def test_event_tags_validation_duplicate_detection():
    """Test that duplicate tags in the original list are normalized (deduplicated)."""
    # Note: build_spec normalizes by deduplicating, this is not an error
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["reset", "important", "reset"],
            }
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    # Duplicates should be removed, and result should be sorted
    event = spec.events[0]
    assert event.tags == ("important", "reset")  # deduplicated and sorted


def test_tag_index_single_tag():
    """Test that tag index is built correctly for single-tagged events."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x", "y": "a * y"}},
        "events": {
            "reset_x": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["reset"],
            },
            "reset_y": {
                "phase": "post",
                "cond": "y > 1.5",
                "action": {"y": "0.5"},
                "tags": ["reset"],
            },
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    # Both events have "reset" tag
    assert "reset" in spec.tag_index
    assert spec.tag_index["reset"] == ("reset_x", "reset_y")


def test_tag_index_multiple_tags():
    """Test that tag index is built correctly for multi-tagged events."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0, "y": 0.5},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x", "y": "a * y"}},
        "events": {
            "event_a": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["tag1", "tag2"],
            },
            "event_b": {
                "phase": "post",
                "cond": "y > 1.5",
                "action": {"y": "0.5"},
                "tags": ["tag2", "tag3"],
            },
            "event_c": {
                "phase": "pre",
                "cond": "x + y > 2.0",
                "tags": ["tag1"],
            },
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    # tag1 -> event_a, event_c
    assert "tag1" in spec.tag_index
    assert spec.tag_index["tag1"] == ("event_a", "event_c")
    
    # tag2 -> event_a, event_b
    assert "tag2" in spec.tag_index
    assert spec.tag_index["tag2"] == ("event_a", "event_b")
    
    # tag3 -> event_b
    assert "tag3" in spec.tag_index
    assert spec.tag_index["tag3"] == ("event_b",)


def test_tag_index_empty_when_no_tags():
    """Test that tag index is empty when no events have tags."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
            }
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    assert spec.tag_index == {}


def test_tag_index_preserves_event_order():
    """Test that tag index preserves event declaration order."""
    doc = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "third": {
                "phase": "post",
                "cond": "x < 0.3",
                "action": {"x": "1.0"},
                "tags": ["common"],
            },
            "first": {
                "phase": "post",
                "cond": "x < 0.1",
                "action": {"x": "1.0"},
                "tags": ["common"],
            },
            "second": {
                "phase": "post",
                "cond": "x < 0.2",
                "action": {"x": "1.0"},
                "tags": ["common"],
            },
        },
    }
    
    normal = parse_model_v2(doc)
    spec = build_spec(normal)
    
    # Tag index should preserve declaration order (TOML dict order)
    assert "common" in spec.tag_index
    assert spec.tag_index["common"] == ("third", "first", "second")


def test_tagged_events_from_toml_file():
    """Test loading a complete model with tagged events from a TOML file."""
    data_dir = Path(__file__).parent.parent / "data" / "models"
    model_path = data_dir / "tagged_events.toml"
    
    with open(model_path, "rb") as f:
        data = tomllib.load(f)
    
    normal = parse_model_v2(data)
    spec = build_spec(normal)
    
    # Check events
    event_names = [e.name for e in spec.events]
    assert "reset_x" in event_names
    assert "reset_y" in event_names
    assert "monitor" in event_names
    assert "boundary" in event_names
    
    # Check reset_x tags
    reset_x = [e for e in spec.events if e.name == "reset_x"][0]
    assert reset_x.tags == ("critical", "reset", "state-x")  # sorted
    
    # Check reset_y tags
    reset_y = [e for e in spec.events if e.name == "reset_y"][0]
    assert reset_y.tags == ("reset",)
    
    # Check monitor tags (explicitly empty)
    monitor = [e for e in spec.events if e.name == "monitor"][0]
    assert monitor.tags == ()
    
    # Check boundary tags (implicitly empty)
    boundary = [e for e in spec.events if e.name == "boundary"][0]
    assert boundary.tags == ()
    
    # Check tag index
    assert "reset" in spec.tag_index
    assert spec.tag_index["reset"] == ("reset_x", "reset_y")
    
    assert "critical" in spec.tag_index
    assert spec.tag_index["critical"] == ("reset_x",)
    
    assert "state-x" in spec.tag_index
    assert spec.tag_index["state-x"] == ("reset_x",)


def test_spec_hash_includes_tags():
    """Test that spec hash changes when tags change."""
    from dynlib.dsl.spec import compute_spec_hash
    
    doc1 = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["tag1"],
            }
        },
    }
    
    doc2 = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": ["tag2"],  # Different tag
            }
        },
    }
    
    spec1 = build_spec(parse_model_v2(doc1))
    spec2 = build_spec(parse_model_v2(doc2))
    
    hash1 = compute_spec_hash(spec1)
    hash2 = compute_spec_hash(spec2)
    
    assert hash1 != hash2, "Hashes should differ when tags differ"


def test_spec_hash_stable_with_empty_tags():
    """Test that spec hash is consistent for empty vs absent tags."""
    doc1 = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                "tags": [],
            }
        },
    }
    
    doc2 = {
        "model": {"type": "ode"},
        "states": {"x": 1.0},
        "params": {"a": 1.0},
        "equations": {"rhs": {"x": "-a * x"}},
        "events": {
            "reset": {
                "phase": "post",
                "cond": "x < 0.5",
                "action": {"x": "1.0"},
                # No tags field
            }
        },
    }
    
    spec1 = build_spec(parse_model_v2(doc1))
    spec2 = build_spec(parse_model_v2(doc2))
    
    from dynlib.dsl.spec import compute_spec_hash
    hash1 = compute_spec_hash(spec1)
    hash2 = compute_spec_hash(spec2)
    
    assert hash1 == hash2, "Hashes should be identical for empty vs absent tags"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
