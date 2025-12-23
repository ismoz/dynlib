from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynlib import cli


def test_model_validate_success(capsys):
    code = cli.main(["model", "validate", "builtin://ode/expdecay.toml"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Model OK" in captured.out


def test_model_validate_missing_file(tmp_path: Path, capsys):
    missing = tmp_path / "missing.toml"
    code = cli.main(["model", "validate", str(missing)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Model not found" in captured.err


def test_steppers_list_filters(capsys):
    code = cli.main(["steppers", "list", "--kind", "map"])
    captured = capsys.readouterr()
    assert code == 0
    assert "map" in captured.out
    assert "kind=map" in captured.out

    # Test filtering by jit_capable - should find jittable steppers
    code = cli.main(["steppers", "list", "--jit_capable"])
    captured = capsys.readouterr()
    assert code == 0
    assert "jit_capable=True" in captured.out
    # Should find common jittable steppers like euler, rk4, etc.
    assert any(name in captured.out for name in ["euler", "rk4", "bdf2"])


def _write_meta(root: Path, family: str, *, stepper: str, dtype: str, spec_hash: str, digest: str, components=None) -> Path:
    directory = root / "jit" / family / digest
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "hash": digest,
        "inputs": {"stepper": stepper, "dtype": dtype, "spec_hash": spec_hash},
    }
    if components is not None:
        meta["components"] = list(components)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / "payload.bin").write_bytes(b"\x00" * 16)
    return directory


def test_cache_list_and_clear(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(cli, "resolve_cache_root", lambda: tmp_path)
    entry = _write_meta(
        tmp_path,
        "steppers",
        stepper="rk4",
        dtype="float64",
        spec_hash="abcd1234",
        digest="deadbeef",
    )
    _write_meta(
        tmp_path,
        "triplets",
        stepper="rk4",
        dtype="float64",
        spec_hash="abcd1234",
        digest="feedface",
        components=("rhs", "events_pre"),
    )

    code = cli.main(["cache", "path"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == str(tmp_path)

    code = cli.main(["cache", "list"])
    captured = capsys.readouterr()
    assert code == 0
    assert "stepper=rk4" in captured.out
    assert "components=rhs,events_pre" in captured.out

    code = cli.main(["cache", "clear"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--stepper" in captured.err

    code = cli.main(["cache", "clear", "--stepper", "rk4", "--dry_run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "[dry-run]" in captured.out
    assert entry.exists()

    code = cli.main(["cache", "clear", "--stepper", "rk4"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Deleted steppers cache" in captured.out
    assert not entry.exists()


def test_cache_clear_all(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(cli, "resolve_cache_root", lambda: tmp_path)
    _write_meta(
        tmp_path,
        "runners",
        stepper="rk4",
        dtype="float64",
        spec_hash="abcd1234",
        digest="aaabbbcc",
    )
    code = cli.main(["cache", "clear", "--all", "--dry_run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "[dry-run]" in captured.out or "does not exist" in captured.out

    code = cli.main(["cache", "clear", "--all"])
    captured = capsys.readouterr()
    assert code == 0
    assert not tmp_path.exists()
