"""Tests for disk-backed runner caching."""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
import tomllib
import pytest

pytest.importorskip("numba")

PYTHON_BIN = os.path.expanduser("~/.virtualenvs/pydefault/bin/python3")

from dynlib.compiler.build import build
from dynlib.compiler.paths import load_config, resolve_cache_root
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.runtime.model import Model
from dynlib.runtime.sim import Sim


MODEL_SRC = """
[model]
type = "ode"
dtype = "float64"

[sim]
t0 = 0.0
t_end = 0.1
dt = 0.05
record = true
record_interval = 1
stepper = "euler"

[states]
x = 1.0

[params]
a = 0.5

[equations.rhs]
x = "-a*x"
"""


def _build_spec():
    doc = tomllib.loads(MODEL_SRC)
    normal = parse_model_v2(doc)
    return build_spec(normal)


def _to_runtime(full) -> Model:
    return Model(
        spec=full.spec,
        stepper_name=full.stepper_name,
        workspace_sig=full.workspace_sig,
        rhs=full.rhs,
        events_pre=full.events_pre,
        events_post=full.events_post,
        stepper=full.stepper,
        runner=full.runner,
        spec_hash=full.spec_hash,
        dtype=full.dtype,
        rhs_source=full.rhs_source,
        events_pre_source=full.events_pre_source,
        events_post_source=full.events_post_source,
        stepper_source=full.stepper_source,
        lag_state_info=full.lag_state_info,
        uses_lag=full.uses_lag,
        equations_use_lag=full.equations_use_lag,
        make_stepper_workspace=full.make_stepper_workspace,
    )


def _configure_cache(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    cache_root = root / "jit-cache"
    config_path = root / "dynlib.toml"
    config_path.write_text(f'cache_root = "{cache_root}"\n[paths]\n', encoding="utf-8")
    monkeypatch.setenv("DYNLIB_CONFIG", str(config_path))
    # Reload config to capture the resolved cache root used by build()
    return resolve_cache_root(load_config())


def _run_sim(full_model) -> None:
    model = _to_runtime(full_model)
    sim = Sim(model)
    sim.run(max_steps=8)


def _single_cache_dir(cache_root: Path) -> Path:
    candidates = list(cache_root.glob("jit/runners/**/runner_mod.py"))
    assert candidates, "expected at least one cached runner"
    parents = {path.parent for path in candidates}
    assert len(parents) == 1, "expected exactly one digest"
    return next(iter(parents))


def _runner_build_script() -> str:
    return f"""
import tomllib
from dynlib.compiler.build import build
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.runtime.model import Model
from dynlib.runtime.sim import Sim

MODEL = tomllib.loads({MODEL_SRC!r})
spec = build_spec(parse_model_v2(MODEL))
full = build(spec, stepper='euler', jit=True, disk_cache=True)
model = Model(
    spec=full.spec,
    stepper_name=full.stepper_name,
    workspace_sig=full.workspace_sig,
    rhs=full.rhs,
    events_pre=full.events_pre,
    events_post=full.events_post,
    stepper=full.stepper,
    runner=full.runner,
    spec_hash=full.spec_hash,
    dtype=full.dtype,
    rhs_source=full.rhs_source,
    events_pre_source=full.events_pre_source,
    events_post_source=full.events_post_source,
    stepper_source=full.stepper_source,
    lag_state_info=full.lag_state_info,
    uses_lag=full.uses_lag,
    equations_use_lag=full.equations_use_lag,
    make_stepper_workspace=full.make_stepper_workspace,
)
Sim(model).run(max_steps=4)
"""


def _run_subprocess_build():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["DYNLIB_CONFIG"] = os.environ["DYNLIB_CONFIG"]
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    subprocess.run(
        [PYTHON_BIN, "-c", _runner_build_script()],
        cwd=str(repo_root),
        env=env,
        check=True,
    )


def test_disk_cache_materializes(tmp_path, monkeypatch):
    cache_root = _configure_cache(monkeypatch, tmp_path)
    full = build(_build_spec(), stepper="euler", jit=True, disk_cache=True)
    _run_sim(full)

    digest_dir = _single_cache_dir(cache_root)
    assert (digest_dir / "__init__.py").exists()
    assert (digest_dir / "runner_mod.py").exists()
    assert (digest_dir / "meta.json").exists()
    assert (digest_dir / "__pycache__").exists()


def test_disk_cache_cross_process_reuse(tmp_path, monkeypatch):
    cache_root = _configure_cache(monkeypatch, tmp_path)
    full = build(_build_spec(), stepper="euler", jit=True, disk_cache=True)
    _run_sim(full)

    digest_dir = _single_cache_dir(cache_root)
    runner_mod = digest_dir / "runner_mod.py"
    before = runner_mod.stat().st_mtime
    _run_subprocess_build()

    after = runner_mod.stat().st_mtime
    assert before == after, "runner_mod.py should be reused without rewrite"


def test_disk_cache_recovers_from_corruption(tmp_path, monkeypatch):
    cache_root = _configure_cache(monkeypatch, tmp_path)
    full = build(_build_spec(), stepper="euler", jit=True, disk_cache=True)
    _run_sim(full)

    digest_dir = _single_cache_dir(cache_root)
    runner_mod = digest_dir / "runner_mod.py"
    runner_mod.write_text("raise RuntimeError('corrupt cache')\n", encoding="utf-8")

    _run_subprocess_build()
    rebuilt = build(_build_spec(), stepper="euler", jit=True, disk_cache=True)
    _run_sim(rebuilt)

    contents = runner_mod.read_text(encoding="utf-8")
    assert "Auto-generated by dynlib" in contents


def test_env_pin_changes_create_new_digest(tmp_path, monkeypatch):
    cache_root = _configure_cache(monkeypatch, tmp_path)
    base_spec = _build_spec()

    model_f64 = build(base_spec, stepper="euler", jit=True, disk_cache=True, dtype="float64")
    _run_sim(model_f64)

    model_f32 = build(base_spec, stepper="euler", jit=True, disk_cache=True, dtype="float32")
    _run_sim(model_f32)

    digest_dirs = {path.parent for path in cache_root.glob("jit/runners/**/runner_mod.py")}
    assert len(digest_dirs) == 2


def test_disk_cache_fallback_when_unwritable(tmp_path, monkeypatch, request):
    if os.name == "nt":
        pytest.skip("chmod-based permission test not supported on Windows")
    locked = tmp_path / "locked"
    locked.mkdir()
    cache_root = locked / "cache"
    config_path = tmp_path / "dynlib.toml"
    config_path.write_text(f'cache_root = "{cache_root}"\n[paths]\n', encoding="utf-8")
    monkeypatch.setenv("DYNLIB_CONFIG", str(config_path))

    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    request.addfinalizer(lambda: locked.chmod(stat.S_IRWXU))

    spec = _build_spec()
    with pytest.warns(RuntimeWarning, match="disk runner cache disabled"):
        full = build(spec, stepper="euler", jit=True, disk_cache=True)
    _run_sim(full)

    assert not cache_root.exists(), "cache root should remain absent when unwritable"


def test_disk_cache_disabled_creates_no_files(tmp_path, monkeypatch):
    cache_root = _configure_cache(monkeypatch, tmp_path)
    spec = _build_spec()

    full = build(spec, stepper="euler", jit=True, disk_cache=False)
    _run_sim(full)
    assert not (cache_root / "jit").exists()

    second = build(spec, stepper="euler", jit=True, disk_cache=False)
    _run_sim(second)
    assert not (cache_root / "jit").exists()
