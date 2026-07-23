from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynlib.runtime import parallel as parallel_policy


def test_windows_auto_never_selects_process(monkeypatch):
    monkeypatch.setattr(parallel_policy.sys, "platform", "win32")

    assert not parallel_policy.should_use_process_parallel(
        "auto",
        total=5000,
        n_workers=4,
    )


def test_posix_auto_selects_process_for_large_batches(monkeypatch):
    monkeypatch.setattr(parallel_policy.sys, "platform", "linux")

    assert parallel_policy.should_use_process_parallel(
        "auto",
        total=5000,
        n_workers=4,
    )


def test_explicit_process_ignores_auto_size_threshold(monkeypatch):
    monkeypatch.setattr(parallel_policy.sys, "platform", "linux")

    assert parallel_policy.should_use_process_parallel(
        "process",
        total=2,
        n_workers=2,
    )


def test_windows_process_in_spawned_child_raises_clear_error(monkeypatch):
    monkeypatch.setattr(parallel_policy.sys, "platform", "win32")
    monkeypatch.setattr(
        parallel_policy.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="SpawnPoolWorker-1"),
    )

    with pytest.raises(RuntimeError, match='parallel_mode="process" uses multiprocessing on Windows'):
        parallel_policy.should_use_process_parallel(
            "process",
            total=2,
            n_workers=2,
        )


def test_windows_auto_resolves_to_threads_only_when_effective(monkeypatch):
    monkeypatch.setattr(parallel_policy.sys, "platform", "win32")

    assert parallel_policy.resolve_thread_backend("auto", threads_are_effective=True) == "threads"
    assert parallel_policy.resolve_thread_backend("auto", threads_are_effective=False) == "none"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
