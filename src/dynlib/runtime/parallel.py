from __future__ import annotations

import multiprocessing
import sys
from typing import Literal

ParallelMode = Literal["auto", "threads", "process", "none"]


WINDOWS_PROCESS_MAIN_GUARD_MESSAGE = (
    'parallel_mode="process" uses multiprocessing on Windows. Put the simulation '
    'or analysis call inside a main() function and run it under an '
    'if __name__ == "__main__": guard. For frozen applications, call '
    "multiprocessing.freeze_support() inside that guard before main()."
)


def is_windows() -> bool:
    return sys.platform == "win32"


def validate_parallel_mode(parallel_mode: str) -> None:
    if parallel_mode not in {"auto", "threads", "process", "none"}:
        raise ValueError(f"Unknown parallel_mode {parallel_mode!r}")


def require_windows_process_main_guard(parallel_mode: str) -> None:
    if parallel_mode != "process" or not is_windows():
        return
    if multiprocessing.current_process().name != "MainProcess":
        raise RuntimeError(WINDOWS_PROCESS_MAIN_GUARD_MESSAGE)


def should_use_process_parallel(
    parallel_mode: str,
    *,
    total: int,
    n_workers: int,
    auto_threshold: int = 1000,
) -> bool:
    validate_parallel_mode(parallel_mode)
    require_windows_process_main_guard(parallel_mode)
    if parallel_mode == "none" or parallel_mode == "threads":
        return False
    if total <= 1 or n_workers <= 1:
        return False
    if parallel_mode == "process":
        return True
    if is_windows():
        return False
    return total > auto_threshold


def resolve_thread_backend(
    parallel_mode: str,
    *,
    threads_are_effective: bool,
) -> Literal["threads", "none"]:
    validate_parallel_mode(parallel_mode)
    require_windows_process_main_guard(parallel_mode)
    if parallel_mode == "none":
        return "none"
    if parallel_mode == "threads":
        return "threads"
    if parallel_mode == "auto":
        return "threads" if threads_are_effective else "none"
    raise ValueError(
        'parallel_mode="process" requires a process-capable execution path; '
        'use an analysis sweep/basin helper for multiprocessing, or use '
        'parallel_mode="threads" for fast-path batch execution.'
    )
