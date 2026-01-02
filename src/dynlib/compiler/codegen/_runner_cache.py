# src/dynlib/compiler/codegen/_runner_cache.py
"""Shared helpers for runner disk cache management."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, MutableMapping, Optional, Tuple


__all__ = [
    "RunnerCacheRequest",
    "RunnerCacheConfig",
    "RunnerDiskCache",
    "CacheLock",
    "DiskCacheUnavailable",
    "canonical_dtype_name",
    "dtype_token",
    "sanitize_token",
    "platform_triple",
    "hash_payload",
    "gather_env_pins",
]

_RUNNER_ABI_VERSION = 3


@dataclass(frozen=True)
class RunnerCacheRequest:
    spec_hash: str
    stepper_name: str
    structsig: Tuple[int, ...]
    dtype: str
    cache_root: Path
    variant: Optional[str] = None
    template_version: Optional[str] = None
    runner_kind: Optional[str] = None


@dataclass(frozen=True)
class RunnerCacheConfig:
    module_prefix: str
    export_name: str
    render_module_source: Callable[[RunnerCacheRequest], str]
    env_pins_factory: Callable[[str], Dict[str, str]]


class DiskCacheUnavailable(RuntimeError):
    """Raised when runner disk cache cannot be used."""


class RunnerDiskCache:
    """Disk-backed cache for runner callables shared by continuous/discrete runners."""

    def __init__(
        self,
        request: RunnerCacheRequest,
        *,
        inproc_cache: MutableMapping[str, Callable],
        config: RunnerCacheConfig,
    ) -> None:
        self.request = request
        self._config = config
        self._inproc_cache = inproc_cache
        self.stepper_token = sanitize_token(request.stepper_name)
        self.dtype_token = dtype_token(request.dtype)
        self.platform_token = platform_triple()
        self.env_pins = config.env_pins_factory(self.platform_token)
        self.payload = self._build_digest_payload()
        self.digest = hash_payload(self.payload)
        shard = self.digest[:2]
        self.cache_dir = (
            request.cache_root
            / "jit"
            / "runners"
            / self.stepper_token
            / self.dtype_token
            / self.platform_token
            / shard
            / self.digest
        )
        self.module_name = f"{config.module_prefix}_{self.digest}"

    def get_or_build(self) -> Tuple[Callable, bool]:
        cached = self._inproc_cache.get(self.digest)
        module_path = self.cache_dir / "runner_mod.py"
        if cached is not None:
            if module_path.exists():
                return cached, True
            self._materialize()
            return cached, False
        runner_fn, from_disk = self._load_or_build()
        self._inproc_cache[self.digest] = runner_fn
        return runner_fn, from_disk

    def _load_or_build(self) -> Tuple[Callable, bool]:
        regen_attempted = False
        built = False
        while True:
            runner_fn = self._try_import()
            if runner_fn is not None:
                return runner_fn, not built
            if regen_attempted:
                raise DiskCacheUnavailable(
                    f"runner cache at {self.cache_dir} is corrupt and could not be rebuilt"
                )
            self._materialize()
            regen_attempted = True
            built = True

    def _try_import(self) -> Optional[Callable]:
        module_path = self.cache_dir / "runner_mod.py"
        if not module_path.exists():
            return None
        try:
            return self._import_runner(module_path)
        except DiskCacheUnavailable:
            raise
        except Exception:
            self._delete_cache_dir()
            return None

    def _import_runner(self, module_path: Path) -> Callable:
        spec = importlib.util.spec_from_file_location(self.module_name, module_path)
        if spec is None or spec.loader is None:
            raise DiskCacheUnavailable(f"unable to load runner module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except RuntimeError as exc:
            message = str(exc)
            if "cannot cache function" in message:
                raise DiskCacheUnavailable(
                    f"Numba cannot cache runner under {module_path.parent}: {message}"
                ) from exc
            # Other RuntimeErrors are treated as cache corruption
            with contextlib.suppress(KeyError):
                del sys.modules[self.module_name]
            raise
        except Exception:
            with contextlib.suppress(KeyError):
                del sys.modules[self.module_name]
            raise
        runner_fn = getattr(module, self._config.export_name, None)
        if runner_fn is None:
            raise DiskCacheUnavailable(
                f"Cached runner module missing '{self._config.export_name}' callable"
            )
        return runner_fn

    def _materialize(self) -> None:
        parent = self.cache_dir.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DiskCacheUnavailable(
                f"cannot create cache directory {parent}: {exc}"
            ) from exc

        lock_path = parent / f".{self.cache_dir.name}.lock"
        lock = CacheLock(lock_path)
        acquired = lock.acquire()
        try:
            if not acquired and self._wait_for_existing_builder():
                return
            if self.cache_dir.exists():
                return

            tmp_dir = parent / f".{self.cache_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir()
            try:
                self._write_runner_package(tmp_dir)
                tmp_dir.replace(self.cache_dir)
            finally:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError as exc:
            raise DiskCacheUnavailable(
                f"failed to materialize runner cache at {self.cache_dir}: {exc}"
            ) from exc
        finally:
            lock.release()

    def _wait_for_existing_builder(self, timeout: float = 5.0) -> bool:
        module_path = self.cache_dir / "runner_mod.py"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if module_path.exists():
                return True
            time.sleep(0.05)
        return False

    def _write_runner_package(self, tmp_dir: Path) -> None:
        init_path = tmp_dir / "__init__.py"
        init_path.write_text(
            f"__all__ = ['{self._config.export_name}']\n",
            encoding="utf-8",
        )

        module_source = self._config.render_module_source(self.request)
        (tmp_dir / "runner_mod.py").write_text(module_source, encoding="utf-8")

        meta_payload = {
            "hash": self.digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": self.payload,
        }
        meta_text = json.dumps(meta_payload, indent=2, sort_keys=True) + "\n"
        (tmp_dir / "meta.json").write_text(meta_text, encoding="utf-8")

    def _delete_cache_dir(self) -> None:
        if not self.cache_dir.exists():
            return
        tombstone = self.cache_dir.with_name(
            f"{self.cache_dir.name}.corrupt-{uuid.uuid4().hex[:6]}"
        )
        try:
            self.cache_dir.replace(tombstone)
        except OSError:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            return
        shutil.rmtree(tombstone, ignore_errors=True)

    def _build_digest_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "spec_hash": self.request.spec_hash,
            "stepper": self.request.stepper_name,
            "structsig": [int(x) for x in self.request.structsig],
            "dtype": canonical_dtype_name(self.request.dtype),
            "abi": _RUNNER_ABI_VERSION,
            "env": self.env_pins,
        }
        if self.request.variant is not None:
            payload["variant"] = self.request.variant
        if self.request.template_version is not None:
            payload["template_version"] = self.request.template_version
        if self.request.runner_kind is not None:
            payload["runner_kind"] = self.request.runner_kind
        return payload


class CacheLock:
    def __init__(self, path: Path):
        self.path = path
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        try:
            self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode())
            return True
        except FileExistsError:
            return False
        except OSError:
            return False

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        finally:
            self._fd = None
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.path)


def canonical_dtype_name(dtype: str) -> str:
    token = dtype.strip().lower()
    if token.startswith("f") and token[1:].isdigit():
        return f"float{token[1:]}"
    return token


def dtype_token(dtype: str) -> str:
    canonical = canonical_dtype_name(dtype)
    if canonical.startswith("float") and canonical[5:].isdigit():
        return f"f{canonical[5:]}"
    return canonical.replace("/", "-").replace(" ", "_")


def sanitize_token(value: str) -> str:
    token = value.strip().lower()
    safe = [ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in token]
    collapsed = "".join(safe).strip("-")
    return collapsed or "default"


def platform_triple() -> str:
    os_part = {
        "darwin": "macos",
        "linux": "linux",
        "win32": "windows",
    }.get(sys.platform, sys.platform)
    arch = platform.machine().lower() or "unknown"
    arch = arch.replace(" ", "-")
    endian = sys.byteorder
    return f"{os_part}-{arch}-{endian}"


def hash_payload(payload: Dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.blake2b(blob.encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def gather_env_pins(
    *,
    platform_token: str,
    dynlib_version: str,
    python_version: str,
    numba_version: Optional[str],
    llvmlite_version: Optional[str],
    cpu_name: Optional[str],
) -> Dict[str, str]:
    pins = {
        "dynlib": dynlib_version,
        "python": python_version,
        "platform": platform_token,
        "numba": numba_version or "unknown",
        "llvmlite": llvmlite_version or "unknown",
    }
    if cpu_name:
        pins["cpu_name"] = cpu_name.strip()
    return pins
