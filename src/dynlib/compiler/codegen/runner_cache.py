# src/dynlib/compiler/codegen/runner_cache.py
"""Disk cache helpers for runner variants and compiled callables."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import platform
import shutil
import sys
import textwrap
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import tomllib

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata  # type: ignore

from dynlib.compiler.codegen._runner_cache import (
    CacheLock,
    DiskCacheUnavailable,
    RunnerCacheConfig,
    RunnerCacheRequest,
    RunnerDiskCache,
    canonical_dtype_name,
    dtype_token,
    gather_env_pins,
    hash_payload,
    platform_triple,
    sanitize_token,
)
from dynlib.compiler.guards import _render_guards_inline_source
from dynlib.runtime.softdeps import softdeps

__all__ = [
    "configure_runner_disk_cache",
    "disable_runner_disk_cache",
    "get_runner_cache_token",
    "get_cached_runner",
    "configure_triplet_disk_cache",
    "configure_stepper_disk_cache",
    "consume_callable_disk_cache_request",
    "DiskCacheUnavailable",
    "JitTripletCache",
    "_StepperDiskCache",
]

_SOFTDEPS = softdeps()
_NUMBA_VERSION = _SOFTDEPS.numba_version
_LLVMLITE_VERSION = _SOFTDEPS.llvmlite_version


@dataclass(frozen=True)
class _RunnerCacheContext:
    spec_hash: str
    stepper_name: str
    structsig: Tuple[int, ...]
    dtype: str
    cache_root: Path


@dataclass(frozen=True)
class _CallableDiskCacheRequest:
    family: str            # "triplet" or "stepper"
    component: str
    function_name: str
    spec_hash: str
    stepper_name: str
    structsig: Tuple[int, ...]
    dtype: str
    cache_root: Path
    source: str


_runner_cache_contexts: Dict[Tuple[str, str], _RunnerCacheContext] = {}
_pending_callable_cache_request: Optional[_CallableDiskCacheRequest] = None
_inproc_runner_cache: Dict[str, Callable] = {}
_inproc_callable_cache: Dict[Tuple[str, str], Callable] = {}
_warned_reasons: set[str] = set()
_last_runner_cache_hit: bool = False


def _discover_dynlib_version() -> str:
    """Best-effort dynlib version lookup."""
    version = _read_pyproject_version()
    if version is not None:
        return version
    try:
        return importlib_metadata.version("dynlib")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0+local"


def _read_pyproject_version() -> Optional[str]:
    root = Path(__file__).resolve()
    for parent in root.parents:
        candidate = parent / "pyproject.toml"
        if not candidate.exists():
            continue
        try:
            with open(candidate, "rb") as fh:
                data = tomllib.load(fh)
        except Exception:
            continue
        project = data.get("project", {})
        version = project.get("version")
        if isinstance(version, str):
            return version
    return None


_DYNLIB_VERSION = _discover_dynlib_version()


def _env_pins(platform_token: str) -> Dict[str, str]:
    cpu_name = platform.processor() or platform.machine()
    return gather_env_pins(
        platform_token=platform_token,
        dynlib_version=_DYNLIB_VERSION,
        python_version=platform.python_version(),
        numba_version=_NUMBA_VERSION,
        llvmlite_version=_LLVMLITE_VERSION,
        cpu_name=cpu_name,
    )


def configure_runner_disk_cache(
    *,
    model_hash: str,
    stepper_name: str,
    structsig: Tuple[int, ...],
    dtype: str,
    cache_root: Path,
) -> None:
    """Register disk cache settings for a model's runner variants."""
    key = (model_hash, stepper_name)
    _runner_cache_contexts[key] = _RunnerCacheContext(
        spec_hash=model_hash,
        stepper_name=stepper_name,
        structsig=tuple(int(x) for x in structsig),
        dtype=str(dtype),
        cache_root=Path(cache_root).expanduser().resolve(),
    )


def disable_runner_disk_cache(*, model_hash: str, stepper_name: str) -> None:
    """Disable runner disk cache for a model (removes stored settings)."""
    key = (model_hash, stepper_name)
    _runner_cache_contexts.pop(key, None)


def get_runner_cache_token(
    model_hash: str,
    stepper_name: str,
) -> Optional[Tuple[str, Tuple[int, ...], str]]:
    """Return a cache token for variant caching based on current disk cache context."""
    context = _runner_cache_contexts.get((model_hash, stepper_name))
    if context is None:
        return None
    return (
        str(context.cache_root),
        context.structsig,
        canonical_dtype_name(context.dtype),
    )


def _render_runner_module_source(
    request: RunnerCacheRequest,
    *,
    source: str,
    function_name: str,
) -> str:
    runner_src = textwrap.dedent(source).lstrip()
    # Analysis variants inject hooks as globals, which numba won't cache reliably.
    cache_enabled = not request.variant.endswith("ANALYSIS")
    decorator = "@njit(cache=True, nogil=True)" if cache_enabled else "@njit(cache=False, nogil=True)"
    decorated = runner_src.replace(
        f"def {function_name}(",
        f"{decorator}\ndef {function_name}(",
        1,
    )
    guards_src = _render_guards_inline_source()

    header = "\n".join(
        [
            "# Auto-generated by dynlib.compiler.codegen.runner_cache",
            f"# variant={request.variant} kind={request.runner_kind} template={request.template_version}",
            "from __future__ import annotations",
            "import math",
            "from numba import njit",
            "from dynlib.runtime.runner_api import (",
            "    OK, STEPFAIL, NAN_DETECTED,",
            "    EARLY_EXIT, DONE, GROW_REC, GROW_EVT, USER_BREAK, TRACE_OVERFLOW",
            ")",
        ]
    )
    sections = [header, guards_src, decorated]
    return "\n\n".join(part for part in sections if part).strip() + "\n"


def get_cached_runner(
    *,
    model_hash: str,
    stepper_name: str,
    variant: str,
    template_version: str,
    runner_kind: str,
    source: str,
    function_name: str,
) -> Optional[Callable]:
    """
    Return a disk-cached runner if configured; otherwise None.

    The caller is responsible for injecting analysis globals (ANALYSIS_PRE/POST)
    before first invocation when using analysis-capable templates.
    """
    global _last_runner_cache_hit
    context = _runner_cache_contexts.get((model_hash, stepper_name))
    if context is None:
        return None

    request = RunnerCacheRequest(
        spec_hash=context.spec_hash,
        stepper_name=context.stepper_name,
        structsig=context.structsig,
        dtype=context.dtype,
        cache_root=context.cache_root,
        variant=variant,
        template_version=template_version,
        runner_kind=runner_kind,
    )

    cache_config = RunnerCacheConfig(
        module_prefix="dynlib_runner",
        export_name=function_name,
        render_module_source=lambda req: _render_runner_module_source(
            req,
            source=source,
            function_name=function_name,
        ),
        env_pins_factory=_env_pins,
    )

    cache = RunnerDiskCache(
        request,
        inproc_cache=_inproc_runner_cache,
        config=cache_config,
    )
    try:
        cached, from_disk = cache.get_or_build()
        _last_runner_cache_hit = from_disk
        return cached
    except DiskCacheUnavailable as exc:
        _warn_disk_cache_disabled(str(exc))
        _last_runner_cache_hit = False
        return None


def configure_triplet_disk_cache(
    *,
    component: str,
    spec_hash: str,
    stepper_name: str,
    structsig: Tuple[int, ...],
    dtype: str,
    cache_root: Path,
    source: str,
    function_name: Optional[str] = None,
) -> None:
    """Store disk cache context for the next RHS/events/update_aux JIT build."""
    global _pending_callable_cache_request
    _pending_callable_cache_request = _CallableDiskCacheRequest(
        family="triplet",
        component=component,
        function_name=function_name or component,
        spec_hash=spec_hash,
        stepper_name=stepper_name,
        structsig=tuple(int(x) for x in structsig),
        dtype=str(dtype),
        cache_root=Path(cache_root).expanduser().resolve(),
        source=source,
    )


def configure_stepper_disk_cache(
    *,
    spec_hash: str,
    stepper_name: str,
    structsig: Tuple[int, ...],
    dtype: str,
    cache_root: Path,
    source: str,
    function_name: str,
) -> None:
    """Store disk cache context for the next stepper JIT build."""
    global _pending_callable_cache_request
    _pending_callable_cache_request = _CallableDiskCacheRequest(
        family="stepper",
        component="stepper",
        function_name=function_name,
        spec_hash=spec_hash,
        stepper_name=stepper_name,
        structsig=tuple(int(x) for x in structsig),
        dtype=str(dtype),
        cache_root=Path(cache_root).expanduser().resolve(),
        source=source,
    )


def consume_callable_disk_cache_request() -> Optional[_CallableDiskCacheRequest]:
    """Consume pending callable disk cache request (triplet/stepper)."""
    global _pending_callable_cache_request
    req = _pending_callable_cache_request
    _pending_callable_cache_request = None
    return req


class JitTripletCache:
    def __init__(self, request: _CallableDiskCacheRequest):
        self.request = request
        self.component = request.component
        self.function_name = request.function_name
        self.source = request.source
        self.stepper_token = sanitize_token(request.stepper_name)
        self.dtype_token = dtype_token(request.dtype)
        self.platform_token = platform_triple()
        self.payload = self._build_digest_payload()
        self.digest = hash_payload(self.payload)
        shard = self.digest[:2]
        self.cache_dir = (
            request.cache_root
            / "jit"
            / "triplets"
            / self.stepper_token
            / self.dtype_token
            / self.platform_token
            / shard
            / self.digest
        )
        self.module_name = f"dynlib_{self.component}_{self.digest}"

    def get_or_build(self) -> Tuple[Callable, str, bool]:
        key = (self.component, self.digest)
        module_path = self._module_path()
        cached = _inproc_callable_cache.get(key)
        if cached is not None and module_path.exists():
            return cached, self.digest, True
        fn, hit = self._load_or_build(module_path)
        _inproc_callable_cache[key] = fn
        return fn, self.digest, hit

    def _module_path(self) -> Path:
        return self.cache_dir / f"{self.component}_mod.py"

    def _load_or_build(self, module_path: Path) -> Tuple[Callable, bool]:
        regen_attempted = False
        built = False
        while True:
            fn = self._try_import(module_path)
            if fn is not None:
                return fn, not built
            if regen_attempted:
                raise DiskCacheUnavailable(
                    f"{self.component} cache at {self.cache_dir} is corrupt and could not be rebuilt"
                )
            self._materialize(module_path)
            regen_attempted = True
            built = True

    def _try_import(self, module_path: Path) -> Optional[Callable]:
        if not module_path.exists():
            return None
        spec = importlib.util.spec_from_file_location(self.module_name, module_path)
        if spec is None or spec.loader is None:
            self._delete_cache_dir()
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception:
            with contextlib.suppress(KeyError):
                del sys.modules[self.module_name]
            self._delete_cache_dir()
            return None
        fn = getattr(module, self.function_name, None)
        if fn is None:
            self._delete_cache_dir()
            return None
        return fn

    def _materialize(self, module_path: Path) -> None:
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
            if not acquired and self._wait_for_existing_builder(module_path):
                return
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = module_path.with_suffix(f".tmp-{uuid.uuid4().hex[:8]}")
            try:
                rendered = _render_callable_module_source(self.source, self.function_name)
                tmp_path.write_text(rendered, encoding="utf-8")
                tmp_path.replace(module_path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
            self._write_metadata(self.component)
        except OSError as exc:
            raise DiskCacheUnavailable(
                f"failed to materialize callable cache at {self.cache_dir}: {exc}"
            ) from exc
        finally:
            lock.release()

    def _wait_for_existing_builder(self, module_path: Path, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if module_path.exists():
                return True
            time.sleep(0.05)
        return False

    def _write_metadata(self, component: str) -> None:
        meta_path = self.cache_dir / "meta.json"
        components: set[str] = set()
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                for entry in existing.get("components", []):
                    if isinstance(entry, str):
                        components.add(entry)
            except Exception:
                components = set()
        components.add(component)
        payload = {
            "hash": self.digest,
            "inputs": self.payload,
            "components": sorted(components),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        return {
            "spec_hash": self.request.spec_hash,
            "stepper": self.request.stepper_name,
            "structsig": list(self.request.structsig),
            "dtype": canonical_dtype_name(self.request.dtype),
            "env": _env_pins(self.platform_token),
        }


class _StepperDiskCache:
    def __init__(self, request: _CallableDiskCacheRequest):
        self.request = request
        self.function_name = request.function_name
        self.source = request.source
        self.stepper_token = sanitize_token(request.stepper_name)
        self.dtype_token = dtype_token(request.dtype)
        self.platform_token = platform_triple()
        self.payload = self._build_digest_payload()
        self.digest = hash_payload(self.payload)
        shard = self.digest[:2]
        self.cache_dir = (
            request.cache_root
            / "jit"
            / "steppers"
            / self.stepper_token
            / self.dtype_token
            / self.platform_token
            / shard
            / self.digest
        )
        self.module_name = f"dynlib_stepper_{self.digest}"

    def get_or_build(self) -> Tuple[Callable, str, bool]:
        key = ("stepper", self.digest)
        module_path = self._module_path()
        cached = _inproc_callable_cache.get(key)
        if cached is not None and module_path.exists():
            return cached, self.digest, True
        fn, hit = self._load_or_build(module_path)
        _inproc_callable_cache[key] = fn
        return fn, self.digest, hit

    def _module_path(self) -> Path:
        return self.cache_dir / "stepper_mod.py"

    def _load_or_build(self, module_path: Path) -> Tuple[Callable, bool]:
        regen_attempted = False
        built = False
        while True:
            fn = self._try_import(module_path)
            if fn is not None:
                return fn, not built
            if regen_attempted:
                raise DiskCacheUnavailable(
                    f"stepper cache at {self.cache_dir} is corrupt and could not be rebuilt"
                )
            self._materialize(module_path)
            regen_attempted = True
            built = True

    def _try_import(self, module_path: Path) -> Optional[Callable]:
        if not module_path.exists():
            return None
        spec = importlib.util.spec_from_file_location(self.module_name, module_path)
        if spec is None or spec.loader is None:
            self._delete_cache_dir()
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception:
            with contextlib.suppress(KeyError):
                del sys.modules[self.module_name]
            self._delete_cache_dir()
            return None
        fn = getattr(module, "stepper", None)
        if fn is None:
            self._delete_cache_dir()
            return None
        return fn

    def _materialize(self, module_path: Path) -> None:
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
            if not acquired and self._wait_for_existing_builder(module_path):
                return
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = module_path.with_suffix(f".tmp-{uuid.uuid4().hex[:8]}")
            try:
                rendered = _render_stepper_module_source(self.source, self.function_name)
                tmp_path.write_text(rendered, encoding="utf-8")
                tmp_path.replace(module_path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
            self._write_metadata()
        except OSError as exc:
            raise DiskCacheUnavailable(
                f"failed to materialize stepper cache at {self.cache_dir}: {exc}"
            ) from exc
        finally:
            lock.release()

    def _wait_for_existing_builder(self, module_path: Path, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if module_path.exists():
                return True
            time.sleep(0.05)
        return False

    def _write_metadata(self) -> None:
        meta_path = self.cache_dir / "meta.json"
        payload = {
            "hash": self.digest,
            "inputs": self.payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        return {
            "spec_hash": self.request.spec_hash,
            "stepper": self.request.stepper_name,
            "structsig": list(self.request.structsig),
            "dtype": canonical_dtype_name(self.request.dtype),
            "env": _env_pins(self.platform_token),
        }


def _render_callable_module_source(source: str, function_name: str) -> str:
    body = textwrap.dedent(source).strip()
    guards_src = _render_guards_inline_source()

    header = "\n".join(
        [
            "# Auto-generated by dynlib.compiler.codegen.runner_cache (callable cache)",
            "from __future__ import annotations",
            "import math",
            "from math import inf, nan",
            "from numba import njit",
            "from dynlib.runtime.runner_api import OK, STEPFAIL, NAN_DETECTED",
        ]
    )
    footer = [
        f"_callable_py = {function_name}",
        f"{function_name} = njit(cache=True)(_callable_py)",
        f"__all__ = [\"{function_name}\"]",
    ]
    sections = [header, guards_src, body, "\n".join(footer)]
    return "\n\n".join(part for part in sections if part).strip() + "\n"


def _render_stepper_module_source(source: str, function_name: str) -> str:
    body = textwrap.dedent(source).strip()
    guards_src = _render_guards_inline_source()

    header = "\n".join(
        [
            "# Auto-generated by dynlib.compiler.codegen.runner_cache (stepper cache)",
            "from __future__ import annotations",
            "import math",
            "from math import inf, nan",
            "from numba import njit",
            "from dynlib.runtime.runner_api import OK, STEPFAIL, NAN_DETECTED",
        ]
    )
    footer = [
        f"_stepper_py = {function_name}",
        "stepper = njit(cache=True)(_stepper_py)",
        "__all__ = [\"stepper\"]",
    ]
    sections = [header, guards_src, body, "\n".join(footer)]
    return "\n\n".join(part for part in sections if part).strip() + "\n"


def _warn_disk_cache_disabled(reason: str) -> None:
    if reason in _warned_reasons:
        return
    _warned_reasons.add(reason)
    warnings.warn(
        f"dynlib disk runner cache disabled: {reason}. Falling back to in-memory JIT.",
        RuntimeWarning,
        stacklevel=3,
    )
