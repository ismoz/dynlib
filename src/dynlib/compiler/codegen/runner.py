# src/dynlib/compiler/codegen/runner.py
"""
Generic runner.

Defines a single runner function with the frozen ABI that:
  1. Pre-events on committed state
  2. Stepper loop (single attempt for fixed-step; adaptive may retry internally)
  3. Commit: y_prev, y_curr, t
  4. Post-events on committed state
  5. Record (with capacity checks)
  6. Loop until t >= t_end or max_steps
  7. Return status codes per runner_api.py

The runner accepts the stepper as a callable parameter, so steppers can be
regular Python functions instead of generated source code.

Per guardrails: same function body is used with/without JIT; decoration happens
only in compiler/jit/*.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import os
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
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata  # type: ignore

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib  # type: ignore

if TYPE_CHECKING:
    from dynlib.steppers.base import StructSpec

# Import status codes from canonical source
from dynlib.runtime.runner_api import (
    OK, STEPFAIL, NAN_DETECTED,
    DONE, GROW_REC, GROW_EVT, USER_BREAK
)
from dynlib.runtime import guards

# Import centralized JIT compilation helper
from dynlib.compiler.jit.compile import jit_compile
from dynlib.compiler.codegen._runner_cache import (
    RunnerCacheRequest,
    RunnerCacheConfig,
    RunnerDiskCache,
    CacheLock,
    DiskCacheUnavailable,
    canonical_dtype_name,
    dtype_token,
    sanitize_token,
    platform_triple,
    gather_env_pins,
    hash_payload,
)

__all__ = [
    "runner",
    "get_runner",
    "configure_runner_disk_cache",
    "configure_triplet_disk_cache",
    "configure_stepper_disk_cache",
    "consume_callable_disk_cache_request",
    "last_runner_cache_hit",
]

try:
    import numba  # type: ignore
    _NUMBA_AVAILABLE = True
    _NUMBA_VERSION = getattr(numba, "__version__", "unknown")
except Exception:
    numba = None  # type: ignore
    _NUMBA_AVAILABLE = False
    _NUMBA_VERSION = None

try:
    import llvmlite  # type: ignore
    _LLVMLITE_VERSION = getattr(llvmlite, "__version__", "unknown")
except Exception:
    llvmlite = None  # type: ignore
    _LLVMLITE_VERSION = None


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


def runner(
    # scalars
    t0, t_end, dt_init,
    max_steps, n_state, record_interval,
    # state/params
    y_curr, y_prev, params,
    # struct banks (views)
    sp, ss,
    sw0, sw1, sw2, sw3,
    iw0, bw0,
    # stepper configuration (read-only)
    stepper_config,
    # proposals/outs (len-1 arrays where applicable)
    y_prop, t_prop, dt_next, err_est,
    # recording
    T, Y, STEP, FLAGS,
    # event log (present; cap may be 1 if disabled)
    EVT_CODE, EVT_INDEX, EVT_LOG_DATA,
    # event log scratch (for writing log values before copying)
    evt_log_scratch,
    # cursors & caps
    i_start, step_start, cap_rec, cap_evt,
    # control/outs (len-1)
    user_break_flag, status_out, hint_out,
    i_out, step_out, t_out,
    # function symbols (jittable callables)
    stepper, rhs, events_pre, events_post
):
    """
    Generic runner: fixed-step execution with events and recording.
    
    Frozen ABI signature - must match runner_api.py specification.
    
    Returns status code (int32).
    """
    # Initialize loop state
    t = float(t0)
    dt = float(dt_init)
    i = int(i_start)         # record cursor
    step = int(step_start)   # global step counter
    
    # Event log cursor: hint_out[0] is used to pass m between re-entries
    # On first call, hint_out[0] is 0; on re-entry after GROW_EVT, it contains the saved m
    m = int(hint_out[0])     # event log cursor (resume from hint)
    
    # Recording at t0 (if record_interval > 0)
    if record_interval > 0 and step == 0:
        # Record initial condition
        if i >= cap_rec:
            # Need growth before recording
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = GROW_REC
            hint_out[0] = m
            return GROW_REC
        
        T[i] = t
        for k in range(n_state):
            Y[k, i] = y_curr[k]
        STEP[i] = step
        FLAGS[i] = OK
        i += 1
    
    # Main integration loop
    while step < max_steps and t < t_end:
        # Check if we need to record a pending step from before growth
        # This happens when step_start > i_start (we've advanced steps but not recorded)
        if step > 0 and record_interval > 0 and (step % record_interval == 0) and step == step_start:
            # Re-entering after GROW_REC: attempt the pending record first
            if i >= cap_rec:
                # Still not enough space (should not happen with geometric growth)
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_state):
                Y[k, i] = y_curr[k]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1
        # 1. Pre-events on committed state
        event_code_pre, log_width_pre = events_pre(t, y_curr, params, evt_log_scratch)
        
        # Record pre-event if it fired and has log data
        if event_code_pre >= 0 and log_width_pre > 0:
            if m >= cap_evt:
                # Need event buffer growth
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            # Copy log data to buffers
            for log_idx in range(log_width_pre):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            
            EVT_CODE[m] = event_code_pre
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1
        
        # 2. Clip dt to avoid overshooting t_end
        if t + dt > t_end:
            dt = t_end - t
        
        # 3. Stepper attempt (fixed-step: single call; adaptive may loop internally)
        step_status = stepper(
            t, dt, y_curr, rhs, params,
            sp, ss, sw0, sw1, sw2, sw3, iw0, bw0,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        )
        
        # Check for stepper failure/termination
        # Steppers return: OK (accepted step) or terminal codes (STEPFAIL, NAN_DETECTED)
        # Fixed-step: single attempt; adaptive: internal accept/reject loop
        if step_status != OK:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = step_status
            hint_out[0] = m
            return step_status

        # Universal guard: never commit non-finite proposals
        if not guards.allfinite1d(y_prop):
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = NAN_DETECTED
            hint_out[0] = m
            return NAN_DETECTED
        
        # 4. Commit: y_prev <- y_curr, y_curr <- y_prop, t <- t_prop
        for k in range(n_state):
            y_prev[k] = y_curr[k]
            y_curr[k] = y_prop[k]
        t = t_prop[0]
        dt = dt_next[0]
        step += 1
        
        # 5. Post-events on committed state
        event_code_post, log_width_post = events_post(t, y_curr, params, evt_log_scratch)
        
        # Record post-event if it fired and has log data
        if event_code_post >= 0 and log_width_post > 0:
            if m >= cap_evt:
                # Need event buffer growth
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_EVT
                hint_out[0] = m
                return GROW_EVT
            
            # Copy log data to buffers
            for log_idx in range(log_width_post):
                EVT_LOG_DATA[m, log_idx] = evt_log_scratch[log_idx]
            
            EVT_CODE[m] = event_code_post
            EVT_INDEX[m] = (i - 1) if i > 0 else -1
            m += 1
        
        # 6. Record (if enabled and step matches record_interval)
        if record_interval > 0 and (step % record_interval == 0):
            if i >= cap_rec:
                # Need growth
                i_out[0] = i
                step_out[0] = step
                t_out[0] = t
                status_out[0] = GROW_REC
                hint_out[0] = m
                return GROW_REC
            
            T[i] = t
            for k in range(n_state):
                Y[k, i] = y_curr[k]
            STEP[i] = step
            FLAGS[i] = OK
            i += 1
        
        # Check user break flag (if implemented)
        if user_break_flag[0] != 0:
            i_out[0] = i
            step_out[0] = step
            t_out[0] = t
            status_out[0] = USER_BREAK
            hint_out[0] = m
            return USER_BREAK
        
        # Check for completion
        if t >= t_end:
            break
    
    # Successful completion
    i_out[0] = i
    step_out[0] = step
    t_out[0] = t
    status_out[0] = DONE
    hint_out[0] = m
    return DONE


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


_pending_cache_request: Optional[RunnerCacheRequest] = None
_pending_callable_cache_request: Optional[_CallableDiskCacheRequest] = None
_inproc_runner_cache: Dict[str, Callable] = {}
_inproc_callable_cache: Dict[Tuple[str, str], Callable] = {}
_warned_reasons: set[str] = set()
_last_runner_cache_hit: bool = False


def configure_runner_disk_cache(
    *,
    spec_hash: str,
    stepper_name: str,
    structsig: Tuple[int, ...],
    dtype: str,
    cache_root: Path,
) -> None:
    """Store the cache context for the next disk-backed runner build."""
    global _pending_cache_request
    _pending_cache_request = RunnerCacheRequest(
        spec_hash=spec_hash,
        stepper_name=stepper_name,
        structsig=tuple(int(x) for x in structsig),
        dtype=str(dtype),
        cache_root=Path(cache_root).expanduser().resolve(),
    )


def _consume_cache_request() -> Optional[RunnerCacheRequest]:
    global _pending_cache_request
    req = _pending_cache_request
    _pending_cache_request = None
    return req


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
    """Store disk cache context for the next RHS/events JIT build."""
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


def get_runner(*, jit: bool = True, disk_cache: bool = True) -> Callable:
    """
    Get the runner function, optionally JIT-compiled.
    
    Per guardrails: JIT decoration happens only in compiler/jit/*.
    This is a convenience wrapper for build.py to avoid direct JIT logic there.
    
    Args:
        jit: Whether to apply JIT compilation (default True)
    
    Returns:
        Runner function (JIT-compiled if requested and available)
    
    Behavior:
        - If jit=False: returns pure Python runner
        - If jit=True and numba not installed: warns and returns pure Python runner
        - If jit=True and numba installed but compilation fails: raises RuntimeError
    """
    global _last_runner_cache_hit
    _last_runner_cache_hit = False

    if not jit:
        return runner

    if not disk_cache:
        return jit_compile(runner, jit=True).fn

    if not _NUMBA_AVAILABLE:
        # No disk cache support without numba
        return jit_compile(runner, jit=True).fn

    request = _consume_cache_request()
    if request is None:
        raise RuntimeError(
            "get_runner(disk_cache=True) called without configure_runner_disk_cache()"
        )

    cache_config = RunnerCacheConfig(
        module_prefix="dynlib_runner",
        export_name="runner",
        render_module_source=_render_runner_module_source,
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
        return jit_compile(runner, jit=True).fn


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
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception:
            with contextlib.suppress(KeyError):
                del sys.modules[self.module_name]
            self._delete_cache_dir()
            return None
        sys.modules[self.module_name] = module
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
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception:
            with contextlib.suppress(KeyError):
                del sys.modules[self.module_name]
            self._delete_cache_dir()
            return None
        sys.modules[self.module_name] = module
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


def _render_runner_module_source() -> str:
    runner_src = textwrap.dedent(inspect.getsource(runner)).lstrip()
    decorated = runner_src.replace("def runner(", "@njit(cache=True)\ndef runner(", 1)
    header = inspect.cleandoc(
        """
        # Auto-generated by dynlib.compiler.codegen.runner
        from __future__ import annotations
        from typing import TYPE_CHECKING

        from numba import njit
        from dynlib.runtime.runner_api import (
            OK, STEPFAIL, NAN_DETECTED,
            DONE, GROW_REC, GROW_EVT, USER_BREAK
        )
        from dynlib.runtime import guards

        if TYPE_CHECKING:
            from dynlib.steppers.base import StructSpec

        __all__ = ["runner"]
        """
    )
    return f"{header}\n\n{decorated}\n"


def _render_callable_module_source(source: str, function_name: str) -> str:
    body = textwrap.dedent(source).strip()
    header = "\n".join(
        [
            "# Auto-generated by dynlib.compiler.codegen.runner (triplet cache)",
            "from __future__ import annotations",
            "from numba import njit",
        ]
    )
    footer = [
        f"_{function_name}_py = {function_name}",
        f"{function_name} = njit(cache=True)(_{function_name}_py)",
        f"__all__ = [\"{function_name}\"]",
    ]
    sections = [header, body, "\n".join(footer)]
    return "\n\n".join(part for part in sections if part).strip() + "\n"


def _render_stepper_module_source(source: str, function_name: str) -> str:
    body = textwrap.dedent(source).strip()
    header = "\n".join(
        [
            "# Auto-generated by dynlib.compiler.codegen.runner (stepper cache)",
            "from __future__ import annotations",
            "from numba import njit",
            "from dynlib.runtime.runner_api import OK, STEPFAIL, NAN_DETECTED",
        ]
    )
    footer = [
        f"_stepper_py = {function_name}",
        "stepper = njit(cache=True)(_stepper_py)",
        "__all__ = [\"stepper\"]",
    ]
    sections = [header, body, "\n".join(footer)]
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


def last_runner_cache_hit() -> bool:
    return _last_runner_cache_hit
