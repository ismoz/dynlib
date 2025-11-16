# src/dynlib/compiler/codegen/cache_importer.py
"""
Import hook that lets Python locate dynlib disk cache modules by name.

Numba stores references to generated module names like ``dynlib_stepper_<digest>``
inside its disk cache metadata.  When those cached artifacts are loaded in a
fresh interpreter, Python attempts to import the module by name.  Since these
modules live under dynlib's private cache directories, they are not otherwise
importable which leads to ``ModuleNotFoundError`` during cache loads.

This module registers a light-weight meta path finder that knows how to locate
the on-disk module for a given digest and hand it back to Python's import
machinery.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

__all__ = ["register_cache_root"]


_COMPONENT_FAMILIES: Dict[str, Tuple[str, str]] = {
    # component -> (family directory, filename)
    "stepper": ("steppers", "stepper_mod.py"),
    "rhs": ("triplets", "rhs_mod.py"),
    "events_pre": ("triplets", "events_pre_mod.py"),
    "events_post": ("triplets", "events_post_mod.py"),
    "runner": ("runners", "runner_mod.py"),
    "runner_discrete": ("runners", "runner_mod.py"),
}


def _is_hex_digest(value: str) -> bool:
    return len(value) == 32 and all(ch in "0123456789abcdef" for ch in value.lower())


class _DynlibCacheModuleFinder(importlib.abc.MetaPathFinder):
    """Locate cached dynlib modules like dynlib_stepper_<digest> on disk."""

    def __init__(self) -> None:
        self._roots: Set[Path] = set()
        self._module_paths: Dict[str, Path] = {}
        self._lock = threading.RLock()

    def add_root(self, root: Path) -> None:
        root = root.expanduser().resolve()
        with self._lock:
            self._roots.add(root)

    def find_spec(self, fullname: str, path, target=None):
        if not fullname.startswith("dynlib_"):
            return None
        suffix = fullname[7:]
        component, sep, digest = suffix.rpartition("_")
        if not sep or not component or not _is_hex_digest(digest):
            return None
        module_path = self._resolve_module_path(fullname, component, digest)
        if module_path is None:
            return None
        loader = importlib.machinery.SourceFileLoader(fullname, str(module_path))
        return importlib.util.spec_from_loader(fullname, loader)

    def _resolve_module_path(self, fullname: str, component: str, digest: str) -> Optional[Path]:
        with self._lock:
            cached = self._module_paths.get(fullname)
            if cached and cached.exists():
                return cached
        family_info = _COMPONENT_FAMILIES.get(component)
        if family_info is None:
            return None
        family, filename = family_info
        shard = digest[:2]
        for root in list(self._roots):
            base = root / "jit" / family
            module_path = self._search_family(base, shard, digest, filename)
            if module_path is not None:
                with self._lock:
                    self._module_paths[fullname] = module_path
                return module_path
        return None

    def _search_family(
        self,
        base: Path,
        shard: str,
        digest: str,
        filename: str,
    ) -> Optional[Path]:
        try:
            level1 = list(base.iterdir())
        except FileNotFoundError:
            return None
        except OSError:
            return None

        for dir1 in level1:
            if not dir1.is_dir():
                continue
            try:
                level2 = list(dir1.iterdir())
            except OSError:
                continue
            for dir2 in level2:
                if not dir2.is_dir():
                    continue
                try:
                    level3 = list(dir2.iterdir())
                except OSError:
                    continue
                for dir3 in level3:
                    if not dir3.is_dir():
                        continue
                    shard_dir = dir3 / shard
                    if not shard_dir.is_dir():
                        continue
                    digest_dir = shard_dir / digest
                    candidate = digest_dir / filename
                    if candidate.exists():
                        return candidate
        return None


_IMPORTER: Optional[_DynlibCacheModuleFinder] = None
_IMPORTER_LOCK = threading.Lock()


def register_cache_root(root: Optional[Path]) -> None:
    """Register *root* so cached modules under it can be imported by name."""
    if root is None:
        return
    with _IMPORTER_LOCK:
        global _IMPORTER
        if _IMPORTER is None:
            _IMPORTER = _DynlibCacheModuleFinder()
            sys.meta_path.insert(0, _IMPORTER)
        _IMPORTER.add_root(root)
