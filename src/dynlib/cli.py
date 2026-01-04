"""Command-line interface for dynlib."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, fields
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Sequence

import tomllib

# Ensure stepper registry populates via side effects
import dynlib.steppers  # noqa: F401

from dynlib.compiler.build import load_model_from_uri
from dynlib.compiler.paths import resolve_cache_root
from dynlib.errors import DynlibError
from dynlib.steppers.base import StepperCaps
from dynlib.steppers.registry import select_steppers

__all__ = ["main"]


@dataclass(frozen=True)
class CacheEntry:
    family: str
    path: Path
    stepper: str
    dtype: str
    spec_hash: str
    digest: str
    size_bytes: int
    components: tuple[str, ...] = ()


def _read_pyproject_version() -> str | None:
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
        project = data.get("project")
        if isinstance(project, dict):
            version = project.get("version")
            if isinstance(version, str):
                return version
    return None


def _discover_version() -> str:
    try:
        return importlib_metadata.version("dynlib")
    except importlib_metadata.PackageNotFoundError:
        fallback = _read_pyproject_version()
        if fallback:
            return fallback
    return "0.0.0+local"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dynlib", description="dynlib command-line interface")
    parser.add_argument("--version", action="version", version=_discover_version())

    subparsers = parser.add_subparsers(dest="command", required=True)

    # model
    p_model = subparsers.add_parser("model", help="Model tooling")
    sp_model = p_model.add_subparsers(dest="model_cmd", required=True)
    p_validate = sp_model.add_parser("validate", help="Validate a model file or URI")
    p_validate.add_argument("uri", help="Model URI or path")
    p_validate.set_defaults(handler=_cmd_model_validate)

    # steppers
    p_steppers = subparsers.add_parser("steppers", help="Stepper registry introspection")
    sp_steppers = p_steppers.add_subparsers(dest="steppers_cmd", required=True)
    p_list = sp_steppers.add_parser("list", help="List registered steppers")
    p_list.add_argument("--kind", help="Filter by StepperMeta.kind")

    bool_caps, value_caps = _caps_fields()
    for field_name in bool_caps:
        p_list.add_argument(
            f"--{field_name}",
            action="store_true",
            help=f"Require StepperCaps.{field_name}=True",
        )
    for field_name in value_caps:
        p_list.add_argument(
            f"--{field_name}",
            help=f"Filter StepperCaps.{field_name} by value",
        )
    p_list.set_defaults(handler=_cmd_steppers_list, bool_caps=bool_caps, value_caps=value_caps)

    # cache
    p_cache = subparsers.add_parser("cache", help="Dynlib JIT cache management")
    sp_cache = p_cache.add_subparsers(dest="cache_cmd", required=True)

    p_cache_path = sp_cache.add_parser("path", help="Show cache root directory")
    p_cache_path.set_defaults(handler=_cmd_cache_path)

    p_cache_list = sp_cache.add_parser("list", help="List cache entries")
    p_cache_list.add_argument("--stepper", help="Filter cache by stepper name")
    p_cache_list.add_argument("--dtype", help="Filter cache by dtype token")
    p_cache_list.add_argument("--hash", dest="spec_hash", help="Filter by spec hash prefix")
    p_cache_list.set_defaults(handler=_cmd_cache_list)

    p_cache_clear = sp_cache.add_parser("clear", help="Delete cache entries")
    p_cache_clear.add_argument("--all", action="store_true", help="Delete the entire cache directory")
    p_cache_clear.add_argument("--stepper", help="Delete entries for one stepper")
    p_cache_clear.add_argument("--dtype", help="Delete entries for dtype token")
    p_cache_clear.add_argument("--hash", dest="spec_hash", help="Delete entries with matching spec hash prefix")
    p_cache_clear.add_argument("--dry_run", action="store_true", help="Show what would be deleted")
    p_cache_clear.set_defaults(handler=_cmd_cache_clear)

    return parser


def _caps_fields() -> tuple[tuple[str, ...], tuple[str, ...]]:
    bool_fields: list[str] = []
    value_fields: list[str] = []
    for field in fields(StepperCaps):
        annotation = field.type
        default = field.default
        if annotation is bool or isinstance(default, bool):
            bool_fields.append(field.name)
        else:
            value_fields.append(field.name)
    return tuple(bool_fields), tuple(value_fields)


def _cmd_model_validate(args: argparse.Namespace) -> int:
    try:
        spec = load_model_from_uri(args.uri)
    except DynlibError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Model OK: kind={spec.kind} dtype={spec.dtype} states={len(spec.states)} stepper={spec.sim.stepper}")
    return 0


def _cmd_steppers_list(args: argparse.Namespace) -> int:
    filters: dict[str, object] = {}
    if args.kind:
        filters["kind"] = args.kind
    for name in args.bool_caps:
        if getattr(args, name):
            filters[name] = True
    for name in args.value_caps:
        value = getattr(args, name)
        if value is not None:
            filters[name] = value
    infos = select_steppers(**filters)
    if not infos:
        print("No steppers found.")
        return 0
    infos.sort(key=lambda meta: meta.name)
    caps_fields = [field.name for field in fields(StepperCaps)]
    for meta in infos:
        caps_values = [f"{field}={getattr(meta.caps, field)}" for field in caps_fields]
        caps_str = " ".join(caps_values)
        print(
            f"{meta.name:<12} kind={meta.kind:<4} scheme={meta.scheme:<8} "
            f"order={meta.order:<2} stiff={str(meta.stiff):<5} {caps_str}"
        )
    return 0


def _cmd_cache_path(_args: argparse.Namespace) -> int:
    root = resolve_cache_root()
    print(str(root))
    return 0


def _cmd_cache_list(args: argparse.Namespace) -> int:
    root = resolve_cache_root()
    entries = _filter_cache_entries(_load_cache_entries(root), args.stepper, args.dtype, args.spec_hash)
    if not entries:
        print("No cache entries found.")
        return 0
    entries.sort(key=lambda entry: (entry.family, entry.stepper, entry.dtype, entry.digest))
    for entry in entries:
        components = f" components={','.join(entry.components)}" if entry.components else ""
        print(
            f"[{entry.family:<8}] stepper={entry.stepper:<12} dtype={entry.dtype:<8} "
            f"spec={entry.spec_hash} digest={entry.digest} size={_format_size(entry.size_bytes)} "
            f"path={entry.path}{components}"
        )
    return 0


def _cmd_cache_clear(args: argparse.Namespace) -> int:
    root = resolve_cache_root()
    if not args.all and not any([args.stepper, args.dtype, args.spec_hash]):
        print(
            "cache clear requires --all or one of --stepper/--dtype/--hash",
            file=sys.stderr,
        )
        return 2
    if args.all:
        if args.dry_run:
            if not root.exists():
                print("Cache root does not exist.")
            else:
                print(f"[dry-run] Would delete: {root}")
            return 0
        if not root.exists():
            print("Cache root does not exist.")
            return 0
        try:
            shutil.rmtree(root)
            print(f"Deleted cache root: {root}")
            return 0
        except OSError as exc:
            print(f"Failed to delete cache root {root}: {exc}", file=sys.stderr)
            return 1

    entries = _filter_cache_entries(_load_cache_entries(root), args.stepper, args.dtype, args.spec_hash)
    if not entries:
        print("No matching cache entries found.")
        return 0

    errors = False
    for entry in entries:
        if args.dry_run:
            print(f"[dry-run] Would delete: {entry.path}")
            continue
        try:
            shutil.rmtree(entry.path)
            print(f"Deleted {entry.family} cache: stepper={entry.stepper} digest={entry.digest}")
        except OSError as exc:
            errors = True
            print(f"Failed to delete {entry.path}: {exc}", file=sys.stderr)
    return 1 if errors else 0


def _filter_cache_entries(
    entries: Sequence[CacheEntry],
    stepper: str | None,
    dtype: str | None,
    spec_hash: str | None,
) -> list[CacheEntry]:
    filtered: list[CacheEntry] = []
    stepper_lower = stepper.lower() if stepper else None
    dtype_lower = dtype.lower() if dtype else None
    spec_lower = spec_hash.lower() if spec_hash else None
    for entry in entries:
        if stepper_lower and entry.stepper.lower() != stepper_lower:
            continue
        if dtype_lower and entry.dtype.lower() != dtype_lower:
            continue
        if spec_lower and not entry.spec_hash.lower().startswith(spec_lower):
            continue
        filtered.append(entry)
    return filtered


def _load_cache_entries(root: Path) -> list[CacheEntry]:
    base = root / "jit"
    if not base.exists():
        return []
    entries: list[CacheEntry] = []
    for family in ("triplets", "steppers", "runners"):
        family_dir = base / family
        if not family_dir.exists():
            continue
        for meta_path in family_dir.rglob("meta.json"):
            entry = _parse_cache_entry(family, meta_path)
            if entry:
                entries.append(entry)
    return entries


def _parse_cache_entry(family: str, meta_path: Path) -> CacheEntry | None:
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return None
    stepper = inputs.get("stepper")
    dtype = inputs.get("dtype")
    spec_hash = inputs.get("spec_hash")
    if not all(isinstance(val, str) and val for val in (stepper, dtype, spec_hash)):
        return None
    digest = payload.get("hash") or meta_path.parent.name
    components_raw = payload.get("components", [])
    components: tuple[str, ...] = ()
    if isinstance(components_raw, list):
        components = tuple(str(item) for item in components_raw if isinstance(item, str))
    size_bytes = _dir_size(meta_path.parent)
    return CacheEntry(
        family=family,
        path=meta_path.parent,
        stepper=stepper,
        dtype=dtype,
        spec_hash=spec_hash,
        digest=str(digest),
        size_bytes=size_bytes,
        components=components,
    )


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
    except FileNotFoundError:
        return 0
    return total


def _format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("missing handler")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
