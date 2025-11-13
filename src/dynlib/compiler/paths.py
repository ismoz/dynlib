# src/dynlib/compiler/paths.py
"""
Model path resolution and configuration for dynlib v2.

Implements TAG:// URIs, config file loading, and environment variable overrides.
See design.md "Model Paths / Registry" for the full specification.
"""
from __future__ import annotations
import os
import sys
import tempfile
import uuid
import warnings
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tomllib

from dynlib.errors import ModelNotFoundError, ConfigError, PathTraversalError, AmbiguousModelError

__all__ = [
    "PathConfig",
    "load_config",
    "resolve_uri",
    "parse_uri",
    "resolve_cache_root",
]

_CACHE_ROOT_FALLBACK_WARNED = False


@dataclass(frozen=True)
class PathConfig:
    """Configuration for model path resolution and cache behavior."""
    # Map of tag name -> list of directory roots
    tags: Dict[str, List[str]]
    # Optional cache root for JIT artifacts
    cache_root: Optional[str] = None


def _builtin_models_dir() -> Optional[str]:
    """
    Return the absolute path to the bundled builtin models directory.
    
    This resolves relative to the installed dynlib package so that it works
    regardless of where the library is located on disk.
    """
    models_dir = Path(__file__).resolve().parents[1] / "models"
    if models_dir.is_dir():
        return str(models_dir)
    return None


def _get_config_path() -> Path:
    """
    Get platform-specific config file path.
    
    - Linux: ${XDG_CONFIG_HOME:-~/.config}/dynlib/config.toml
    - macOS: ~/Library/Application Support/dynlib/config.toml
    - Windows: %APPDATA%/dynlib/config.toml
    
    Can be overridden with DYNLIB_CONFIG environment variable.
    """
    env_override = os.environ.get("DYNLIB_CONFIG")
    if env_override:
        return Path(env_override).expanduser().resolve()
    
    if sys.platform == "darwin":
        # macOS
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        # Windows
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise ConfigError("APPDATA environment variable not set on Windows")
        base = Path(appdata)
    else:
        # Linux and other Unix-like
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            base = Path(xdg_config)
        else:
            base = Path.home() / ".config"
    
    return base / "dynlib" / "config.toml"


def _parse_env_model_path() -> Dict[str, List[str]]:
    """
    Parse DYN_MODEL_PATH environment variable.
    
    Format (POSIX): TAG1=/path1,/path2:TAG2=/path3
    Format (Windows): TAG1=C:\\path1,C:\\path2;TAG2=C:\\path3
    
    Returns dict of tag -> list of roots.
    """
    env_val = os.environ.get("DYN_MODEL_PATH", "").strip()
    if not env_val:
        return {}
    
    # Use ; on Windows, : on Unix
    tag_sep = ";" if sys.platform == "win32" else ":"
    path_sep = ","
    
    result: Dict[str, List[str]] = {}
    
    for tag_entry in env_val.split(tag_sep):
        tag_entry = tag_entry.strip()
        if not tag_entry:
            continue
        
        if "=" not in tag_entry:
            raise ConfigError(f"Invalid DYN_MODEL_PATH entry (missing '='): {tag_entry}")
        
        tag, paths_str = tag_entry.split("=", 1)
        tag = tag.strip()
        paths_str = paths_str.strip()
        
        if not tag:
            raise ConfigError(f"Empty tag in DYN_MODEL_PATH: {tag_entry}")
        
        paths = [p.strip() for p in paths_str.split(path_sep) if p.strip()]
        if not paths:
            raise ConfigError(f"No paths for tag '{tag}' in DYN_MODEL_PATH")
        
        result[tag] = paths
    
    return result


def _default_cache_root() -> Path:
    """Return the platform-specific default cache root."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "dynlib"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            base = Path(local)
        else:
            base = Path.home() / "AppData" / "Local"
        return base / "dynlib" / "Cache"
    # Linux / Unix fallback uses XDG_CACHE_HOME when available
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache).expanduser()
    else:
        base = Path.home() / ".cache"
    return base / "dynlib"


def _ensure_writable_dir(root: Path) -> bool:
    """Return True if *root* can be created and written to."""
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    probe = root / f".dynlib-write-test-{uuid.uuid4().hex}"
    try:
        with open(probe, "wb") as fh:
            fh.write(b"0")
        probe.unlink()
    except OSError:
        with contextlib.suppress(FileNotFoundError):
            probe.unlink()
        return False
    return True


def _warn_cache_root_fallback(primary: Path, fallback: Path) -> None:
    global _CACHE_ROOT_FALLBACK_WARNED
    if _CACHE_ROOT_FALLBACK_WARNED:
        return
    _CACHE_ROOT_FALLBACK_WARNED = True
    warnings.warn(
        f"dynlib cache root {primary} is not writable; using {fallback} instead",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_cache_root_unwritable(primary: Path) -> None:
    warnings.warn(
        f"dynlib cache root {primary} is not writable; falling back to in-memory JIT",
        RuntimeWarning,
        stacklevel=3,
    )


def resolve_cache_root(config: Optional[PathConfig] = None) -> Path:
    """Resolve the cache root based on config or platform defaults."""
    cfg = config or load_config()
    if cfg.cache_root:
        return Path(cfg.cache_root).expanduser().resolve()
    primary = _default_cache_root().expanduser().resolve()
    if _ensure_writable_dir(primary):
        return primary
    fallback = (Path(tempfile.gettempdir()) / "dynlib-cache").expanduser().resolve()
    if _ensure_writable_dir(fallback):
        _warn_cache_root_fallback(primary, fallback)
        return fallback
    _warn_cache_root_unwritable(primary)
    return primary


def load_config() -> PathConfig:
    """
    Load configuration from file and environment.
    
    Environment variable DYN_MODEL_PATH entries are prepended to config file tags.
    If config file doesn't exist, returns empty config (only env paths are used).
    
    Raises:
        ConfigError: If config file is malformed or env var is invalid.
    """
    config_path = _get_config_path()
    
    # Start with empty tags
    tags: Dict[str, List[str]] = {}
    
    cache_root: Optional[str] = None

    # Load from config file if it exists
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load config from {config_path}: {e}")
        
        # Parse [paths] section
        paths_section = data.get("paths", {})
        if not isinstance(paths_section, dict):
            raise ConfigError(f"[paths] section must be a table in {config_path}")
        
        for tag, roots in paths_section.items():
            if isinstance(roots, str):
                # Single string -> convert to list
                roots = [roots]
            elif not isinstance(roots, list):
                raise ConfigError(
                    f"[paths].{tag} must be a string or list of strings in {config_path}"
                )
            
            # Validate all are strings
            for root in roots:
                if not isinstance(root, str):
                    raise ConfigError(
                        f"[paths].{tag} contains non-string entry in {config_path}"
                    )
            
            tags[tag] = list(roots)

        # Optional cache_root can appear either as a top-level key or within a [cache] table
        top_level_cache = data.get("cache_root")
        if top_level_cache is not None:
            if not isinstance(top_level_cache, str):
                raise ConfigError("cache_root must be a string path in config")
            cache_root = top_level_cache
        else:
            cache_section = data.get("cache")
            if isinstance(cache_section, dict):
                cache_value = cache_section.get("root")
                if cache_value is not None:
                    if not isinstance(cache_value, str):
                        raise ConfigError("[cache].root must be a string path")
                    cache_root = cache_value
    
    # Parse environment override (prepend to existing tags)
    env_tags = _parse_env_model_path()
    for tag, env_roots in env_tags.items():
        existing = tags.get(tag, [])
        # Prepend env roots so they win on first match
        tags[tag] = env_roots + existing

    # Ensure builtin models are always available
    builtin_root = _builtin_models_dir()
    if builtin_root:
        builtin_list = tags.setdefault("builtin", [])
        if builtin_root not in builtin_list:
            builtin_list.append(builtin_root)
    
    return PathConfig(tags=tags, cache_root=cache_root)


def parse_uri(uri: str) -> Tuple[str, Optional[str]]:
    """
    Parse a URI into (base_uri, fragment).
    
    Supports both inline declaration styles:
        - Same line: "inline: [model]\\ntype='ode'"
        - Separate lines (cleaner): "inline:\\n    [model]\\n    type='ode'"
    
    Examples:
        "model.toml#mod=drive" -> ("model.toml", "mod=drive")
        "proj://ho.toml" -> ("proj://ho.toml", None)
        "inline: [model]\\ntype='ode'" -> ("inline: [model]\\ntype='ode'", None)
        "inline:\\n    [model]\\n    type='ode'" -> ("inline: [model]\\ntype='ode'", None)
    
    Args:
        uri: URI string
    
    Returns:
        Tuple of (base_uri, fragment or None)
    """
    # Normalize inline: declarations - support both formats:
    # 1. "inline: [model]\ntype='ode'"  (on same line)
    # 2. "inline:\n    [model]\n    type='ode'"  (on separate lines)
    if uri.strip().startswith("inline:"):
        # If inline: is on its own line, strip it and keep the rest
        stripped = uri.strip()
        if stripped == "inline:" or stripped.startswith("inline:\n"):
            # Extract content after "inline:" and strip leading whitespace from each line
            content = stripped[7:].lstrip()  # Remove "inline:" prefix
            return (f"inline: {content}", None)
        # Otherwise keep as-is
        return (uri, None)
    
    if "#" not in uri:
        return (uri, None)
    
    base, fragment = uri.split("#", 1)
    return (base.strip(), fragment.strip() if fragment else None)


def _normalize_path(p: Path) -> Path:
    """Normalize path: expand user, expand vars, resolve to absolute."""
    return Path(os.path.expandvars(str(p))).expanduser().resolve()


def _check_traversal(resolved: Path, root: Path, uri: str) -> None:
    """
    Check that resolved path doesn't traverse outside root.
    
    Raises:
        PathTraversalError: If traversal is detected.
    """
    try:
        # relative_to raises ValueError if resolved is not under root
        resolved.relative_to(root)
    except ValueError:
        raise PathTraversalError(uri, str(resolved), str(root))


def _try_resolve_with_extension(base_path: Path, allow_extensionless: bool) -> Optional[Path]:
    """
    Try to resolve a path, optionally trying .toml extension.
    
    Args:
        base_path: Path to try
        allow_extensionless: If True and base_path has no suffix, try adding .toml
    
    Returns:
        Resolved path if exactly one match found, None otherwise.
    
    Raises:
        AmbiguousModelError: If multiple matches found.
    """
    candidates = []
    
    # Try exact path
    if base_path.exists():
        candidates.append(base_path)
    
    # Try with .toml extension if no extension present
    if allow_extensionless and not base_path.suffix:
        with_toml = base_path.with_suffix(".toml")
        if with_toml.exists() and with_toml not in candidates:
            candidates.append(with_toml)
    
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return candidates[0]
    else:
        # Multiple matches
        raise AmbiguousModelError(str(base_path), [str(c) for c in candidates])


def resolve_uri(
    uri: str,
    config: Optional[PathConfig] = None,
    cwd: Optional[Path] = None,
    allow_extensionless: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Resolve a model URI to an absolute file path.
    
    Supports:
        - inline: ... -> returns as-is
        - Absolute paths -> normalized and validated
        - Relative paths -> resolved from cwd
        - TAG://relpath -> search in config tags
    
    Args:
        uri: URI string (may include #fragment)
        config: PathConfig to use (loads default if None)
        cwd: Current working directory (uses os.getcwd if None)
        allow_extensionless: If True, try adding .toml for paths without extension
    
    Returns:
        Tuple of (resolved_content_or_path, fragment)
        - For inline: uri, returns the inline content
        - For file paths, returns absolute path string
        - fragment is the part after # (or None)
    
    Raises:
        ModelNotFoundError: If file cannot be found
        PathTraversalError: If traversal outside root is detected
        ConfigError: If TAG is unknown or config is invalid
        AmbiguousModelError: If extensionless match is ambiguous
    """
    # Parse fragment
    base_uri, fragment = parse_uri(uri)
    
    # Handle inline: prefix
    # Note: parse_uri already normalized this to "inline: <content>" format
    if base_uri.startswith("inline:"):
        content = base_uri[7:].strip()  # Remove "inline:" prefix and trim
        return (content, fragment)
    
    if config is None:
        config = load_config()
    
    if cwd is None:
        cwd = Path.cwd()
    
    # Handle TAG://relpath
    if "://" in base_uri:
        tag, relpath = base_uri.split("://", 1)
        tag = tag.strip()
        relpath = relpath.strip()
        
        if tag not in config.tags:
            known_tags = list(config.tags.keys())
            msg = f"Unknown tag '{tag}' in URI: {base_uri}"
            if known_tags:
                msg += f"\nKnown tags: {', '.join(known_tags)}"
            else:
                msg += "\nNo tags configured. Check your config file or DYN_MODEL_PATH."
            raise ConfigError(msg)
        
        roots = config.tags[tag]
        candidates = []
        
        for root_str in roots:
            root = _normalize_path(Path(root_str))
            candidate = root / relpath
            candidate = _normalize_path(candidate)
            
            # Security check: prevent traversal outside root
            _check_traversal(candidate, root, base_uri)
            
            # Try to resolve with optional .toml extension
            resolved = _try_resolve_with_extension(candidate, allow_extensionless)
            if resolved:
                return (str(resolved), fragment)
            
            # Record candidate for error message
            candidates.append(str(candidate))
            if allow_extensionless and not candidate.suffix:
                candidates.append(str(candidate.with_suffix(".toml")))
        
        # Not found in any root
        raise ModelNotFoundError(base_uri, candidates)
    
    # Handle absolute or relative path
    path = Path(base_uri)
    
    # Normalize FIRST (expand ~ and env vars before making absolute)
    path = _normalize_path(path)
    
    # Make absolute if relative (after normalization)
    if not path.is_absolute():
        path = cwd / path
        path = path.resolve()
    
    # Try to resolve with optional .toml extension
    resolved = _try_resolve_with_extension(path, allow_extensionless)
    if resolved:
        return (str(resolved), fragment)
    
    # Not found
    candidates = [str(path)]
    if allow_extensionless and not path.suffix:
        candidates.append(str(path.with_suffix(".toml")))
    
    raise ModelNotFoundError(base_uri, candidates)
