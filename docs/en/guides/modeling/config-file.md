# Model registry configuration

Dynlib keeps its model registry details in a small TOML file so you can assign tags to directories, override builtin models, and control where the JIT cache lives. `load_config()` merges the file with `DYN_MODEL_PATH` entries before handing the final `PathConfig` (tag map + optional cache root) to every resolver.

## Where the config lives

- **Default path** (when `DYNLIB_CONFIG` is not set):
  - Linux/Unix: `${XDG_CONFIG_HOME:-~/.config}/dynlib/config.toml`
  - macOS: `~/Library/Application Support/dynlib/config.toml`
  - Windows: `%APPDATA%/dynlib/config.toml`
- **Override:** set `DYNLIB_CONFIG` to a custom TOML file and dynlib loads that instead.
- **Missing file:** `load_config()` quietly returns an empty config so dynlib still works with `DYNLIB_CONFIG` or `DYN_MODEL_PATH` overrides.

## File format

```toml
[paths]
custom = ["~/repos/dynlib-models", "/opt/dynlib/models"]
builtin = ["~/custom/builtin"] # This extends the built-in model path, does not replace it.

cache_root = "~/Library/Caches/dynlib"
# or the alternate form
[cache]
root = "~/Library/Caches/dynlib"
```

- `[paths]` maps a tag name (like `builtin` or `custom`) to one or more directory roots. Each entry can be a string or a list of strings. Dynlib resolves a URI such as `custom://circuit/srn` by searching each root in order.
- The `[cache]` table (or top-level `cache_root`) lets you pin the JIT cache location passed to `resolve_cache_root()`. Provide an absolute or `~/`-expanded path, and dynlib validates writability before using it.
- The file is guarded by `ConfigError` if the TOML is malformed, `[paths]` contains non-string entries, or a required value is missing.

## Environment overrides

- `DYN_MODEL_PATH` lets you prepend tag roots without editing the file. Its syntax is `TAG=/path/one:/path/two` on POSIX and `TAG=C:\path1;TAG2=C:\path2` on Windows.
- Entries are parsed into a map and inserted before the ones declared in the config file, so environment paths win when multiple directories share a tag.
- Dynlib keeps the builtin models folder appended to the `builtin` tag list after all overrides so `builtin://` always resolves even if you override it.

## Resolution order and behavior

1. `load_config()` loads the TOML file (if present) and builds the tag map.
2. `DYN_MODEL_PATH` entries are prepended to each tag, letting temporary overrides shadow the file-backed roots.
3. The builtin models directory is appended to the `builtin` tag to guarantee `builtin://` URIs exist even when you redefine the tag.
4. The resulting `PathConfig` is cached by resolver helpers, so restarting the CLI or process re-reads any on-disk changes.

When dynlib can’t find a tag or the requested model path, it raises a `ConfigError` (unknown tag) or `ModelNotFoundError` (file search failed), listing the candidates it tried.

## Troubleshooting tips

- Run `dynlib model validate <uri>` to confirm the registry resolves a model before you run a simulation.
- Inspect `DYNLIB_CONFIG` and `DYN_MODEL_PATH` to ensure they point to writable directories.
- If the cache root in the config is unwritable, dynlib falls back to a platform default (Linux: `~/.cache/dynlib`, macOS: `~/Library/Caches/dynlib`, Windows: `%LOCALAPPDATA%/dynlib/Cache`). It emits a `RuntimeWarning` when this happens.
