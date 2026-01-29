# Model registry

Model files live on disk, but dynlib exposes them through a small registry that lets you refer to any model with a stable URI instead of juggling absolute paths. The registry handles the builtin models that ship with dynlib, lets you define your own tags, and transparently resolves relative paths, fragments, and inline models.

## Built-in models

Dynlib preloads the `src/dynlib/models` package so that a `builtin://` tag always exists. That means you can drop any of the models below into `setup(...)`, `dynlib model validate`, or any other entry point without writing a config file.

### Map models
- `builtin://map/logistic`
- `builtin://map/henon`
- `builtin://map/henon2`
- `builtin://map/ikeda`
- `builtin://map/lozi`
- `builtin://map/sine`
- `builtin://map/standard`

### ODE models
- `builtin://ode/duffing`
- `builtin://ode/eto-circular`
- `builtin://ode/expdecay`
- `builtin://ode/exp-if`
- `builtin://ode/fitzhugh-nagumo`
- `builtin://ode/hodgkin-huxley`
- `builtin://ode/izhikevich`
- `builtin://ode/leaky-if`
- `builtin://ode/lorenz`
- `builtin://ode/quadratic-if`
- `builtin://ode/resonate-if`
- `builtin://ode/vanderpol`

The registry adds that builtin directory automatically (see `dynlib/compiler/paths.py` for the exact logic) so you rarely need to worry about paths under `builtin://` — just write `builtin://ode/vanderpol` (without `.toml`) and dynlib checks for the file and throws a helpful `ModelNotFoundError` if it cannot be located.

Use the CLI when you need to inspect or validate a builtin model:

```bash
dynlib model validate builtin://ode/expdecay
```

This command parses the URI, resolves the file, validates the DSL, and reports any parsing errors before you run a simulation.

## URI usage

`resolve_uri` (the same logic behind the CLI and `setup(...)`) understands several URI forms:

1. **Inline declarations**: Start a string with `inline:` and dynlib keeps the DSL snippet in memory. Handy for throwaway models in notebooks or tests.
2. **Tag URIs**: `TAG://relative/path` looks for the model under any root registered for `TAG`. The builtin models use `TAG=builtin`, but you can add your own tags with custom directories (see the next section).
3. **Absolute or relative paths**: A literal file path works too, and dynlib normalizes it (expands `~`, environment variables, `.toml` extension, and resolves the absolute path relative to `cwd`).

Tag URIs can also carry fragments to select mods or sections inside a file:

```
builtin://ode/duffing#mod=odd
```

The parser strips the `#mod=...` before resolving the file and hands the fragment back so the compiler can use it when calling `build(..., mods=[...])`.

`resolve_uri` also tries to append `.toml` when the provided path has no suffix, so both `builtin://ode/vanderpol` and `builtin://ode/vanderpol.toml` are accepted. Security checks prevent traversal outside the registered root, so `TAG://../foo.toml` raises a `PathTraversalError` before any file is read.

## Configuring tag roots

Dynlib keeps the registry configuration in a tiny TOML file and supplements it with environment variables:

- `DYNLIB_CONFIG` overrides the config path (default: `~/.config/dynlib/config.toml` on Linux, `~/Library/Application Support/dynlib/config.toml` on macOS, `%APPDATA%/dynlib/config.toml` on Windows).
- `DYN_MODEL_PATH` lets you prepend tag roots on the fly with a shell-friendly syntax. On POSIX systems use `TAG=/path/one,/path/two:OTHER=/path/three`, and on Windows use `;` between tags.

A `config.toml` looks like this:

```toml
[paths]
myproj = ["~/repos/dynlib-models", "/opt/models"]
builtin = ["/custom/builtin/overrides"]  # keep dynlib builtins accessible

cache_root = "~/Library/Caches/dynlib"
```

`load_config()` parses that file, then prepends any `DYN_MODEL_PATH` entries so environment roots win when multiple directories share the same tag. After that, the builtin models folder is appended to the `builtin` tag list to guarantee `builtin://` URIs resolve even if you override the tag elsewhere.

## Adding your own paths

1. Choose a tag (e.g., `myproj`) and create a directory tree that mirrors the tag URI structure. For example, `myproj://circuit/srn.toml` resolves to `.../<root>/circuit/srn.toml`.
2. Add the root to `DYNLIB_CONFIG`’s `[paths]` table, or set `DYN_MODEL_PATH="myproj=~/models/myproj"` in your shell for temporary overrides.
3. Validate the setup with `dynlib model validate myproj://circuit/srn`.
4. Use the URI inside scripts, `setup(...)`, or your own tooling — dynlib resolves tags, tries `.toml`, and reports missing files with a list of candidates.

If you maintain multiple registries, remember that `DYN_MODEL_PATH` entries take precedence over config file entries, and both of those are searched before the builtin folder. This ordering lets you override `builtin://` models by putting a directory with the same structure earlier in the `builtin` tag list.

## Tips

- Run `dynlib model validate <uri>` before running a simulation to ensure the registry actually resolves the file.
- Use `mytag://path/to/model#mod=variant` when composing models with variants stored in separate `[[mods]]` tables.
- Keep reusable models under a well-known tag directory so collaborators can rely on the same URIs without editing their local config.

With this registry in place, you can freely mix builtin models, shared libraries, and project-specific files while letting dynlib handle the lookup semantics for you.
