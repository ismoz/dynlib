# Reference

The reference section collects generated artifacts and registry helpers that complement the conceptual guides. Most of the reference content is produced automatically when you run `mkdocs build` so it always matches the stepper/model implementations in the source tree.

## Built-in models

The `reference/models` subfolder is populated by `tools/gen_model_docs.py` via `mkdocs-gen-files`. Each model under `src/dynlib/models/{map,ode}` gets a dedicated page showing the TOML source plus links into the literate navigation tree (`reference/models/SUMMARY.md`).

- [Built-in model overview](models/index.md) — choose between the map and ODE collections and open any generated TOML listing.


## Generating docs locally
- The documentation relies on `mkdocs`. To regenerate or serve the documentation locally:

1. Install MkDocs and required plugins:
   ```bash
   pip install mkdocs mkdocs-material mkdocs-gen-files mkdocs-literate-nav "mkdocstrings[python]" mkdocstrings mkdocs-static-i18n
   ```

2. Install additional Markdown extensions:
   ```bash
   pip install pymdown-extensions
   ```

3. From the project root, serve the docs:
   ```bash
   mkdocs serve
   ```
   Or build them:
   ```bash
   mkdocs build
   ```

4. To manually update the auto-generated doc files run:
   ```bash
   python tools/gen_model_docs.py
   ```

The generated site will be in the `site/` directory.