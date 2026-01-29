# State, URI, and source-management demos

## Overview

These scripts highlight the ways dynlib exposes the compiled artifacts, externalizes simulation state, and resolves models from different URI schemes. Use them when you want to checkpoint a run, save or reload presets, export generated Python source for inspection, or understand how `setup()` picks a model from inline text, files, or config-based tags.

## Example scripts

### Snapshots & export/import

```python
--8<-- "examples/snapshot_demo.py"
```
Builds a simple exponential-decay model, runs to a few time points, and then demonstrates `sim.create_snapshot`, `sim.export_snapshot`, and `sim.import_snapshot`. The script prints snapshot metadata, shows how workspaces survive across exports, and verifies that `sim.results()` is cleared after restoring a snapshot so you can safely resume from arbitrary points.

### Exporting the compiled sources

```python
--8<-- "examples/export_sources_demo.py"
```
Builds a JIT-enabled simulation from `tests/data/models/decay.toml`, toggles `disk_cache=False`, and calls `sim.model.export_sources()` into a temporary directory. The demo checks that the RHS, events, and stepper sources exist, prints their sizes, and shows the contents so you can review what dynlib emitted for debugging or compliance.

### Preset workflows

```python
--8<-- "examples/presets_demo.py"
```
Builds an inline Izhikevich neuron and then lists/apply the inline `regular_spiking`, `fast_spiking`, and `bursting` presets. It saves the presets to a temporary file, loads them back into a second `Sim`, and proves that the loaded presets reproduce the spike counts while showcasing glob-style preset matching.

### URI and path resolution

```python
--8<-- "examples/uri_demo.py"
```
Covers the URI resolver itself by:
 - running inline TOML via the `inline:` scheme,
 - loading from absolute and relative filesystem paths,
 - showing how to use extensionless names, fragments (`#mod=`), and `proj://`/`TAG://` references,
 - and reminding you where config files live (`~/.config/dynlib/config.toml`, `%APPDATA%` on Windows, etc.).
The script prints what each invocation loaded to make debug easier when a model path is ambiguous.
