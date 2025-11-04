## Changelog

---

## [2.1.0] – 2025-11-04
### Added
- Introduced `src/dynlib/__init__.py` to re-export constants, types, steppers, and utilities 
  for stable imports.
- Added `src/dynlib/runtime/runner_api.py` defining stable exit/status codes and the frozen 
  runner ABI.
- Added `src/dynlib/runtime/types.py` for type literals like `Kind`, `TimeCtrl`, and `Scheme`.
- Added `src/dynlib/steppers/base.py` with metadata and struct specifications for steppers.
- Added `src/dynlib/steppers/registry.py` to manage stepper registration and retrieval.
- Added `src/dynlib/utils/arrays.py` with utility functions for array validation and slicing.
- Added `tests/unit/test_numba_probe.py` to validate future JIT-compiled runner's numba
  compliance.

---

## [2.0.0] – 2025-11-03
### Changed
- v1 hit a dead end and v0 was a failure. Starting a new design from scratch. Only plot 
  tools are preserved. They should be modified in the future for the new model format.