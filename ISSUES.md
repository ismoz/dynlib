# Known Issues

This file lists the known issues in the project. Each issue includes a brief description, status, and any relevant notes or workarounds.

## Issues

1. **StructSpec Validation**
   - **Description**: The guardrail checklist demands build-time validation of StructSpec sizes, lane counts, dtype usage, and bank semantics (docs/guardrails.md (lines 236-280)). The current validator only checks assignment targets and basic slicing (src/dynlib/compiler/codegen/validate.py (lines 1-210)); it does not enforce non‑negative lane sizes, forbid storing floats in iw0/bw0, or detect persistence misuse, so several mandated checks are missing.
   - **Status**: CLOSED.
   - **Notes**:


