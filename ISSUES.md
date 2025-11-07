# Known Issues

This file lists the known issues in the project. Each issue includes a brief description, status, and any relevant notes or workarounds.

## Issues

1. **StructSpec Validation**
   - **Description**: The guardrail checklist demands build-time validation of StructSpec sizes, lane counts, dtype usage, and bank semantics (docs/guardrails.md (lines 236-280)). The current validator only checks assignment targets and basic slicing (src/dynlib/compiler/codegen/validate.py (lines 1-210)); it does not enforce non‑negative lane sizes, forbid storing floats in iw0/bw0, or detect persistence misuse, so several mandated checks are missing.
   - **Status**: Open.
   - **Notes**:

2. **Stepper-Model Compatibility**
   - **Description**: docs/guardrails.md (lines 231-234) requires the chosen stepper’s kind to match the model (ode vs map). build() never checks stepper_spec.meta.kind before wiring the model (src/dynlib/compiler/build.py (lines 300-345)), so picking an incompatible stepper isn’t blocked.
   - **Status**: Open.
   - **Notes**:

