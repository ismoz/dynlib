# src/dynlib/runtime/stepper_checks.py
from __future__ import annotations

import numpy as np

from dynlib.errors import ConfigError, StepperJitCapabilityError, JITUnavailableError
from dynlib.runtime.softdeps import SoftDepState
from dynlib.steppers.base import StepperCaps, StepperSpec

__all__ = ["check_stepper"]


def check_stepper(
    *,
    stepper_name: str,
    jit: bool,
    dtype: np.dtype,
    stepper_spec: StepperSpec,
    deps: SoftDepState,
) -> None:
    """
    Centralized capability/availability reconciliation for steppers.

    Args:
        stepper_name: Registry name of the selected stepper.
        jit: Whether build() requested JIT compilation.
        dtype: numpy dtype selected for the model build.
        stepper_spec: Registry spec describing metadata/caps.
        deps: Snapshot of soft dependency availability.
    """

    meta = stepper_spec.meta
    caps: StepperCaps = meta.caps
    dtype_np = np.dtype(dtype)

    # ------------------------------------------------------------------ #
    # JIT vs availability / stepper support
    # ------------------------------------------------------------------ #
    if jit:
        if not deps.numba:
            raise JITUnavailableError(
                f"Stepper '{stepper_name}' requested with jit=True, "
                "but numba is not available. Install numba or pass jit=False."
            )
        elif not caps.jit_capable:
            raise StepperJitCapabilityError(stepper_name=stepper_name)
    # If a stepper explicitly requires numba (not a single stepper does yet):
    elif getattr(caps, "requires_numba", False) and not deps.numba:
        raise ConfigError(
            f"Stepper '{stepper_name}' requires numba but it is not available."
        )

    # ------------------------------------------------------------------ #
    # Other soft dependency requirements (SciPy, CUDA, etc.)
    # ------------------------------------------------------------------ #
    if getattr(caps, "requires_scipy", False) and not deps.scipy:
        raise ConfigError(
            f"Stepper '{stepper_name}' requires SciPy but it is not available."
        )

    # ------------------------------------------------------------------ #
    # dtype compatibility helpers
    # ------------------------------------------------------------------ #
    # NOTE: Not a single stepper currently uses these, but they're here
    #       for future-proofing.
    allowed_dtypes = getattr(caps, "allowed_dtypes", None)
    if allowed_dtypes is not None:
        normalized = {np.dtype(entry) for entry in allowed_dtypes}
        if dtype_np not in normalized:
            allowed_names = ", ".join(sorted({dt.name for dt in normalized}))
            raise ConfigError(
                f"Stepper '{stepper_name}' does not support dtype "
                f"{dtype_np.name} (allowed: {allowed_names})."
            )

    # NOTE: Again no stepper uses this yet.
    if getattr(caps, "real_only", False) and dtype_np.kind == "c":
        raise ConfigError(
            f"Stepper '{stepper_name}' does not support complex dtypes."
        )

    # ------------------------------------------------------------------ #
    # Placeholder for additional caps-based rules. Keep all stepper
    # compatibility logic in this module so future capability checks are
    # discoverable in one place.
    # ------------------------------------------------------------------ #
