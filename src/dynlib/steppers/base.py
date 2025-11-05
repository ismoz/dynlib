# src/dynlib/steppers/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import FrozenSet, Protocol, Callable

from dynlib.runtime.types import Kind, TimeCtrl, Scheme

__all__ = [
    "StepperMeta", "StepperInfo", "StructSpec", "StepperSpec",
]

@dataclass(frozen=True)
class StepperMeta:
    """
    Public metadata for a stepper.
    """
    name: str
    kind: Kind
    time_control: TimeCtrl = "fixed"
    scheme: Scheme = "explicit"
    geometry: FrozenSet[str] = frozenset()
    family: str = ""
    order: int = 1
    embedded_order: int | None = None
    dense_output: bool = False
    stiff_ok: bool = False
    aliases: tuple[str, ...] = ()

# Alias requested by guardrails
StepperInfo = StepperMeta


@dataclass(frozen=True)
class StructSpec:
    """
    Sole extension point for stepper storage. Sizes are counts in elements.
    Flags guide tiny runner maintenance hooks (e.g., history rings).
    """
    sp_size: int = 0
    ss_size: int = 0
    sw0_size: int = 0
    sw1_size: int = 0
    sw2_size: int = 0
    sw3_size: int = 0
    iw0_size: int = 0
    bw0_size: int = 0

    # Flags (behavioral)
    use_history: bool = False
    use_f_history: bool = False
    dense_output: bool = False
    needs_jacobian: bool = False

    # Capabilities (mirror metadata where relevant)
    embedded_order: int | None = None
    stiff_ok: bool = False


class StepperSpec(Protocol):
    """
    Abstract interface for stepper specs used by the build/codegen layer.
    Implementations MUST:
      - accept `meta: StepperMeta` in __init__
      - provide `struct_spec() -> StructSpec`
      - provide `emit(rhs_fn, struct: StructSpec, model_spec=None) -> Callable` that returns a jittable stepper function
    """

    meta: StepperMeta

    def __init__(self, meta: StepperMeta) -> None: ...
    def struct_spec(self) -> StructSpec: ...
    def emit(self, rhs_fn: Callable, struct: StructSpec, model_spec=None) -> Callable:
        """
        Generate a jittable stepper function.
        
        Args:
            rhs_fn: The compiled RHS function (for inlining or reference)
            struct: StructSpec for this stepper (for workspace allocation info)
            model_spec: Optional ModelSpec for accessing sim defaults (e.g., tolerances)
        
        Returns:
            A callable Python function that implements the stepper with the frozen ABI signature:
                stepper(t, dt, y_curr, rhs, params, sp, ss, sw0, sw1, sw2, sw3, 
                       iw0, bw0, y_prop, t_prop, dt_next, err_est) -> int32
        """
        ...
