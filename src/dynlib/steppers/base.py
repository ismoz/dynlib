# src/dynlib/steppers/base.py
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import FrozenSet, Protocol, Callable, Optional
import numpy as np

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
    
    # Lag system metadata (reserved prefix in iw0/ss)
    ss_lag_reserved: int = 0  # Number of ss lanes reserved for lag buffers
    iw0_lag_reserved: int = 0  # Number of iw0 slots reserved for lag heads


class StepperSpec(Protocol):
    """
    Abstract interface for stepper specs used by the build/codegen layer.
    Implementations MUST:
      - accept `meta: StepperMeta` in __init__
      - provide `struct_spec() -> StructSpec`
      - provide `config_spec() -> type | None` (dataclass type for runtime config, or None)
      - provide `default_config(model_spec=None)` (create default config instance)
      - provide `pack_config(config) -> np.ndarray` (pack config to float64 array)
      - provide `emit(rhs_fn, struct: StructSpec, model_spec=None) -> Callable` that returns a jittable stepper function
    """

    meta: StepperMeta

    def __init__(self, meta: StepperMeta) -> None: ...
    def struct_spec(self) -> StructSpec: ...
    
    def config_spec(self) -> type | None:
        """
        Return dataclass type for runtime configuration, or None.
        
        If None, stepper has no runtime config (e.g., fixed-step methods).
        If a dataclass, it should contain only numeric fields (float/int).
        
        Example:
            @dataclass
            class RK45Config:
                atol: float = 1e-8
                rtol: float = 1e-5
                safety: float = 0.9
                min_factor: float = 0.2
                max_factor: float = 10.0
                max_tries: int = 10
                min_step: float = 1e-12
            
            return RK45Config
        """
        ...
    
    def default_config(self, model_spec=None):
        """
        Create default config instance, optionally reading from model_spec.
        
        Args:
            model_spec: Optional ModelSpec to read defaults from
        
        Returns:
            Instance of config dataclass, or None if no config needed
        """
        ...
    
    def pack_config(self, config) -> np.ndarray:
        """
        Pack config dataclass into float64 array.
        
        Args:
            config: Instance of config dataclass (or None)
        
        Returns:
            1D float64 array. Empty array if config is None.
        """
        ...
    
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
                       iw0, bw0, stepper_config, y_prop, t_prop, dt_next, err_est) -> int32
        """
        ...
