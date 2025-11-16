from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Protocol, Callable, Optional
import numpy as np

from dynlib.runtime.types import Kind, TimeCtrl, Scheme

__all__ = [
    "StepperCaps", "StepperMeta", "StepperInfo", "StepperSpec",
]

JacobianPolicy = Literal["none", "optional", "required"]

# NOTE: When you need to add a stepper with a new capability add a field below.
#       You don't need to change any existing stepper. You only define the
#       capabilities you want for that stepper. The defaulta are applied to the
#       fields you don't specify.
@dataclass(frozen=True)
class StepperCaps:
    """
    Optional / implementation-level capabilities.
    These are things you *can* add or remove without changing the mathematical
    identity of the method.
    """
    dense_output: bool = False           # has continuous interpolation / dense output
    jacobian: JacobianPolicy = "none"    # how this impl uses external Jacobian
    # future:
    # mass_matrix: bool = False
    # fsal: bool = False
    # new_feature: NewFeatureType = DefaultValue
    # ...

@dataclass(frozen=True)
class StepperMeta:
    """
    Public metadata for a stepper.
    Fundamental classification + suitability, plus a caps block
    for optional capabilities.
    """
    name: str
    kind: Kind
    time_control: TimeCtrl = "fixed"
    scheme: Scheme = "explicit"
    geometry: FrozenSet[str] = frozenset()
    family: str = ""
    order: int = 1
    embedded_order: int | None = None
    stiff_ok: bool = False               # fundamental: intended for stiff use?
    aliases: tuple[str, ...] = ()
    caps: StepperCaps = field(default_factory=StepperCaps)

# Alias requested by guardrails
StepperInfo = StepperMeta


class StepperSpec(Protocol):
    """
    Abstract interface for stepper specs used by the build/codegen layer.
    Implementations MUST:
      - accept `meta: StepperMeta` in __init__
      - provide `workspace_type() -> type | None` for NamedTuple layout
      - provide `make_workspace(n_state, dtype, model_spec=None) -> object`
      - provide `config_spec() -> type | None` (dataclass type for runtime config, or None)
      - provide `default_config(model_spec=None)` (create default config instance)
      - provide `pack_config(config) -> np.ndarray` (pack config to float64 array)
      - provide `emit(rhs_fn, model_spec=None) -> Callable` that returns a jittable stepper function
    """

    meta: StepperMeta

    def __init__(self, meta: StepperMeta) -> None: ...
    def workspace_type(self) -> type | None: ...
    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> object: ...
    
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
    
    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable stepper function.
        
        Args:
            rhs_fn: The compiled RHS function (for inlining or reference)
            model_spec: Optional ModelSpec for accessing sim defaults (e.g., tolerances)
        
        Returns:
            A callable Python function that implements the stepper with the frozen ABI signature:
                stepper(t, dt, y_curr, rhs, params, runtime_ws,
                        stepper_ws, stepper_config,
                        y_prop, t_prop, dt_next, err_est) -> int32
        """
        ...
