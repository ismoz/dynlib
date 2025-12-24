# src/dynlib/steppers/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Protocol, Callable, Optional
import numpy as np

from dynlib.runtime.types import Kind, TimeCtrl, Scheme

__all__ = [
    "StepperCaps", "StepperMeta", "StepperInfo", "StepperSpec",
]

JacobianPolicy = Literal["none", "internal", "optional", "required"]

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
    jit_capable: bool = True             # can be jitted
    requires_scipy: bool = False
    variational_stepping: bool = False   # supports emit_step_with_variational for Lyapunov analysis
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
    stiff: bool = False               # fundamental: intended for stiff use?
    aliases: tuple[str, ...] = ()
    caps: StepperCaps = field(default_factory=StepperCaps)

# Alias requested by guardrails
StepperInfo = StepperMeta


class StepperSpec(Protocol):
    """
    Abstract interface for stepper specs used by the build/codegen layer.
    
    Implementations MUST provide:
      - `meta: StepperMeta` attribute
      - `__init__(meta)` constructor
      - `workspace_type() -> type | None` for NamedTuple layout
      - `make_workspace(n_state, dtype, model_spec=None) -> object`
      - `emit(rhs_fn, model_spec=None) -> Callable` that returns a jittable stepper function
    
    For runtime configuration support, either:
      1. Inherit from ConfigMixin and set `Config` class attribute (recommended), OR
      2. Manually implement: config_spec(), default_config(), pack_config(), config_enum_maps()
    
    ConfigMixin provides automatic implementations based on a single Config declaration.
    See steppers.config_base.ConfigMixin for usage examples.
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
        
        NOTE: If using ConfigMixin, this is auto-implemented from Config attribute.
        Only override manually if not using ConfigMixin.
        
        Returns:
            Config dataclass type or None
        """
        ...
    
    def default_config(self, model_spec=None):
        """
        Create default config instance, optionally reading from model_spec.
        
        NOTE: If using ConfigMixin, this is auto-implemented.
        Only override manually if not using ConfigMixin or need model_spec overrides.
        
        Args:
            model_spec: Optional ModelSpec to read defaults from
        
        Returns:
            Instance of config dataclass, or None if no config needed
        """
        ...
    
    def pack_config(self, config) -> np.ndarray:
        """
        Pack config dataclass into float64 array.
        
        NOTE: If using ConfigMixin, this is auto-implemented using pack_config_auto().
        Only override manually if you need custom packing logic.
        
        Args:
            config: Instance of config dataclass (or None)
        
        Returns:
            1D float64 array. Empty array if config is None.
        """
        ...
    
    def config_enum_maps(self) -> dict[str, dict[str, int]] | None:
        """
        Return string→int enum mappings for config fields (optional).
        
        NOTE: If using ConfigMixin, this is auto-implemented from Config.__enums__.
        Only override manually if not using ConfigMixin.
        
        Returns:
            Dict mapping field names to {string_value: int_code} dicts, or None.
        
        Example (in Config class):
            @dataclass
            class Config:
                method: str = 'hybr'
                __enums__ = {"method": {"hybr": 0, "lm": 1, "broyden1": 2}}
        """
        ...
    
    def emit(
        self,
        rhs_fn: Callable,
        model_spec=None,
        jacobian_fn: Callable | None = None,
        jvp_fn: Callable | None = None,
    ) -> Callable:
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
