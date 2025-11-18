# src/dynlib/steppers/config_base.py
"""
Base mixin providing default config protocol implementations via introspection.

Steppers can inherit from ConfigMixin to get automatic implementation of:
- config_spec()
- default_config()
- config_enum_maps()
- pack_config()

All based on a single `Config` class attribute.
"""
from __future__ import annotations
import dataclasses
from typing import Any
import numpy as np

from .config_utils import pack_config_auto

__all__ = ["ConfigMixin", "NoConfig"]


class NoConfig:
    """Sentinel type indicating a stepper has no runtime configuration."""
    pass


class ConfigMixin:
    """
    Mixin providing automatic config protocol implementations.
    
    Usage:
        # No config stepper:
        class MyStepperSpec(ConfigMixin):
            Config = None
            # ... rest of stepper
        
        # Simple config (no enums):
        class MyStepperSpec(ConfigMixin):
            @dataclass
            class Config:
                tol: float = 1e-8
                max_iter: int = 50
            # ... rest of stepper
        
        # Config with string enums:
        class MyStepperSpec(ConfigMixin):
            @dataclass
            class Config:
                tol: float = 1e-8
                method: str = 'hybr'
                
                # Declare enum mappings inline
                __enums__ = {
                    "method": {"hybr": 0, "lm": 1, "broyden1": 2}
                }
            # ... rest of stepper
    
    Benefits:
    - No boilerplate: declare config once, get all 4 protocol methods
    - Enum mappings co-located with config definition
    - Easy to see at a glance if a stepper has config or not
    - Backwards compatible: can still override methods manually
    """
    
    Config: type | None = None
    
    def config_spec(self) -> type | None:
        """Return the Config dataclass type, or None if no config."""
        if self.Config is None or self.Config is NoConfig:
            return None
        return self.Config
    
    def default_config(self, model_spec=None):
        """
        Create default config instance, or None if no config.
        
        Automatically merges stepper config defaults with matching fields from
        model_spec.sim (if provided). This enables TOML [sim] section to override
        stepper-specific defaults without requiring manual implementation.
        
        Precedence: Config defaults < model_spec.sim.* < user run() kwargs
        """
        config_type = self.config_spec()
        if config_type is None:
            return None
        
        # Instantiate with stepper defaults
        config = config_type()
        
        # Merge from model_spec.sim if available
        if model_spec is not None and hasattr(model_spec, 'sim'):
            import dataclasses
            updates = {}
            
            # Check each config field to see if model_spec.sim has a matching attribute
            for field in dataclasses.fields(config):
                field_name = field.name
                if hasattr(model_spec.sim, field_name):
                    sim_value = getattr(model_spec.sim, field_name)
                    # Cast to field type for safety
                    try:
                        if field.type in (int, 'int'):
                            updates[field_name] = int(sim_value)
                        elif field.type in (float, 'float'):
                            updates[field_name] = float(sim_value)
                        elif field.type in (str, 'str'):
                            updates[field_name] = str(sim_value)
                        else:
                            # For other types (bool, etc), try direct assignment
                            updates[field_name] = sim_value
                    except (ValueError, TypeError):
                        # Skip fields that can't be converted
                        pass
            
            if updates:
                config = dataclasses.replace(config, **updates)
        
        return config
    
    def config_enum_maps(self) -> dict[str, dict[str, int]] | None:
        """
        Return enum mappings from Config.__enums__, if present.
        
        Returns:
            Dict mapping field names to {str: int} enum dicts, or None
        """
        config_type = self.config_spec()
        if config_type is None:
            return None
        
        # Check for inline __enums__ attribute
        if hasattr(config_type, '__enums__'):
            return config_type.__enums__
        
        return None
    
    def pack_config(self, config) -> np.ndarray:
        """
        Pack config to float64 array using automated field introspection.
        
        Uses pack_config_auto() which handles enum conversion automatically.
        """
        if config is None:
            return np.array([], dtype=np.float64)
        
        enum_maps = self.config_enum_maps()
        return pack_config_auto(config, enum_maps=enum_maps)
