# src/dynlib/steppers/config_utils.py
"""
Utilities for stepper configuration packing with automated field introspection.

Provides helpers for:
- Automatic packing of dataclass configs to float64 arrays
- String enum handling with validation
- Order-preserving field iteration
"""
from __future__ import annotations
import dataclasses
from typing import Any, Dict, Optional, Type
import numpy as np
import warnings

__all__ = ["pack_config_auto", "ConfigEnumMap"]


class ConfigEnumMap:
    """
    Declarative string→int enum mapping for stepper config fields.
    
    Example:
        class MyStepperSpec:
            ENUM_MAPS = {
                "method": {"hybr": 0, "lm": 1, "broyden1": 2},
                "mode": {"fast": 0, "accurate": 1},
            }
            
            def config_enum_maps(self) -> dict[str, dict[str, int]] | None:
                return self.ENUM_MAPS
    """
    
    @staticmethod
    def validate_and_convert(
        value: Any,
        field_name: str,
        enum_map: Dict[str, int],
        context: str = "config",
    ) -> int:
        """
        Convert string value to int using enum map, or pass through if already int.
        
        Args:
            value: String or int value
            field_name: Name of the config field
            enum_map: String→int mapping
            context: Context for error messages
        
        Returns:
            Integer enum value
        
        Raises:
            ValueError: If string value not in enum_map
            TypeError: If value is neither str nor int
        """
        if isinstance(value, str):
            if value not in enum_map:
                valid = ", ".join(f"'{k}'" for k in sorted(enum_map.keys()))
                raise ValueError(
                    f"Invalid {context} value '{value}' for field '{field_name}'. "
                    f"Valid options: {valid}"
                )
            return enum_map[value]
        elif isinstance(value, (int, np.integer)):
            return int(value)
        else:
            raise TypeError(
                f"Config field '{field_name}' must be str or int, got {type(value).__name__}"
            )


def pack_config_auto(
    config: Any,
    enum_maps: Optional[Dict[str, Dict[str, int]]] = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Automatically pack a dataclass config into a numpy array.
    
    Introspects dataclass fields in declaration order and packs values sequentially.
    Handles string→int enum conversion for specified fields.
    
    Args:
        config: Dataclass instance to pack (or None)
        enum_maps: Optional dict mapping field names to {str: int} enum maps
        dtype: Target numpy dtype (default: float64)
    
    Returns:
        1D numpy array with packed config values in field order.
        Returns empty array if config is None.
    
    Example:
        @dataclass
        class MyConfig:
            tol: float = 1e-8
            max_iter: int = 50
            method: str = 'hybr'
        
        ENUM_MAPS = {"method": {"hybr": 0, "lm": 1}}
        
        cfg = MyConfig(tol=1e-6, max_iter=100, method="lm")
        arr = pack_config_auto(cfg, ENUM_MAPS)
        # arr = [1e-6, 100.0, 1.0]
    """
    if config is None:
        return np.array([], dtype=dtype)
    
    if not dataclasses.is_dataclass(config):
        raise TypeError(f"config must be a dataclass instance, got {type(config).__name__}")
    
    enum_maps = enum_maps or {}
    values = []
    
    for field in dataclasses.fields(config):
        field_name = field.name
        value = getattr(config, field_name)
        
        # Handle string→int enum conversion
        if field_name in enum_maps:
            enum_map = enum_maps[field_name]
            value = ConfigEnumMap.validate_and_convert(
                value, field_name, enum_map, context="stepper config"
            )
        
        # Convert to float (handles int, float, np types)
        try:
            float_value = float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot convert config field '{field_name}' with value {value!r} "
                f"to float. Ensure all config fields are numeric or have enum mappings."
            ) from e
        
        values.append(float_value)
    
    return np.array(values, dtype=dtype)


def convert_config_enums(
    kwargs: Dict[str, Any],
    enum_maps: Optional[Dict[str, Dict[str, int]]],
    stepper_name: str = "stepper",
) -> Dict[str, Any]:
    """
    Convert string values in kwargs to ints using enum maps.
    
    Used by sim.py to preprocess user-provided stepper config kwargs
    before building the dataclass instance.
    
    Args:
        kwargs: User-provided config overrides
        enum_maps: Field name → {str: int} mappings
        stepper_name: Name for error messages
    
    Returns:
        New dict with string values converted to ints where applicable.
        Non-enum fields and int values pass through unchanged.
    
    Example:
        kwargs = {"method": "lm", "tol": 1e-6}
        enum_maps = {"method": {"hybr": 0, "lm": 1}}
        result = convert_config_enums(kwargs, enum_maps)
        # result = {"method": 1, "tol": 1e-6}
    """
    if not enum_maps:
        return kwargs
    
    converted = {}
    for key, value in kwargs.items():
        if key in enum_maps and isinstance(value, str):
            enum_map = enum_maps[key]
            try:
                converted[key] = ConfigEnumMap.validate_and_convert(
                    value, key, enum_map, context=f"stepper '{stepper_name}'"
                )
            except ValueError as e:
                # Re-raise with more context
                raise ValueError(str(e)) from None
        else:
            converted[key] = value
    
    return converted


def unpack_config_field(
    config_array: np.ndarray,
    field_index: int,
    field_name: str,
    reverse_enum_map: Optional[Dict[int, str]] = None,
    as_int: bool = False,
) -> Any:
    """
    Extract a single field value from a packed config array.
    
    Args:
        config_array: Packed float64 config array
        field_index: Index of the field in the array
        field_name: Name of the field (for error messages)
        reverse_enum_map: Optional {int: str} mapping to convert back to string
        as_int: If True, cast to int (for integer config fields)
    
    Returns:
        Field value (float, int, or str depending on args)
    """
    if field_index >= config_array.size:
        raise IndexError(
            f"Config field '{field_name}' at index {field_index} not found. "
            f"Config array size: {config_array.size}"
        )
    
    value = config_array[field_index]
    
    if reverse_enum_map is not None:
        int_val = int(value)
        if int_val in reverse_enum_map:
            return reverse_enum_map[int_val]
        else:
            warnings.warn(
                f"Unknown enum value {int_val} for field '{field_name}'. "
                f"Valid values: {list(reverse_enum_map.keys())}",
                RuntimeWarning,
            )
            return int_val
    
    if as_int:
        return int(value)
    
    return float(value)
