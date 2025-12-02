# src/dynlib/runtime/initial_step.py
"""
This file will contain hooks that will run before wrapper starts the runners.
Right now it contains Hairer/Shampine’s WRMS initial step size selection 
heuristic for ODE steppers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from dynlib.steppers.base import StepperMeta

Array = np.ndarray

__all__ = [
    "WRMSConfig",
    "wrms_norm",
    "choose_initial_dt_wrms",
    "make_wrms_config_from_stepper",
]


@dataclass
class WRMSConfig:
    atol: float
    rtol: float
    order: int
    safety: float = 0.01
    min_dt: float = 1e-16
    max_dt: Optional[float] = None


def wrms_norm(err: Array, state: Array, cfg: WRMSConfig) -> float:
    """
    Weighted RMS norm used by Hairer/Shampine heuristics.
    """
    scale = cfg.atol + cfg.rtol * np.abs(state)
    scaled = err / scale
    return float(np.sqrt(np.mean(scaled * scaled)))


def choose_initial_dt_wrms(
    *,
    rhs: Callable[[float, Array, Array, Array, object], None],
    runtime_ws: object,
    t0: float,
    y0: Array,
    params: Array,
    t_end: Optional[float],
    cfg: WRMSConfig,
) -> float:
    """
    Compute an initial step using the WRMS heuristic.

    Uses one Euler trial step and two RHS evaluations to estimate a safe dt.
    """
    f0 = np.empty_like(y0)
    rhs(t0, y0, f0, params, runtime_ws)

    d0 = wrms_norm(y0, y0, cfg)
    d1 = wrms_norm(f0, y0, cfg)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6 if t_end is None else min(1e-6, abs(t_end - t0))
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    f1 = np.empty_like(y0)
    rhs(t0 + h0, y1, f1, params, runtime_ws)

    d2 = wrms_norm(f1 - f0, y0, cfg) / max(h0, cfg.min_dt)

    d = max(d1, d2, 1e-15)
    exponent = 1.0 / (cfg.order + 1.0)
    h = cfg.safety * (1.0 / d) ** exponent

    if cfg.max_dt is not None:
        h = min(h, cfg.max_dt)
    if t_end is not None:
        h = min(h, abs(t_end - t0))

    h = max(h, cfg.min_dt)
    return h


def make_wrms_config_from_stepper(
    meta: StepperMeta,
    config_obj: object | Mapping[str, float] | None,
    *,
    safety: float = 0.01,
    min_dt: float = 1e-16,
    max_dt: Optional[float] = None,
) -> Optional[WRMSConfig]:
    """
    Derive WRMS parameters from a stepper's metadata and runtime config.
    """
    if meta.kind != "ode":
        return None
    if meta.time_control != "adaptive":
        return None
    atol_raw = _config_value(config_obj, "atol")
    rtol_raw = _config_value(config_obj, "rtol")
    if atol_raw is None or rtol_raw is None:
        return None

    try:
        atol = float(atol_raw)
        rtol = float(rtol_raw)
    except (TypeError, ValueError):
        return None

    order = int(meta.order)
    
    # If dt_max is specified in stepper config, use it (override max_dt parameter)
    dt_max_raw = _config_value(config_obj, "dt_max")
    if dt_max_raw is not None:
        try:
            dt_max_from_config = float(dt_max_raw)
            if np.isfinite(dt_max_from_config):
                max_dt = dt_max_from_config
        except (TypeError, ValueError):
            pass

    return WRMSConfig(
        atol=atol,
        rtol=rtol,
        order=order,
        safety=safety,
        min_dt=min_dt,
        max_dt=max_dt,
    )


def _config_value(source: object | Mapping[str, float] | None, field: str) -> object | None:
    if source is None:
        return None
    if hasattr(source, field):
        return getattr(source, field)
    if isinstance(source, Mapping):
        return source.get(field)
    return None
