# src/dynlib/runtime/types.py
from __future__ import annotations
from typing import Literal

__all__ = ["Kind", "TimeCtrl", "Scheme"]

Kind = Literal["ode", "map"]
TimeCtrl = Literal["fixed", "adaptive"]
Scheme = Literal["explicit", "implicit", "splitting"]
