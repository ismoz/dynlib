# src/dynlib/runtime/model.py
#TODO: THIS IS A STUB. EXPAND AS NEEDED LATER.
from __future__ import annotations
from dataclasses import dataclass
from dynlib.dsl.spec import ModelSpec

@dataclass(frozen=True)
class Model:
    spec: ModelSpec
    # Slice 3: we only guarantee rhs/events exist via compiler.build().
    rhs: callable
    events_pre: callable
    events_post: callable
    stepper: str
    spec_hash: str
