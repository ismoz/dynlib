# src/dynlib/runtime/sim.py
#TODO: THIS IS A STUB. EXPAND AS NEEDED LATER.
from __future__ import annotations
from .model import Model

class Sim:
    """
    Slice 3 placeholder. The full runner wiring lands in Slice 4.
    """
    def __init__(self, model: Model):
        self.model = model

    def dry_run(self):
        """Tiny helper to assert callability."""
        return callable(self.model.rhs) and callable(self.model.events_pre) and callable(self.model.events_post)
