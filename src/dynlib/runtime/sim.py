# src/dynlib/runtime/sim.py
from __future__ import annotations
from typing import Optional
import numpy as np

from .model import Model
from .wrapper import run_with_wrapper
from .results import Results
from .results_api import ResultsView

__all__ = ["Sim"]


class Sim:
    """
    Simulation facade around a compiled Model.
    
    Delegates to the wrapper for execution.
    """
    def __init__(self, model: Model):
        self.model = model
        self._raw_results: Optional[Results] = None
        self._results_view: Optional[ResultsView] = None

    def dry_run(self) -> bool:
        """Tiny helper to assert callability."""
        return (
            callable(self.model.rhs) 
            and callable(self.model.events_pre) 
            and callable(self.model.events_post)
            and callable(self.model.runner)
            and callable(self.model.stepper)
        )
    
    def run(
        self,
        *,
        t0: Optional[float] = None,
        t_end: Optional[float] = None,
        dt: Optional[float] = None,
        max_steps: int = 100000,
        record: Optional[bool] = None,
        record_every_step: int = 1,
        y0: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        cap_rec: int = 1024,
        cap_evt: int = 1,
    ) -> None:
        """
        Run the simulation using the compiled model.
        
        Args:
            t0: Initial time (default from spec.sim.t0)
            t_end: End time (default from spec.sim.t_end)
            dt: Initial time step (default from spec.sim.dt)
            max_steps: Maximum number of steps
            record: Enable recording (default from spec.sim.record)
            record_every_step: Record every N steps (1 = every step)
            y0: Initial state (default from spec.state_ic)
            params: Parameters (default from spec.param_vals)
            cap_rec: Initial recording buffer capacity
            cap_evt: Initial event log capacity
        """
        # Use defaults from spec.sim if not provided
        sim_defaults = self.model.spec.sim
        t0 = t0 if t0 is not None else sim_defaults.t0
        t_end = t_end if t_end is not None else sim_defaults.t_end
        dt = dt if dt is not None else sim_defaults.dt
        record = record if record is not None else sim_defaults.record
        
        # Initial state and params
        if y0 is None:
            y0 = np.array(self.model.spec.state_ic, dtype=self.model.model_dtype)
        else:
            y0 = np.array(y0, dtype=self.model.model_dtype)
        
        if params is None:
            params = np.array(self.model.spec.param_vals, dtype=self.model.model_dtype)
        else:
            params = np.array(params, dtype=self.model.model_dtype)
        
        n_state = len(self.model.spec.states)
        
        # Calculate max_log_width from events
        max_log_width = 0
        for event in self.model.spec.events:
            if event.log:
                max_log_width = max(max_log_width, len(event.log))
        
        # Call the wrapper
        result = run_with_wrapper(
            runner=self.model.runner,
            stepper=self.model.stepper,
            rhs=self.model.rhs,
            events_pre=self.model.events_pre,
            events_post=self.model.events_post,
            struct=self.model.struct,
            model_dtype=self.model.model_dtype,
            n_state=n_state,
            t0=t0,
            t_end=t_end,
            dt_init=dt,
            max_steps=max_steps,
            record=record,
            record_every_step=record_every_step,
            y0=y0,
            params=params,
            cap_rec=cap_rec,
            cap_evt=cap_evt,
            max_log_width=max_log_width,
        )
        self._raw_results = result
        self._results_view = None

    def raw_results(self) -> Results:
        """Return the latest raw Results façade (raises if run() not yet called)."""
        if self._raw_results is None:
            raise RuntimeError("No simulation results available; call run() first.")
        return self._raw_results

    def results(self) -> ResultsView:
        """Return a cached ResultsView wrapper over the latest run."""
        if self._results_view is None:
            raw = self.raw_results()
            self._results_view = ResultsView(raw, self.model.spec)
        return self._results_view
