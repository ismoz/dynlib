# src/dynlib/runtime/sim.py
from __future__ import annotations
from typing import Optional
import dataclasses
import warnings
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
        record_interval: int = 1, # in steps
        ic: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        cap_rec: int = 1024,
        cap_evt: int = 1,
        **stepper_kwargs,
    ) -> None:
        """
        Run the simulation using the compiled model.
        
        Args:
            t0: Initial time (default from spec.sim.t0)
            t_end: End time (default from spec.sim.t_end)
            dt: Initial time step (default from spec.sim.dt)
            max_steps: Maximum number of steps
            record: Enable recording (default from spec.sim.record)
            record_interval: Record every N steps (1 = every step)
            ic: Initial state (default from spec.state_ic)
            params: Parameters (default from spec.param_vals)
            cap_rec: Initial recording buffer capacity
            cap_evt: Initial event log capacity
            **stepper_kwargs: Stepper-specific runtime parameters
            
        Stepper-specific parameters:
            The available parameters depend on the stepper used by the model.
            
            For RK45 (adaptive):
                atol: Absolute tolerance (overrides model_spec.sim.atol)
                rtol: Relative tolerance (overrides model_spec.sim.rtol)
                safety: Safety factor for step size control
                min_factor: Minimum step size reduction factor
                max_factor: Maximum step size increase factor
                max_tries: Maximum number of adaptive retries
                min_step: Minimum allowed step size
            
            For Euler/RK4 (fixed-step):
                No runtime parameters (stepper_kwargs ignored with warning)
                
        Examples:
            sim.run(t0=0, t_end=10)  # Use defaults
            sim.run(t0=0, t_end=10, atol=1e-10, rtol=1e-8)  # Override RK45 tolerances
            sim.run(max_tries=50, min_step=1e-15)  # Override RK45 retry limits
        
        Note:
            Stepper-specific parameters that don't apply to the current stepper
            will trigger a warning and be ignored (e.g., atol/rtol for Euler).
        """
        # Use defaults from spec.sim if not provided
        sim_defaults = self.model.spec.sim
        t0 = t0 if t0 is not None else sim_defaults.t0
        t_end = t_end if t_end is not None else sim_defaults.t_end
        dt = dt if dt is not None else sim_defaults.dt
        record = record if record is not None else sim_defaults.record
        
        # Initial state and params
        if ic is None:
            ic = np.array(self.model.spec.state_ic, dtype=self.model.dtype)
        else:
            ic = np.array(ic, dtype=self.model.dtype)
        
        if params is None:
            params = np.array(self.model.spec.param_vals, dtype=self.model.dtype)
        else:
            params = np.array(params, dtype=self.model.dtype)
        
        n_state = len(self.model.spec.states)
        
        # Calculate max_log_width from events
        max_log_width = 0
        for event in self.model.spec.events:
            if event.log:
                max_log_width = max(max_log_width, len(event.log))
        
        # Build stepper config from kwargs
        stepper_config = self._build_stepper_config(stepper_kwargs)
        
        # Call the wrapper
        result = run_with_wrapper(
            runner=self.model.runner,
            stepper=self.model.stepper,
            rhs=self.model.rhs,
            events_pre=self.model.events_pre,
            events_post=self.model.events_post,
            struct=self.model.struct,
            dtype=self.model.dtype,
            n_state=n_state,
            t0=t0,
            t_end=t_end,
            dt_init=dt,
            max_steps=max_steps,
            record=record,
            record_interval=record_interval,
            ic=ic,
            params=params,
            cap_rec=cap_rec,
            cap_evt=cap_evt,
            max_log_width=max_log_width,
            stepper_config=stepper_config,  # NEW: pass stepper config
        )
        self._raw_results = result
        self._results_view = None
    
    def _build_stepper_config(self, kwargs: dict) -> np.ndarray:
        """
        Build stepper config array from run() kwargs and model_spec.
        
        Priority: run() kwargs > model_spec values > stepper defaults
        
        Args:
            kwargs: Dictionary of stepper-specific parameters from **stepper_kwargs
        
        Returns:
            Packed float64 array for stepper configuration
        """
        from dynlib.steppers.registry import get_stepper
        
        # CRITICAL: Use the actual compiled stepper, not the spec's default
        # The model may have been built with an explicit stepper_name override
        stepper_name = self.model.stepper_name
        stepper_spec = get_stepper(stepper_name)
        
        # Get default config (includes model_spec overrides)
        default_config = stepper_spec.default_config(self.model.spec)
        
        if default_config is None:
            # Stepper has no runtime config
            if kwargs:
                warnings.warn(
                    f"Stepper '{stepper_name}' does not accept runtime parameters. "
                    f"Ignoring: {list(kwargs.keys())}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return np.array([], dtype=np.float64)
        
        # Filter valid kwargs
        valid_fields = {f.name for f in dataclasses.fields(default_config)}
        config_updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        
        # Warn about invalid kwargs
        invalid = set(kwargs.keys()) - valid_fields
        if invalid:
            warnings.warn(
                f"Unknown stepper parameters for '{stepper_name}': {invalid}. "
                f"Valid parameters: {valid_fields}",
                RuntimeWarning,
                stacklevel=3,
            )
        
        # Apply overrides to default config
        if config_updates:
            final_config = dataclasses.replace(default_config, **config_updates)
        else:
            final_config = default_config
        
        # Pack to array
        return stepper_spec.pack_config(final_config)

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
