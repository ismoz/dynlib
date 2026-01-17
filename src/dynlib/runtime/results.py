from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import numpy as np

from dynlib.runtime.runner_api import DONE, EARLY_EXIT

__all__ = ["Results"]

@dataclass(frozen=True)
class Results:
    """
    Thin view-only façade returned by the wrapper.

    Fields:
      - T, Y, STEP, FLAGS: backing arrays (not copies)
      - AUX: aux variables array (may be None)
      - state_names, aux_names: names of recorded variables
      - EVT_CODE, EVT_INDEX, EVT_LOG_DATA: event arrays (may be size 1 if disabled)
      - n: number of valid records
      - m: number of valid event entries
      - status: runner exit status code (see runner_api.Status)

    Notes:
      - All accessors return *views* limited to n/m (no copying).
      - Y has shape (n_rec_states, n); states are columns per record index.
      - AUX has shape (n_rec_aux, n) if aux recorded, else None
      - EVT_LOG_DATA has shape (cap_evt, max_log_width); interpret column counts using the event definition (len(log)).
      - EVT_INDEX stores the owning record index (0-based). Events without a materialized record use -1.
    """
    # recording (backing arrays)
    T: np.ndarray          # float64, shape (cap_rec,)
    Y: np.ndarray          # model dtype, shape (n_rec_states, cap_rec)
    AUX: np.ndarray | None # model dtype, shape (n_rec_aux, cap_rec) or None
    STEP: np.ndarray       # int64,   shape (cap_rec,)
    FLAGS: np.ndarray      # int32,   shape (cap_rec,)

    # event log (backing arrays)
    EVT_CODE: np.ndarray      # int32,   shape (cap_evt,)
    EVT_INDEX: np.ndarray     # int32,   shape (cap_evt,) - stores record index owner (or -1)
    EVT_LOG_DATA: np.ndarray  # model dtype, shape (cap_evt, max_log_width)

    # filled lengths (cursors)
    n: int                 # filled records
    m: int                 # filled events
    status: int            # runner exit status
    final_state: np.ndarray  # final committed state (length = n_state)
    final_params: np.ndarray  # final committed params (length = n_params)
    t_final: float         # final committed time
    final_dt: float        # integrator's next dt suggestion
    step_count_final: int  # last accepted global step (relative to run)
    final_workspace: Mapping[str, Mapping[str, object]]  # snapshots of stepper/runtime workspaces
    
    # metadata for recorded variables
    state_names: list[str] # names of recorded states (ordered)
    aux_names: list[str]   # names of recorded aux (ordered, without "aux." prefix)
    # optional observer artifacts
    analysis_out: np.ndarray | None = None
    analysis_trace: np.ndarray | None = None
    analysis_trace_filled: int | None = None
    analysis_trace_stride: int | None = None
    analysis_trace_offset: int | None = None
    analysis_modules: tuple[object, ...] | None = None
    analysis_meta: Mapping[str, object] | None = None

    # ---------------- views ----------------

    @property
    def T_view(self) -> np.ndarray:
        return self.T[: self.n]

    @property
    def Y_view(self) -> np.ndarray:
        return self.Y[:, : self.n]

    @property
    def AUX_view(self) -> np.ndarray | None:
        return self.AUX[:, : self.n] if self.AUX is not None else None

    @property
    def STEP_view(self) -> np.ndarray:
        return self.STEP[: self.n]

    @property
    def FLAGS_view(self) -> np.ndarray:
        return self.FLAGS[: self.n]

    @property
    def EVT_CODE_view(self) -> np.ndarray:
        return self.EVT_CODE[: self.m]

    @property
    def EVT_INDEX_view(self) -> np.ndarray:
        return self.EVT_INDEX[: self.m]

    @property
    def EVT_LOG_DATA_view(self) -> np.ndarray:
        """Return event log data view (m, max_log_width)."""
        return self.EVT_LOG_DATA[: self.m, :]

    @property
    def final_state_view(self) -> np.ndarray:
        """View of the final committed state vector."""
        return self.final_state

    @property
    def final_params_view(self) -> np.ndarray:
        """View of the final committed parameter vector."""
        return self.final_params

    @property
    def final_workspace_view(self) -> Mapping[str, Mapping[str, object]]:
        """Read-only snapshot of captured workspaces."""
        return self.final_workspace

    @property
    def final_stepper_ws(self) -> Mapping[str, object]:
        """Backward-compatible view of the stepper workspace snapshot."""
        return self.final_workspace.get("stepper", {})

    @property
    def analysis_out_view(self) -> np.ndarray | None:
        """View of online observer outputs, if provided."""
        return self.analysis_out

    @property
    def analysis_trace_view(self) -> np.ndarray | None:
        """View of filled observer trace rows, if provided."""
        if self.analysis_trace is None or self.analysis_trace_filled is None:
            return None
        filled = int(self.analysis_trace_filled)
        return self.analysis_trace[:filled, :]

    @property
    def observers(self) -> Mapping[str, Mapping[str, object]]:
        """Runtime observer outputs keyed by observer key."""
        modules = self.analysis_modules
        if not modules:
            return {}
        out = self.analysis_out_view
        trace = self.analysis_trace_view
        stride = self.analysis_trace_stride
        result: dict[str, Mapping[str, object]] = {}
        out_offset = 0
        trace_offset = 0
        for mod in modules:
            out_slice = None
            if out is not None and mod.output_size > 0:
                out_slice = out[out_offset : out_offset + mod.output_size]
            trace_slice = None
            if trace is not None and mod.trace is not None and trace.size > 0:
                trace_slice = trace[:, trace_offset : trace_offset + mod.trace.width]
            result[mod.key] = {
                "out": out_slice,
                "trace": trace_slice,
                "stride": int(stride) if stride is not None and mod.trace is not None else None,
                "output_names": mod.output_names,
                "trace_names": mod.trace_names,
            }
            out_offset += mod.output_size
            if mod.trace is not None:
                trace_offset += mod.trace.width
        return result

    @property
    def observer_metadata(self) -> Mapping[str, object] | None:
        """Metadata describing attached observer modules."""
        return self.analysis_meta

    # --------------- helpers (out of hot path) ---------------

    def to_pandas(self, state_names: Iterable[str] | None = None):
        """
        Build a tidy pandas.DataFrame (optional dependency).
        Columns: 't', 'step', 'flag', and per-state columns, and per-aux columns.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pandas is required for Results.to_pandas()") from e

        n = self.n
        data = {
            "t": self.T[:n],
            "step": self.STEP[:n],
            "flag": self.FLAGS[:n],
        }
        y = self.Y[:, :n]
        if state_names is None:
            # Use recorded state names if available
            state_names = self.state_names if self.state_names else [f"s{i}" for i in range(y.shape[0])]
        for idx, name in enumerate(state_names):
            data[str(name)] = y[idx]
        
        # Add aux variables if present
        if self.AUX is not None:
            aux = self.AUX[:, :n]
            aux_names = self.aux_names if self.aux_names else [f"aux{i}" for i in range(aux.shape[0])]
            for idx, name in enumerate(aux_names):
                data[f"aux.{name}"] = aux[idx]
        
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return self.n

    @property
    def ok(self) -> bool:
        """Return True when the runner exited cleanly (DONE or EARLY_EXIT)."""
        return int(self.status) in (DONE, EARLY_EXIT)

    @property
    def exited_early(self) -> bool:
        """Return True when the runner stopped due to sim.stop (status == EARLY_EXIT)."""
        return int(self.status) == EARLY_EXIT
    
    def get_var(self, name: str) -> np.ndarray:
        """
        Get recorded variable by name.
        
        Args:
            name: State name ("x"), aux name ("energy"), or aux with prefix ("aux.energy")
        
        Returns:
            Array of shape (n,) with recorded values
        
        Raises:
            KeyError: If variable not recorded
            
        Note:
            Variables are auto-detected: states first, then aux variables.
            Aux variables can be accessed with or without "aux." prefix.
        """
        if name.startswith("aux."):
            # Explicit aux variable with prefix
            aux_name = name[4:]
            if self.AUX is None or aux_name not in self.aux_names:
                raise KeyError(f"Aux variable '{aux_name}' not recorded")
            idx = self.aux_names.index(aux_name)
            return self.AUX[idx, :self.n]
        elif name in self.state_names:
            # State variable
            idx = self.state_names.index(name)
            return self.Y[idx, :self.n]
        elif self.AUX is not None and name in self.aux_names:
            # Aux variable without prefix (auto-detected)
            idx = self.aux_names.index(name)
            return self.AUX[idx, :self.n]
        else:
            # Not found - provide helpful error
            raise KeyError(
                f"Unknown variable: '{name}'. "
                f"Available states: {self.state_names}. "
                f"Available aux: {self.aux_names}."
            )
    
    def __getitem__(self, name: str) -> np.ndarray:
        """Shorthand for get_var: res["x"] or res["aux.energy"]"""
        return self.get_var(name)
