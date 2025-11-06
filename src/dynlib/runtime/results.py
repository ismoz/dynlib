from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np

from dynlib.runtime.runner_api import DONE

__all__ = ["Results"]

@dataclass(frozen=True)
class Results:
    """
    Thin view-only façade returned by the wrapper.

    Fields:
      - T, Y, STEP, FLAGS: backing arrays (not copies)
      - EVT_CODE, EVT_INDEX, EVT_LOG_DATA: event arrays (may be size 1 if disabled)
      - n: number of valid records
      - m: number of valid event entries
      - status: runner exit status code (see runner_api.Status)

    Notes:
      - All accessors return *views* limited to n/m (no copying).
      - Y has shape (n_state, n); states are columns per record index.
      - EVT_LOG_DATA has shape (cap_evt, max_log_width); use EVT_INDEX to know how many values are valid per event.
    """
    # recording (backing arrays)
    T: np.ndarray          # float64, shape (cap_rec,)
    Y: np.ndarray          # model dtype, shape (n_state, cap_rec)
    STEP: np.ndarray       # int64,   shape (cap_rec,)
    FLAGS: np.ndarray      # int32,   shape (cap_rec,)

    # event log (backing arrays)
    EVT_CODE: np.ndarray      # int32,   shape (cap_evt,)
    EVT_INDEX: np.ndarray     # int32,   shape (cap_evt,) - stores log_width
    EVT_LOG_DATA: np.ndarray  # model dtype, shape (cap_evt, max_log_width)

    # filled lengths (cursors)
    n: int                 # filled records
    m: int                 # filled events
    status: int            # runner exit status

    # ---------------- views ----------------

    @property
    def T_view(self) -> np.ndarray:
        return self.T[: self.n]

    @property
    def Y_view(self) -> np.ndarray:
        return self.Y[:, : self.n]

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
        """
        Return event log data view (m, max_log_width).
        Use EVT_INDEX_view to know how many values are valid per event row.
        """
        return self.EVT_LOG_DATA[: self.m, :]

    # --------------- helpers (out of hot path) ---------------

    def to_pandas(self, state_names: Iterable[str] | None = None):
        """
        Build a tidy pandas.DataFrame (optional dependency).
        Columns: 't', 'step', 'flag', and per-state columns.
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
            state_names = [f"s{i}" for i in range(y.shape[0])]
        for idx, name in enumerate(state_names):
            data[str(name)] = y[idx]
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return self.n

    @property
    def ok(self) -> bool:
        """Return True when the runner exited cleanly (status == DONE)."""
        return int(self.status) == DONE
