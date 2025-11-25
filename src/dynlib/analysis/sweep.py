from dataclasses import dataclass, field
from typing import Literal, Sequence
import numpy as np
from dynlib.runtime.sim import Sim
from dynlib.runtime.results_api import ResultsView

@dataclass
class ParamSweepScalarResult:
    param_name: str
    values: np.ndarray       # (M,)
    var: str                 # "x"
    mode: str                # "final" | "mean" | "max" | ...
    y: np.ndarray            # (M,)
    meta: dict

@dataclass
class ParamSweepTrajResult:
    param_name: str
    values: np.ndarray               # (M,)
    vars: tuple[str, ...]            # ("x", "y", "z")
    t_runs: list[np.ndarray]         # length M, each shape (n_i,)
    data: list[np.ndarray]           # length M, each shape (n_i, len(vars))
    meta: dict
    _var_index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._var_index = {name: i for i, name in enumerate(self.vars)}

    def _stacked_data(self) -> np.ndarray | None:
        """Return stacked data (M, N, len(vars)) if lengths match, else None."""
        if not self.data:
            return np.empty((0, 0, len(self.vars)), dtype=float)
        lengths = {arr.shape[0] for arr in self.data}
        if len(lengths) == 1:
            return np.stack(self.data, axis=0)
        return None

    def stack(self) -> np.ndarray:
        """Force a stacked 3-D view (M, N, len(vars)); raises if runs are ragged."""
        stacked = self._stacked_data()
        if stacked is None:
            raise ValueError("Cannot stack sweep trajectories: run lengths differ.")
        return stacked

    def __getitem__(self, key: str | Sequence[str]) -> np.ndarray | list[np.ndarray]:
        """
        Named access to trajectories by recorded variable.

        Time is the leading axis when stacking succeeds:
        - ``res["x"]`` -> shape (N, M) when runs share a length, else list of (n_i,)
        - ``res[["x","y"]]`` -> shape (N, M, 2) when stackable, else list of (n_i, 2)
        """
        names = (key,) if isinstance(key, str) else tuple(key)
        missing = [nm for nm in names if nm not in self._var_index]
        if missing:
            raise KeyError(f"Unknown variable(s) {missing}; available: {self.vars}")

        cols = [self._var_index[nm] for nm in names]
        stacked = self._stacked_data()
        if stacked is not None:
            if len(cols) == 1:
                return stacked[:, :, cols[0]].T
            return np.transpose(stacked[:, :, cols], (1, 0, 2))

        # Ragged fallback: preserve per-run lengths
        out: list[np.ndarray] = []
        for arr in self.data:
            if len(cols) == 1:
                out.append(arr[:, cols[0]])
            else:
                out.append(arr[:, cols])
        return out

    # ---- time axis convenience ----
    @property
    def t(self) -> np.ndarray:
        """Primary time axis (first run). Useful for plotting when grids match."""
        if not self.t_runs:
            return np.empty((0,), dtype=float)
        return self.t_runs[0]

    @property
    def t_all(self) -> list[np.ndarray]:
        """All per-run time axes (adaptive runs may differ)."""
        return self.t_runs


def _param_index(sim: Sim, name: str) -> int:
    params = list(sim.model.spec.params)
    try:
        return params.index(name)
    except ValueError:
        raise ValueError(
            f"Unknown param {name!r}. Available params: {params}"
        ) from None


def _run_one(
    sim: Sim,
    *,
    param_idx: int,
    param_value: float,
    base_states: np.ndarray,
    base_params: np.ndarray,
    record_vars: list[str],
    T: float | None,
    N: int | None,
    dt: float | None,
    transient: float,
    record_interval: int | None,
    max_steps: int | None,
) -> ResultsView:
    # Build per-run ic/params
    ic = base_states.copy()
    params = base_params.copy()
    params[param_idx] = param_value

    kwargs: dict[str, object] = dict(
        record=True,
        record_vars=record_vars,
        transient=transient,
        resume=False,
    )
    if T is not None:
        kwargs["T"] = float(T)
    if N is not None:
        kwargs["N"] = int(N)
    if dt is not None:
        kwargs["dt"] = float(dt)
    if record_interval is not None:
        kwargs["record_interval"] = int(record_interval)
    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)

    sim.run(ic=ic, params=params, **kwargs)
    return sim.results()


def scalar(
    sim: Sim,
    *,
    param: str,
    values,
    var: str,
    mode: Literal["final", "mean", "max", "min"] = "final",
    T: float | None = None,
    N: int | None = None,
    dt: float | None = None,
    transient: float = 0.0,
    record_interval: int | None = None,
    max_steps: int | None = None,
) -> ParamSweepScalarResult:
    vals = np.asarray(values, dtype=float)
    M = vals.size

    p_idx = _param_index(sim, param)
    base_states = sim.state_vector(source="session", copy=True)
    base_params = sim.param_vector(source="session", copy=True)

    out = np.zeros(M, dtype=float)

    for i, v in enumerate(vals):
        res = _run_one(
            sim,
            param_idx=p_idx,
            param_value=float(v),
            base_states=base_states,
            base_params=base_params,
            record_vars=[var],
            T=T,
            N=N,
            dt=dt,
            transient=transient,
            record_interval=record_interval,
            max_steps=max_steps,
        )
        seg = res.segment[-1]    # isolate this run's samples
        series = seg[var]        # 1D (n,)
        if series.size == 0:
            raise RuntimeError("No samples recorded; adjust T/N/record_interval.")
        if mode == "final":
            out[i] = float(series[-1])
        elif mode == "mean":
            out[i] = float(series.mean())
        elif mode == "max":
            out[i] = float(series.max())
        elif mode == "min":
            out[i] = float(series.min())
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    meta = dict(
        stepper=sim.model.stepper_name,
        kind=sim.model.spec.kind,
        T=T,
        N=N,
        dt=dt,
        transient=transient,
        record_interval=record_interval,
    )
    return ParamSweepScalarResult(
        param_name=param,
        values=vals,
        var=var,
        mode=mode,
        y=out,
        meta=meta,
    )


def traj(
    sim: Sim,
    *,
    param: str,
    values,
    vars: Sequence[str],
    T: float | None = None,
    N: int | None = None,
    dt: float | None = None,
    transient: float = 0.0,
    record_interval: int | None = None,
    max_steps: int | None = None,
) -> ParamSweepTrajResult:
    vals = np.asarray(values, dtype=float)
    M = vals.size
    vars = tuple(vars)

    p_idx = _param_index(sim, param)
    base_states = sim.state_vector(source="session", copy=True)
    base_params = sim.param_vector(source="session", copy=True)

    t_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    record_vars = list(vars)

    for v in vals:
        res = _run_one(
            sim,
            param_idx=p_idx,
            param_value=float(v),
            base_states=base_states,
            base_params=base_params,
            record_vars=record_vars,
            T=T,
            N=N,
            dt=dt,
            transient=transient,
            record_interval=record_interval,
            max_steps=max_steps,
        )

        seg = res.segment[-1]      # isolate this run's samples
        t_full = seg.t             # 1D (n,)
        if t_full.size == 0:
            raise RuntimeError("No samples recorded; adjust T/N/record_interval.")
        series = seg[list(vars)]   # 2D (n, len(vars))

        t_list.append(t_full)
        data_list.append(series)

    meta = dict(
        stepper=sim.model.stepper_name,
        kind=sim.model.spec.kind,
        T=T,
        N=N,
        dt=dt,
        transient=transient,
        record_interval=record_interval,
    )
    return ParamSweepTrajResult(
        param_name=param,
        values=vals,
        vars=vars,
        t_runs=t_list,
        data=data_list,
        meta=meta,
    )
