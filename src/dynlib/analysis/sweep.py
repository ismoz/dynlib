from dataclasses import dataclass
from typing import Literal, Sequence, TYPE_CHECKING
import numpy as np
from dynlib.runtime.sim import Sim
if TYPE_CHECKING:
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
    t: list[np.ndarray]              # length M, each shape (n_i,)
    data: list[np.ndarray]           # length M, each shape (n_i, len(vars))
    meta: dict


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
        t=t_list,
        data=data_list,
        meta=meta,
    )
