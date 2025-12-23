# src/dynlib/analysis/sweep.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence, TYPE_CHECKING
import warnings
import numpy as np

if TYPE_CHECKING:
    from dynlib.runtime.sim import Sim
from dynlib.runtime.results_api import ResultsView
from dynlib.runtime.fastpath import FixedStridePlan, fastpath_for_sim, fastpath_batch_for_sim
from dynlib.runtime.fastpath.capability import FastpathSupport, assess_capability


class SweepRun:
    """Single run from a parameter sweep with convenient access to time and variables."""

    def __init__(
        self,
        param_value: float,
        t: np.ndarray,
        data: np.ndarray,
        var_index: dict[str, int],
        record_vars: tuple[str, ...],
    ):
        self.param_value = param_value
        self.t = t
        self._data = data
        self._var_index = var_index
        self._record_vars = record_vars

    def __getitem__(self, key: str | Sequence[str]) -> np.ndarray:
        """Access trajectory data by variable name(s)."""
        names = (key,) if isinstance(key, str) else tuple(key)
        missing = [nm for nm in names if nm not in self._var_index]
        if missing:
            raise KeyError(f"Unknown variable(s) {missing}; available: {self._record_vars}")

        cols = [self._var_index[nm] for nm in names]
        if len(cols) == 1:
            return self._data[:, cols[0]]
        return self._data[:, cols]

    def __repr__(self) -> str:
        return f"SweepRun(param={self.param_value}, t={self.t.shape[0]} points, vars={self._record_vars})"


class SweepRunsView:
    """List-like view of all runs in a parameter sweep."""

    def __init__(self, *, values: np.ndarray, payload: "TrajectoryPayload"):
        self._values = values
        self._payload = payload

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: int) -> SweepRun:
        if idx < 0 or idx >= len(self._values):
            raise IndexError(f"Run index {idx} out of range [0, {len(self._values)})")
        return SweepRun(
            param_value=float(self._values[idx]),
            t=self._payload.t_runs[idx],
            data=self._payload.data[idx],
            var_index=self._payload._var_index,
            record_vars=self._payload.record_vars,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"SweepRunsView({len(self)} runs)"


@dataclass
class TrajectoryPayload:
    """Encapsulates trajectory data for parameter sweeps."""

    record_vars: tuple[str, ...]
    t_runs: list[np.ndarray]
    data: list[np.ndarray]
    values: np.ndarray
    _var_index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._var_index = {name: i for i, name in enumerate(self.record_vars)}

    def _stacked_data(self) -> np.ndarray | None:
        """Return stacked data (M, N, len(record_vars)) if lengths match, else None."""
        if not self.data:
            return np.empty((0, 0, len(self.record_vars)), dtype=float)
        lengths = {arr.shape[0] for arr in self.data}
        if len(lengths) == 1:
            return np.stack(self.data, axis=0)
        return None

    def stack(self) -> np.ndarray:
        """Force a stacked 3-D view (M, N, len(record_vars)); raises if runs are ragged."""
        stacked = self._stacked_data()
        if stacked is None:
            raise ValueError("Cannot stack sweep trajectories: run lengths differ.")
        return stacked

    def __getitem__(self, key: str | Sequence[str]) -> np.ndarray | list[np.ndarray]:
        """Named access to trajectories by recorded variable."""
        names = (key,) if isinstance(key, str) else tuple(key)
        missing = [nm for nm in names if nm not in self._var_index]
        if missing:
            raise KeyError(f"Unknown variable(s) {missing}; available: {self.record_vars}")

        cols = [self._var_index[nm] for nm in names]
        stacked = self._stacked_data()
        if stacked is not None:
            if len(cols) == 1:
                return stacked[:, :, cols[0]].T
            return np.transpose(stacked[:, :, cols], (1, 0, 2))

        out: list[np.ndarray] = []
        for arr in self.data:
            if len(cols) == 1:
                out.append(arr[:, cols[0]])
            else:
                out.append(arr[:, cols])
        return out

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

    @property
    def runs(self) -> SweepRunsView:
        """Access individual sweep runs with convenient time and variable access."""
        return SweepRunsView(values=self.values, payload=self)


class SweepResult(Mapping[str, object]):
    """Unified result for parameter sweeps (scalar, trajectory, Lyapunov, etc.)."""

    def __init__(
        self,
        *,
        param_name: str,
        values: np.ndarray,
        kind: str,
        outputs: Mapping[str, object] | None = None,
        traces: Mapping[str, object] | None = None,
        meta: dict | None = None,
        payload: object | None = None,
    ):
        self.param_name = param_name
        self.values = np.asarray(values, dtype=float)
        self.kind = kind
        self.meta = meta or {}
        self._outputs = dict(outputs or {})
        self._traces = dict(traces or {})
        self._payload = payload

        self._output_names: tuple[str, ...] = tuple(self._outputs.keys())
        self._trace_names: tuple[str, ...] = tuple(self._traces.keys())
        self._core_keys = frozenset(
            ["values", "param_name", "kind", "meta", "outputs", "traces", "payload", "output_names", "trace_names"]
        )

    # Mapping interface
    def __getitem__(self, key: str) -> object:
        if key == "values":
            return self.values
        if key == "param_name":
            return self.param_name
        if key == "kind":
            return self.kind
        if key == "meta":
            return self.meta
        if key == "outputs":
            return self._outputs
        if key == "traces":
            return self._traces
        if key == "payload":
            return self._payload
        if key == "output_names":
            return self._output_names
        if key == "trace_names":
            return self._trace_names

        if key in self._outputs:
            return self._outputs[key]
        if key in self._traces:
            return self._traces[key]
        if isinstance(self._payload, TrajectoryPayload) and key in self._payload._var_index:
            return self._payload[key]

        available = (
            list(self._core_keys)
            + list(self._output_names)
            + list(self._trace_names)
            + (list(self._payload.record_vars) if isinstance(self._payload, TrajectoryPayload) else [])
        )
        raise KeyError(f"Unknown field '{key}'. Available: {available}")

    def __iter__(self):
        yield from self._core_keys
        yield from self._output_names
        yield from self._trace_names
        if isinstance(self._payload, TrajectoryPayload):
            yield from self._payload.record_vars

    def __len__(self) -> int:
        payload_len = len(self._payload.record_vars) if isinstance(self._payload, TrajectoryPayload) else 0
        return len(self._core_keys) + len(self._output_names) + len(self._trace_names) + payload_len

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return (
            key in self._core_keys
            or key in self._outputs
            or key in self._traces
            or (isinstance(self._payload, TrajectoryPayload) and key in self._payload._var_index)
        )

    # Attribute access
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._outputs:
            return self._outputs[name]
        if name in self._traces:
            return self._traces[name]
        if isinstance(self._payload, TrajectoryPayload) and name in self._payload._var_index:
            return self._payload[name]
        available = (
            list(self._output_names)
            + list(self._trace_names)
            + (list(self._payload.record_vars) if isinstance(self._payload, TrajectoryPayload) else [])
        )
        raise AttributeError(f"SweepResult has no attribute '{name}'. Available: {available}")

    def __repr__(self) -> str:
        return (
            f"<SweepResult kind={self.kind!r}, param={self.param_name!r}, "
            f"values={self.values.shape}, outputs={self._output_names}, traces={self._trace_names}>"
        )

    # Convenience passthroughs for trajectory payloads
    @property
    def payload(self) -> object | None:
        return self._payload

    @property
    def outputs(self) -> Mapping[str, object]:
        return self._outputs

    @property
    def traces(self) -> Mapping[str, object]:
        return self._traces

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def trace_names(self) -> tuple[str, ...]:
        return self._trace_names

    @property
    def record_vars(self) -> tuple[str, ...]:
        if isinstance(self._payload, TrajectoryPayload):
            return self._payload.record_vars
        return tuple()

    @property
    def t(self) -> np.ndarray:
        if isinstance(self._payload, TrajectoryPayload):
            return self._payload.t
        return np.empty((0,), dtype=float)

    @property
    def t_all(self) -> list[np.ndarray]:
        if isinstance(self._payload, TrajectoryPayload):
            return self._payload.t_all
        return []

    @property
    def data(self) -> list[np.ndarray]:
        if isinstance(self._payload, TrajectoryPayload):
            return self._payload.data
        return []

    @property
    def runs(self) -> SweepRunsView:
        if not isinstance(self._payload, TrajectoryPayload):
            raise AttributeError("runs is only available for trajectory sweep results")
        return self._payload.runs

    def stack(self) -> np.ndarray:
        if not isinstance(self._payload, TrajectoryPayload):
            raise AttributeError("stack is only available for trajectory sweep results")
        return self._payload.stack()

    def series(self, key: str | Sequence[str]) -> object:
        """Named access helper mirroring __getitem__."""
        return self[key]

    def bifurcation(self, var: str):
        """Create a bifurcation extractor for trajectory sweeps."""
        if not isinstance(self._payload, TrajectoryPayload):
            raise AttributeError("bifurcation is only available for trajectory sweep results")
        from dynlib.analysis.post.bifurcation import BifurcationExtractor

        return BifurcationExtractor(self, var)


def _param_index(sim: Sim, name: str) -> int:
    params = list(sim.model.spec.params)
    try:
        return params.index(name)
    except ValueError:
        raise ValueError(
            f"Unknown param {name!r}. Available params: {params}"
        ) from None


def _assess_fastpath_support(
    sim: Sim,
    *,
    plan: FixedStridePlan,
    record_vars: Sequence[str] | None,
    dt: float | None,
    transient: float | None,
) -> FastpathSupport:
    sim_defaults = sim.model.spec.sim
    dt_use = float(dt if dt is not None else sim._nominal_dt if sim._nominal_dt else sim_defaults.dt)
    transient_use = float(transient) if transient is not None else 0.0
    adaptive = getattr(sim._stepper_spec.meta, "time_control", "fixed") == "adaptive"
    return assess_capability(
        sim,
        plan=plan,
        record_vars=record_vars,
        dt=dt_use,
        transient=transient_use,
        adaptive=adaptive,
    )


def _warn_fastpath_fallback(support: FastpathSupport, *, stacklevel: int) -> None:
    reason = f" ({support.reason})" if support.reason else ""
    warnings.warn(
        "Parameter sweep falling back to Sim.run() (fast-path unavailable"
        f"{reason}). For better performance, use jit=True with fixed-step steppers and explicit dt.",
        stacklevel=stacklevel,
    )


def _run_batch_fast(
    sim: Sim,
    *,
    param_idx: int,
    values: np.ndarray,
    base_states: np.ndarray,
    base_params: np.ndarray,
    record_vars: list[str],
    t0: float | None,
    T: float | None,
    N: float | None,
    dt: float | None,
    transient: float | None,
    record_interval: int | None,
    max_steps: int | None,
    parallel_mode: str = "auto",
    max_workers: int | None = None,
) -> tuple[list[ResultsView] | None, FastpathSupport]:
    stride = int(record_interval) if record_interval is not None else 1
    plan = FixedStridePlan(stride=stride)
    support = _assess_fastpath_support(
        sim,
        plan=plan,
        record_vars=record_vars,
        dt=dt,
        transient=transient,
    )

    ic_stack = np.repeat(base_states[np.newaxis, :], values.size, axis=0)
    params_stack = np.repeat(base_params[np.newaxis, :], values.size, axis=0)
    params_stack[:, param_idx] = values

    result = fastpath_batch_for_sim(
        sim,
        plan=plan,
        t0=t0,
        T=T,
        N=int(N) if N is not None else None,
        dt=dt,
        record_vars=record_vars,
        transient=transient,
        record_interval=record_interval,
        max_steps=max_steps,
        ic=ic_stack,
        params=params_stack,
        parallel_mode=parallel_mode,  # type: ignore[arg-type]
        max_workers=max_workers,
    )
    
    return result, support


def _run_one(
    sim: Sim,
    *,
    param_idx: int,
    param_value: float,
    base_states: np.ndarray,
    base_params: np.ndarray,
    record_vars: list[str],
    t0: float | None,
    T: float | None,
    N: int | None,
    dt: float | None,
    transient: float | None,
    record_interval: int | None,
    max_steps: int | None,
    fastpath_support: FastpathSupport | None = None,
    warn_fallback: bool = True,
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
    if t0 is not None:
        kwargs["t0"] = float(t0)
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

    # Try fastpath runner when eligible; fall back to full Sim.run otherwise.
    stride = int(record_interval) if record_interval is not None else 1
    plan = FixedStridePlan(stride=stride)
    fast_res = fastpath_for_sim(
        sim,
        plan=plan,
        t0=t0,
        T=T,
        N=N,
        dt=dt,
        record_vars=record_vars,
        transient=transient,
        record_interval=record_interval,
        max_steps=max_steps,
        ic=ic,
        params=params,
        support=fastpath_support,
    )
    if fast_res is not None:
        return fast_res

    if warn_fallback:
        warnings.warn(
            "Sweep iteration falling back to Sim.run() (fast-path unavailable). "
            "For better performance, use jit=True with fixed-step steppers and explicit dt.",
            stacklevel=2
        )
    sim.run(ic=ic, params=params, **kwargs)
    return sim.results()


def scalar(
    sim: Sim,
    *,
    param: str,
    values,
    var: str,
    mode: Literal["final", "mean", "max", "min"] = "final",
    t0: float | None = None,
    T: float | None = None,
    N: int | None = None,
    dt: float | None = None,
    transient: float | None = None,
    record_interval: int | None = None,
    max_steps: int | None = None,
) -> SweepResult:
    """Sweep a parameter and reduce each run to a single scalar value.
    
    Use this when you only need summary statistics (equilibria, averages, extrema)
    rather than full time-series. Ideal for bifurcation diagrams, parameter
    sensitivity analysis, or equilibrium surfaces.
    
    Args:
        sim: Simulation instance (uses current session state as baseline)
        param: Name of parameter to sweep
        values: Array-like of parameter values to test
        var: Single state variable to record and reduce (e.g., "x")
        mode: How to reduce each trajectory to a scalar:
            - "final": Last recorded value (equilibrium/endpoint)
            - "mean": Time average over the recording window
            - "max": Maximum value reached
            - "min": Minimum value reached
        t0: Initial time (default from sim config)
        T: Absolute end time for continuous systems
        N: Number of iterations (discrete maps)
        dt: Time step (overrides stepper default)
        transient: Time/iterations to discard before recording (default from sim config)
        record_interval: Record every Nth step (memory optimization)
        max_steps: Safety limit on total steps
    
    Returns:
        SweepResult(kind="scalar") with:
            - values: parameter values (M,)
            - y: scalar outputs (M,)
    
    Example:
        >>> # Bifurcation diagram: equilibrium vs parameter
        >>> res = sweep.scalar(sim, param="r", values=np.linspace(2.5, 4.0, 100),
        ...                    var="x", mode="final", N=1000, transient=500)
        >>> plot.curve(res.values, res.y, xlabel="r", ylabel="x*")
    """
    vals = np.asarray(values, dtype=float)
    M = vals.size

    p_idx = _param_index(sim, param)
    base_states = sim.state_vector(source="session", copy=True)
    base_params = sim.param_vector(source="session", copy=True)

    out = np.zeros(M, dtype=float)

    batch_views, batch_support = _run_batch_fast(
        sim,
        param_idx=p_idx,
        values=vals,
        base_states=base_states,
        base_params=base_params,
        record_vars=[var],
        t0=t0,
        T=T,
        N=N,
        dt=dt,
        transient=transient,
        record_interval=record_interval,
        max_steps=max_steps,
    )

    if batch_views is not None:
        runs_iter: Iterable[ResultsView] = batch_views
    else:
        _warn_fastpath_fallback(batch_support, stacklevel=2)
        runs_iter = (
            _run_one(
                sim,
                param_idx=p_idx,
                param_value=float(v),
                base_states=base_states,
                base_params=base_params,
                record_vars=[var],
                t0=t0,
                T=T,
                N=N,
                dt=dt,
                transient=transient,
                record_interval=record_interval,
                max_steps=max_steps,
                fastpath_support=batch_support,
                warn_fallback=False,
            )
            for v in vals
        )

    for i, res in enumerate(runs_iter):
        seg = res.segment[-1]  # isolate this run's samples
        series = seg[var]  # 1D (n,)
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
        t0=t0,
        T=T,
        N=N,
        dt=dt,
        transient=transient,
        record_interval=record_interval,
    )
    meta.update(var=var, mode=mode)
    return SweepResult(
        param_name=param,
        values=vals,
        kind="scalar",
        outputs={"y": out},
        traces={},
        meta=meta,
    )


def traj(
    sim: Sim,
    *,
    param: str,
    values,
    record_vars: Sequence[str],
    t0: float | None = None,
    T: float | None = None,
    N: int | None = None,
    dt: float | None = None,
    transient: float | None = None,
    record_interval: int | None = None,
    max_steps: int | None = None,
    parallel_mode: Literal["auto", "threads", "none"] = "auto",
    max_workers: int | None = None,
) -> SweepResult:
    """Sweep a parameter and collect full time-series trajectories for each run.
    
    Use this when you need complete dynamical behavior: transients, oscillations,
    phase portraits, or any time-dependent phenomena. Records multiple state
    variables over time for each parameter value.
    
    Args:
        sim: Simulation instance (uses current session state as baseline)
        param: Name of parameter to sweep
        values: Array-like of parameter values to test
        record_vars: Sequence of state variable names to record (e.g., ["x", "y", "z"])
                     Can record multiple variables to capture full phase space
        t0: Initial time (default from sim config)
        T: Absolute end time for continuous systems
        N: Number of iterations (discrete maps)
        dt: Time step (overrides stepper default)
        transient: Time/iterations to discard before recording (default from sim config)
        record_interval: Record every Nth step (memory optimization)
        max_steps: Safety limit on total steps
        parallel_mode: Parallel execution mode for fast-path batch runs ("auto", "threads", "none")
        max_workers: Maximum worker threads when parallel_mode uses threads (None = default)
    
    Returns:
        SweepResult(kind="trajectory") with:
            - values: parameter values (M,)
            - payload: TrajectoryPayload containing record_vars, t_runs, data
        
        Access via indexing: res["x"] or res[["x", "y"]]
    
    Example:
        >>> # Phase portraits across parameter values
        >>> res = sweep.traj(sim, param="r", values=[2.5, 3.0, 3.5, 4.0],
        ...                  record_vars=["x", "y"], T=50, transient=10)
        >>> for i, r_val in enumerate(res.values):
        ...     plot.phase2d(res.data[i][:, 0], res.data[i][:, 1],
        ...                  label=f"r={r_val}")
        
        >>> # Or use named access (time-leading axis when stackable)
        >>> x_traces = res["x"]  # shape (N, M) if all runs same length
        >>> plot.traces(res.t, x_traces)  # overlay all parameter traces
    """
    vals = np.asarray(values, dtype=float)
    M = vals.size
    record_vars_tuple = tuple(record_vars)

    p_idx = _param_index(sim, param)
    base_states = sim.state_vector(source="session", copy=True)
    base_params = sim.param_vector(source="session", copy=True)

    t_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    record_vars_list = list(record_vars)

    batch_views, batch_support = _run_batch_fast(
        sim,
        param_idx=p_idx,
        values=vals,
        base_states=base_states,
        base_params=base_params,
        record_vars=record_vars_list,
        t0=t0,
        T=T,
        N=N,
        dt=dt,
        transient=transient,
        record_interval=record_interval,
        max_steps=max_steps,
        parallel_mode=parallel_mode,
        max_workers=max_workers,
    )

    run_iter: Iterable[ResultsView]
    if batch_views is not None:
        run_iter = batch_views
    else:
        _warn_fastpath_fallback(batch_support, stacklevel=2)
        run_iter = (
            _run_one(
                sim,
                param_idx=p_idx,
                param_value=float(v),
                base_states=base_states,
                base_params=base_params,
                record_vars=record_vars_list,
                t0=t0,
                T=T,
                N=N,
                dt=dt,
                transient=transient,
                record_interval=record_interval,
                max_steps=max_steps,
                fastpath_support=batch_support,
                warn_fallback=False,
            )
            for v in vals
        )

    for res in run_iter:
        seg = res.segment[-1]  # isolate this run's samples
        t_full = seg.t  # 1D (n,)
        if t_full.size == 0:
            raise RuntimeError("No samples recorded; adjust T/N/record_interval.")
        series = seg[list(record_vars_tuple)]  # 2D (n, len(record_vars))

        t_list.append(t_full)
        data_list.append(series)

    meta = dict(
        stepper=sim.model.stepper_name,
        kind=sim.model.spec.kind,
        t0=t0,
        T=T,
        N=N,
        dt=dt,
        transient=transient,
        record_interval=record_interval,
        parallel_mode=parallel_mode,
        max_workers=max_workers,
    )
    payload = TrajectoryPayload(
        record_vars=record_vars_tuple,
        t_runs=t_list,
        data=data_list,
        values=vals,
    )
    meta.update(record_vars=record_vars_tuple)
    return SweepResult(
        param_name=param,
        values=vals,
        kind="trajectory",
        outputs={},
        traces={},
        meta=meta,
        payload=payload,
    )


def lyapunov_mle(
    sim: Sim,
    *,
    param: str,
    values,
    t0: float | None = None,
    T: float | None = None,
    N: int | None = None,
    dt: float | None = None,
    transient: float | None = None,
    record_interval: int | None = None,
    max_steps: int | None = None,
    parallel_mode: Literal["auto", "threads", "none"] = "auto",
    max_workers: int | None = None,
    analysis_kind: int = 1,
) -> SweepResult:
    """Sweep a parameter and compute maximum Lyapunov exponent for each value.
    
    Combines parameter sweep with Lyapunov analysis to characterize how chaos
    transitions vary with parameters. Returns both final converged MLE values
    and optional convergence traces.
    
    Args:
        sim: Simulation instance (must be JIT-compiled with fixed-step stepper)
        param: Name of parameter to sweep
        values: Array-like of parameter values to test
        t0: Initial time (default from sim config)
        T: Absolute end time for continuous systems
        N: Number of iterations (discrete maps)
        dt: Time step (required for MLE analysis - must be fixed)
        transient: Time/iterations to discard before recording
        record_interval: Record trace every Nth step (1 = every step)
        max_steps: Safety limit on total steps
        parallel_mode: Parallel execution ("auto", "threads", "none")
        max_workers: Maximum worker threads (None = default)
        analysis_kind: Lyapunov algorithm variant (default 1)
    
    Returns:
        SweepResult(kind="mle") with:
            - values: parameter values (M,)
            - outputs: mle, log_growth, steps
            - traces: optional list of convergence traces (key 'mle')
    
    Example:
        >>> # Characterize chaos onset in logistic map
        >>> r_vals = np.linspace(3.0, 4.0, 100)
        >>> res = sweep.lyapunov_mle(sim, param="r", values=r_vals, 
        ...                          N=5000, transient=1000, record_interval=10)
        >>> series.plot(x=res.values, y=res.mle, xlabel="r", ylabel="λ")
        >>> # λ transitions from negative (stable) to positive (chaotic)
    """
    from dynlib.analysis.runtime import lyapunov_mle as lyapunov_mle_module
    
    vals = np.asarray(values, dtype=float)
    M = vals.size

    p_idx = _param_index(sim, param)
    base_states = sim.state_vector(source="session", copy=True)
    base_params = sim.param_vector(source="session", copy=True)

    # Build analysis module for Lyapunov MLE
    analysis = lyapunov_mle_module(
        model=sim.model,
        record_interval=record_interval,
        analysis_kind=analysis_kind,
    )

    mle_values = np.zeros(M, dtype=float)
    log_growth_list: list[float] = []
    steps_values = np.zeros(M, dtype=float)
    trace_list: list[np.ndarray] | None = [] if record_interval else None

    # Try batch fast-path execution
    ic_stack = np.repeat(base_states[np.newaxis, :], values.size, axis=0)
    params_stack = np.repeat(base_params[np.newaxis, :], values.size, axis=0)
    params_stack[:, p_idx] = vals

    stride = int(record_interval) if record_interval is not None else 1
    plan = FixedStridePlan(stride=stride)
    support = _assess_fastpath_support(
        sim,
        plan=plan,
        record_vars=None,
        dt=dt,
        transient=transient,
    )

    batch_views = fastpath_batch_for_sim(
        sim,
        plan=plan,
        t0=t0,
        T=T,
        N=int(N) if N is not None else None,
        dt=dt,
        record_vars=None,
        transient=transient,
        record_interval=record_interval,
        max_steps=max_steps,
        ic=ic_stack,
        params=params_stack,
        parallel_mode=parallel_mode,  # type: ignore[arg-type]
        max_workers=max_workers,
        analysis=analysis,
    )

    run_iter: Iterable[ResultsView]
    if batch_views is not None:
        run_iter = batch_views
    else:
        _warn_fastpath_fallback(support, stacklevel=2)
        # Fallback to sequential execution
        def _sequential_runs():
            for v in vals:
                ic = base_states.copy()
                params = base_params.copy()
                params[p_idx] = v

                kwargs: dict[str, object] = dict(
                    record=False,
                    transient=transient,
                    resume=False,
                    analysis=analysis,
                )
                if t0 is not None:
                    kwargs["t0"] = float(t0)
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
                yield sim.results()
        
        run_iter = _sequential_runs()

    # Extract analysis results from each run
    for i, res in enumerate(run_iter):
        lyap_result = res.analysis["lyapunov_mle"]
        mle_values[i] = float(lyap_result.mle)
        log_growth_list.append(float(lyap_result.log_growth))
        steps_values[i] = float(lyap_result.steps)
        
        if trace_list is not None:
            trace_array = lyap_result["mle"]
            trace_list.append(trace_array)

    meta = dict(
        stepper=sim.model.stepper_name,
        kind=sim.model.spec.kind,
        t0=t0,
        T=T,
        N=N,
        dt=dt,
        transient=transient,
        record_interval=record_interval,
        parallel_mode=parallel_mode,
        max_workers=max_workers,
        analysis_kind=analysis_kind,
    )
    outputs = dict(
        mle=mle_values,
        log_growth=np.array(log_growth_list, dtype=float),
        steps=steps_values,
    )
    traces: dict[str, object] = {}
    if trace_list is not None:
        traces["mle"] = trace_list

    return SweepResult(
        param_name=param,
        values=vals,
        kind="mle",
        outputs=outputs,
        traces=traces,
        meta=meta,
    )
