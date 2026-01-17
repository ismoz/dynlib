# src/dynlib/runtime/results_api.py
# High-level, name-aware wrapper around the low-level Results façade.
#
# - Do NOT modify results.py; this module composes over it.
# - Zero changes to ABI / buffers; read-only views where possible.
# - States & time:
#     res.t                 -> recorded time axis (length n)
#     res["v"] / res[[...]] -> recorded state series / stacked series
# - Events (single doorway):
#     ev = res.event("spike")
#     ev.t                  -> event times if logged (else informative error)
#     ev["id"] / ev[[...]]  -> logged fields
#     grp = res.event(tag="spiking")  -> grouped multi-event view
#
# Note on copying semantics:
# - Trajectory/state access returns NumPy *views* of the backing arrays.
# - Event row selection in EVT_LOG_DATA requires fancy indexing and will
#   allocate a compact array (NumPy cannot provide a view for arbitrary
#   scattered rows). We keep this path tight and documented.

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from dynlib.runtime.results import Results as _RawResults
from dynlib.dsl.spec import ModelSpec

if TYPE_CHECKING:
    from .sim import Segment
    from dynlib.runtime.observers import ObserverModule

__all__ = [
    "ObserverResult",
    "ResultsView",
]

# ------------------------------ utilities ------------------------------------

def _ensure_tuple(x: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
    if x is None:
        return tuple()
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def _friendly_key_error(kind: str, name: str, options: Iterable[str]) -> KeyError:
    opts = ", ".join(options)
    return KeyError(f"Unknown {kind} '{name}'. Available: {opts}")


# ------------------------------ observer results ----------------------------

class ObserverResult(Mapping[str, object]):
    """Smart wrapper for runtime observer results with named access.
    
    Provides both Mapping interface (backward compat) and attribute access (ergonomic).
    Automatically exposes output_names and trace_names as attributes/keys.
    
    Mapping keys: "out", "trace", "trace_steps", "trace_time", "record_interval", "stride", "output_names", "trace_names"
    Named access: Any name from output_names or trace_names
    
    Examples:
        >>> lyap = result_view.observers["lyapunov_mle"]
        >>> # Mapping interface (backward compat)
        >>> lyap["out"][0]  # raw access
        >>> lyap["trace"][:, 0]
        >>> # Named access (ergonomic)
        >>> lyap.log_growth  # from output_names
        >>> lyap.steps
        >>> lyap.mle  # from trace_names
        >>> # Discovery
        >>> lyap.output_names  # ('log_growth', 'steps')
        >>> lyap.trace_names   # ('mle',)
        >>> list(lyap)  # all keys including named outputs
    """
    
    def __init__(
        self,
        *,
        name: str,
        out: np.ndarray | None,
        trace: np.ndarray | None,
        stride: int | None,
        trace_steps: np.ndarray | None,
        trace_time: np.ndarray | None,
        output_names: Tuple[str, ...] | None,
        trace_names: Tuple[str, ...] | None,
    ):
        self._name = name
        self._out = out
        self._trace = trace
        self._stride = stride
        self._trace_steps = trace_steps
        self._trace_time = trace_time
        self._output_names = output_names or ()
        self._trace_names = trace_names or ()
        
        # Build index maps for O(1) lookup
        self._output_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self._output_names)
        }
        self._trace_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self._trace_names)
        }
        
        # Core mapping keys (record_interval is the user-facing alias)
        self._keys = frozenset(
            ["out", "trace", "trace_steps", "trace_time", "record_interval", "stride", "output_names", "trace_names"]
        )
    
    # ---- Mapping interface (backward compat) ----
    
    def __getitem__(self, key: str) -> object:
        """Bracket access: supports both raw keys and named outputs/traces."""
        # Core mapping keys
        if key == "out":
            return self._out
        if key == "trace":
            return self._trace
        if key == "trace_steps":
            return self._trace_steps
        if key == "trace_time":
            return self._trace_time
        if key == "record_interval":
            return self._stride
        if key == "stride":
            return self._stride
        if key == "output_names":
            return self._output_names
        if key == "trace_names":
            return self._trace_names
        
        # Named outputs
        if key in self._output_idx:
            if self._out is None:
                return None
            return self._out[self._output_idx[key]]
        
        # Named traces
        if key in self._trace_idx:
            if self._trace is None or self._trace.size == 0:
                raise KeyError(
                    f"Trace '{key}' not recorded for observer '{self._name}'. "
                    "Enable recording with record_interval or trace_plan."
                )
            return self._trace[:, self._trace_idx[key]]
        
        # Unknown
        available = list(self._keys) + list(self._output_names) + list(self._trace_names)
        raise _friendly_key_error("field", key, available)
    
    def __iter__(self):
        """Iterate over all keys: core + output_names + trace_names."""
        yield from self._keys
        yield from self._output_names
        yield from self._trace_names
    
    def __len__(self) -> int:
        """Total number of accessible keys."""
        return len(self._keys) + len(self._output_names) + len(self._trace_names)
    
    def __contains__(self, key: object) -> bool:
        """Check if key exists."""
        if not isinstance(key, str):
            return False
        return (
            key in self._keys
            or key in self._output_idx
            or key in self._trace_idx
        )
    
    # ---- Attribute access (ergonomic) ----
    
    def __getattr__(self, name: str) -> object:
        """Attribute access for named outputs and traces.
        
        Special handling for trace names accessed as attributes:
        - Returns final scalar value if trace is 1D
        - Enables ergonomic access like `lyap.mle` returning final converged value
        
        Raises AttributeError with helpful message for non-identifier names.
        """
        # Avoid recursion on private/special attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Named outputs
        if name in self._output_idx:
            if self._out is None:
                return None
            return self._out[self._output_idx[name]]
        
        # Named traces - return final scalar value for 1D traces
        if name in self._trace_idx:
            if self._trace is None or self._trace.size == 0:
                raise AttributeError(
                    f"Trace '{name}' not recorded for observer '{self._name}'. "
                    "Enable recording with record_interval or trace_plan."
                )
            col = self._trace[:, self._trace_idx[name]]
            # Return final scalar value for ergonomic access
            # (for full trace array, use bracket access: result["mle"])
            return float(col[-1]) if col.size > 0 else 0.0
        
        # Check if it's a non-identifier name in output/trace names
        all_names = list(self._output_names) + list(self._trace_names)
        non_identifiers = [n for n in all_names if not n.isidentifier()]
        if non_identifiers:
            raise AttributeError(
                f"'{name}' not found. Non-identifier names ({non_identifiers}) "
                f"must use bracket access: result['{non_identifiers[0]}']"
            )
        
        # Not found
        available = list(self._output_names) + list(self._trace_names)
        raise AttributeError(
            f"Observer '{self._name}' has no attribute '{name}'. "
            f"Available: {available}"
        )
    
    def __dir__(self) -> List[str]:
        """Include output/trace names for tab-completion discovery."""
        # Standard attributes
        base = list(super().__dir__())
        # Add valid identifier names from outputs/traces
        identifiers = [
            name for name in list(self._output_names) + list(self._trace_names)
            if name.isidentifier()
        ]
        return sorted(set(base + identifiers))
    
    # ---- Properties ----
    
    @property
    def name(self) -> str:
        """Observer name."""
        return self._name
    
    @property
    def out(self) -> np.ndarray | None:
        """Raw output array."""
        return self._out
    
    @property
    def trace(self) -> np.ndarray | None:
        """Raw trace array (all columns)."""
        return self._trace

    @property
    def trace_steps(self) -> np.ndarray | None:
        """Step indices corresponding to trace rows."""
        return self._trace_steps

    @property
    def trace_time(self) -> np.ndarray | None:
        """Time values corresponding to trace rows (if available)."""
        return self._trace_time
    
    @property
    def stride(self) -> int | None:
        """Trace recording stride."""
        return self._stride

    @property
    def record_interval(self) -> int | None:
        """Trace recording interval (user-facing alias of stride)."""
        return self._stride
    
    @property
    def output_names(self) -> Tuple[str, ...]:
        """Names of output components."""
        return self._output_names
    
    @property
    def trace_names(self) -> Tuple[str, ...]:
        """Names of trace components."""
        return self._trace_names
    
    def __repr__(self) -> str:
        out_info = f"{len(self._output_names)} outputs" if self._output_names else "no outputs"
        trace_info = f"{len(self._trace_names)} traces" if self._trace_names else "no traces"
        return f"<ObserverResult '{self._name}': {out_info}, {trace_info}>"


# ------------------------------ main wrapper ---------------------------------

class ResultsView:
    """User-facing, name-aware view on top of the low-level :class:`Results`.

    Construct with the raw results and a frozen :class:`ModelSpec` to enable
    named access. No mutation, no copies for trajectory access. Per-run
    segments are exposed via ``res.segment[...]`` mirroring the main view API.
    """

    # ---- construction ----
    def __init__(
        self,
        raw: _RawResults,
        spec: ModelSpec,
        *,
        event_code_map: Optional[Mapping[str, int]] = None,
        segments: Optional[Sequence["Segment"]] = None,
    ) -> None:
        self._raw = raw
        self._spec = spec

        # State names in recorded order (fall back to spec if missing)
        recorded_states = tuple(raw.state_names) if getattr(raw, "state_names", None) is not None else tuple(spec.states)
        self._state_names: Tuple[str, ...] = recorded_states
        self._state_index: Dict[str, int] = {name: i for i, name in enumerate(self._state_names)}

        # Aux names in recorded order (fall back to spec if missing)
        aux_from_spec = tuple(spec.aux.keys()) if getattr(spec, "aux", None) else tuple()
        self._aux_names: Tuple[str, ...] = (
            tuple(raw.aux_names) if getattr(raw, "aux_names", None) is not None else aux_from_spec
        )

        # Event name <-> code mapping.
        # If not provided, assume codes follow declaration order (0..E-1).
        if event_code_map is None:
            self._ev_name_to_code: Dict[str, int] = {ev.name: i for i, ev in enumerate(spec.events)}
        else:
            self._ev_name_to_code = dict(event_code_map)
        self._ev_code_to_name: Dict[int, str] = {code: name for name, code in self._ev_name_to_code.items()}

        # Per-event field layout from the DSL spec
        self._ev_fields: Dict[str, Tuple[str, ...]] = {ev.name: tuple(ev.log) for ev in spec.events}

        # Tag index taken directly from the spec (tag -> tuple of event names)
        self._tag_index: Dict[str, Tuple[str, ...]] = dict(spec.tag_index)

        # Single accessor object, callable + discovery helpers
        self.event: EventAccessor = EventAccessor(self)
        self.segment = SegmentsView(self, list(segments) if segments else [])

    # ---- core trajectory access ----
    @property
    def t(self) -> np.ndarray:
        """Recorded times (length ``n``). View into backing buffer."""
        return self._raw.T_view

    @property
    def step(self) -> np.ndarray:
        """Recorded step indices (length ``n``). View into backing buffer."""
        return self._raw.STEP_view

    @property
    def flags(self) -> np.ndarray:
        """Recorded flags (length ``n``). View into backing buffer."""
        return self._raw.FLAGS_view

    def __len__(self) -> int:  # number of recorded rows
        return int(self._raw.n)

    @property
    def state_names(self) -> Tuple[str, ...]:
        return self._state_names

    @property
    def aux_names(self) -> Tuple[str, ...]:
        return self._aux_names

    def __getitem__(self, key: Union[str, Sequence[str]]) -> np.ndarray:
        """Access recorded state or aux variable series by name.

        - ``res["v"]`` -> 1-D view (length ``n``) for state or aux variable
        - ``res["aux.energy"]`` -> 1-D view with explicit aux prefix
        - ``res[["v","w"]]`` -> 2-D array (shape ``(n, k)``) stacked in request
          order. This requires fancy indexing and therefore returns a compact
          copy, unlike the single-variable view above.
          
        Variables are auto-detected: states first, then aux variables.
        Aux variables can be accessed with or without the "aux." prefix.
        """
        if isinstance(key, str):
            # Single variable - delegate to raw results which handles auto-detection
            return self._raw.get_var(key)
        
        # sequence of names - need to handle each one
        names = _ensure_tuple(key)
        arrays: List[np.ndarray] = []
        for nm in names:
            arrays.append(self._raw.get_var(nm))
        
        # Stack as columns (each array is already shape (n,))
        return np.column_stack(arrays)

    # ---- low-level passthrough ----
    @property
    def ok(self) -> bool:
        return bool(self._raw.ok)

    @property
    def exited_early(self) -> bool:
        return bool(self._raw.exited_early)

    @property
    def status(self) -> int:
        return int(self._raw.status)

    @property
    def step_count_final(self) -> int:
        return int(self._raw.step_count_final)

    @property
    def n(self) -> int:
        return int(self._raw.n)

    @property
    def m(self) -> int:
        return int(self._raw.m)

    # ---- runtime observers (during-run diagnostics) ----
    @property
    def observers(self) -> Dict[str, ObserverResult]:
        """
        Runtime observer outputs keyed by observer key.

        Each result is an :class:`ObserverResult` providing:
        
        - Mapping interface (backward compat): ``result["out"]``, ``result["trace"]``
        - Named access (ergonomic): ``result.log_growth``, ``result.steps``, ``result.mle``
        - Discovery: ``result.output_names``, ``result.trace_names``, ``list(result)``
        
        Examples:
            >>> lyap = res.observers["lyapunov_mle"]
            >>> # Named access (auto-discovered from output_names)
            >>> lyap.log_growth  # instead of lyap["out"][0]
            >>> lyap.steps       # instead of lyap["out"][1]
            >>> lyap.mle         # trace value from trace_names
            >>> # Mapping interface still works
            >>> lyap["out"], lyap["trace"]
            >>> # Discovery
            >>> lyap.output_names  # ('log_growth', 'steps')
            >>> list(lyap)  # all available keys
        """
        modules: Tuple["ObserverModule", ...] | None = getattr(self._raw, "analysis_modules", None)  # type: ignore[attr-defined]
        if not modules:
            return {}
        out_attr = getattr(self._raw, "analysis_out_view", None)
        out = out_attr() if callable(out_attr) else out_attr
        trace_attr = getattr(self._raw, "analysis_trace_view", None)
        trace = trace_attr() if callable(trace_attr) else trace_attr
        stride = getattr(self._raw, "analysis_trace_stride", None)
        trace_start_offset = getattr(self._raw, "analysis_trace_offset", None)
        result: Dict[str, ObserverResult] = {}
        out_offset = 0
        trace_col_offset = 0
        for mod in modules:
            out_slice = None
            if out is not None and mod.output_size > 0:
                out_slice = out[out_offset : out_offset + mod.output_size]
            trace_slice = None
            if trace is not None and mod.trace is not None and trace.size > 0:
                trace_slice = trace[:, trace_col_offset : trace_col_offset + mod.trace.width]
            
            trace_steps = None
            trace_time = None
            if trace_slice is not None and trace_slice.size > 0 and stride is not None:
                steps = self._raw.STEP_view
                stride_val = int(stride)
                if steps.size > 0 and stride_val > 0:
                    step_start = int(steps[0])
                    offset_val = int(trace_start_offset or 0)
                    first_step = step_start + ((stride_val - (step_start % stride_val)) % stride_val)
                    trace_len = trace_slice.shape[0]
                    trace_steps = first_step + (offset_val + np.arange(trace_len)) * stride_val
                    if self._raw.T_view.size > 1:
                        t_vals = self._raw.T_view
                        step_vals = steps
                        step_delta = int(step_vals[1] - step_vals[0])
                        if step_delta != 0:
                            dt = (t_vals[1] - t_vals[0]) / float(step_delta)
                            t0 = t_vals[0] - dt * float(step_vals[0])
                            trace_time = t0 + dt * trace_steps

            # Wrap in ObserverResult for ergonomic access
            result[mod.key] = ObserverResult(
                name=mod.name,
                out=out_slice,
                trace=trace_slice,
                stride=int(stride) if stride is not None and mod.trace is not None else None,
                trace_steps=trace_steps,
                trace_time=trace_time,
                output_names=mod.output_names,
                trace_names=mod.trace_names,
            )
            
            out_offset += mod.output_size
            if mod.trace is not None:
                trace_col_offset += mod.trace.width
        return result

    @property
    def observer_metadata(self) -> Mapping[str, object] | None:
        """Return observer metadata attached to the raw results."""
        raw_meta = getattr(self._raw, "observer_metadata", None)
        if callable(raw_meta):
            return raw_meta()
        return getattr(self._raw, "analysis_meta", None)

    # ---- discovery helpers (states/events/tags) ----
    def event_names(self) -> Tuple[str, ...]:
        return tuple(self._ev_fields.keys())

    def event_fields(self, name: str) -> Tuple[str, ...]:
        if name not in self._ev_fields:
            raise _friendly_key_error("event", name, self._ev_fields.keys())
        return self._ev_fields[name]

    def tag_names(self) -> Tuple[str, ...]:
        return tuple(self._tag_index.keys())

    def events_by_tag(self, tag: str) -> Tuple[str, ...]:
        if tag not in self._tag_index:
            raise _friendly_key_error("tag", tag, self._tag_index.keys())
        return self._tag_index[tag]

    # ---- trajectory analysis ----
    def analyze(self, var: Union[str, Sequence[str], None] = None):
        """Get analyzer for trajectory variable(s).
        
        Returns a TrajectoryAnalyzer (for single variable) or MultiVarAnalyzer
        (for multiple variables) providing statistical and temporal analysis
        methods like max, min, argmax, crossings, etc.
        
        Args:
            var: Variable name, list of names, or None for recorded variables
                 (prefers states, falls back to recorded aux)
        
        Returns:
            TrajectoryAnalyzer or MultiVarAnalyzer instance
        
        Examples:
            >>> # Analyze single variable
            >>> res.analyze("x").max()
            3.14
            >>> res.analyze("x").argmax()
            (45.2, 3.14)
            
            >>> # Analyze multiple variables
            >>> res.analyze(["x", "y"]).max()
            {'x': 3.14, 'y': 2.71}
            
            >>> # Analyze recorded variables (states if present, else aux)
            >>> res.analyze().summary()
            {'x': {'min': -1.0, 'max': 3.14, ...}, 'y': {...}}
            
            >>> # Find crossings
            >>> res.analyze("x").crossing_times(threshold=0.5, direction="up")
            array([12.3, 24.7, 36.1])
        """
        from dynlib.analysis.post import TrajectoryAnalyzer, MultiVarAnalyzer
        
        if var is None:
            # Prefer recorded states; if none, fall back to recorded aux
            var_names: Tuple[str, ...] = self.state_names or self.aux_names
            if not var_names:
                raise ValueError("No recorded variables available to analyze (record_vars=[]).")
            data = self[list(var_names)]
            return MultiVarAnalyzer(self, var_names, data, self.t)
        elif isinstance(var, str):
            # Single variable
            data = self[var]
            return TrajectoryAnalyzer(self, var, data, self.t)
        else:
            # Multiple variables
            var_names = tuple(var)
            data = self[list(var_names)]
            return MultiVarAnalyzer(self, var_names, data, self.t)

    # Internal helpers for EventAccessor
    # ----------------------------------
    def _event_code_for(self, name: str) -> int:
        if name not in self._ev_name_to_code:
            raise _friendly_key_error("event", name, self._ev_name_to_code.keys())
        return int(self._ev_name_to_code[name])

    def _event_fields_for(self, name: str) -> Tuple[str, ...]:
        return self.event_fields(name)

    def _mask_or_indices_for_code(self, code: int) -> np.ndarray:
        """Return row indices for occurrences of the given event code.

        We compute and return the *indices* of rows in EVT_LOG_DATA where
        EVT_CODE == code, restricted to the filled region ``m``.
        """
        codes = self._raw.EVT_CODE_view  # shape (m,)
        # Boolean mask then nonzero -> indices; unavoidable allocation in NumPy.
        m = int(self._raw.m)
        return np.nonzero(codes[:m] == code)[0]


# ------------------------------ segment views --------------------------------

def _segment_auto_name(seg: "Segment") -> str:
    return f"run#{seg.id}"


def _segment_effective_name(seg: "Segment") -> str:
    return seg.name if seg.name is not None else _segment_auto_name(seg)


class SegmentView:
    """Read-only view over a single recorded segment."""

    def __init__(self, owner: "ResultsView", seg: "Segment") -> None:
        self._o = owner
        self._s = seg

    @property
    def id(self) -> int:
        return int(self._s.id)

    @property
    def name(self) -> str:
        return _segment_effective_name(self._s)

    @property
    def meta(self) -> "Segment":
        return self._s

    @property
    def t(self) -> np.ndarray:
        i0, n = self._s.rec_start, self._s.rec_len
        return self._o.t[i0 : i0 + n]

    @property
    def step(self) -> np.ndarray:
        i0, n = self._s.rec_start, self._s.rec_len
        return self._o.step[i0 : i0 + n]

    @property
    def flags(self) -> np.ndarray:
        i0, n = self._s.rec_start, self._s.rec_len
        return self._o.flags[i0 : i0 + n]

    def events(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        j0, m = self._s.evt_start, self._s.evt_len
        codes = self._o._raw.EVT_CODE_view[j0 : j0 + m]
        idx = self._o._raw.EVT_INDEX_view[j0 : j0 + m]
        logs = self._o._raw.EVT_LOG_DATA_view[j0 : j0 + m, :]
        return codes, idx, logs

    def __getitem__(self, key: Union[str, Sequence[str]]) -> np.ndarray:
        arr = self._o[key]
        i0, n = self._s.rec_start, self._s.rec_len
        return arr[i0 : i0 + n, ...] if arr.ndim >= 1 else arr


class SegmentsView:
    """Container providing subscriptable access to recorded segments."""

    def __init__(self, owner: "ResultsView", segments: Sequence["Segment"]) -> None:
        self._o = owner
        self._segments: List["Segment"] = list(segments)
        self._name_to_idx: Dict[str, int] = {}
        for idx, seg in enumerate(self._segments):
            # Auto alias always available
            auto = _segment_auto_name(seg)
            self._name_to_idx.setdefault(auto, idx)
            effective = _segment_effective_name(seg)
            self._name_to_idx[effective] = idx

    def __getitem__(self, key: Union[int, str]) -> SegmentView:
        if isinstance(key, int):
            seg = self._segments[key]
            return SegmentView(self._o, seg)
        if key not in self._name_to_idx:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown segment '{key}'. Known names: {known}")
        idx = self._name_to_idx[key]
        return SegmentView(self._o, self._segments[idx])

    def __len__(self) -> int:
        return len(self._segments)

    def __iter__(self):
        for seg in self._segments:
            yield SegmentView(self._o, seg)

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    def names(self) -> List[str]:
        return [_segment_effective_name(seg) for seg in self._segments]

# ----------------------------- event accessor ---------------------------------

class EventAccessor:
    """Callable accessor and discovery hub for event data.

    Usage:
        ev = res.event("spike")       # -> EventView
        grp = res.event(tag="spiking")  # -> EventGroupView

    Discovery helpers live here as methods: ``names()``, ``fields(name)``,
    ``tags()``, ``by_tag(tag)``, ``summary()``.
    """
    def __init__(self, parent: ResultsView) -> None:
        self._p = parent

    # ---- callable doorway ----
    def __call__(self, name: Optional[str] = None, *, tag: Optional[str] = None):
        if (name is None) == (tag is None):
            raise ValueError("Use exactly one of: name or tag")
        if name is not None:
            return EventView(self._p, name)
        # tag
        names = self._p.events_by_tag(tag)  # validates tag
        return EventGroupView(self._p, names)

    # ---- discovery ----
    def names(self) -> Tuple[str, ...]:
        return self._p.event_names()

    def fields(self, name: str) -> Tuple[str, ...]:
        return self._p.event_fields(name)

    def tags(self) -> Tuple[str, ...]:
        return self._p.tag_names()

    def by_tag(self, tag: str) -> Tuple[str, ...]:
        return self._p.events_by_tag(tag)

    def summary(self) -> Dict[str, int]:
        """Counts per event name across the entire log (filled region)."""
        counts: Dict[str, int] = {}
        m = int(self._p._raw.m)
        codes = self._p._raw.EVT_CODE_view[:m]
        # Map codes -> counts, then to names (unknown codes are ignored)
        unique, freq = np.unique(codes, return_counts=True)
        for code, cnt in zip(unique.tolist(), freq.tolist()):
            name = self._p._ev_code_to_name.get(int(code))
            if name is not None:
                counts[name] = cnt
        return counts


# ----------------------------- per-event view ---------------------------------

class EventView:
    """Read-only view of one event type.

    Provides attribute ``.t`` (if logged) and bracket access for any logged
    field: ``ev["id"]`` or ``ev[["t","id"]]``. Row selection uses indices
    and returns compact arrays (NumPy cannot view scattered rows).
    """
    def __init__(self, parent: ResultsView, name: str, *, _row_idx: Optional[np.ndarray] = None):
        self._p = parent
        self._name = name
        self._fields = parent._event_fields_for(name)
        self._code = parent._event_code_for(name)
        self._row_idx = _row_idx  # optional pre-filtered indices
        if self._row_idx is None:
            self._row_idx = parent._mask_or_indices_for_code(self._code)
        # Field -> column offset map for quick lookup
        self._col_ofs: Dict[str, int] = {f: i for i, f in enumerate(self._fields)}

    # ---- structural info ----
    @property
    def name(self) -> str:
        return self._name

    @property
    def fields(self) -> Tuple[str, ...]:
        return self._fields

    @property
    def count(self) -> int:
        return int(self._row_idx.shape[0])

    # ---- time sugar ----
    @property
    def t(self) -> np.ndarray:
        if "t" not in self._col_ofs:
            raise ValueError(
                f"Event '{self._name}' has no 't' in its log. Add log=['t', ...] in the DSL.")
        col = self._col_ofs["t"]
        return self._p._raw.EVT_LOG_DATA_view[self._row_idx, col]

    # ---- data access ----
    def __getitem__(self, key: Union[str, Sequence[str]]) -> np.ndarray:
        if isinstance(key, str):
            if key not in self._col_ofs:
                raise _friendly_key_error("field", key, self._fields)
            col = self._col_ofs[key]
            return self._p._raw.EVT_LOG_DATA_view[self._row_idx, col]
        names = _ensure_tuple(key)
        cols: List[int] = []
        for nm in names:
            if nm not in self._col_ofs:
                raise _friendly_key_error("field", nm, self._fields)
            cols.append(self._col_ofs[nm])
        return self._p._raw.EVT_LOG_DATA_view[self._row_idx[:, None], np.array(cols, dtype=int)]

    # ---- filtering / chaining ----
    def time(self, t0: float, t1: float) -> "EventView":
        if "t" not in self._col_ofs:
            raise ValueError("Cannot filter by time: 't' not logged for this event")
        tcol = self._p._raw.EVT_LOG_DATA_view[self._row_idx, self._col_ofs["t"]]
        mask = (tcol >= t0) & (tcol <= t1)
        return EventView(self._p, self._name, _row_idx=self._row_idx[mask])

    def head(self, k: int) -> "EventView":
        return EventView(self._p, self._name, _row_idx=self._row_idx[:k])

    def tail(self, k: int) -> "EventView":
        return EventView(self._p, self._name, _row_idx=self._row_idx[-k:])

    def sort(self, by: str = "t") -> "EventView":
        if by not in self._col_ofs:
            raise _friendly_key_error("field", by, self._fields)
        vals = self._p._raw.EVT_LOG_DATA_view[self._row_idx, self._col_ofs[by]]
        order = np.argsort(vals, kind="stable")
        return EventView(self._p, self._name, _row_idx=self._row_idx[order])

    # Optional convenience: materialize a named table (2-D array) of all fields
    def table(self) -> np.ndarray:
        cols = list(range(len(self._fields)))
        return self._p._raw.EVT_LOG_DATA_view[self._row_idx[:, None], np.array(cols, dtype=int)]


# --------------------------- grouped events view ------------------------------

class EventGroupView:
    """Concatenated view across multiple event types selected by a tag.

    Default field policy is intersection across member events for bracket access.
    Use ``select`` / ``table`` for explicit intersection or union reads.
    """
    def __init__(self, parent: ResultsView, names: Sequence[str]):
        self._p = parent
        self._names: Tuple[str, ...] = tuple(names)
        # Build EventViews now (row indices & mappings cached per member)
        self._members: Tuple[EventView, ...] = tuple(EventView(parent, nm) for nm in self._names)

    # ---- info ----
    @property
    def names(self) -> Tuple[str, ...]:
        return self._names

    def counts(self) -> Dict[str, int]:
        return {ev.name: ev.count for ev in self._members}

    def fields(self, mode: str = "intersection") -> Tuple[str, ...]:
        if mode not in {"intersection", "union"}:
            raise ValueError("mode must be 'intersection' or 'union'")
        seqs = [set(ev.fields) for ev in self._members]
        if not seqs:
            return tuple()
        if mode == "intersection":
            common = set.intersection(*seqs)
            # Preserve a deterministic order by following the first member
            return tuple([f for f in self._members[0].fields if f in common])
        # union
        ordered = []
        seen = set()
        for ev in self._members:
            for f in ev.fields:
                if f not in seen:
                    ordered.append(f)
                    seen.add(f)
        return tuple(ordered)

    # ---- attribute sugar for 't' when universally present ----
    @property
    def t(self) -> np.ndarray:
        fields = self.fields(mode="intersection")
        if "t" not in fields:
            raise ValueError("Not all grouped events logged 't'; cannot read .t uniformly")
        return self["t"]

    # ---- data access ----
    def __getitem__(self, key: Union[str, Sequence[str]]) -> np.ndarray:
        inter_fields = self.fields(mode="intersection")
        if isinstance(key, str):
            field = key
            if field not in inter_fields:
                raise ValueError(
                    f"Field '{field}' is not common to all events. "
                    "Use group.select([...], mode='union') to allow per-event fields.")
            parts = [mem[field] for mem in self._members]
            dtype = self._p._raw.EVT_LOG_DATA.dtype
            return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=dtype)
        # list/tuple of fields -> horizontally stack each member's 2-D slice, then concatenate rows
        req = _ensure_tuple(key)
        inter = set(inter_fields)
        missing = [f for f in req if f not in inter]
        if missing:
            raise ValueError(
                f"Fields {missing} are not common to all events. "
                "Use group.select(..., mode='union') for heterogeneous fields.")
        member_blocks = [mem[req] for mem in self._members]
        dtype = self._p._raw.EVT_LOG_DATA.dtype
        if not member_blocks:
            return np.empty((0, len(req)), dtype=dtype)
        return np.concatenate(member_blocks, axis=0)

    def select(
        self,
        fields: Union[str, Sequence[str]],
        *,
        mode: str = "intersection",
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """Explicit intersection/union selection. Returns 1-D for a single field."""
        if mode not in {"intersection", "union"}:
            raise ValueError("mode must be 'intersection' or 'union'")
        is_scalar = isinstance(fields, str)
        req = _ensure_tuple(fields)
        if mode == "intersection":
            available = set(self.fields(mode="intersection"))
            missing = [f for f in req if f not in available]
            if missing:
                raise ValueError(f"Fields {missing} are not common to all events")
            if is_scalar:
                parts = [mem[req[0]] for mem in self._members]
                base_dtype = self._p._raw.EVT_LOG_DATA.dtype
                return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=base_dtype)
            member_blocks = [mem[req] for mem in self._members]
            base_dtype = self._p._raw.EVT_LOG_DATA.dtype
            if not member_blocks:
                return np.empty((0, len(req)), dtype=base_dtype)
            return np.concatenate(member_blocks, axis=0)

        # union mode
        if not req:
            base_dtype = dtype or self._p._raw.EVT_LOG_DATA.dtype
            return np.empty((0,), dtype=base_dtype) if is_scalar else np.empty((0, 0), dtype=base_dtype)
        base_dtype = self._p._raw.EVT_LOG_DATA.dtype
        fill_dtype = np.asarray(fill_value).dtype
        target_dtype = dtype or np.result_type(base_dtype, fill_dtype)
        col_count = 1 if is_scalar else len(req)
        blocks: List[np.ndarray] = []
        for mem in self._members:
            rows = mem.count
            if rows == 0:
                continue
            block = np.full((rows, col_count), fill_value, dtype=target_dtype)
            for j, field in enumerate(req if not is_scalar else (req[0],)):
                if field in mem.fields:
                    data = mem[field]
                    if is_scalar:
                        block[:, 0] = data.astype(target_dtype, copy=False)
                    else:
                        block[:, j] = data.astype(target_dtype, copy=False)
            blocks.append(block)
        if not blocks:
            empty_shape = (0,) if is_scalar else (0, col_count)
            return np.empty(empty_shape, dtype=target_dtype)
        out = np.concatenate(blocks, axis=0)
        return out[:, 0] if is_scalar else out

    def table(
        self,
        fields: Optional[Union[str, Sequence[str]]] = None,
        *,
        mode: str = "intersection",
        sort_by: Optional[str] = None,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """2-D convenience wrapper over ``select`` with optional sorting."""
        if fields is None:
            req = self.fields(mode=mode)
        else:
            req = _ensure_tuple(fields)
        data = self.select(req if req else tuple(), mode=mode, fill_value=fill_value, dtype=dtype)
        if data.ndim == 1:
            data = data[:, None]
        if sort_by is not None:
            if sort_by not in req:
                raise ValueError(f"sort_by '{sort_by}' not requested in table fields")
            col = req.index(sort_by)
            order = np.argsort(data[:, col], kind="stable")
            data = data[order]
        return data

    # Sorting across members (requires a common key)
    def sort(self, by: str = "t") -> "EventGroupView":
        if by not in self.fields(mode="intersection"):
            raise ValueError(f"Cannot sort by '{by}': not present in all member events")
        # Build a concatenated table with an auxiliary index to recover membership
        blocks = []
        offsets = [0]
        for mem in self._members:
            tbl = mem[[by]]  # column as 2-D
            blocks.append(tbl)
            offsets.append(offsets[-1] + tbl.shape[0])
        cat = np.concatenate(blocks, axis=0) if blocks else np.empty((0, 1), dtype=self._p._raw.EVT_LOG_DATA.dtype)
        order = np.argsort(cat[:, 0], kind="stable")
        # Re-slice each member's indices by the global order
        new_members: List[EventView] = []
        for i, mem in enumerate(self._members):
            lo, hi = offsets[i], offsets[i+1]
            if hi > lo:
                sub = order[(order >= lo) & (order < hi)] - lo
                new_members.append(EventView(self._p, mem.name, _row_idx=mem._row_idx[sub]))
            else:
                new_members.append(mem)
        out = EventGroupView(self._p, self._names)
        out._members = tuple(new_members)  # type: ignore[attr-defined]
        return out
