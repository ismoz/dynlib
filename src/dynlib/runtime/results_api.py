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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import numpy as np

from dynlib.runtime.results import Results as _RawResults
from dynlib.dsl.spec import ModelSpec, EventSpec

__all__ = [
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


# ------------------------------ main wrapper ---------------------------------

class ResultsView:
    """User-facing, name-aware view on top of the low-level :class:`Results`.

    Construct with the raw results and a frozen :class:`ModelSpec` to enable
    named access. No mutation, no copies for trajectory access.
    """

    # ---- construction ----
    def __init__(
        self,
        raw: _RawResults,
        spec: ModelSpec,
        *,
        event_code_map: Optional[Mapping[str, int]] = None,
    ) -> None:
        self._raw = raw
        self._spec = spec

        # State names in canonical order
        self._state_names: Tuple[str, ...] = tuple(spec.states)
        self._state_index: Dict[str, int] = {name: i for i, name in enumerate(self._state_names)}

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

    def __getitem__(self, key: Union[str, Sequence[str]]) -> np.ndarray:
        """Access recorded state series by name.

        - ``res["v"]`` -> 1-D view (length ``n``)
        - ``res[["v","w"]]`` -> 2-D array (shape ``(n, k)``) stacked in request
          order. This requires fancy indexing and therefore returns a compact
          copy, unlike the single-state view above.
        """
        n = int(self._raw.n)
        Y = self._raw.Y_view  # shape (n_state, n)
        if isinstance(key, str):
            if key not in self._state_index:
                raise _friendly_key_error("state", key, self._state_names)
            return Y[self._state_index[key], :n]
        # sequence of names
        names = _ensure_tuple(key)
        idxs: List[int] = []
        for nm in names:
            if nm not in self._state_index:
                raise _friendly_key_error("state", nm, self._state_names)
            idxs.append(self._state_index[nm])
        # Stack chosen states as columns (transpose the row slice).
        # Fancy indexing allocates a compact copy (documented in the docstring).
        return Y[np.array(idxs, dtype=int), :n].T

    # ---- low-level passthrough ----
    @property
    def ok(self) -> bool:
        return bool(self._raw.ok)

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
