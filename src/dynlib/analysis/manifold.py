# src/dynlib/analysis/manifold.py
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Literal, TYPE_CHECKING

import numpy as np

from dynlib.runtime.softdeps import softdeps
from dynlib.runtime.workspace import make_runtime_workspace, initialize_lag_runtime_workspace
from dynlib.runtime.fastpath.plans import FixedStridePlan
from dynlib.runtime.fastpath.capability import assess_capability

if TYPE_CHECKING:  # pragma: no cover
    from dynlib.compiler.build import FullModel
    from dynlib.runtime.sim import Sim

_SOFTDEPS = softdeps()
_NUMBA_AVAILABLE = _SOFTDEPS.numba

__all__ = ["ManifoldTraceResult", "trace_manifold_1d_map"]


@dataclass
class ManifoldTraceResult:
    kind: str
    fixed_point: np.ndarray
    branches: tuple[list[np.ndarray], list[np.ndarray]]
    eigenvalue: complex
    eigenvector: np.ndarray
    eig_index: int
    step_mul: int
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def branch_pos(self) -> list[np.ndarray]:
        return self.branches[0]

    @property
    def branch_neg(self) -> list[np.ndarray]:
        return self.branches[1]


def _format_spectrum_report(w: np.ndarray, unit_tol: float) -> str:
    mags = np.abs(w)
    stable = mags < (1.0 - unit_tol)
    unstable = mags > (1.0 + unit_tol)
    near = ~(stable | unstable)

    def pack(mask, label: str, reverse_mag: bool) -> str:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return f"{label}: (none)"
        m = mags[idx]
        order = np.argsort(m)
        if reverse_mag:
            order = order[::-1]
        idx = idx[order]
        parts: list[str] = []
        for r, j in enumerate(idx):
            lam_val = w[j]
            if np.iscomplexobj(lam_val):
                lam_str = f"{lam_val.real:.6g}{lam_val.imag:+.6g}j"
            else:
                lam_str = f"{float(lam_val):.6g}"
            parts.append(
                f"{label}[{r}] idx={int(j)}  lam={lam_str}  |lam|={mags[j]:.6g}"
            )
        return "\n".join(parts)

    s1 = pack(unstable, "unstable", True)
    s2 = pack(stable, "stable", False)
    s3 = pack(near, "near1", False)
    return f"{s1}\n{s2}\n{s3}"


def _select_eig_direction_map(
    J: np.ndarray,
    *,
    kind: str,
    eig_rank: int | None = None,
    unit_tol: float = 1e-10,
    imag_tol: float = 1e-12,
    strict_1d: bool = True,
) -> tuple[complex, np.ndarray, int, int]:
    if kind not in ("stable", "unstable"):
        raise ValueError("kind must be 'stable' or 'unstable'")

    w, V = np.linalg.eig(J)
    mags = np.abs(w)

    stable_mask = mags < (1.0 - unit_tol)
    unstable_mask = mags > (1.0 + unit_tol)

    if kind == "stable":
        idx = np.flatnonzero(stable_mask)
        idx = idx[np.argsort(mags[idx])]
    else:
        idx = np.flatnonzero(unstable_mask)
        idx = idx[np.argsort(mags[idx])[::-1]]

    if eig_rank is None:
        if strict_1d and idx.size != 1:
            report = _format_spectrum_report(w, unit_tol)
            raise ValueError(
                f"Cannot auto-select 1D {kind} direction: count is {idx.size} (needs 1).\n"
                f"Spectrum (ranked):\n{report}"
            )
        if idx.size == 0:
            report = _format_spectrum_report(w, unit_tol)
            raise ValueError(
                f"No {kind} eigenvalues detected (check unit_tol).\nSpectrum:\n{report}"
            )
        i = int(idx[0])
    else:
        if eig_rank < 0 or eig_rank >= idx.size:
            report = _format_spectrum_report(w, unit_tol)
            raise ValueError(
                f"eig_rank={eig_rank} out of range for kind='{kind}' (count={idx.size}).\n"
                f"Spectrum (ranked):\n{report}"
            )
        i = int(idx[eig_rank])

    lam = w[i]
    v = V[:, i]

    if np.max(np.abs(v.imag)) > imag_tol * max(1.0, float(np.max(np.abs(v.real)))):
        report = _format_spectrum_report(w, unit_tol)
        raise ValueError(
            "Selected eigenvector is not numerically real; cannot seed a real 1D branch.\n"
            f"Spectrum (ranked):\n{report}"
        )

    v = np.real(v)
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm == 0.0:
        raise ValueError("Selected eigenvector has zero/invalid norm.")
    v /= nrm

    lam_is_real = abs(lam.imag) < imag_tol
    step_mul = 2 if (kind == "unstable" and lam_is_real and lam.real < 0.0) else 1

    return lam, v, i, step_mul


def _normalize_bounds(bounds: Sequence[Sequence[float]] | np.ndarray, d: int) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float)
    if arr.shape != (d, 2):
        raise ValueError(f"bounds must have shape ({d}, 2)")
    if np.any(~np.isfinite(arr)):
        raise ValueError("bounds must be finite")
    if np.any(arr[:, 1] <= arr[:, 0]):
        raise ValueError("bounds must satisfy max > min for each dimension")
    return arr


def _inside_bounds(P: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(P), axis=1)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    in_box = np.all((P >= lo) & (P <= hi), axis=1)
    return finite & in_box


def _split_contiguous_fast(P: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    if P.shape[0] < 2:
        return []
    if not np.any(mask):
        return []

    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1

    out: list[np.ndarray] = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            out.append(P[s:e])
    return out


# Numba-accelerated subdivision (defined conditionally)
_refine_subdivide_numba = None

if _NUMBA_AVAILABLE:  # pragma: no cover - optional dependency
    try:
        from numba import njit  # type: ignore

        @njit(cache=True)
        def _refine_subdivide_numba_impl(P, hmax, max_points):
            n, d = P.shape
            if n < 2:
                return P

            h2 = hmax * hmax
            out = np.empty((max_points, d), dtype=P.dtype)
            for j in range(d):
                out[0, j] = P[0, j]
            k = 1

            for i in range(n - 1):
                d2 = 0.0
                for j in range(d):
                    diff = P[i + 1, j] - P[i, j]
                    d2 += diff * diff

                if d2 <= h2:
                    if k >= max_points:
                        break
                    for j in range(d):
                        out[k, j] = P[i + 1, j]
                    k += 1
                else:
                    seg_len = np.sqrt(d2)
                    subdiv = int(np.ceil(seg_len / hmax))
                    if subdiv < 1:
                        subdiv = 1
                    inv = 1.0 / subdiv
                    for j in range(1, subdiv + 1):
                        if k >= max_points:
                            break
                        t = j * inv
                        for m in range(d):
                            out[k, m] = P[i, m] + t * (P[i + 1, m] - P[i, m])
                        k += 1
                    if k >= max_points:
                        break

            return out[:k]

        _refine_subdivide_numba = _refine_subdivide_numba_impl
    except ImportError:  # pragma: no cover
        pass


def _refine_subdivide(P: np.ndarray, hmax: float, max_points: int) -> np.ndarray:
    if P.shape[0] < 2:
        return P

    d = np.diff(P, axis=0)
    if np.max(np.sum(d * d, axis=1)) <= (hmax * hmax):
        return P

    if _refine_subdivide_numba is not None:
        return _refine_subdivide_numba(P, float(hmax), int(max_points))

    n, dim = P.shape
    out = np.empty((max_points, dim), dtype=P.dtype)
    out[0] = P[0]
    k = 1
    h2 = hmax * hmax
    for i in range(n - 1):
        diff = P[i + 1] - P[i]
        d2 = float(np.dot(diff, diff))
        if d2 <= h2:
            if k >= max_points:
                break
            out[k] = P[i + 1]
            k += 1
        else:
            seg_len = float(np.sqrt(d2))
            subdiv = int(np.ceil(seg_len / hmax))
            if subdiv < 1:
                subdiv = 1
            inv = 1.0 / subdiv
            for j in range(1, subdiv + 1):
                if k >= max_points:
                    break
                t = j * inv
                out[k] = P[i] + t * diff
                k += 1
            if k >= max_points:
                break
    return out[:k]


def _resolve_model(sim: "Sim") -> "FullModel":
    """Extract FullModel from Sim object."""
    model = sim.model
    if not hasattr(model, "spec"):
        raise TypeError("trace_manifold_1d_map expects a Sim instance with a valid model")
    return model  # type: ignore[return-value]


def _resolve_params(model: "FullModel", params) -> np.ndarray:
    dtype = model.dtype
    n_params = len(model.spec.params)
    base_params = np.asarray(model.spec.param_vals, dtype=dtype)
    if params is None:
        return np.array(base_params, copy=True)
    if isinstance(params, Mapping):
        params_vec = np.array(base_params, copy=True)
        param_index = {name: i for i, name in enumerate(model.spec.params)}
        for key, val in params.items():
            if key not in param_index:
                raise KeyError(f"Unknown param '{key}'.")
            params_vec[param_index[key]] = float(val)
        return params_vec
    params_arr = np.asarray(params, dtype=dtype)
    if params_arr.ndim != 1 or params_arr.shape[0] != n_params:
        raise ValueError(f"params must have shape ({n_params},)")
    return params_arr


def _resolve_fixed_point(model: "FullModel", fp) -> np.ndarray:
    dtype = model.dtype
    n_state = len(model.spec.states)
    base_state = np.asarray(model.spec.state_ic, dtype=dtype)
    if isinstance(fp, Mapping):
        fp_vec = np.array(base_state, copy=True)
        state_index = {name: i for i, name in enumerate(model.spec.states)}
        for key, val in fp.items():
            if key not in state_index:
                raise KeyError(f"Unknown state '{key}'.")
            fp_vec[state_index[key]] = float(val)
        return fp_vec
    fp_arr = np.asarray(fp, dtype=dtype)
    if fp_arr.ndim != 1 or fp_arr.shape[0] != n_state:
        raise ValueError(f"fp must have shape ({n_state},)")
    return fp_arr


def trace_manifold_1d_map(
    sim: "Sim",
    *,
    fp: Mapping[str, float] | Sequence[float] | np.ndarray,
    kind: Literal["stable", "unstable"] = "stable",
    params: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    bounds: Sequence[Sequence[float]] | np.ndarray | None = None,
    clip_margin: float = 0.25,
    seed_delta: float = 1e-7,
    steps: int = 60,
    hmax: float = 2e-3,
    max_points_per_segment: int = 20000,
    max_segments: int = 200,
    eig_rank: int | None = None,
    strict_1d: bool = True,
    eig_unit_tol: float = 1e-10,
    eig_imag_tol: float = 1e-12,
    jac: Literal["auto", "fd", "analytic"] = "auto",
    fd_eps: float = 1e-6,
    fp_check_tol: float | None = 1e-6,
    t: float | None = None,
) -> ManifoldTraceResult:
    """
    Trace a 1D stable or unstable manifold for a discrete-time map.

    The map must be autonomous and the target stable/unstable subspace must be 1D.
    For stable manifolds, an analytic inverse map is required.

    Parameters
    ----------
    sim : Sim
        Simulation object containing the map model. For best performance, build
        the model with ``jit=True``. If numba is unavailable or the model was
        built without JIT, a warning is issued and execution falls back to
        slower Python loops.
    """
    if kind not in ("stable", "unstable"):
        raise ValueError("kind must be 'stable' or 'unstable'")
    if bounds is None:
        raise ValueError("bounds is required")
    if clip_margin < 0.0:
        raise ValueError("clip_margin must be non-negative")
    if seed_delta <= 0.0:
        raise ValueError("seed_delta must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if hmax <= 0.0:
        raise ValueError("hmax must be positive")
    if max_points_per_segment < 2:
        raise ValueError("max_points_per_segment must be >= 2")
    if max_segments < 1:
        raise ValueError("max_segments must be >= 1")
    if eig_unit_tol < 0.0:
        raise ValueError("eig_unit_tol must be non-negative")
    if eig_imag_tol < 0.0:
        raise ValueError("eig_imag_tol must be non-negative")
    if jac not in ("auto", "fd", "analytic"):
        raise ValueError("jac must be 'auto', 'fd', or 'analytic'")
    if fd_eps <= 0.0:
        raise ValueError("fd_eps must be positive")
    if fp_check_tol is not None and fp_check_tol < 0.0:
        raise ValueError("fp_check_tol must be non-negative or None")

    model = _resolve_model(sim)
    if model.spec.kind != "map":
        raise ValueError("trace_manifold_1d_map requires model.spec.kind == 'map'")
    if not np.issubdtype(model.dtype, np.floating):
        raise ValueError("trace_manifold_1d_map requires a floating-point model dtype.")
    if model.rhs is None:
        raise ValueError("Map RHS is not available on the model.")

    if kind == "stable" and model.inv_rhs is None:
        raise ValueError("Stable manifold requires an inverse map (model.inv_rhs).")

    n_state = len(model.spec.states)
    n_params = len(model.spec.params)
    bounds_arr = _normalize_bounds(bounds, n_state)
    params_vec = _resolve_params(model, params)
    fp_vec = _resolve_fixed_point(model, fp)

    t_eval = float(model.spec.sim.t0 if t is None else t)
    map_fn = model.rhs
    inv_map_fn = model.inv_rhs
    jac_fn = model.jacobian

    # ---------------------------------------------------------------------------
    # Setup workspace for map evaluation
    # ---------------------------------------------------------------------------
    stop_phase_mask = 0
    if model.spec.sim.stop is not None:
        phase = model.spec.sim.stop.phase
        if phase in ("pre", "both"):
            stop_phase_mask |= 1
        if phase in ("post", "both"):
            stop_phase_mask |= 2

    runtime_ws = make_runtime_workspace(
        lag_state_info=model.lag_state_info or (),
        dtype=model.dtype,
        n_aux=len(model.spec.aux or {}),
        stop_enabled=stop_phase_mask != 0,
        stop_phase_mask=stop_phase_mask,
    )
    lag_state_info = model.lag_state_info or ()
    needs_prep = bool(lag_state_info) or runtime_ws.aux_values.size > 0 or runtime_ws.stop_flag.size > 0

    def _prep_ws(y_vec: np.ndarray) -> None:
        if lag_state_info:
            initialize_lag_runtime_workspace(
                runtime_ws,
                lag_state_info=lag_state_info,
                y_curr=y_vec,
            )
        if runtime_ws.aux_values.size:
            runtime_ws.aux_values[:] = 0
        if runtime_ws.stop_flag.size:
            runtime_ws.stop_flag[0] = 0

    # ---------------------------------------------------------------------------
    # Determine execution strategy
    # ---------------------------------------------------------------------------
    def _is_numba_dispatcher(fn: object) -> bool:
        return hasattr(fn, "py_func") and hasattr(fn, "signatures")

    # Strategy 1: Direct JIT batch loop (fastest, but bypasses events/stops)
    # Use when no workspace prep needed and map is JIT-compiled
    use_direct_jit = (
        not needs_prep
        and _NUMBA_AVAILABLE
        and _is_numba_dispatcher(map_fn)
    )

    # Strategy 2: Fastpath runner with preallocated workspace bundle
    # Use when events/stops matter but fastpath is available
    use_fastpath_bundle = False
    bundle = None

    if not use_direct_jit:
        # Check if fastpath is available
        plan = FixedStridePlan(stride=1)
        support = assess_capability(
            sim, plan=plan, record_vars=None, dt=1.0, transient=0.0, adaptive=False
        )
        if support.ok:
            use_fastpath_bundle = True
            # Import and create the workspace bundle (preallocated, reusable)
            from dynlib.runtime.fastpath.executor import _WorkspaceBundle, _RunContext

            ctx = _RunContext(
                t0=t_eval,
                t_end=t_eval + 1.0,
                target_steps=1,  # Single map iteration
                dt=1.0,
                max_steps=1,
                transient=0.0,
                record_interval=0,  # No recording needed
            )
            state_rec_indices = np.array([], dtype=np.int32)
            aux_rec_indices = np.array([], dtype=np.int32)

            stepper_config = None
            if model.stepper_spec is not None:
                default_cfg = model.stepper_spec.default_config(model.spec)
                stepper_config = model.stepper_spec.pack_config(default_cfg)

            bundle = _WorkspaceBundle(
                model=model,
                plan=plan,
                ctx=ctx,
                state_rec_indices=state_rec_indices,
                aux_rec_indices=aux_rec_indices,
                state_names=[],
                aux_names=[],
                stepper_config=stepper_config,
                analysis=None,
            )
        else:
            # Neither direct JIT nor fastpath available - warn about fallback
            reason = f" ({support.reason})" if support.reason else ""
            warnings.warn(
                f"trace_manifold_1d_map: fast execution path unavailable{reason}. "
                "Falling back to Python loops. For best performance, build the model "
                "with jit=True and ensure numba is installed.",
                stacklevel=2,
            )

    # ---------------------------------------------------------------------------
    # Build batch map functions based on strategy
    # ---------------------------------------------------------------------------

    # Direct JIT batch mapper (for simple cases)
    batch_map_jit = None
    batch_inv_map_jit = None

    if use_direct_jit:
        try:
            from numba import njit  # type: ignore

            @njit(cache=True)
            def _batch_map_jit_impl(P, t_eval, params_vec, runtime_ws):
                Q = np.empty_like(P)
                for i in range(P.shape[0]):
                    map_fn(t_eval, P[i], Q[i], params_vec, runtime_ws)
                return Q

            batch_map_jit = _batch_map_jit_impl

            if inv_map_fn is not None and _is_numba_dispatcher(inv_map_fn):
                @njit(cache=True)
                def _batch_inv_map_jit_impl(P, t_eval, params_vec, runtime_ws):
                    Q = np.empty_like(P)
                    for i in range(P.shape[0]):
                        inv_map_fn(t_eval, P[i], Q[i], params_vec, runtime_ws)
                    return Q

                batch_inv_map_jit = _batch_inv_map_jit_impl
        except Exception:  # pragma: no cover
            use_direct_jit = False

    # ---------------------------------------------------------------------------
    # Forward map batch evaluation
    # ---------------------------------------------------------------------------
    def _map_points_python(P: np.ndarray) -> np.ndarray:
        """Apply forward map using Python loop (slowest fallback)."""
        Q = np.empty_like(P)
        for i in range(P.shape[0]):
            if needs_prep:
                _prep_ws(P[i])
            map_fn(t_eval, P[i], Q[i], params_vec, runtime_ws)
        return Q

    def _map_points_bundle(P: np.ndarray) -> np.ndarray:
        """Apply forward map using preallocated fastpath bundle."""
        Q = np.empty_like(P)
        for i in range(P.shape[0]):
            bundle.reset(P[i])
            bundle.runner(
                float(bundle.ctx.t0),
                float(bundle.ctx.target_steps),
                float(bundle.ctx.dt),
                int(bundle.ctx.max_steps),
                int(bundle.n_state),
                int(bundle.rec_every),
                bundle.y_curr,
                bundle.y_prev,
                params_vec,
                bundle.runtime_ws,
                bundle.stepper_ws,
                bundle.stepper_config,
                bundle.y_prop,
                bundle.t_prop,
                bundle.dt_next,
                bundle.err_est,
                bundle.T,
                bundle.Y,
                bundle.AUX,
                bundle.STEP,
                bundle.FLAGS,
                bundle.EVT_CODE,
                bundle.EVT_INDEX,
                bundle.EVT_LOG_DATA,
                bundle.evt_log_scratch,
                bundle.analysis_ws,
                bundle.analysis_out,
                bundle.analysis_trace,
                bundle.analysis_trace_count,
                int(bundle.analysis_trace_cap),
                int(bundle.analysis_trace_stride),
                int(bundle.variational_step_enabled),
                bundle.variational_step_fn,
                bundle.i_start,
                bundle.step_start,
                int(bundle.cap_rec),
                int(bundle.cap_evt),
                bundle.user_break_flag,
                bundle.status_out,
                bundle.hint_out,
                bundle.i_out,
                bundle.step_out,
                bundle.t_out,
                bundle.model.stepper,
                bundle.model.rhs,
                bundle.model.events_pre,
                bundle.model.events_post,
                bundle.model.update_aux,
                bundle.state_rec_indices,
                bundle.aux_rec_indices,
                bundle.n_rec_states,
                bundle.n_rec_aux,
            )
            Q[i] = bundle.y_curr
        return Q

    def _map_points(P: np.ndarray) -> np.ndarray:
        """Apply forward map to batch of points."""
        if batch_map_jit is not None:
            return batch_map_jit(P, t_eval, params_vec, runtime_ws)
        if use_fastpath_bundle and bundle is not None:
            return _map_points_bundle(P)
        return _map_points_python(P)

    # ---------------------------------------------------------------------------
    # Inverse map batch evaluation
    # ---------------------------------------------------------------------------
    def _inv_map_points(P: np.ndarray) -> np.ndarray:
        """Apply inverse map to batch of points."""
        if inv_map_fn is None:
            raise ValueError("Inverse map requested but model.inv_rhs is missing.")
        if batch_inv_map_jit is not None:
            return batch_inv_map_jit(P, t_eval, params_vec, runtime_ws)
        # For inverse maps, use Python loop (no fastpath support for custom rhs)
        Q = np.empty_like(P)
        for i in range(P.shape[0]):
            if needs_prep:
                _prep_ws(P[i])
            inv_map_fn(t_eval, P[i], Q[i], params_vec, runtime_ws)
        return Q

    # ---------------------------------------------------------------------------
    # Single point evaluation (for Jacobian and fixed-point check)
    # ---------------------------------------------------------------------------
    def _map_point(y_vec: np.ndarray, out: np.ndarray) -> None:
        if needs_prep:
            _prep_ws(y_vec)
        map_fn(t_eval, y_vec, out, params_vec, runtime_ws)

    if fp_check_tol is not None:
        fp_img = np.empty((n_state,), dtype=model.dtype)
        _map_point(fp_vec, fp_img)
        diff = fp_img - fp_vec
        err = float(np.linalg.norm(diff))
        if not np.isfinite(err) or err > fp_check_tol:
            raise ValueError(
                f"Provided fp is not a fixed point; |F(fp)-fp|={err:.6g} exceeds tol={fp_check_tol}."
            )

    def _jacobian_at(x: np.ndarray) -> np.ndarray:
        if jac == "fd" or (jac == "auto" and jac_fn is None):
            fx = np.empty((n_state,), dtype=model.dtype)
            _map_point(x, fx)
            J = np.zeros((n_state, n_state), dtype=float)
            for j in range(n_state):
                step = fd_eps * (1.0 + abs(float(x[j])))
                if step == 0.0:
                    step = fd_eps
                x_step = np.array(x, copy=True)
                x_step[j] += step
                f_step = np.empty((n_state,), dtype=model.dtype)
                _map_point(x_step, f_step)
                J[:, j] = (f_step - fx) / step
            return J

        if jac == "analytic":
            if jac_fn is None:
                raise ValueError("jac='analytic' requires a model Jacobian.")
        if jac_fn is None:
            raise ValueError("Jacobian is not available (jac='auto' found none).")

        jac_out = np.zeros((n_state, n_state), dtype=model.dtype)
        _prep_ws(x)
        jac_fn(t_eval, x, params_vec, jac_out, runtime_ws)
        return np.array(jac_out, copy=True)

    J = _jacobian_at(fp_vec)
    lam, v, eig_index, step_mul = _select_eig_direction_map(
        J,
        kind=kind,
        eig_rank=eig_rank,
        unit_tol=eig_unit_tol,
        imag_tol=eig_imag_tol,
        strict_1d=strict_1d,
    )

    use_inverse = kind == "stable"
    seed = np.asarray(fp_vec, dtype=model.dtype)
    v = np.asarray(v, dtype=model.dtype)

    segs_pos = [np.vstack([seed, seed + seed_delta * v])]
    segs_neg = [np.vstack([seed, seed - seed_delta * v])]

    extent = bounds_arr[:, 1] - bounds_arr[:, 0]
    clip_pad = extent * clip_margin
    clip_bounds = np.stack((bounds_arr[:, 0] - clip_pad, bounds_arr[:, 1] + clip_pad), axis=1)

    def _step_segments(segs: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for P in segs:
            Q = _inv_map_points(P) if use_inverse else _map_points(P)

            if step_mul == 2:
                Q = _inv_map_points(Q) if use_inverse else _map_points(Q)

            mask = _inside_bounds(Q, clip_bounds)
            parts = _split_contiguous_fast(Q, mask)
            for R in parts:
                R2 = _refine_subdivide(R, hmax=hmax, max_points=max_points_per_segment)
                if R2.shape[0] >= 2:
                    out.append(R2)
                    if len(out) >= max_segments:
                        return out[:max_segments]
        return out[:max_segments]

    for _ in range(steps):
        segs_pos = _step_segments(segs_pos)
        segs_neg = _step_segments(segs_neg)
        if not segs_pos and not segs_neg:
            break

    def _final_clip(segs: list[np.ndarray]) -> list[np.ndarray]:
        clipped: list[np.ndarray] = []
        for P in segs:
            mask = _inside_bounds(P, bounds_arr)
            clipped.extend(_split_contiguous_fast(P, mask))
        return [s.copy() for s in clipped]

    return ManifoldTraceResult(
        kind=kind,
        fixed_point=np.array(fp_vec, copy=True),
        branches=(_final_clip(segs_pos), _final_clip(segs_neg)),
        eigenvalue=lam,
        eigenvector=np.array(v, copy=True),
        eig_index=eig_index,
        step_mul=step_mul,
        meta={
            "bounds": bounds_arr,
            "clip_margin": float(clip_margin),
            "seed_delta": float(seed_delta),
            "steps": int(steps),
            "hmax": float(hmax),
            "max_points_per_segment": int(max_points_per_segment),
            "max_segments": int(max_segments),
            "eig_rank": eig_rank,
            "strict_1d": bool(strict_1d),
            "eig_unit_tol": float(eig_unit_tol),
            "eig_imag_tol": float(eig_imag_tol),
            "jac": str(jac),
            "fd_eps": float(fd_eps),
            "t_eval": float(t_eval),
            "uses_inverse": bool(use_inverse),
        },
    )
