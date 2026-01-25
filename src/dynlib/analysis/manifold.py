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

__all__ = ["ManifoldTraceResult", "trace_manifold_1d_map", "trace_manifold_1d_ode"]


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


# =============================================================================
# ODE Manifold Tracing
# =============================================================================


def _format_spectrum_report_ode(w: np.ndarray, real_tol: float) -> str:
    """Format eigenvalue spectrum report for ODE systems (Re(λ) classification)."""
    re = w.real
    stable = re < -real_tol
    unstable = re > +real_tol
    center = ~(stable | unstable)

    def pack(mask, label: str, sort_descending: bool) -> str:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return f"{label}: (none)"
        r = re[idx]
        order = np.argsort(r)
        if sort_descending:
            order = order[::-1]
        idx = idx[order]
        parts: list[str] = []
        for rank, j in enumerate(idx):
            lam_val = w[j]
            if np.iscomplexobj(lam_val) and abs(lam_val.imag) > 1e-14:
                lam_str = f"{lam_val.real:.6g}{lam_val.imag:+.6g}j"
            else:
                lam_str = f"{float(lam_val.real):.6g}"
            parts.append(
                f"{label}[{rank}] idx={int(j)}  λ={lam_str}  Re(λ)={re[j]:.6g}"
            )
        return "\n".join(parts)

    s1 = pack(unstable, "unstable", True)  # Most positive first
    s2 = pack(stable, "stable", False)     # Most negative first
    s3 = pack(center, "center", False)
    return f"{s1}\n{s2}\n{s3}"


def _select_eig_direction_ode(
    J: np.ndarray,
    *,
    kind: str,
    eig_rank: int | None = None,
    real_tol: float = 1e-10,
    imag_tol: float = 1e-12,
    strict_1d: bool = True,
) -> tuple[complex, np.ndarray, int]:
    """
    Select a real unit eigenvector for the 1D stable/unstable direction at an ODE equilibrium.

    Classification is by sign of Re(λ):
      - "stable": Re(λ) < -real_tol (trajectories contract toward equilibrium)
      - "unstable": Re(λ) > +real_tol (trajectories expand away from equilibrium)

    Parameters
    ----------
    J : ndarray
        Jacobian matrix at the equilibrium.
    kind : str
        Either "stable" or "unstable".
    eig_rank : int or None
        If None, auto-select (requires exactly one eigenvalue of the requested kind).
        Otherwise, select the eig_rank-th eigenvalue (0-indexed) sorted by |Re(λ)|.
    real_tol : float
        Tolerance for classifying eigenvalues as stable/unstable.
    imag_tol : float
        Tolerance for considering an eigenvector as numerically real.
    strict_1d : bool
        If True, raise error when auto-selecting and count != 1.

    Returns
    -------
    lam : complex
        The selected eigenvalue.
    v : ndarray
        Unit eigenvector (real).
    eig_index : int
        Index of the selected eigenvalue in the original spectrum.
    """
    if kind not in ("stable", "unstable"):
        raise ValueError("kind must be 'stable' or 'unstable'")

    w, V = np.linalg.eig(J)
    re = w.real

    if kind == "stable":
        # Re(λ) < -real_tol, sorted from most negative
        idx = np.flatnonzero(re < -real_tol)
        idx = idx[np.argsort(re[idx])]  # most negative first
    else:
        # Re(λ) > +real_tol, sorted from most positive
        idx = np.flatnonzero(re > +real_tol)
        idx = idx[np.argsort(re[idx])[::-1]]  # most positive first

    if eig_rank is None:
        if strict_1d and idx.size != 1:
            report = _format_spectrum_report_ode(w, real_tol)
            raise ValueError(
                f"Cannot auto-select 1D {kind} direction: count is {idx.size} (needs 1).\n"
                f"Spectrum (ranked):\n{report}"
            )
        if idx.size == 0:
            report = _format_spectrum_report_ode(w, real_tol)
            raise ValueError(
                f"No {kind} eigenvalues detected (check real_tol).\nSpectrum:\n{report}"
            )
        i = int(idx[0])
    else:
        if eig_rank < 0 or eig_rank >= idx.size:
            report = _format_spectrum_report_ode(w, real_tol)
            raise ValueError(
                f"eig_rank={eig_rank} out of range for kind='{kind}' (count={idx.size}).\n"
                f"Spectrum (ranked):\n{report}"
            )
        i = int(idx[eig_rank])

    lam = w[i]
    v = V[:, i]

    # Enforce a real direction (for a real 1D manifold branch)
    if np.max(np.abs(v.imag)) > imag_tol * max(1.0, float(np.max(np.abs(v.real)))):
        report = _format_spectrum_report_ode(w, real_tol)
        raise ValueError(
            "Selected eigenvector is not numerically real; cannot seed a real 1D branch.\n"
            f"Spectrum (ranked):\n{report}"
        )

    v = np.real(v)
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm == 0.0:
        raise ValueError("Selected eigenvector has zero/invalid norm.")
    v /= nrm

    return lam, v, i


# Numba-accelerated RK4 branch tracing for ODEs
_trace_branch_rk4_numba = None

if _NUMBA_AVAILABLE:  # pragma: no cover - optional dependency
    try:
        from numba import njit  # type: ignore

        @njit(cache=True)
        def _rk4_step_numba(t, z, dt, k1, k2, k3, k4, z_stage, rhs_fn, params, runtime_ws):
            """Single RK4 step with preallocated work arrays (numba version)."""
            n = z.size

            # k1 = f(t, z)
            rhs_fn(t, z, k1, params, runtime_ws)

            # k2 = f(t + dt/2, z + dt/2*k1)
            for i in range(n):
                z_stage[i] = z[i] + 0.5 * dt * k1[i]
            rhs_fn(t + 0.5 * dt, z_stage, k2, params, runtime_ws)

            # k3 = f(t + dt/2, z + dt/2*k2)
            for i in range(n):
                z_stage[i] = z[i] + 0.5 * dt * k2[i]
            rhs_fn(t + 0.5 * dt, z_stage, k3, params, runtime_ws)

            # k4 = f(t + dt, z + dt*k3)
            for i in range(n):
                z_stage[i] = z[i] + dt * k3[i]
            rhs_fn(t + dt, z_stage, k4, params, runtime_ws)

            # Combine: z = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            for i in range(n):
                z[i] = z[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

        @njit(cache=True)
        def _trace_branch_rk4_numba_impl(
            z0, t0, dt, max_steps, bounds_lo, bounds_hi,
            rhs_fn, params, runtime_ws, update_aux, do_aux_pre, do_aux_post,
            out_buf, k1, k2, k3, k4, z_stage
        ):
            """
            Trace single ODE branch using RK4. Returns number of valid points.
            """
            n = z0.size
            z = np.empty(n, dtype=z0.dtype)
            for i in range(n):
                z[i] = z0[i]
                out_buf[0, i] = z0[i]

            m = 1
            t = t0

            stop_mask = 0
            if runtime_ws.stop_phase_mask.shape[0] > 0:
                stop_mask = runtime_ws.stop_phase_mask[0]

            if do_aux_pre:
                update_aux(t, z, params, runtime_ws.aux_values, runtime_ws)
            if (stop_mask & 1) != 0 and runtime_ws.stop_flag.shape[0] > 0:
                if runtime_ws.stop_flag[0] != 0:
                    return m

            for _ in range(max_steps):
                _rk4_step_numba(t, z, dt, k1, k2, k3, k4, z_stage, rhs_fn, params, runtime_ws)
                t = t + dt

                # Check bounds and validity
                valid = True
                for i in range(n):
                    if not np.isfinite(z[i]):
                        valid = False
                        break
                    if z[i] < bounds_lo[i] or z[i] > bounds_hi[i]:
                        valid = False
                        break

                if not valid:
                    break

                if do_aux_post:
                    update_aux(t, z, params, runtime_ws.aux_values, runtime_ws)

                lag_info = runtime_ws.lag_info
                if lag_info.shape[0] > 0:
                    lag_ring = runtime_ws.lag_ring
                    lag_head = runtime_ws.lag_head
                    for j in range(lag_info.shape[0]):
                        state_idx = lag_info[j, 0]
                        depth = lag_info[j, 1]
                        offset = lag_info[j, 2]
                        head = int(lag_head[j]) + 1
                        if head >= depth:
                            head = 0
                        lag_head[j] = head
                        lag_ring[offset + head] = z[state_idx]

                for i in range(n):
                    out_buf[m, i] = z[i]
                m += 1

                if m >= out_buf.shape[0]:
                    break

                if (stop_mask & 2) != 0 and runtime_ws.stop_flag.shape[0] > 0:
                    if runtime_ws.stop_flag[0] != 0:
                        break

            return m

        _trace_branch_rk4_numba = _trace_branch_rk4_numba_impl

    except ImportError:  # pragma: no cover
        pass


def _rk4_step_python(t, z, dt, k1, k2, k3, k4, z_stage, rhs_fn, params, runtime_ws):
    """Single RK4 step with preallocated work arrays (pure Python version)."""
    n = z.size

    # k1 = f(t, z)
    rhs_fn(t, z, k1, params, runtime_ws)

    # k2 = f(t + dt/2, z + dt/2*k1)
    for i in range(n):
        z_stage[i] = z[i] + 0.5 * dt * k1[i]
    rhs_fn(t + 0.5 * dt, z_stage, k2, params, runtime_ws)

    # k3 = f(t + dt/2, z + dt/2*k2)
    for i in range(n):
        z_stage[i] = z[i] + 0.5 * dt * k2[i]
    rhs_fn(t + 0.5 * dt, z_stage, k3, params, runtime_ws)

    # k4 = f(t + dt, z + dt*k3)
    for i in range(n):
        z_stage[i] = z[i] + dt * k3[i]
    rhs_fn(t + dt, z_stage, k4, params, runtime_ws)

    # Combine: z = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    for i in range(n):
        z[i] = z[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


def _trace_branch_rk4_python(
    z0, t0, dt, max_steps, bounds_lo, bounds_hi,
    rhs_fn, params, runtime_ws, update_aux, do_aux_pre, do_aux_post,
    out_buf, k1, k2, k3, k4, z_stage
):
    """
    Trace single ODE branch using RK4 (pure Python version). Returns number of valid points.
    """
    n = z0.size
    z = np.array(z0, copy=True)
    out_buf[0] = z0

    m = 1
    t = t0

    stop_mask = int(runtime_ws.stop_phase_mask[0]) if runtime_ws.stop_phase_mask.size else 0
    if do_aux_pre:
        update_aux(t, z, params, runtime_ws.aux_values, runtime_ws)
    if (stop_mask & 1) != 0 and runtime_ws.stop_flag.size:
        if runtime_ws.stop_flag[0] != 0:
            return m

    for _ in range(max_steps):
        _rk4_step_python(t, z, dt, k1, k2, k3, k4, z_stage, rhs_fn, params, runtime_ws)
        t = t + dt

        # Check bounds and validity
        if not np.all(np.isfinite(z)):
            break
        if np.any(z < bounds_lo) or np.any(z > bounds_hi):
            break

        if do_aux_post:
            update_aux(t, z, params, runtime_ws.aux_values, runtime_ws)

        lag_info = runtime_ws.lag_info
        if lag_info.size:
            lag_ring = runtime_ws.lag_ring
            lag_head = runtime_ws.lag_head
            for j in range(lag_info.shape[0]):
                state_idx = lag_info[j, 0]
                depth = lag_info[j, 1]
                offset = lag_info[j, 2]
                head = int(lag_head[j]) + 1
                if head >= depth:
                    head = 0
                lag_head[j] = head
                lag_ring[offset + head] = z[state_idx]

        out_buf[m] = z
        m += 1

        if m >= out_buf.shape[0]:
            break

        if (stop_mask & 2) != 0 and runtime_ws.stop_flag.size:
            if runtime_ws.stop_flag[0] != 0:
                break

    return m


def _resample_by_arclength(P: np.ndarray, h: float) -> np.ndarray:
    """
    Resample polyline P to approximately uniform spacing h using linear interpolation
    along cumulative arc-length.
    """
    if P.shape[0] < 2:
        return P

    d = np.diff(P, axis=0)
    seg = np.sqrt(np.sum(d * d, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if not np.isfinite(L) or L == 0.0:
        return P

    n = int(np.floor(L / h)) + 1
    if n < 2:
        return P

    t = np.linspace(0.0, L, n)
    out = np.empty((n, P.shape[1]), dtype=P.dtype)
    for j in range(P.shape[1]):
        out[:, j] = np.interp(t, s, P[:, j])
    return out


def trace_manifold_1d_ode(
    sim: "Sim",
    *,
    fp: Mapping[str, float] | Sequence[float] | np.ndarray,
    kind: Literal["stable", "unstable"] = "stable",
    params: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    bounds: Sequence[Sequence[float]] | np.ndarray,
    clip_margin: float = 0.25,
    seed_delta: float = 1e-6,
    dt: float = 0.01,
    max_time: float = 100.0,
    resample_h: float | None = 0.01,
    max_points: int = 50000,
    eig_rank: int | None = None,
    strict_1d: bool = True,
    eig_real_tol: float = 1e-10,
    eig_imag_tol: float = 1e-12,
    jac: Literal["auto", "fd", "analytic"] = "auto",
    fd_eps: float = 1e-6,
    fp_check_tol: float | None = 1e-6,
    t: float | None = None,
) -> ManifoldTraceResult:
    """
    Trace a 1D stable or unstable manifold for an ODE system.

    Uses an internal RK4 integrator to trace manifold branches forward (unstable)
    or backward (stable) in time from an equilibrium point.

    Parameters
    ----------
    sim : Sim
        Simulation object containing the ODE model. For best performance, build
        the model with ``jit=True``. If numba is unavailable or the model was
        built without JIT, a warning is issued and execution falls back to
        slower Python loops.
    fp : dict or array-like
        The equilibrium (fixed point) coordinates. Can be a dict mapping state
        names to values, or a sequence/array of values in state declaration order.
    kind : {"stable", "unstable"}
        Which manifold to trace. Stable manifolds are traced backward in time,
        unstable manifolds are traced forward.
    params : dict or array-like, optional
        Parameter overrides. If None, uses model's current parameters.
    bounds : array-like of shape (n_state, 2)
        Bounding box ``[[x_min, x_max], [y_min, y_max], ...]`` for each state.
        Integration terminates when trajectory leaves bounds.
    clip_margin : float
        Fractional margin added to bounds during integration (clipped to exact
        bounds in final output).
    seed_delta : float
        Distance from equilibrium to seed initial conditions along eigenvector.
    dt : float
        Integration step size for the internal RK4 stepper.
    max_time : float
        Maximum integration time per branch.
    resample_h : float or None
        If not None, resample output curves to approximately uniform arc-length
        spacing. Helps produce cleaner curves for plotting.
    max_points : int
        Maximum number of points to store per branch.
    eig_rank : int or None
        If None, auto-select the unique stable/unstable eigenvalue (requires
        exactly one). Otherwise, select the eig_rank-th eigenvalue (0-indexed)
        sorted by |Re(λ)|.
    strict_1d : bool
        If True and eig_rank is None, raise error when the selected subspace
        is not exactly 1-dimensional.
    eig_real_tol : float
        Tolerance for classifying eigenvalues: |Re(λ)| must exceed this to be
        considered stable or unstable.
    eig_imag_tol : float
        Tolerance for considering an eigenvector as numerically real.
    jac : {"auto", "fd", "analytic"}
        How to compute the Jacobian at the equilibrium:
        - "auto": use model's analytic Jacobian if available, else finite differences
        - "fd": always use finite differences
        - "analytic": require model's analytic Jacobian
    fd_eps : float
        Step size for finite-difference Jacobian approximation.
    fp_check_tol : float or None
        If not None, verify that ``|f(fp)| < fp_check_tol`` (i.e., fp is actually
        an equilibrium). Set to None to skip this check.
    t : float or None
        Time value for RHS evaluation. If None, uses model's t0.

    Returns
    -------
    ManifoldTraceResult
        Contains the traced branches and metadata.

    Notes
    -----
    This function uses an internal RK4 integrator regardless of the stepper
    configured on the Sim object. RK4 is well-suited for manifold tracing due
    to its balance of accuracy and stability. A warning is issued if the Sim
    uses a different stepper, for informational purposes.

    For stable manifolds, integration proceeds backward in time (negative dt).
    For unstable manifolds, integration proceeds forward in time (positive dt).
    """
    # ---------------------------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------------------------
    if kind not in ("stable", "unstable"):
        raise ValueError("kind must be 'stable' or 'unstable'")
    if bounds is None:
        raise ValueError("bounds is required")
    if clip_margin < 0.0:
        raise ValueError("clip_margin must be non-negative")
    if seed_delta <= 0.0:
        raise ValueError("seed_delta must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if max_time <= 0.0:
        raise ValueError("max_time must be positive")
    if max_points < 2:
        raise ValueError("max_points must be >= 2")
    if eig_real_tol < 0.0:
        raise ValueError("eig_real_tol must be non-negative")
    if eig_imag_tol < 0.0:
        raise ValueError("eig_imag_tol must be non-negative")
    if jac not in ("auto", "fd", "analytic"):
        raise ValueError("jac must be 'auto', 'fd', or 'analytic'")
    if fd_eps <= 0.0:
        raise ValueError("fd_eps must be positive")
    if fp_check_tol is not None and fp_check_tol < 0.0:
        raise ValueError("fp_check_tol must be non-negative or None")
    if resample_h is not None and resample_h <= 0.0:
        raise ValueError("resample_h must be positive or None")

    # ---------------------------------------------------------------------------
    # Extract model
    # ---------------------------------------------------------------------------
    model = _resolve_model(sim)
    if model.spec.kind != "ode":
        raise ValueError("trace_manifold_1d_ode requires model.spec.kind == 'ode'")
    if not np.issubdtype(model.dtype, np.floating):
        raise ValueError("trace_manifold_1d_ode requires a floating-point model dtype.")
    if model.rhs is None:
        raise ValueError("ODE RHS is not available on the model.")

    # Warn if stepper is not RK4
    stepper_name = model.stepper_name.lower() if model.stepper_name else ""
    if stepper_name not in ("rk4", "rk4_classic", "classical_rk4"):
        warnings.warn(
            f"trace_manifold_1d_ode uses an internal RK4 integrator, but Sim is "
            f"configured with stepper '{model.stepper_name}'. This is informational "
            f"only; the internal RK4 will be used regardless.",
            stacklevel=2,
        )

    n_state = len(model.spec.states)
    bounds_arr = _normalize_bounds(bounds, n_state)
    params_vec = _resolve_params(model, params)
    fp_vec = _resolve_fixed_point(model, fp)

    t_eval = float(model.spec.sim.t0 if t is None else t)
    rhs_fn = model.rhs
    update_aux_fn = model.update_aux
    jac_fn = model.jacobian

    # ---------------------------------------------------------------------------
    # Setup workspace for RHS evaluation
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

    # Initialize workspace
    _prep_ws(fp_vec)

    stop_mask = int(runtime_ws.stop_phase_mask[0]) if runtime_ws.stop_phase_mask.size else 0
    has_aux = runtime_ws.aux_values.size > 0
    do_aux_pre = has_aux or (stop_mask & 1) != 0
    do_aux_post = has_aux or (stop_mask & 2) != 0

    # ---------------------------------------------------------------------------
    # Determine execution strategy (JIT or Python)
    # ---------------------------------------------------------------------------
    def _is_numba_dispatcher(fn: object) -> bool:
        return hasattr(fn, "py_func") and hasattr(fn, "signatures")

    use_jit = _NUMBA_AVAILABLE and _is_numba_dispatcher(rhs_fn)

    if not use_jit:
        reason = ""
        if not _NUMBA_AVAILABLE:
            reason = "numba is not available"
        elif not _is_numba_dispatcher(rhs_fn):
            reason = "model was built with jit=False"
        warnings.warn(
            f"trace_manifold_1d_ode: fast execution path unavailable ({reason}). "
            "Falling back to Python loops. For best performance, build the model "
            "with jit=True and ensure numba is installed.",
            stacklevel=2,
        )

    # ---------------------------------------------------------------------------
    # Verify equilibrium
    # ---------------------------------------------------------------------------
    if fp_check_tol is not None:
        f_at_fp = np.empty((n_state,), dtype=model.dtype)
        rhs_fn(t_eval, fp_vec, f_at_fp, params_vec, runtime_ws)
        err = float(np.linalg.norm(f_at_fp))
        if not np.isfinite(err) or err > fp_check_tol:
            raise ValueError(
                f"Provided fp is not an equilibrium; |f(fp)|={err:.6g} exceeds tol={fp_check_tol}."
            )

    # ---------------------------------------------------------------------------
    # Compute Jacobian at equilibrium
    # ---------------------------------------------------------------------------
    def _jacobian_at(x: np.ndarray) -> np.ndarray:
        if jac == "fd" or (jac == "auto" and jac_fn is None):
            fx = np.empty((n_state,), dtype=model.dtype)
            _prep_ws(x)
            rhs_fn(t_eval, x, fx, params_vec, runtime_ws)
            J = np.zeros((n_state, n_state), dtype=float)
            for j in range(n_state):
                step = fd_eps * (1.0 + abs(float(x[j])))
                if step == 0.0:
                    step = fd_eps
                x_step = np.array(x, copy=True)
                x_step[j] += step
                f_step = np.empty((n_state,), dtype=model.dtype)
                _prep_ws(x_step)
                rhs_fn(t_eval, x_step, f_step, params_vec, runtime_ws)
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
    lam, v, eig_index = _select_eig_direction_ode(
        J,
        kind=kind,
        eig_rank=eig_rank,
        real_tol=eig_real_tol,
        imag_tol=eig_imag_tol,
        strict_1d=strict_1d,
    )

    # ---------------------------------------------------------------------------
    # Prepare integration
    # ---------------------------------------------------------------------------
    # Stable manifold: backward time (dt < 0)
    # Unstable manifold: forward time (dt > 0)
    dt_trace = -abs(dt) if kind == "stable" else +abs(dt)
    max_steps = int(np.ceil(max_time / abs(dt)))

    # Compute clipped bounds (with margin for integration, clipped later)
    extent = bounds_arr[:, 1] - bounds_arr[:, 0]
    clip_pad = extent * clip_margin
    clip_lo = bounds_arr[:, 0] - clip_pad
    clip_hi = bounds_arr[:, 1] + clip_pad

    # Seed points
    v = np.asarray(v, dtype=model.dtype)
    seed_pos = fp_vec + seed_delta * v
    seed_neg = fp_vec - seed_delta * v

    # Allocate output and work buffers
    out_buf = np.empty((max_points, n_state), dtype=model.dtype)
    k1 = np.empty((n_state,), dtype=model.dtype)
    k2 = np.empty((n_state,), dtype=model.dtype)
    k3 = np.empty((n_state,), dtype=model.dtype)
    k4 = np.empty((n_state,), dtype=model.dtype)
    z_stage = np.empty((n_state,), dtype=model.dtype)

    # ---------------------------------------------------------------------------
    # Trace branches
    # ---------------------------------------------------------------------------
    def _trace_branch(z0: np.ndarray) -> np.ndarray:
        _prep_ws(z0)
        if use_jit and _trace_branch_rk4_numba is not None:
            n_pts = _trace_branch_rk4_numba(
                z0, t_eval, dt_trace, max_steps, clip_lo, clip_hi,
                rhs_fn, params_vec, runtime_ws, update_aux_fn, do_aux_pre, do_aux_post,
                out_buf, k1, k2, k3, k4, z_stage
            )
        else:
            n_pts = _trace_branch_rk4_python(
                z0, t_eval, dt_trace, max_steps, clip_lo, clip_hi,
                rhs_fn, params_vec, runtime_ws, update_aux_fn, do_aux_pre, do_aux_post,
                out_buf, k1, k2, k3, k4, z_stage
            )
        return np.array(out_buf[:n_pts], copy=True)

    branch_pos = _trace_branch(seed_pos)
    branch_neg = _trace_branch(seed_neg)

    # ---------------------------------------------------------------------------
    # Post-process: clip to exact bounds and resample
    # ---------------------------------------------------------------------------
    def _clip_to_bounds(P: np.ndarray) -> list[np.ndarray]:
        if P.shape[0] < 2:
            return []
        mask = _inside_bounds(P, bounds_arr)
        return _split_contiguous_fast(P, mask)

    def _process_branch(P: np.ndarray) -> list[np.ndarray]:
        segments = _clip_to_bounds(P)
        if resample_h is not None:
            segments = [_resample_by_arclength(seg, resample_h) for seg in segments]
        return [seg for seg in segments if seg.shape[0] >= 2]

    branches_pos = _process_branch(branch_pos)
    branches_neg = _process_branch(branch_neg)

    return ManifoldTraceResult(
        kind=kind,
        fixed_point=np.array(fp_vec, copy=True),
        branches=(branches_pos, branches_neg),
        eigenvalue=lam,
        eigenvector=np.array(v, copy=True),
        eig_index=eig_index,
        step_mul=1,  # Not applicable to ODEs
        meta={
            "bounds": bounds_arr,
            "clip_margin": float(clip_margin),
            "seed_delta": float(seed_delta),
            "dt": float(dt),
            "max_time": float(max_time),
            "resample_h": resample_h,
            "max_points": int(max_points),
            "eig_rank": eig_rank,
            "strict_1d": bool(strict_1d),
            "eig_real_tol": float(eig_real_tol),
            "eig_imag_tol": float(eig_imag_tol),
            "jac": str(jac),
            "fd_eps": float(fd_eps),
            "t_eval": float(t_eval),
            "kind": kind,
        },
    )
