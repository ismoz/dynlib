# src/dynlib/analysis/manifold.py
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence, Literal, NamedTuple, TYPE_CHECKING

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

__all__ = [
    "ManifoldTraceResult",
    "trace_manifold_1d_map",
    "trace_manifold_1d_ode",
    "HeteroclinicRK45Config",
    "HeteroclinicBranchConfig",
    "HeteroclinicFinderConfig2D",
    "HeteroclinicFinderConfigND",
    "HeteroclinicTraceEvent",
    "HeteroclinicMissResult2D",
    "HeteroclinicMissResultND",
    "HeteroclinicFinderResult",
    "HeteroclinicTraceMeta",
    "HeteroclinicTraceResult",
    "HeteroclinicPreset",
    "heteroclinic_finder",
    "heteroclinic_tracer",
    "HomoclinicRK45Config",
    "HomoclinicBranchConfig",
    "HomoclinicFinderConfig",
    "HomoclinicMissResult",
    "HomoclinicFinderResult",
    "HomoclinicTraceEvent",
    "HomoclinicTraceMeta",
    "HomoclinicTraceResult",
    "HomoclinicPreset",
    "homoclinic_finder",
    "homoclinic_tracer",
]


def _brief_warning_format(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


def _warn_rk45_mismatch(caller: str, stepper_name: str) -> None:
    old_format = warnings.formatwarning
    try:
        warnings.formatwarning = _brief_warning_format
        warnings.warn(
            f"{caller} uses an internal RK45 integrator, but Sim is configured with "
            f"stepper '{stepper_name}'. This is informational only; the internal "
            "RK45 will be used regardless.",
            stacklevel=2,
        )
    finally:
        warnings.formatwarning = old_format


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


# =============================================================================
# Heteroclinic finder/tracer (ODE)
# =============================================================================


@dataclass(frozen=True)
class HeteroclinicRK45Config:
    dt0: float = 1e-3
    min_step: float = 1e-12
    dt_max: float = 1e-1
    atol: float = 1e-10
    rtol: float = 1e-7
    safety: float = 0.9
    max_steps: int = 2_000_000


@dataclass(frozen=True)
class HeteroclinicBranchConfig:
    eq_tol: float = 1e-12
    eq_max_iter: int = 40
    eq_track_max_dist: float | None = None

    eps_mode: Literal["leave", "fixed"] = "leave"
    eps: float = 1e-6
    eps_min: float = 1e-10
    eps_max: float = 1e-2
    r_leave: float = 1e-2
    t_leave_target: float = 0.05

    t_max: float = 500.0
    s_max: float = 1e6
    r_blow: float = 200.0

    window_min: Sequence[float] | np.ndarray | None = None
    window_max: Sequence[float] | np.ndarray | None = None

    t_min_event: float = 0.1
    require_leave_before_event: bool = True

    eig_real_tol: float = 1e-10
    eig_imag_tol: float = 1e-12
    strict_1d: bool = True
    jac: Literal["auto", "fd", "analytic"] = "auto"
    fd_eps: float = 1e-6

    rk: HeteroclinicRK45Config = field(default_factory=HeteroclinicRK45Config)


@dataclass(frozen=True)
class HeteroclinicFinderConfig2D:
    eq_tol: float = 1e-12
    eq_max_iter: int = 40
    eq_track_max_dist: float | None = None

    trace_u: HeteroclinicBranchConfig = field(default_factory=HeteroclinicBranchConfig)
    trace_s: HeteroclinicBranchConfig = field(default_factory=HeteroclinicBranchConfig)

    scan_n: int = 61
    max_bisect: int = 60

    x_tol: float = 1e-4
    gap_tol: float = 1e-6
    gap_fac: float = 2.0

    branch_mode: Literal["auto", "fixed"] = "auto"
    sign_u: int = +1
    sign_s: int = +1

    r_sec: float | None = None
    r_sec_mult: float = 20.0
    r_sec_min_mult: float = 2.0


@dataclass(frozen=True)
class HeteroclinicFinderConfigND:
    eq_tol: float = 1e-12
    eq_max_iter: int = 40
    eq_track_max_dist: float | None = None

    trace_u: HeteroclinicBranchConfig = field(default_factory=HeteroclinicBranchConfig)
    trace_s: HeteroclinicBranchConfig = field(default_factory=HeteroclinicBranchConfig)

    scan_n: int = 61
    max_bisect: int = 60

    x_tol: float = 1e-4
    gap_tol: float = 1e-4
    gap_fac: float = 2.0

    branch_mode: Literal["auto", "fixed"] = "auto"
    sign_u: int = +1
    sign_s: int = +1

    r_sec: float | None = None
    r_sec_mult: float = 20.0
    r_sec_min_mult: float = 2.0

    tau_min: float = 1e-10


@dataclass
class HeteroclinicTraceEvent:
    kind: str
    t: float
    x: np.ndarray
    info: dict[str, object] = field(default_factory=dict)


@dataclass
class _TraceResult:
    points: np.ndarray
    t: np.ndarray
    status: str
    event: HeteroclinicTraceEvent | None
    meta: dict[str, object]


@dataclass
class HeteroclinicMissResult2D:
    qualified: bool
    param_value: float
    source_eq: np.ndarray
    target_eq: np.ndarray
    sign_u: int
    sign_s: int
    x_u_cross: np.ndarray
    x_s_cross: np.ndarray
    theta_u_raw: float
    theta_s_raw: float
    theta_u: float
    theta_s: float
    g: float
    q: float
    status_u: str
    status_s: str
    status: str
    diag: dict[str, object]


@dataclass
class HeteroclinicMissResultND:
    qualified: bool
    param_value: float
    source_eq: np.ndarray
    target_eq: np.ndarray
    sign_u: int
    sign_s: int
    n_B: np.ndarray
    tau: np.ndarray
    r_sec: float
    x_u_cross: np.ndarray
    x_s_cross: np.ndarray
    s_u: float
    s_s: float
    g: float
    q: float
    status_u: str
    status_s: str
    status: str
    diag: dict[str, object]


class HeteroclinicFinderResult(NamedTuple):
    success: bool
    param_found: float
    miss: HeteroclinicMissResult2D | HeteroclinicMissResultND | None
    info: dict[str, object]


@dataclass
class HeteroclinicTraceMeta:
    param_value: float
    source_eq: np.ndarray
    target_eq: np.ndarray
    sign_u: int
    eps_used: float
    status: str
    success: bool
    event: HeteroclinicTraceEvent | None
    diag: dict[str, object]


class HeteroclinicTraceResult(NamedTuple):
    t: np.ndarray
    X: np.ndarray
    meta: HeteroclinicTraceMeta

    @property
    def branches(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return ([self.X], [])

    @property
    def branch_pos(self) -> list[np.ndarray]:
        return [self.X]

    @property
    def branch_neg(self) -> list[np.ndarray]:
        return []

    @property
    def kind(self) -> str:
        return "heteroclinic"

    @property
    def success(self) -> bool:
        return bool(self.meta.success)

# =============================================================================
# Presets for heteroclinic finder/tracer
# =============================================================================


@dataclass(frozen=True)
class HeteroclinicPreset:
    """
    Preset configurations for heteroclinic finder/tracer.

    Available presets:
    - "fast": Quick scan with lower accuracy, good for exploration.
    - "default": Balanced accuracy and speed for typical use.
    - "precise": High accuracy with more iterations, slower but robust.

    Users can create custom presets by instantiating this class directly.
    """
    name: str
    rk: HeteroclinicRK45Config
    branch: HeteroclinicBranchConfig
    scan_n: int
    max_bisect: int
    gap_tol: float
    x_tol: float


_HETEROCLINIC_PRESETS: dict[str, HeteroclinicPreset] = {
    "fast": HeteroclinicPreset(
        name="fast",
        rk=HeteroclinicRK45Config(
            dt0=1e-2,
            dt_max=1e-1,
            atol=1e-8,
            rtol=1e-5,
            max_steps=500_000,
        ),
        branch=HeteroclinicBranchConfig(
            r_leave=1e-2,
            t_leave_target=0.05,
            t_max=100.0,
            r_blow=200.0,
        ),
        scan_n=31,
        max_bisect=30,
        gap_tol=1e-3,
        x_tol=1e-3,
    ),
    "default": HeteroclinicPreset(
        name="default",
        rk=HeteroclinicRK45Config(
            dt0=1e-3,
            dt_max=5e-2,
            atol=1e-10,
            rtol=1e-7,
            max_steps=1_000_000,
        ),
        branch=HeteroclinicBranchConfig(
            r_leave=1e-2,
            t_leave_target=0.05,
            t_max=200.0,
            r_blow=200.0,
        ),
        scan_n=61,
        max_bisect=60,
        gap_tol=1e-4,
        x_tol=1e-4,
    ),
    "precise": HeteroclinicPreset(
        name="precise",
        rk=HeteroclinicRK45Config(
            dt0=1e-4,
            dt_max=1e-2,
            atol=1e-12,
            rtol=1e-9,
            max_steps=3_000_000,
        ),
        branch=HeteroclinicBranchConfig(
            r_leave=1e-3,
            t_leave_target=0.02,
            t_max=500.0,
            r_blow=500.0,
        ),
        scan_n=121,
        max_bisect=80,
        gap_tol=1e-6,
        x_tol=1e-6,
    ),
}


def _get_heteroclinic_preset(preset: str | HeteroclinicPreset) -> HeteroclinicPreset:
    """Resolve a preset name or instance to a HeteroclinicPreset."""
    if isinstance(preset, HeteroclinicPreset):
        return preset
    if preset not in _HETEROCLINIC_PRESETS:
        available = ", ".join(sorted(_HETEROCLINIC_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    return _HETEROCLINIC_PRESETS[preset]


def _build_branch_config_from_preset(
    preset: HeteroclinicPreset,
    *,
    window: Sequence[tuple[float, float]] | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
) -> HeteroclinicBranchConfig:
    """Build a HeteroclinicBranchConfig from a preset with optional overrides."""
    window_min = None
    window_max = None
    if window is not None:
        window_min = np.array([lo for lo, _ in window], dtype=float)
        window_max = np.array([hi for _, hi in window], dtype=float)

    return HeteroclinicBranchConfig(
        eq_tol=preset.branch.eq_tol,
        eq_max_iter=preset.branch.eq_max_iter,
        eq_track_max_dist=preset.branch.eq_track_max_dist,
        eps_mode=preset.branch.eps_mode,
        eps=preset.branch.eps,
        eps_min=preset.branch.eps_min,
        eps_max=preset.branch.eps_max,
        r_leave=preset.branch.r_leave,
        t_leave_target=preset.branch.t_leave_target,
        t_max=t_max if t_max is not None else preset.branch.t_max,
        s_max=preset.branch.s_max,
        r_blow=r_blow if r_blow is not None else preset.branch.r_blow,
        window_min=window_min,
        window_max=window_max,
        t_min_event=preset.branch.t_min_event,
        require_leave_before_event=preset.branch.require_leave_before_event,
        eig_real_tol=preset.branch.eig_real_tol,
        eig_imag_tol=preset.branch.eig_imag_tol,
        strict_1d=preset.branch.strict_1d,
        jac=preset.branch.jac,
        fd_eps=preset.branch.fd_eps,
        rk=preset.rk,
    )


def _build_finder_config_from_kwargs(
    mode: Literal["2d", "nd"],
    *,
    preset: str | HeteroclinicPreset | None = None,
    trace_cfg: HeteroclinicBranchConfig | None = None,
    window: Sequence[tuple[float, float]] | None = None,
    scan_n: int | None = None,
    max_bisect: int | None = None,
    gap_tol: float | None = None,
    x_tol: float | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
) -> HeteroclinicFinderConfig2D | HeteroclinicFinderConfigND:
    """Build a finder config from kwargs, preset, or trace_cfg."""
    # Resolve preset
    pset = _get_heteroclinic_preset(preset if preset is not None else "default")

    # Build trace config
    if trace_cfg is not None:
        # Apply window override to provided trace_cfg if needed
        if window is not None:
            window_min = np.array([lo for lo, _ in window], dtype=float)
            window_max = np.array([hi for _, hi in window], dtype=float)
            trace_cfg = HeteroclinicBranchConfig(
                eq_tol=trace_cfg.eq_tol,
                eq_max_iter=trace_cfg.eq_max_iter,
                eq_track_max_dist=trace_cfg.eq_track_max_dist,
                eps_mode=trace_cfg.eps_mode,
                eps=trace_cfg.eps,
                eps_min=trace_cfg.eps_min,
                eps_max=trace_cfg.eps_max,
                r_leave=trace_cfg.r_leave,
                t_leave_target=trace_cfg.t_leave_target,
                t_max=t_max if t_max is not None else trace_cfg.t_max,
                s_max=trace_cfg.s_max,
                r_blow=r_blow if r_blow is not None else trace_cfg.r_blow,
                window_min=window_min,
                window_max=window_max,
                t_min_event=trace_cfg.t_min_event,
                require_leave_before_event=trace_cfg.require_leave_before_event,
                eig_real_tol=trace_cfg.eig_real_tol,
                eig_imag_tol=trace_cfg.eig_imag_tol,
                strict_1d=trace_cfg.strict_1d,
                jac=trace_cfg.jac,
                fd_eps=trace_cfg.fd_eps,
                rk=trace_cfg.rk,
            )
        elif t_max is not None or r_blow is not None:
            trace_cfg = HeteroclinicBranchConfig(
                eq_tol=trace_cfg.eq_tol,
                eq_max_iter=trace_cfg.eq_max_iter,
                eq_track_max_dist=trace_cfg.eq_track_max_dist,
                eps_mode=trace_cfg.eps_mode,
                eps=trace_cfg.eps,
                eps_min=trace_cfg.eps_min,
                eps_max=trace_cfg.eps_max,
                r_leave=trace_cfg.r_leave,
                t_leave_target=trace_cfg.t_leave_target,
                t_max=t_max if t_max is not None else trace_cfg.t_max,
                s_max=trace_cfg.s_max,
                r_blow=r_blow if r_blow is not None else trace_cfg.r_blow,
                window_min=trace_cfg.window_min,
                window_max=trace_cfg.window_max,
                t_min_event=trace_cfg.t_min_event,
                require_leave_before_event=trace_cfg.require_leave_before_event,
                eig_real_tol=trace_cfg.eig_real_tol,
                eig_imag_tol=trace_cfg.eig_imag_tol,
                strict_1d=trace_cfg.strict_1d,
                jac=trace_cfg.jac,
                fd_eps=trace_cfg.fd_eps,
                rk=trace_cfg.rk,
            )
    else:
        trace_cfg = _build_branch_config_from_preset(
            pset, window=window, t_max=t_max, r_blow=r_blow
        )

    # Resolve tolerances with fallback to preset
    final_scan_n = scan_n if scan_n is not None else pset.scan_n
    final_max_bisect = max_bisect if max_bisect is not None else pset.max_bisect
    final_gap_tol = gap_tol if gap_tol is not None else pset.gap_tol
    final_x_tol = x_tol if x_tol is not None else pset.x_tol

    if mode == "2d":
        return HeteroclinicFinderConfig2D(
            trace_u=trace_cfg,
            trace_s=trace_cfg,
            scan_n=final_scan_n,
            max_bisect=final_max_bisect,
            gap_tol=final_gap_tol,
            x_tol=final_x_tol,
        )
    else:
        return HeteroclinicFinderConfigND(
            trace_u=trace_cfg,
            trace_s=trace_cfg,
            scan_n=final_scan_n,
            max_bisect=final_max_bisect,
            gap_tol=final_gap_tol,
            x_tol=final_x_tol,
        )


@dataclass(frozen=True)
class _HeteroclinicContext:
    model: "FullModel"
    t_eval: float
    runtime_ws: object
    prep_ws: Callable[[np.ndarray], None]
    update_aux_fn: Callable
    do_aux_pre: bool
    do_aux_post: bool
    params_base: np.ndarray
    param_index: int
    rhs_fn: Callable
    jac_fn: Callable | None
    use_jit: bool


def _is_numba_dispatcher(fn: object) -> bool:
    return hasattr(fn, "py_func") and hasattr(fn, "signatures")


def _resolve_param_index(model: "FullModel", param: str | int | None) -> int:
    params = list(model.spec.params)
    if param is None:
        if len(params) != 1:
            raise ValueError("param must be specified when model has multiple parameters.")
        return 0
    if isinstance(param, int):
        if param < 0 or param >= len(params):
            raise ValueError(f"param index {param} out of range [0, {len(params)})")
        return int(param)
    param_index = {name: i for i, name in enumerate(params)}
    if param not in param_index:
        raise KeyError(f"Unknown param '{param}'.")
    return int(param_index[param])


def _params_with_override(params_base: np.ndarray, index: int, value: float) -> np.ndarray:
    params_vec = np.array(params_base, copy=True)
    params_vec[index] = float(value)
    return params_vec


def _track_max_dist_default(r_leave: float) -> float:
    return 0.25 * float(r_leave)


def _validate_rk45_cfg(cfg: HeteroclinicRK45Config) -> None:
    if cfg.dt0 <= 0.0:
        raise ValueError("rk.dt0 must be positive")
    if cfg.min_step <= 0.0:
        raise ValueError("rk.min_step must be positive")
    if cfg.dt_max <= 0.0:
        raise ValueError("rk.dt_max must be positive")
    if cfg.min_step > cfg.dt_max:
        raise ValueError("rk.min_step must be <= rk.dt_max")
    if cfg.atol <= 0.0:
        raise ValueError("rk.atol must be positive")
    if cfg.rtol <= 0.0:
        raise ValueError("rk.rtol must be positive")
    if cfg.safety <= 0.0:
        raise ValueError("rk.safety must be positive")
    if cfg.max_steps <= 0:
        raise ValueError("rk.max_steps must be positive")


def _validate_branch_cfg(cfg: HeteroclinicBranchConfig) -> None:
    if cfg.eq_tol <= 0.0:
        raise ValueError("eq_tol must be positive")
    if cfg.eq_max_iter <= 0:
        raise ValueError("eq_max_iter must be positive")
    if cfg.eps <= 0.0:
        raise ValueError("eps must be positive")
    if cfg.eps_min <= 0.0:
        raise ValueError("eps_min must be positive")
    if cfg.eps_max <= 0.0:
        raise ValueError("eps_max must be positive")
    if cfg.eps_min > cfg.eps_max:
        raise ValueError("eps_min must be <= eps_max")
    if cfg.r_leave <= 0.0:
        raise ValueError("r_leave must be positive")
    if cfg.t_leave_target <= 0.0:
        raise ValueError("t_leave_target must be positive")
    if cfg.t_max <= 0.0:
        raise ValueError("t_max must be positive")
    if cfg.s_max <= 0.0:
        raise ValueError("s_max must be positive")
    if cfg.r_blow <= 0.0:
        raise ValueError("r_blow must be positive")
    if cfg.t_min_event < 0.0:
        raise ValueError("t_min_event must be non-negative")
    if cfg.eig_real_tol < 0.0:
        raise ValueError("eig_real_tol must be non-negative")
    if cfg.eig_imag_tol < 0.0:
        raise ValueError("eig_imag_tol must be non-negative")
    if cfg.fd_eps <= 0.0:
        raise ValueError("fd_eps must be positive")
    if cfg.jac not in ("auto", "fd", "analytic"):
        raise ValueError("jac must be 'auto', 'fd', or 'analytic'")
    _validate_rk45_cfg(cfg.rk)


def _validate_finder_cfg_2d(cfg: HeteroclinicFinderConfig2D) -> None:
    if cfg.scan_n < 2:
        raise ValueError("scan_n must be >= 2")
    if cfg.max_bisect < 0:
        raise ValueError("max_bisect must be >= 0")
    if cfg.x_tol <= 0.0:
        raise ValueError("x_tol must be positive")
    if cfg.gap_tol < 0.0:
        raise ValueError("gap_tol must be non-negative")
    if cfg.gap_fac < 0.0:
        raise ValueError("gap_fac must be non-negative")
    if cfg.r_sec is not None and cfg.r_sec <= 0.0:
        raise ValueError("r_sec must be positive")
    if cfg.r_sec_mult <= 0.0:
        raise ValueError("r_sec_mult must be positive")
    if cfg.r_sec_min_mult <= 0.0:
        raise ValueError("r_sec_min_mult must be positive")
    _validate_branch_cfg(cfg.trace_u)
    _validate_branch_cfg(cfg.trace_s)


def _validate_finder_cfg_nd(cfg: HeteroclinicFinderConfigND) -> None:
    if cfg.scan_n < 2:
        raise ValueError("scan_n must be >= 2")
    if cfg.max_bisect < 0:
        raise ValueError("max_bisect must be >= 0")
    if cfg.x_tol <= 0.0:
        raise ValueError("x_tol must be positive")
    if cfg.gap_tol < 0.0:
        raise ValueError("gap_tol must be non-negative")
    if cfg.gap_fac < 0.0:
        raise ValueError("gap_fac must be non-negative")
    if cfg.r_sec is not None and cfg.r_sec <= 0.0:
        raise ValueError("r_sec must be positive")
    if cfg.r_sec_mult <= 0.0:
        raise ValueError("r_sec_mult must be positive")
    if cfg.r_sec_min_mult <= 0.0:
        raise ValueError("r_sec_min_mult must be positive")
    if cfg.tau_min <= 0.0:
        raise ValueError("tau_min must be positive")
    _validate_branch_cfg(cfg.trace_u)
    _validate_branch_cfg(cfg.trace_s)


def _build_heteroclinic_context(
    sim: "Sim",
    *,
    params: Mapping[str, float] | Sequence[float] | np.ndarray | None,
    param: str | int | None,
    t: float | None,
    caller: str,
) -> _HeteroclinicContext:
    model = _resolve_model(sim)
    if model.spec.kind != "ode":
        raise ValueError(f"{caller} requires model.spec.kind == 'ode'")
    if not np.issubdtype(model.dtype, np.floating):
        raise ValueError(f"{caller} requires a floating-point model dtype.")
    if model.rhs is None:
        raise ValueError("ODE RHS is not available on the model.")

    params_base = _resolve_params(model, params)
    param_index = _resolve_param_index(model, param)

    t_eval = float(model.spec.sim.t0 if t is None else t)
    rhs_fn = model.rhs
    jac_fn = model.jacobian
    update_aux_fn = model.update_aux

    # Warn if stepper differs from internal RK45
    stepper_name = model.stepper_name.lower() if model.stepper_name else ""
    if stepper_name and stepper_name not in ("rk45", "dopri5", "rk45_dopri"):
        _warn_rk45_mismatch(caller, model.stepper_name)

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

    stop_mask = int(runtime_ws.stop_phase_mask[0]) if runtime_ws.stop_phase_mask.size else 0
    has_aux = runtime_ws.aux_values.size > 0
    do_aux_pre = has_aux or (stop_mask & 1) != 0
    do_aux_post = has_aux or (stop_mask & 2) != 0

    use_jit = _NUMBA_AVAILABLE and _is_numba_dispatcher(rhs_fn)
    if not use_jit:
        reason = ""
        if not _NUMBA_AVAILABLE:
            reason = "numba is not available"
        elif not _is_numba_dispatcher(rhs_fn):
            reason = "model was built with jit=False"
        warnings.warn(
            f"{caller}: fast execution path unavailable ({reason}). "
            "Falling back to Python loops. For best performance, build the model "
            "with jit=True and ensure numba is installed.",
            stacklevel=2,
        )

    return _HeteroclinicContext(
        model=model,
        t_eval=t_eval,
        runtime_ws=runtime_ws,
        prep_ws=_prep_ws,
        update_aux_fn=update_aux_fn,
        do_aux_pre=do_aux_pre,
        do_aux_post=do_aux_post,
        params_base=params_base,
        param_index=param_index,
        rhs_fn=rhs_fn,
        jac_fn=jac_fn,
        use_jit=use_jit,
    )


def _update_lag_state(runtime_ws, z: np.ndarray) -> None:
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


def _segment_ball_hit_alpha(x0: np.ndarray, x1: np.ndarray, c: np.ndarray, r: float) -> float | None:
    x0 = np.array(x0, float)
    x1 = np.array(x1, float)
    c = np.array(c, float)
    r = float(r)
    if not (
        np.all(np.isfinite(x0))
        and np.all(np.isfinite(x1))
        and np.all(np.isfinite(c))
        and np.isfinite(r)
        and r > 0.0
    ):
        return None

    v0 = x0 - c
    d = x1 - x0
    a = float(d @ d)
    if a <= 0.0:
        return 0.0 if float(v0 @ v0) <= r * r else None

    c0 = float(v0 @ v0) - r * r
    if c0 <= 0.0:
        return 0.0

    b = 2.0 * float(d @ v0)
    disc = b * b - 4.0 * a * c0
    if disc < 0.0:
        return None
    sdisc = math.sqrt(disc)
    r1 = (-b - sdisc) / (2.0 * a)
    r2 = (-b + sdisc) / (2.0 * a)
    cand = [rr for rr in (r1, r2) if 0.0 <= rr <= 1.0]
    if not cand:
        return None
    return float(min(cand))


@dataclass(frozen=True)
class _Section:
    h: Callable[[np.ndarray], float]
    x_ref: np.ndarray
    tau: np.ndarray
    s_fn: Callable[[np.ndarray], float] | None = None
    qualify_fn: Callable[[np.ndarray], bool] | None = None
    kind: str = "generic"
    meta: dict[str, object] = field(default_factory=dict)

    def s(self, x: np.ndarray) -> float:
        x = np.array(x, float)
        if self.s_fn is not None:
            return float(self.s_fn(x))
        return float(np.array(self.tau, float) @ (x - np.array(self.x_ref, float)))

    def qualify(self, x: np.ndarray) -> bool:
        if self.qualify_fn is None:
            return True
        return bool(self.qualify_fn(np.array(x, float)))


def _make_circle_section_2d(*, center: np.ndarray, radius: float) -> _Section:
    c = np.array(center, float).reshape(2,)
    r = float(radius)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be positive")

    def h(x: np.ndarray) -> float:
        x = np.array(x, float).reshape(2,)
        return float(np.linalg.norm(x - c) - r)

    def s_fn(x: np.ndarray) -> float:
        x = np.array(x, float).reshape(2,)
        v = x - c
        return float(math.atan2(float(v[1]), float(v[0])))

    return _Section(
        h=h,
        x_ref=c.copy(),
        tau=np.array([1.0, 0.0], float),
        s_fn=s_fn,
        kind="circle",
        meta={"center": c.copy(), "radius": float(r)},
    )


def _unit(v: np.ndarray) -> tuple[bool, np.ndarray, float]:
    v = np.array(v, float).reshape(-1,)
    n = float(np.linalg.norm(v))
    if (not np.isfinite(n)) or n <= 1e-300:
        return False, v, n
    return True, v / n, n


def _choose_fallback_tau(nB_unit: np.ndarray) -> np.ndarray:
    nB_unit = np.array(nB_unit, float).reshape(-1,)
    n = nB_unit.size
    i = int(np.argmin(np.abs(nB_unit)))
    ei = np.zeros(n, float)
    ei[i] = 1.0
    tau0 = ei - nB_unit * float(nB_unit @ ei)
    return tau0


def _make_hyperplane_section_nd(
    *,
    center: np.ndarray,
    normal_unit: np.ndarray,
    tau_unit: np.ndarray,
    offset: float,
    rho_max: float | None = None,
) -> _Section:
    c = np.array(center, float).reshape(-1,)
    nB = np.array(normal_unit, float).reshape(-1,)
    tau = np.array(tau_unit, float).reshape(-1,)
    off = float(offset)

    x_sec = c + off * nB

    def h(x: np.ndarray) -> float:
        x = np.array(x, float).reshape(-1,)
        return float(nB @ (x - x_sec))

    def qualify_fn(x: np.ndarray) -> bool:
        if rho_max is None:
            return True
        return float(np.linalg.norm(np.array(x, float) - x_sec)) <= float(rho_max)

    return _Section(
        h=h,
        x_ref=c.copy(),
        tau=tau.copy(),
        s_fn=None,
        qualify_fn=qualify_fn,
        kind="plane",
        meta={
            "center": c.copy(),
            "normal": nB.copy(),
            "tau": tau.copy(),
            "x_sec": x_sec.copy(),
            "rho_max": (None if rho_max is None else float(rho_max)),
        },
    )


def _in_window(x: np.ndarray, wmin: np.ndarray | None, wmax: np.ndarray | None) -> bool:
    if wmin is None or wmax is None:
        return True
    return bool(np.all(x >= wmin) and np.all(x <= wmax))


def _polyline_arclength(xs: np.ndarray) -> float:
    if xs.shape[0] < 2:
        return 0.0
    d = xs[1:] - xs[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


def _choose_eps_leave(
    r_leave: float,
    rate: float,
    t_leave_target: float,
    eps_min: float,
    eps_max: float,
    fallback: float,
) -> float:
    rate = float(abs(rate))
    if not np.isfinite(rate) or rate <= 1e-15:
        eps = float(fallback)
    else:
        eps = float(r_leave) * math.exp(-rate * float(t_leave_target))
    eps = max(float(eps_min), min(float(eps_max), eps))
    return eps


def _newton_equilibrium(
    rhs_eval: Callable[[np.ndarray, np.ndarray], None],
    jac_eval: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    damping: float,
    backtrack: bool,
) -> tuple[bool, np.ndarray, dict[str, object]]:
    x = np.array(x0, copy=True)
    info: dict[str, object] = {"iters": 0, "res": None}

    r = np.empty_like(x)
    rhs_eval(x, r)
    rnorm = float(np.linalg.norm(r))
    info["res"] = rnorm
    if rnorm <= tol:
        return True, x, info

    for it in range(1, int(max_iter) + 1):
        A = jac_eval(x)
        try:
            dx = np.linalg.solve(A, -r)
        except np.linalg.LinAlgError:
            info.update({"iters": it, "fail": "singular_jacobian", "res": rnorm})
            return False, x, info

        lam = float(damping)
        if backtrack:
            for _ in range(25):
                x_try = x + lam * dx
                rhs_eval(x_try, r)
                r_try = float(np.linalg.norm(r))
                if r_try < (1.0 - 1e-4 * lam) * rnorm:
                    x = x_try
                    rnorm = r_try
                    break
                lam *= 0.5
            else:
                info.update({"iters": it, "fail": "newton_backtrack_failed", "res": rnorm})
                return False, x, info
        else:
            x = x + lam * dx
            rhs_eval(x, r)
            rnorm = float(np.linalg.norm(r))

        info["iters"] = it
        info["res"] = rnorm
        if rnorm <= tol:
            return True, x, info

    info.update({"fail": "max_iter", "res": rnorm})
    return False, x, info


def _solve_equilibrium_locked(
    rhs_eval: Callable[[np.ndarray, np.ndarray], None],
    jac_eval: Callable[[np.ndarray], np.ndarray],
    x_guess: np.ndarray,
    *,
    x_prev: np.ndarray | None,
    eq_tol: float,
    eq_max_iter: int,
    eq_track_max_dist: float | None,
    r_leave_for_default: float,
) -> tuple[bool, np.ndarray, dict[str, object]]:
    ok, x_eq, info = _newton_equilibrium(
        rhs_eval,
        jac_eval,
        np.array(x_guess, copy=True),
        tol=float(eq_tol),
        max_iter=int(eq_max_iter),
        damping=1.0,
        backtrack=True,
    )
    if not ok:
        info["fail_mode"] = "eq_fail"
        return False, x_eq, info

    if x_prev is not None:
        dist = float(np.linalg.norm(x_eq - np.array(x_prev, float)))
        info["track_dist"] = dist
        maxd = _track_max_dist_default(r_leave_for_default) if eq_track_max_dist is None else float(eq_track_max_dist)
        info["track_max_dist"] = maxd
        if dist > maxd:
            info["fail_mode"] = "eq_jump"
            return False, x_eq, info

    info["fail_mode"] = None
    return True, x_eq, info


def _eig_1d_data_ode(
    J: np.ndarray,
    *,
    kind: Literal["unstable", "stable"],
    real_tol: float,
    imag_tol: float,
    strict_1d: bool,
) -> tuple[bool, tuple[complex, np.ndarray, int] | None, dict[str, object]]:
    w, V = np.linalg.eig(J)
    re = np.real(w)

    unstable = np.where(re > real_tol)[0]
    stable = np.where(re < -real_tol)[0]

    info: dict[str, object] = {
        "eigvals": w,
        "unstable_count": int(unstable.size),
        "stable_count": int(stable.size),
        "kind": kind,
    }

    if kind == "unstable":
        if unstable.size != 1 and strict_1d:
            info["fail"] = "not_1d_unstable"
            return False, None, info
        if unstable.size == 0:
            info["fail"] = "no_unstable"
            return False, None, info
        idx = int(unstable[0])
    else:
        if stable.size != 1 and strict_1d:
            info["fail"] = "not_1d_stable"
            return False, None, info
        if stable.size == 0:
            info["fail"] = "no_stable"
            return False, None, info
        idx = int(stable[0])

    lam = w[idx]
    v = V[:, idx]

    if np.max(np.abs(np.imag(lam))) > imag_tol or np.max(np.abs(np.imag(v))) > imag_tol:
        info["fail"] = "complex_mode_not_supported"
        return False, None, info

    v = np.real(v)
    return True, (lam, v, idx), info


# Numba-accelerated RK45 integration for heteroclinic tracing
_rk45_integrate_numba = None
_update_aux_noop_numba = None

if _NUMBA_AVAILABLE:  # pragma: no cover - optional dependency
    try:
        from numba import njit  # type: ignore

        @njit(cache=True)
        def _update_aux_noop_numba(t, x, params, aux_values, runtime_ws):
            return

        @njit(cache=True)
        def _segment_ball_hit_alpha_numba(x0, x1, c, r):
            if r <= 0.0:
                return -1.0

            a = 0.0
            b = 0.0
            c0 = 0.0
            for i in range(x0.size):
                v0 = x0[i] - c[i]
                d = x1[i] - x0[i]
                a += d * d
                b += d * v0
                c0 += v0 * v0

            if a <= 0.0:
                return 0.0 if c0 <= r * r else -1.0

            c0 = c0 - r * r
            if c0 <= 0.0:
                return 0.0

            b = 2.0 * b
            disc = b * b - 4.0 * a * c0
            if disc < 0.0:
                return -1.0
            sdisc = math.sqrt(disc)
            r1 = (-b - sdisc) / (2.0 * a)
            r2 = (-b + sdisc) / (2.0 * a)
            if 0.0 <= r1 <= 1.0:
                if 0.0 <= r2 <= 1.0:
                    return r1 if r1 <= r2 else r2
                return r1
            if 0.0 <= r2 <= 1.0:
                return r2
            return -1.0

        @njit(cache=True)
        def _rk45_integrate_numba_impl(
            rhs_fn,
            params_vec,
            runtime_ws,
            update_aux_fn,
            do_aux_pre,
            do_aux_post,
            x0,
            t0,
            t1,
            dt0,
            min_step,
            dt_max,
            atol,
            rtol,
            safety,
            max_steps,
            sign_factor,
            x_eq,
            r_blow,
            wmin,
            wmax,
            has_window,
            t_min_event,
            require_leave_before_event,
            r_leave,
            target_center,
            target_r,
            enable_target,
            section_kind,
            section_center,
            section_normal,
            section_tau,
            section_radius,
            section_xsec,
            rho_max,
            enable_section,
            out_t,
            out_x,
            event_info,
        ):
            n = x0.size

            x = np.empty(n, dtype=x0.dtype)
            x_prev = np.empty(n, dtype=x0.dtype)

            k1 = np.empty(n, dtype=x0.dtype)
            k2 = np.empty(n, dtype=x0.dtype)
            k3 = np.empty(n, dtype=x0.dtype)
            k4 = np.empty(n, dtype=x0.dtype)
            k5 = np.empty(n, dtype=x0.dtype)
            k6 = np.empty(n, dtype=x0.dtype)
            k7 = np.empty(n, dtype=x0.dtype)
            x_stage = np.empty(n, dtype=x0.dtype)
            x5 = np.empty(n, dtype=x0.dtype)
            x4 = np.empty(n, dtype=x0.dtype)

            for i in range(n):
                x[i] = x0[i]
                out_x[0, i] = x0[i]
            out_t[0] = t0

            for i in range(event_info.size):
                event_info[i] = math.nan
            event_info[0] = 0.0

            m = 1
            t = t0

            direction = 1.0 if t1 >= t0 else -1.0
            h = dt0 * direction

            stop_mask = 0
            if runtime_ws.stop_phase_mask.shape[0] > 0:
                stop_mask = runtime_ws.stop_phase_mask[0]

            if do_aux_pre:
                update_aux_fn(t, x, params_vec, runtime_ws.aux_values, runtime_ws)
            if (stop_mask & 1) != 0 and runtime_ws.stop_flag.shape[0] > 0:
                if runtime_ws.stop_flag[0] != 0:
                    return 4, m

            left = False
            r_blow_sq = r_blow * r_blow
            r_leave_sq = r_leave * r_leave

            for _ in range(int(max_steps)):
                # Blow/window checks
                dist2 = 0.0
                for i in range(n):
                    xi = x[i]
                    if not np.isfinite(xi):
                        return 2, m
                    dx = xi - x_eq[i]
                    dist2 += dx * dx
                if r_blow_sq > 0.0 and dist2 > r_blow_sq:
                    return 2, m
                if has_window:
                    for i in range(n):
                        if x[i] < wmin[i] or x[i] > wmax[i]:
                            return 2, m

                if direction > 0.0 and t + h > t1:
                    h = t1 - t
                if direction < 0.0 and t + h < t1:
                    h = t1 - t

                rhs_fn(t, x, k1, params_vec, runtime_ws)
                if sign_factor < 0.0:
                    for i in range(n):
                        k1[i] = sign_factor * k1[i]
                for i in range(n):
                    x_stage[i] = x[i] + h * (0.2 * k1[i])
                rhs_fn(t + 0.2 * h, x_stage, k2, params_vec, runtime_ws)
                if sign_factor < 0.0:
                    for i in range(n):
                        k2[i] = sign_factor * k2[i]

                for i in range(n):
                    x_stage[i] = x[i] + h * (0.075 * k1[i] + 0.225 * k2[i])
                rhs_fn(t + 0.3 * h, x_stage, k3, params_vec, runtime_ws)
                if sign_factor < 0.0:
                    for i in range(n):
                        k3[i] = sign_factor * k3[i]

                for i in range(n):
                    x_stage[i] = x[i] + h * (
                        0.9777777777777777 * k1[i]
                        - 3.7333333333333334 * k2[i]
                        + 3.5555555555555554 * k3[i]
                    )
                rhs_fn(t + 0.8 * h, x_stage, k4, params_vec, runtime_ws)
                if sign_factor < 0.0:
                    for i in range(n):
                        k4[i] = sign_factor * k4[i]

                for i in range(n):
                    x_stage[i] = x[i] + h * (
                        2.9525986892242035 * k1[i]
                        - 11.595793324188385 * k2[i]
                        + 9.822892851699436 * k3[i]
                        - 0.2908093278463649 * k4[i]
                    )
                rhs_fn(t + (8.0 / 9.0) * h, x_stage, k5, params_vec, runtime_ws)
                if sign_factor < 0.0:
                    for i in range(n):
                        k5[i] = sign_factor * k5[i]

                for i in range(n):
                    x_stage[i] = x[i] + h * (
                        2.8462752525252526 * k1[i]
                        - 10.757575757575758 * k2[i]
                        + 8.906422717743472 * k3[i]
                        + 0.2784090909090909 * k4[i]
                        - 0.2735313036020583 * k5[i]
                    )
                rhs_fn(t + h, x_stage, k6, params_vec, runtime_ws)
                if sign_factor < 0.0:
                    for i in range(n):
                        k6[i] = sign_factor * k6[i]

                for i in range(n):
                    x5[i] = x[i] + h * (
                        0.09114583333333333 * k1[i]
                        + 0.44923629829290207 * k3[i]
                        + 0.6510416666666666 * k4[i]
                        - 0.322376179245283 * k5[i]
                        + 0.13095238095238096 * k6[i]
                    )
                rhs_fn(t + h, x5, k7, params_vec, runtime_ws)
                if sign_factor < 0.0:
                    for i in range(n):
                        k7[i] = sign_factor * k7[i]

                for i in range(n):
                    x4[i] = x[i] + h * (
                        0.08991319444444444 * k1[i]
                        + 0.4534890685834082 * k3[i]
                        + 0.6140625 * k4[i]
                        - 0.2715123820754717 * k5[i]
                        + 0.08904761904761905 * k6[i]
                        + 0.025 * k7[i]
                    )

                en = 0.0
                for i in range(n):
                    e = x5[i] - x4[i]
                    scale = atol + rtol * max(abs(x[i]), abs(x[i] + e))
                    val = e / scale
                    en += val * val
                en = math.sqrt(en / n)
                if not np.isfinite(en):
                    return 2, m

                accepted = en <= 1.0
                if accepted:
                    for i in range(n):
                        x_prev[i] = x[i]
                    t_prev = t

                    t = t + h
                    for i in range(n):
                        x[i] = x5[i]
                        out_x[m, i] = x[i]
                    out_t[m] = t
                    m += 1

                    if do_aux_post:
                        update_aux_fn(t, x, params_vec, runtime_ws.aux_values, runtime_ws)

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
                            lag_ring[offset + head] = x[state_idx]

                    if (stop_mask & 2) != 0 and runtime_ws.stop_flag.shape[0] > 0:
                        if runtime_ws.stop_flag[0] != 0:
                            return 4, m

                    if not left:
                        dist2 = 0.0
                        for i in range(n):
                            dx = x[i] - x_eq[i]
                            dist2 += dx * dx
                        if dist2 >= r_leave_sq:
                            left = True

                    if enable_target or enable_section:
                        if t >= t_min_event and (not require_leave_before_event or left):
                            if enable_target and target_r > 0.0:
                                alpha = _segment_ball_hit_alpha_numba(x_prev, x, target_center, target_r)
                                if alpha >= 0.0:
                                    t_ev = t_prev + alpha * (t - t_prev)
                                    for i in range(n):
                                        out_x[m - 1, i] = x_prev[i] + alpha * (x[i] - x_prev[i])
                                    out_t[m - 1] = t_ev
                                    event_info[0] = 1.0
                                    event_info[1] = alpha
                                    d_prev = 0.0
                                    d_curr = 0.0
                                    d_ev = 0.0
                                    for i in range(n):
                                        v0 = x_prev[i] - target_center[i]
                                        v1 = x[i] - target_center[i]
                                        ve = out_x[m - 1, i] - target_center[i]
                                        d_prev += v0 * v0
                                        d_curr += v1 * v1
                                        d_ev += ve * ve
                                    event_info[2] = math.sqrt(d_ev)
                                    event_info[3] = math.sqrt(d_prev)
                                    event_info[4] = math.sqrt(d_curr)
                                    return 1, m

                            if enable_section and section_kind != 0:
                                if section_kind == 1:
                                    h0 = 0.0
                                    h1 = 0.0
                                    for i in range(n):
                                        v0 = x_prev[i] - section_center[i]
                                        v1 = x[i] - section_center[i]
                                        h0 += v0 * v0
                                        h1 += v1 * v1
                                    h0 = math.sqrt(h0) - section_radius
                                    h1 = math.sqrt(h1) - section_radius
                                else:
                                    h0 = 0.0
                                    h1 = 0.0
                                    for i in range(n):
                                        h0 += section_normal[i] * (x_prev[i] - section_xsec[i])
                                        h1 += section_normal[i] * (x[i] - section_xsec[i])

                                if not (h0 == 0.0 and h1 == 0.0):
                                    if (h0 == 0.0) or (h1 == 0.0) or (h0 * h1 < 0.0):
                                        denom = h1 - h0
                                        if denom == 0.0:
                                            alpha = 0.0
                                        else:
                                            alpha = (0.0 - h0) / denom
                                            if alpha < 0.0:
                                                alpha = 0.0
                                            if alpha > 1.0:
                                                alpha = 1.0
                                        t_ev = t_prev + alpha * (t - t_prev)
                                        for i in range(n):
                                            x_stage[i] = x_prev[i] + alpha * (x[i] - x_prev[i])
                                        if section_kind == 2 and rho_max > 0.0 and rho_max == rho_max:
                                            dist2 = 0.0
                                            for i in range(n):
                                                dx = x_stage[i] - section_xsec[i]
                                                dist2 += dx * dx
                                            if dist2 > rho_max * rho_max:
                                                pass
                                            else:
                                                event_info[0] = 2.0
                                        else:
                                            event_info[0] = 2.0

                                        if event_info[0] == 2.0:
                                            out_t[m - 1] = t_ev
                                            for i in range(n):
                                                out_x[m - 1, i] = x_stage[i]
                                            if section_kind == 1:
                                                if n >= 2:
                                                    vx = x_stage[0] - section_center[0]
                                                    vy = x_stage[1] - section_center[1]
                                                    event_info[1] = math.atan2(vy, vx)
                                                else:
                                                    event_info[1] = 0.0
                                            else:
                                                sval = 0.0
                                                for i in range(n):
                                                    sval += section_tau[i] * (x_stage[i] - section_center[i])
                                                event_info[1] = sval
                                            event_info[2] = h0
                                            event_info[3] = h1
                                            return 1, m

                    if (direction > 0.0 and t >= t1) or (direction < 0.0 and t <= t1):
                        return 0, m

                if en == 0.0:
                    fac = 2.0
                else:
                    fac = safety * en ** (-0.2)
                    if fac < 0.2:
                        fac = 0.2
                    if fac > 5.0:
                        fac = 5.0

                h_new = h * fac
                h_mag = dt_max
                if abs(h_new) < h_mag:
                    h_mag = abs(h_new)
                if min_step > h_mag:
                    h_mag = min_step
                h = direction * h_mag

                if abs(h) <= min_step * 1.001 and (not accepted):
                    return 2, m

            return 3, m

        _rk45_integrate_numba = _rk45_integrate_numba_impl

    except ImportError:  # pragma: no cover
        pass


def _rk45_integrate(
    rhs_eval: Callable[[float, np.ndarray, np.ndarray], None],
    params_vec: np.ndarray,
    runtime_ws,
    update_aux_fn: Callable,
    do_aux_pre: bool,
    do_aux_post: bool,
    x0: np.ndarray,
    t0: float,
    t1: float,
    cfg: HeteroclinicRK45Config,
    *,
    blow_fn: Callable[[float, np.ndarray], bool] | None = None,
    event_fn: Callable[[float, np.ndarray, float, np.ndarray], tuple[bool, float, np.ndarray, dict[str, object]]] | None = None,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, object] | None]:
    x = np.array(x0, copy=True)
    t = float(t0)

    direction = 1.0 if t1 >= t0 else -1.0
    h = float(cfg.dt0) * direction

    ts: list[float] = [t]
    xs: list[np.ndarray] = [x.copy()]

    n = x.size
    dtype = x.dtype
    k1 = np.empty(n, dtype=dtype)
    k2 = np.empty(n, dtype=dtype)
    k3 = np.empty(n, dtype=dtype)
    k4 = np.empty(n, dtype=dtype)
    k5 = np.empty(n, dtype=dtype)
    k6 = np.empty(n, dtype=dtype)
    k7 = np.empty(n, dtype=dtype)
    x_stage = np.empty(n, dtype=dtype)
    x5 = np.empty(n, dtype=dtype)
    x4 = np.empty(n, dtype=dtype)

    if do_aux_pre:
        update_aux_fn(t, x, params_vec, runtime_ws.aux_values, runtime_ws)
    stop_mask = int(runtime_ws.stop_phase_mask[0]) if runtime_ws.stop_phase_mask.size else 0
    if (stop_mask & 1) != 0 and runtime_ws.stop_flag.size:
        if runtime_ws.stop_flag[0] != 0:
            return "stop", np.array(ts), np.vstack(xs), None

    def err_norm(e: np.ndarray, x_ref: np.ndarray) -> float:
        scale = cfg.atol + cfg.rtol * np.maximum(np.abs(x_ref), np.abs(x_ref + e))
        return float(np.sqrt(np.mean((e / scale) ** 2)))

    for _ in range(int(cfg.max_steps)):
        if blow_fn is not None and blow_fn(t, x):
            return "blow", np.array(ts), np.vstack(xs), None

        if direction > 0 and t + h > t1:
            h = t1 - t
        if direction < 0 and t + h < t1:
            h = t1 - t

        rhs_eval(t, x, k1)
        x_stage[:] = x + h * (0.2 * k1)
        rhs_eval(t + 0.2 * h, x_stage, k2)

        x_stage[:] = x + h * (0.075 * k1 + 0.225 * k2)
        rhs_eval(t + 0.3 * h, x_stage, k3)

        x_stage[:] = x + h * (0.9777777777777777 * k1 - 3.7333333333333334 * k2 + 3.5555555555555554 * k3)
        rhs_eval(t + 0.8 * h, x_stage, k4)

        x_stage[:] = x + h * (
            2.9525986892242035 * k1
            - 11.595793324188385 * k2
            + 9.822892851699436 * k3
            - 0.2908093278463649 * k4
        )
        rhs_eval(t + (8.0 / 9.0) * h, x_stage, k5)

        x_stage[:] = x + h * (
            2.8462752525252526 * k1
            - 10.757575757575758 * k2
            + 8.906422717743472 * k3
            + 0.2784090909090909 * k4
            - 0.2735313036020583 * k5
        )
        rhs_eval(t + h, x_stage, k6)

        x5[:] = x + h * (
            0.09114583333333333 * k1
            + 0.44923629829290207 * k3
            + 0.6510416666666666 * k4
            - 0.322376179245283 * k5
            + 0.13095238095238096 * k6
        )
        rhs_eval(t + h, x5, k7)

        x4[:] = x + h * (
            0.08991319444444444 * k1
            + 0.4534890685834082 * k3
            + 0.6140625 * k4
            - 0.2715123820754717 * k5
            + 0.08904761904761905 * k6
            + 0.025 * k7
        )

        e = x5 - x4
        en = err_norm(e, x)

        if not np.isfinite(en):
            return "blow", np.array(ts), np.vstack(xs), None

        accepted = en <= 1.0
        t_prev = t
        x_prev = x.copy()

        if accepted:
            t = t + h
            x[:] = x5
            ts.append(float(t))
            xs.append(x.copy())

            if do_aux_post:
                update_aux_fn(t, x, params_vec, runtime_ws.aux_values, runtime_ws)
            _update_lag_state(runtime_ws, x)

            if (stop_mask & 2) != 0 and runtime_ws.stop_flag.size:
                if runtime_ws.stop_flag[0] != 0:
                    return "stop", np.array(ts), np.vstack(xs), None

            if event_fn is not None:
                hit, t_ev, x_ev, evinfo = event_fn(float(t_prev), x_prev, float(t), x)
                if hit:
                    ts_out = np.array(ts, float)
                    xs_out = np.vstack(xs)
                    ts_out[-1] = float(t_ev)
                    xs_out[-1, :] = np.array(x_ev, float)
                    return "event", ts_out, xs_out, evinfo

            if (direction > 0 and t >= t1) or (direction < 0 and t <= t1):
                return "ok", np.array(ts), np.vstack(xs), None

        if en == 0.0:
            fac = 2.0
        else:
            fac = cfg.safety * en ** (-0.2)
            fac = min(5.0, max(0.2, fac))

        h_new = h * fac
        h_mag = min(cfg.dt_max, max(cfg.min_step, abs(h_new)))
        h = direction * h_mag

        if abs(h) <= cfg.min_step * 1.001 and (not accepted):
            return "blow", np.array(ts), np.vstack(xs), None

    return "max_steps", np.array(ts), np.vstack(xs), None


def _rk45_integrate_numba_wrapper(
    rhs_fn: Callable,
    params_vec: np.ndarray,
    runtime_ws,
    update_aux_fn: Callable,
    do_aux_pre: bool,
    do_aux_post: bool,
    x0: np.ndarray,
    t0: float,
    t1: float,
    cfg: HeteroclinicRK45Config,
    *,
    sign_factor: float,
    x_eq: np.ndarray,
    r_blow: float,
    wmin: np.ndarray,
    wmax: np.ndarray,
    has_window: bool,
    t_min_event: float,
    require_leave_before_event: bool,
    r_leave: float,
    target_center: np.ndarray,
    target_r: float,
    enable_target: bool,
    section_kind: int,
    section_center: np.ndarray,
    section_normal: np.ndarray,
    section_tau: np.ndarray,
    section_radius: float,
    section_xsec: np.ndarray,
    rho_max: float,
    enable_section: bool,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, object] | None]:
    if _rk45_integrate_numba is None:
        raise RuntimeError("Numba RK45 integrator is unavailable.")

    n = int(x0.size)
    max_steps = int(cfg.max_steps)
    out_t = np.empty((max_steps + 1,), dtype=float)
    out_x = np.empty((max_steps + 1, n), dtype=x0.dtype)
    event_info = np.empty((8,), dtype=float)

    status_code, n_steps = _rk45_integrate_numba(
        rhs_fn,
        params_vec,
        runtime_ws,
        update_aux_fn,
        bool(do_aux_pre),
        bool(do_aux_post),
        x0,
        float(t0),
        float(t1),
        float(cfg.dt0),
        float(cfg.min_step),
        float(cfg.dt_max),
        float(cfg.atol),
        float(cfg.rtol),
        float(cfg.safety),
        int(cfg.max_steps),
        float(sign_factor),
        x_eq,
        float(r_blow),
        wmin,
        wmax,
        bool(has_window),
        float(t_min_event),
        bool(require_leave_before_event),
        float(r_leave),
        target_center,
        float(target_r),
        bool(enable_target),
        int(section_kind),
        section_center,
        section_normal,
        section_tau,
        float(section_radius),
        section_xsec,
        float(rho_max),
        bool(enable_section),
        out_t,
        out_x,
        event_info,
    )

    status_map = {
        0: "ok",
        1: "event",
        2: "blow",
        3: "max_steps",
        4: "stop",
    }
    status = status_map.get(int(status_code), "max_steps")

    ts = np.array(out_t[:n_steps], copy=True)
    xs = np.array(out_x[:n_steps], copy=True)

    evinfo: dict[str, object] | None = None
    if status == "event":
        event_code = int(round(float(event_info[0])))
        if event_code == 1:
            evinfo = {
                "kind": "target_ball",
                "target_r": float(target_r),
                "alpha": float(event_info[1]),
                "d_event": float(event_info[2]),
                "d_prev": float(event_info[3]),
                "d_curr": float(event_info[4]),
            }
        elif event_code == 2:
            evinfo = {
                "kind": "section",
                "h_prev": float(event_info[2]),
                "h_curr": float(event_info[3]),
                "s": float(event_info[1]),
            }

    return status, ts, xs, evinfo


def _trace_1d_manifold(
    ctx: _HeteroclinicContext,
    params_vec: np.ndarray,
    *,
    param_value: float,
    x_eq: np.ndarray | None = None,
    x_eq_seed: np.ndarray | None = None,
    x_eq_prev: np.ndarray | None = None,
    kind: Literal["unstable", "stable"],
    sign: Literal[+1, -1] = +1,
    cfg: HeteroclinicBranchConfig,
    section: _Section | None = None,
    stop_on_section: bool = False,
    target: np.ndarray | None = None,
    target_r: float = 0.0,
    stop_on_target: bool = False,
) -> _TraceResult:
    meta: dict[str, object] = {"param_value": float(param_value), "kind": kind, "sign": int(sign)}
    diag: dict[str, object] = {}

    if kind not in ("unstable", "stable"):
        raise ValueError("kind must be 'unstable' or 'stable'")
    if cfg.eps_mode not in ("leave", "fixed"):
        raise ValueError("eps_mode must be 'leave' or 'fixed'")

    def rhs_eval_reset(x: np.ndarray, out: np.ndarray) -> None:
        ctx.prep_ws(x)
        ctx.rhs_fn(ctx.t_eval, x, out, params_vec, ctx.runtime_ws)

    def jac_eval(x: np.ndarray) -> np.ndarray:
        if cfg.jac == "fd" or (cfg.jac == "auto" and ctx.jac_fn is None):
            fx = np.empty((x.size,), dtype=ctx.model.dtype)
            rhs_eval_reset(x, fx)
            J = np.zeros((x.size, x.size), dtype=float)
            for j in range(x.size):
                step = cfg.fd_eps * (1.0 + abs(float(x[j])))
                if step == 0.0:
                    step = cfg.fd_eps
                x_step = np.array(x, copy=True)
                x_step[j] += step
                f_step = np.empty((x.size,), dtype=ctx.model.dtype)
                rhs_eval_reset(x_step, f_step)
                J[:, j] = (f_step - fx) / step
            return J

        if cfg.jac == "analytic" and ctx.jac_fn is None:
            raise ValueError("jac='analytic' requires a model Jacobian.")
        if ctx.jac_fn is None:
            raise ValueError("Jacobian is not available (jac='auto' found none).")

        jac_out = np.zeros((x.size, x.size), dtype=ctx.model.dtype)
        ctx.prep_ws(x)
        ctx.jac_fn(ctx.t_eval, x, params_vec, jac_out, ctx.runtime_ws)
        return np.array(jac_out, copy=True)

    if x_eq is None:
        if x_eq_seed is None:
            raise ValueError("Provide x_eq or x_eq_seed.")
        ok_eq, x_eq_star, eqinfo = _solve_equilibrium_locked(
            rhs_eval_reset,
            jac_eval,
            np.array(x_eq_seed, float),
            x_prev=(None if x_eq_prev is None else np.array(x_eq_prev, float)),
            eq_tol=float(cfg.eq_tol),
            eq_max_iter=int(cfg.eq_max_iter),
            eq_track_max_dist=cfg.eq_track_max_dist,
            r_leave_for_default=float(cfg.r_leave),
        )
        diag["eq"] = eqinfo
        if not ok_eq:
            status = "eq_jump" if eqinfo.get("fail_mode") == "eq_jump" else "eq_fail"
            return _TraceResult(
                points=np.array([x_eq_star], float),
                t=np.array([0.0], float),
                status=status,
                event=None,
                meta={"diag": diag, **meta},
            )
        x_eq = np.array(x_eq_star, dtype=ctx.model.dtype, copy=True)
    else:
        x_eq = np.array(x_eq, dtype=ctx.model.dtype, copy=True)

    meta["x_eq"] = np.array(x_eq, float)

    J = jac_eval(x_eq)
    ok_e, ed, einfo = _eig_1d_data_ode(
        J,
        kind=kind,
        real_tol=cfg.eig_real_tol,
        imag_tol=cfg.eig_imag_tol,
        strict_1d=cfg.strict_1d,
    )
    diag["eig"] = einfo
    if not ok_e or ed is None:
        return _TraceResult(
            points=np.array([x_eq], float),
            t=np.array([0.0], float),
            status="eig_fail",
            event=None,
            meta={"diag": diag, **meta},
        )

    lam, v_raw, _ = ed
    lam_val = float(np.real(lam))
    vn = float(np.linalg.norm(v_raw))
    if not np.isfinite(vn) or vn <= 1e-300:
        diag["fail"] = "eigenvector_norm_bad"
        return _TraceResult(
            points=np.array([x_eq], float),
            t=np.array([0.0], float),
            status="eig_fail",
            event=None,
            meta={"diag": diag, **meta},
        )
    v_dir = np.array(v_raw, float) / vn

    growth_rate = lam_val if kind == "unstable" else -lam_val
    meta["lam"] = float(lam_val)
    meta["v_dir"] = np.array(v_dir, float)

    eps = float(cfg.eps) if cfg.eps_mode == "fixed" else _choose_eps_leave(
        r_leave=float(cfg.r_leave),
        rate=growth_rate,
        t_leave_target=float(cfg.t_leave_target),
        eps_min=float(cfg.eps_min),
        eps_max=float(cfg.eps_max),
        fallback=float(cfg.eps),
    )
    meta["eps"] = float(eps)
    x0 = np.array(x_eq + float(sign) * eps * v_dir, dtype=ctx.model.dtype)
    meta["x0"] = np.array(x0, float)

    wmin = None if cfg.window_min is None else np.array(cfg.window_min, float).reshape(-1,)
    wmax = None if cfg.window_max is None else np.array(cfg.window_max, float).reshape(-1,)
    if wmin is not None and wmin.shape != x_eq.shape:
        raise ValueError("window_min must have shape (n_state,)")
    if wmax is not None and wmax.shape != x_eq.shape:
        raise ValueError("window_max must have shape (n_state,)")

    def blow_fn(_t: float, x: np.ndarray) -> bool:
        if not np.all(np.isfinite(x)):
            return True
        if float(np.linalg.norm(np.array(x, float) - x_eq)) > float(cfg.r_blow):
            return True
        if not _in_window(np.array(x, float), wmin, wmax):
            return True
        return False

    left = False

    def event_fn(
        t_prev: float,
        x_prev: np.ndarray,
        t_curr: float,
        x_curr: np.ndarray,
    ) -> tuple[bool, float, np.ndarray, dict[str, object]]:
        nonlocal left
        info: dict[str, object] = {}

        if float(np.linalg.norm(np.array(x_curr, float) - x_eq)) >= float(cfg.r_leave):
            left = True

        if t_curr < float(cfg.t_min_event):
            return False, t_curr, x_curr, info
        if cfg.require_leave_before_event and (not left):
            return False, t_curr, x_curr, info

        if stop_on_target and target is not None and float(target_r) > 0.0:
            x0p = np.array(x_prev, float)
            x1p = np.array(x_curr, float)
            c = np.array(target, float)
            rr = float(target_r)
            alpha = _segment_ball_hit_alpha(x0p, x1p, c, rr)
            if alpha is not None:
                t_ev = float(t_prev + alpha * (t_curr - t_prev))
                x_ev = x0p + alpha * (x1p - x0p)
                d_ev = float(np.linalg.norm(x_ev - c))
                info.update({
                    "kind": "target_ball",
                    "target_r": rr,
                    "alpha": float(alpha),
                    "d_event": d_ev,
                    "d_prev": float(np.linalg.norm(x0p - c)),
                    "d_curr": float(np.linalg.norm(x1p - c)),
                })
                return True, t_ev, np.array(x_ev, float), info

        if stop_on_section and section is not None:
            h0 = float(section.h(np.array(x_prev, float)))
            h1 = float(section.h(np.array(x_curr, float)))
            if (h0 == 0.0) and (h1 == 0.0):
                return False, t_curr, x_curr, info
            if (h0 == 0.0) or (h0 * h1 < 0.0) or (h1 == 0.0):
                denom = (h1 - h0)
                if denom == 0.0:
                    alpha = 0.0
                else:
                    alpha = (0.0 - h0) / denom
                    alpha = min(1.0, max(0.0, alpha))
                t_ev = float(t_prev + alpha * (t_curr - t_prev))
                x_ev = np.array(x_prev, float) + alpha * (np.array(x_curr, float) - np.array(x_prev, float))
                if not section.qualify(x_ev):
                    return False, t_curr, x_curr, info

                info.update({
                    "kind": "section",
                    "h_prev": h0,
                    "h_curr": h1,
                    "s": float(section.s(x_ev)),
                })
                return True, t_ev, x_ev, info

        return False, t_curr, x_curr, info

    sign_factor = -1.0 if kind == "stable" else 1.0
    use_numba = ctx.use_jit and _rk45_integrate_numba is not None
    update_aux_fn = ctx.update_aux_fn
    if use_numba:
        if (ctx.do_aux_pre or ctx.do_aux_post):
            if not _is_numba_dispatcher(ctx.update_aux_fn):
                use_numba = False
        else:
            if _update_aux_noop_numba is not None:
                update_aux_fn = _update_aux_noop_numba
            else:
                use_numba = False

    n_state = int(x_eq.size)
    has_window = wmin is not None and wmax is not None
    wmin_arr = np.array(wmin if wmin is not None else np.zeros(n_state), float).reshape(-1,)
    wmax_arr = np.array(wmax if wmax is not None else np.zeros(n_state), float).reshape(-1,)
    if use_numba and has_window:
        if wmin_arr.size != n_state or wmax_arr.size != n_state:
            use_numba = False

    target_center = np.array(target, float).reshape(-1,) if target is not None else np.zeros(n_state, float)
    enable_target = bool(stop_on_target and target is not None and float(target_r) > 0.0)
    if use_numba and target is not None and target_center.size != n_state:
        use_numba = False

    section_kind = 0
    section_center = np.zeros(n_state, float)
    section_normal = np.zeros(n_state, float)
    section_tau = np.zeros(n_state, float)
    section_radius = 0.0
    section_xsec = np.zeros(n_state, float)
    rho_max = -1.0

    if use_numba and stop_on_section and section is not None:
        if section.kind == "circle":
            section_kind = 1
            sec_center = np.array(section.meta.get("center", section.x_ref), float).reshape(-1,)
            sec_radius = section.meta.get("radius", None)
            if sec_center.size != n_state or sec_radius is None or (not np.isfinite(float(sec_radius))) or float(sec_radius) <= 0.0:
                use_numba = False
            else:
                section_center = sec_center
                section_radius = float(sec_radius)
        elif section.kind == "plane":
            section_kind = 2
            sec_center = np.array(section.meta.get("center", section.x_ref), float).reshape(-1,)
            sec_normal = section.meta.get("normal", None)
            sec_tau = np.array(section.meta.get("tau", section.tau), float).reshape(-1,)
            sec_xsec = section.meta.get("x_sec", None)
            sec_rho = section.meta.get("rho_max", None)
            if sec_normal is None or sec_xsec is None:
                use_numba = False
            else:
                section_center = sec_center
                section_normal = np.array(sec_normal, float).reshape(-1,)
                section_tau = sec_tau
                section_xsec = np.array(sec_xsec, float).reshape(-1,)
                rho_max = -1.0 if sec_rho is None else float(sec_rho)
                if (
                    section_center.size != n_state
                    or section_normal.size != n_state
                    or section_tau.size != n_state
                    or section_xsec.size != n_state
                ):
                    use_numba = False
        else:
            use_numba = False

    ctx.prep_ws(x0)
    if use_numba:
        status, ts, xs, evinfo = _rk45_integrate_numba_wrapper(
            ctx.rhs_fn,
            params_vec,
            ctx.runtime_ws,
            update_aux_fn,
            ctx.do_aux_pre,
            ctx.do_aux_post,
            x0,
            float(ctx.t_eval),
            float(ctx.t_eval) + float(cfg.t_max),
            cfg.rk,
            sign_factor=sign_factor,
            x_eq=x_eq,
            r_blow=float(cfg.r_blow),
            wmin=wmin_arr,
            wmax=wmax_arr,
            has_window=bool(has_window),
            t_min_event=float(cfg.t_min_event),
            require_leave_before_event=bool(cfg.require_leave_before_event),
            r_leave=float(cfg.r_leave),
            target_center=target_center,
            target_r=float(target_r),
            enable_target=bool(enable_target),
            section_kind=int(section_kind),
            section_center=section_center,
            section_normal=section_normal,
            section_tau=section_tau,
            section_radius=float(section_radius),
            section_xsec=section_xsec,
            rho_max=float(rho_max),
            enable_section=bool(stop_on_section and section_kind != 0),
        )
    else:
        def rhs_eval(t: float, x: np.ndarray, out: np.ndarray) -> None:
            ctx.rhs_fn(t, x, out, params_vec, ctx.runtime_ws)
            if kind == "stable":
                out *= -1.0

        status, ts, xs, evinfo = _rk45_integrate(
            rhs_eval,
            params_vec,
            ctx.runtime_ws,
            ctx.update_aux_fn,
            ctx.do_aux_pre,
            ctx.do_aux_post,
            x0,
            float(ctx.t_eval),
            float(ctx.t_eval) + float(cfg.t_max),
            cfg.rk,
            blow_fn=blow_fn,
            event_fn=(event_fn if (stop_on_section or stop_on_target) else None),
        )

    if status == "blow":
        x_last = xs[-1]
        if (wmin is not None and wmax is not None) and (not _in_window(x_last, wmin, wmax)):
            status = "window"
        else:
            status = "blow"

    s_len = _polyline_arclength(xs)
    meta["arclength"] = float(s_len)
    if s_len > float(cfg.s_max) and status in ("ok", "max_steps"):
        status = "smax"

    event_obj: HeteroclinicTraceEvent | None = None
    if status == "event" and evinfo is not None:
        event_obj = HeteroclinicTraceEvent(
            kind=str(evinfo.get("kind", "event")),
            t=float(ts[-1]),
            x=np.array(xs[-1], float),
            info=dict(evinfo),
        )

    meta["diag"] = diag
    return _TraceResult(points=np.array(xs, float), t=np.array(ts, float), status=str(status), event=event_obj, meta=meta)


def _unwrap_angle(theta: float, theta_ref: float | None) -> float:
    theta = float(theta)
    if theta_ref is None or (not np.isfinite(theta_ref)):
        return theta
    k = int(round((float(theta_ref) - theta) / (2.0 * math.pi)))
    return theta + (2.0 * math.pi) * k


def _scan_grid_centered(param_min: float, param_max: float, param_init: float, n: int) -> tuple[np.ndarray, int]:
    grid = np.linspace(float(param_min), float(param_max), int(n))
    idx0 = int(np.argmin(np.abs(grid - float(param_init))))
    return grid, idx0


def _apply_unwrap_inplace(r: HeteroclinicMissResult2D, ref: HeteroclinicMissResult2D | None) -> None:
    if not r.qualified or (not np.isfinite(r.theta_u_raw)) or (not np.isfinite(r.theta_s_raw)):
        return
    ref_u = None if ref is None else float(ref.theta_u)
    ref_s = None if ref is None else float(ref.theta_s)
    r.theta_u = _unwrap_angle(float(r.theta_u_raw), ref_u)
    r.theta_s = _unwrap_angle(float(r.theta_s_raw), ref_s)
    r.g = float(r.theta_u - r.theta_s)


def _unwrap_qualified_sorted(qual_sorted: list[HeteroclinicMissResult2D]) -> None:
    prev: HeteroclinicMissResult2D | None = None
    for r in qual_sorted:
        _apply_unwrap_inplace(r, prev)
        prev = r


def _compute_circle_radius(cfg: HeteroclinicFinderConfig2D) -> float:
    base = float(cfg.trace_s.r_leave)
    r = float(cfg.r_sec) if cfg.r_sec is not None else float(cfg.r_sec_mult) * base
    rmin = float(cfg.r_sec_min_mult) * base
    return float(max(r, rmin))


def _get_r_sec_from_result(cfg: HeteroclinicFinderConfig2D, r: HeteroclinicMissResult2D) -> float:
    try:
        rr = float(r.diag.get("circle", {}).get("r_sec", float("nan")))
    except Exception:
        rr = float("nan")
    if np.isfinite(rr) and rr > 0.0:
        return rr
    return float(_compute_circle_radius(cfg))


def _gap_tol_eff(cfg: HeteroclinicFinderConfig2D, r: HeteroclinicMissResult2D) -> float:
    base = float(cfg.gap_tol)
    if not (r.qualified and np.isfinite(r.q)):
        return base
    r_sec = _get_r_sec_from_result(cfg, r)
    if (not np.isfinite(r_sec)) or r_sec <= 0.0:
        return base
    return float(max(base, float(cfg.gap_fac) * float(r.q) / r_sec))


def _is_success(cfg: HeteroclinicFinderConfig2D, r: HeteroclinicMissResult2D) -> bool:
    if not (r.qualified and np.isfinite(r.q) and np.isfinite(r.g)):
        return False
    if float(r.q) > float(cfg.x_tol):
        return False
    if abs(float(r.g)) > _gap_tol_eff(cfg, r):
        return False
    return True


def _compute_plane_r_sec(cfg: HeteroclinicFinderConfigND) -> float:
    base = float(cfg.trace_s.r_leave)
    r = float(cfg.r_sec) if cfg.r_sec is not None else float(cfg.r_sec_mult) * base
    rmin = float(cfg.r_sec_min_mult) * base
    return float(max(r, rmin))


def _gap_tol_eff_nd(cfg: HeteroclinicFinderConfigND, r: HeteroclinicMissResultND) -> float:
    base = float(cfg.gap_tol)
    if not (r.qualified and np.isfinite(r.q)):
        return base
    return float(max(base, float(cfg.gap_fac) * float(r.q)))


def _is_success_nd(cfg: HeteroclinicFinderConfigND, r: HeteroclinicMissResultND) -> bool:
    if not (r.qualified and np.isfinite(r.q) and np.isfinite(r.g)):
        return False
    if float(r.q) > float(cfg.x_tol):
        return False
    if abs(float(r.g)) > _gap_tol_eff_nd(cfg, r):
        return False
    return True


def _heteroclinic_miss_circleB_2d(
    ctx: _HeteroclinicContext,
    params_vec: np.ndarray,
    param_value: float,
    *,
    source_eq_guess: np.ndarray,
    target_eq_guess: np.ndarray,
    source_eq_prev: np.ndarray | None,
    target_eq_prev: np.ndarray | None,
    sign_u: int,
    sign_s: int,
    cfg: HeteroclinicFinderConfig2D,
) -> HeteroclinicMissResult2D:
    diag: dict[str, object] = {"param_value": float(param_value)}

    def rhs_eval_reset(x: np.ndarray, out: np.ndarray) -> None:
        ctx.prep_ws(x)
        ctx.rhs_fn(ctx.t_eval, x, out, params_vec, ctx.runtime_ws)

    def jac_eval(x: np.ndarray, cfg_branch: HeteroclinicBranchConfig) -> np.ndarray:
        if cfg_branch.jac == "fd" or (cfg_branch.jac == "auto" and ctx.jac_fn is None):
            fx = np.empty((x.size,), dtype=ctx.model.dtype)
            rhs_eval_reset(x, fx)
            J = np.zeros((x.size, x.size), dtype=float)
            for j in range(x.size):
                step = cfg_branch.fd_eps * (1.0 + abs(float(x[j])))
                if step == 0.0:
                    step = cfg_branch.fd_eps
                x_step = np.array(x, copy=True)
                x_step[j] += step
                f_step = np.empty((x.size,), dtype=ctx.model.dtype)
                rhs_eval_reset(x_step, f_step)
                J[:, j] = (f_step - fx) / step
            return J

        if cfg_branch.jac == "analytic" and ctx.jac_fn is None:
            raise ValueError("jac='analytic' requires a model Jacobian.")
        if ctx.jac_fn is None:
            raise ValueError("Jacobian is not available (jac='auto' found none).")

        jac_out = np.zeros((x.size, x.size), dtype=ctx.model.dtype)
        ctx.prep_ws(x)
        ctx.jac_fn(ctx.t_eval, x, params_vec, jac_out, ctx.runtime_ws)
        return np.array(jac_out, copy=True)

    okA, source_eq, infoA = _solve_equilibrium_locked(
        rhs_eval_reset,
        lambda x: jac_eval(x, cfg.trace_u),
        np.array(source_eq_guess, float),
        x_prev=(None if source_eq_prev is None else np.array(source_eq_prev, float)),
        eq_tol=float(cfg.eq_tol),
        eq_max_iter=int(cfg.eq_max_iter),
        eq_track_max_dist=cfg.eq_track_max_dist,
        r_leave_for_default=float(cfg.trace_u.r_leave),
    )
    okB, target_eq, infoB = _solve_equilibrium_locked(
        rhs_eval_reset,
        lambda x: jac_eval(x, cfg.trace_s),
        np.array(target_eq_guess, float),
        x_prev=(None if target_eq_prev is None else np.array(target_eq_prev, float)),
        eq_tol=float(cfg.eq_tol),
        eq_max_iter=int(cfg.eq_max_iter),
        eq_track_max_dist=cfg.eq_track_max_dist,
        r_leave_for_default=float(cfg.trace_s.r_leave),
    )
    diag["eqA"] = infoA
    diag["eqB"] = infoB

    if not okA or not okB:
        status = "eq_fail"
        if infoA.get("fail_mode") == "eq_jump" or infoB.get("fail_mode") == "eq_jump":
            status = "eq_jump"
        zA = np.array(source_eq, float)
        zB = np.array(target_eq, float)
        return HeteroclinicMissResult2D(
            qualified=False,
            param_value=float(param_value),
            source_eq=zA,
            target_eq=zB,
            sign_u=int(sign_u),
            sign_s=int(sign_s),
            x_u_cross=zA.copy(),
            x_s_cross=zB.copy(),
            theta_u_raw=float("nan"),
            theta_s_raw=float("nan"),
            theta_u=float("nan"),
            theta_s=float("nan"),
            g=float("nan"),
            q=float("inf"),
            status_u=status,
            status_s=status,
            status=status,
            diag=diag,
        )

    target_eq = np.array(target_eq, float).reshape(2,)
    r_sec = _compute_circle_radius(cfg)
    circle = _make_circle_section_2d(center=target_eq, radius=r_sec)
    diag["circle"] = {"r_sec": float(r_sec), "target_eq": target_eq.copy()}

    ru = _trace_1d_manifold(
        ctx,
        params_vec,
        param_value=float(param_value),
        x_eq=np.array(source_eq, float),
        kind="unstable",
        sign=+1 if int(sign_u) >= 0 else -1,
        cfg=cfg.trace_u,
        section=circle,
        stop_on_section=True,
    )

    rs = _trace_1d_manifold(
        ctx,
        params_vec,
        param_value=float(param_value),
        x_eq=np.array(target_eq, float),
        kind="stable",
        sign=+1 if int(sign_s) >= 0 else -1,
        cfg=cfg.trace_s,
        section=circle,
        stop_on_section=True,
    )

    diag["trace_u"] = {"status": ru.status, "event": None if ru.event is None else {"kind": ru.event.kind, "s": ru.event.info.get("s")}}
    diag["trace_s"] = {"status": rs.status, "event": None if rs.event is None else {"kind": rs.event.kind, "s": rs.event.info.get("s")}}

    if ru.event is None or rs.event is None or ru.event.kind != "section" or rs.event.kind != "section":
        status = "no_cross"
        if ru.status in ("window", "blow", "smax", "max_steps", "stop"):
            status = f"u_{ru.status}"
        if rs.status in ("window", "blow", "smax", "max_steps", "stop"):
            status = f"s_{rs.status}"
        return HeteroclinicMissResult2D(
            qualified=False,
            param_value=float(param_value),
            source_eq=np.array(source_eq, float),
            target_eq=np.array(target_eq, float),
            sign_u=int(sign_u),
            sign_s=int(sign_s),
            x_u_cross=ru.points[-1].copy(),
            x_s_cross=rs.points[-1].copy(),
            theta_u_raw=float("nan"),
            theta_s_raw=float("nan"),
            theta_u=float("nan"),
            theta_s=float("nan"),
            g=float("nan"),
            q=float("inf"),
            status_u=str(ru.status),
            status_s=str(rs.status),
            status=status,
            diag=diag,
        )

    x_u = np.array(ru.event.x, float).reshape(2,)
    x_s = np.array(rs.event.x, float).reshape(2,)

    theta_u_raw = float(circle.s(x_u))
    theta_s_raw = float(circle.s(x_s))
    g_raw = float(theta_u_raw - theta_s_raw)
    q = float(np.linalg.norm(x_u - x_s))

    return HeteroclinicMissResult2D(
        qualified=True,
        param_value=float(param_value),
        source_eq=np.array(source_eq, float),
        target_eq=np.array(target_eq, float),
        sign_u=int(sign_u),
        sign_s=int(sign_s),
        x_u_cross=x_u,
        x_s_cross=x_s,
        theta_u_raw=theta_u_raw,
        theta_s_raw=theta_s_raw,
        theta_u=theta_u_raw,
        theta_s=theta_s_raw,
        g=g_raw,
        q=q,
        status_u=str(ru.status),
        status_s=str(rs.status),
        status="ok",
        diag=diag,
    )


def _heteroclinic_miss_planeB_nd(
    ctx: _HeteroclinicContext,
    params_vec: np.ndarray,
    param_value: float,
    *,
    source_eq_guess: np.ndarray,
    target_eq_guess: np.ndarray,
    source_eq_prev: np.ndarray | None,
    target_eq_prev: np.ndarray | None,
    sign_u: int,
    sign_s: int,
    cfg: HeteroclinicFinderConfigND,
    nB_ref: np.ndarray | None = None,
    tau_ref: np.ndarray | None = None,
) -> HeteroclinicMissResultND:
    diag: dict[str, object] = {"param_value": float(param_value)}

    def rhs_eval_reset(x: np.ndarray, out: np.ndarray) -> None:
        ctx.prep_ws(x)
        ctx.rhs_fn(ctx.t_eval, x, out, params_vec, ctx.runtime_ws)

    def jac_eval(x: np.ndarray, cfg_branch: HeteroclinicBranchConfig) -> np.ndarray:
        if cfg_branch.jac == "fd" or (cfg_branch.jac == "auto" and ctx.jac_fn is None):
            fx = np.empty((x.size,), dtype=ctx.model.dtype)
            rhs_eval_reset(x, fx)
            J = np.zeros((x.size, x.size), dtype=float)
            for j in range(x.size):
                step = cfg_branch.fd_eps * (1.0 + abs(float(x[j])))
                if step == 0.0:
                    step = cfg_branch.fd_eps
                x_step = np.array(x, copy=True)
                x_step[j] += step
                f_step = np.empty((x.size,), dtype=ctx.model.dtype)
                rhs_eval_reset(x_step, f_step)
                J[:, j] = (f_step - fx) / step
            return J

        if cfg_branch.jac == "analytic" and ctx.jac_fn is None:
            raise ValueError("jac='analytic' requires a model Jacobian.")
        if ctx.jac_fn is None:
            raise ValueError("Jacobian is not available (jac='auto' found none).")

        jac_out = np.zeros((x.size, x.size), dtype=ctx.model.dtype)
        ctx.prep_ws(x)
        ctx.jac_fn(ctx.t_eval, x, params_vec, jac_out, ctx.runtime_ws)
        return np.array(jac_out, copy=True)

    okA, source_eq, infoA = _solve_equilibrium_locked(
        rhs_eval_reset,
        lambda x: jac_eval(x, cfg.trace_u),
        np.array(source_eq_guess, float),
        x_prev=(None if source_eq_prev is None else np.array(source_eq_prev, float)),
        eq_tol=float(cfg.eq_tol),
        eq_max_iter=int(cfg.eq_max_iter),
        eq_track_max_dist=cfg.eq_track_max_dist,
        r_leave_for_default=float(cfg.trace_u.r_leave),
    )
    okB, target_eq, infoB = _solve_equilibrium_locked(
        rhs_eval_reset,
        lambda x: jac_eval(x, cfg.trace_s),
        np.array(target_eq_guess, float),
        x_prev=(None if target_eq_prev is None else np.array(target_eq_prev, float)),
        eq_tol=float(cfg.eq_tol),
        eq_max_iter=int(cfg.eq_max_iter),
        eq_track_max_dist=cfg.eq_track_max_dist,
        r_leave_for_default=float(cfg.trace_s.r_leave),
    )
    diag["eqA"] = infoA
    diag["eqB"] = infoB

    source_eq = np.array(source_eq, float).reshape(-1,)
    target_eq = np.array(target_eq, float).reshape(-1,)

    if not okA or not okB:
        status = "eq_fail"
        if infoA.get("fail_mode") == "eq_jump" or infoB.get("fail_mode") == "eq_jump":
            status = "eq_jump"
        nanv = np.full_like(target_eq, np.nan)
        return HeteroclinicMissResultND(
            qualified=False,
            param_value=float(param_value),
            source_eq=source_eq.copy(),
            target_eq=target_eq.copy(),
            sign_u=int(sign_u),
            sign_s=int(sign_s),
            n_B=nanv.copy(),
            tau=nanv.copy(),
            r_sec=float(_compute_plane_r_sec(cfg)),
            x_u_cross=source_eq.copy(),
            x_s_cross=target_eq.copy(),
            s_u=float("nan"),
            s_s=float("nan"),
            g=float("nan"),
            q=float("inf"),
            status_u=status,
            status_s=status,
            status=status,
            diag=diag,
        )

    okBs, edBs, einfoBs = _eig_1d_data_ode(
        jac_eval(target_eq, cfg.trace_s),
        kind="stable",
        real_tol=cfg.trace_s.eig_real_tol,
        imag_tol=cfg.trace_s.eig_imag_tol,
        strict_1d=cfg.trace_s.strict_1d,
    )
    diag["eigB_stable"] = einfoBs
    if not okBs or edBs is None:
        nanv = np.full_like(target_eq, np.nan)
        return HeteroclinicMissResultND(
            qualified=False,
            param_value=float(param_value),
            source_eq=source_eq.copy(),
            target_eq=target_eq.copy(),
            sign_u=int(sign_u),
            sign_s=int(sign_s),
            n_B=nanv.copy(),
            tau=nanv.copy(),
            r_sec=float(_compute_plane_r_sec(cfg)),
            x_u_cross=source_eq.copy(),
            x_s_cross=target_eq.copy(),
            s_u=float("nan"),
            s_s=float("nan"),
            g=float("nan"),
            q=float("inf"),
            status_u="eig_fail",
            status_s="eig_fail",
            status="eig_fail_B_stable",
            diag=diag,
        )

    okN, nB_raw, _ = _unit(edBs[1])
    if not okN:
        nanv = np.full_like(target_eq, np.nan)
        return HeteroclinicMissResultND(
            qualified=False,
            param_value=float(param_value),
            source_eq=source_eq.copy(),
            target_eq=target_eq.copy(),
            sign_u=int(sign_u),
            sign_s=int(sign_s),
            n_B=nanv.copy(),
            tau=nanv.copy(),
            r_sec=float(_compute_plane_r_sec(cfg)),
            x_u_cross=source_eq.copy(),
            x_s_cross=target_eq.copy(),
            s_u=float("nan"),
            s_s=float("nan"),
            g=float("nan"),
            q=float("inf"),
            status_u="eig_fail",
            status_s="eig_fail",
            status="eig_fail_nB",
            diag=diag,
        )

    sgn_s = +1 if int(sign_s) >= 0 else -1
    nB = float(sgn_s) * np.array(nB_raw, float)

    if nB_ref is not None:
        nB_ref_u = np.array(nB_ref, float).reshape(-1,)
        if nB_ref_u.size == nB.size and np.isfinite(float(nB @ nB_ref_u)) and float(nB @ nB_ref_u) < 0.0:
            nB = -nB

    n = int(target_eq.size)

    def _perp_2d(v: np.ndarray) -> np.ndarray:
        v = np.array(v, float).reshape(2,)
        return np.array([-float(v[1]), float(v[0])], float)

    if n == 2:
        tau0 = _perp_2d(nB)
        okT, tau, _ = _unit(tau0)
        if not okT:
            nanv = np.full_like(target_eq, np.nan)
            return HeteroclinicMissResultND(
                qualified=False,
                param_value=float(param_value),
                source_eq=source_eq.copy(),
                target_eq=target_eq.copy(),
                sign_u=int(sign_u),
                sign_s=int(sign_s),
                n_B=nB.copy(),
                tau=nanv.copy(),
                r_sec=float(_compute_plane_r_sec(cfg)),
                x_u_cross=source_eq.copy(),
                x_s_cross=target_eq.copy(),
                s_u=float("nan"),
                s_s=float("nan"),
                g=float("nan"),
                q=float("inf"),
                status_u="geom_fail",
                status_s="geom_fail",
                status="tau_degenerate_2d",
                diag=diag,
            )
    else:
        okAu, edAu, einfoAu = _eig_1d_data_ode(
            jac_eval(source_eq, cfg.trace_u),
            kind="unstable",
            real_tol=cfg.trace_u.eig_real_tol,
            imag_tol=cfg.trace_u.eig_imag_tol,
            strict_1d=cfg.trace_u.strict_1d,
        )
        diag["eigA_unstable"] = einfoAu
        if not okAu or edAu is None:
            nanv = np.full_like(target_eq, np.nan)
            return HeteroclinicMissResultND(
                qualified=False,
                param_value=float(param_value),
                source_eq=source_eq.copy(),
                target_eq=target_eq.copy(),
                sign_u=int(sign_u),
                sign_s=int(sign_s),
                n_B=nB.copy(),
                tau=nanv.copy(),
                r_sec=float(_compute_plane_r_sec(cfg)),
                x_u_cross=source_eq.copy(),
                x_s_cross=target_eq.copy(),
                s_u=float("nan"),
                s_s=float("nan"),
                g=float("nan"),
                q=float("inf"),
                status_u="eig_fail",
                status_s="eig_fail",
                status="eig_fail_A_unstable",
                diag=diag,
            )

        okVu, v_u, _ = _unit(edAu[1])
        if not okVu:
            nanv = np.full_like(target_eq, np.nan)
            return HeteroclinicMissResultND(
                qualified=False,
                param_value=float(param_value),
                source_eq=source_eq.copy(),
                target_eq=target_eq.copy(),
                sign_u=int(sign_u),
                sign_s=int(sign_s),
                n_B=nB.copy(),
                tau=nanv.copy(),
                r_sec=float(_compute_plane_r_sec(cfg)),
                x_u_cross=source_eq.copy(),
                x_s_cross=target_eq.copy(),
                s_u=float("nan"),
                s_s=float("nan"),
                g=float("nan"),
                q=float("inf"),
                status_u="eig_fail",
                status_s="eig_fail",
                status="eig_fail_vu",
                diag=diag,
            )

        tau0 = v_u - nB * float(nB @ v_u)
        tau0n = float(np.linalg.norm(tau0))

        if (not np.isfinite(tau0n)) or tau0n < float(cfg.tau_min):
            if tau_ref is not None:
                tr = np.array(tau_ref, float).reshape(-1,)
                tau0 = tr - nB * float(nB @ tr)
                tau0n = float(np.linalg.norm(tau0))

        if (not np.isfinite(tau0n)) or tau0n < float(cfg.tau_min):
            tau0 = _choose_fallback_tau(nB)
            tau0n = float(np.linalg.norm(tau0))

        if (not np.isfinite(tau0n)) or tau0n <= 1e-300:
            nanv = np.full_like(target_eq, np.nan)
            return HeteroclinicMissResultND(
                qualified=False,
                param_value=float(param_value),
                source_eq=source_eq.copy(),
                target_eq=target_eq.copy(),
                sign_u=int(sign_u),
                sign_s=int(sign_s),
                n_B=nB.copy(),
                tau=nanv.copy(),
                r_sec=float(_compute_plane_r_sec(cfg)),
                x_u_cross=source_eq.copy(),
                x_s_cross=target_eq.copy(),
                s_u=float("nan"),
                s_s=float("nan"),
                g=float("nan"),
                q=float("inf"),
                status_u="geom_fail",
                status_s="geom_fail",
                status="tau_degenerate",
                diag=diag,
            )

        tau = tau0 / tau0n

    if tau_ref is not None:
        tr = np.array(tau_ref, float).reshape(-1,)
        if tr.size == tau.size and np.isfinite(float(tau @ tr)) and float(tau @ tr) < 0.0:
            tau = -tau

    r_sec = float(_compute_plane_r_sec(cfg))
    rho_max = 2.0 * r_sec
    section = _make_hyperplane_section_nd(center=target_eq, normal_unit=nB, tau_unit=tau, offset=r_sec, rho_max=rho_max)
    diag["plane"] = {"r_sec": r_sec, "target_eq": target_eq.copy(), "n_B": nB.copy(), "tau": tau.copy(), "rho_max": float(rho_max)}

    ru = _trace_1d_manifold(
        ctx,
        params_vec,
        param_value=float(param_value),
        x_eq=source_eq.copy(),
        kind="unstable",
        sign=+1 if int(sign_u) >= 0 else -1,
        cfg=cfg.trace_u,
        section=section,
        stop_on_section=True,
    )
    rs = _trace_1d_manifold(
        ctx,
        params_vec,
        param_value=float(param_value),
        x_eq=target_eq.copy(),
        kind="stable",
        sign=+1 if int(sign_s) >= 0 else -1,
        cfg=cfg.trace_s,
        section=section,
        stop_on_section=True,
    )

    diag["trace_u"] = {"status": ru.status, "event": None if ru.event is None else {"kind": ru.event.kind, "s": ru.event.info.get("s")}}
    diag["trace_s"] = {"status": rs.status, "event": None if rs.event is None else {"kind": rs.event.kind, "s": rs.event.info.get("s")}}

    if ru.event is None or rs.event is None or ru.event.kind != "section" or rs.event.kind != "section":
        status = "no_cross"
        if ru.status in ("window", "blow", "smax", "max_steps", "stop"):
            status = f"u_{ru.status}"
        if rs.status in ("window", "blow", "smax", "max_steps", "stop"):
            status = f"s_{rs.status}"
        return HeteroclinicMissResultND(
            qualified=False,
            param_value=float(param_value),
            source_eq=source_eq.copy(),
            target_eq=target_eq.copy(),
            sign_u=int(sign_u),
            sign_s=int(sign_s),
            n_B=nB.copy(),
            tau=tau.copy(),
            r_sec=r_sec,
            x_u_cross=ru.points[-1].copy(),
            x_s_cross=rs.points[-1].copy(),
            s_u=float("nan"),
            s_s=float("nan"),
            g=float("nan"),
            q=float("inf"),
            status_u=str(ru.status),
            status_s=str(rs.status),
            status=status,
            diag=diag,
        )

    x_u = np.array(ru.event.x, float).reshape(-1,)
    x_s = np.array(rs.event.x, float).reshape(-1,)
    s_u = float(section.s(x_u))
    s_s = float(section.s(x_s))
    g = float(s_u - s_s)
    q = float(np.linalg.norm(x_u - x_s))

    return HeteroclinicMissResultND(
        qualified=True,
        param_value=float(param_value),
        source_eq=source_eq.copy(),
        target_eq=target_eq.copy(),
        sign_u=int(sign_u),
        sign_s=int(sign_s),
        n_B=nB.copy(),
        tau=tau.copy(),
        r_sec=r_sec,
        x_u_cross=x_u,
        x_s_cross=x_s,
        s_u=s_u,
        s_s=s_s,
        g=g,
        q=q,
        status_u=str(ru.status),
        status_s=str(rs.status),
        status="ok",
        diag=diag,
    )


def _bracket_on_qualified(
    qual_sorted: list[HeteroclinicMissResult2D],
    gap_scan_tol: float,
) -> tuple[HeteroclinicMissResult2D, HeteroclinicMissResult2D] | None:
    tol = float(gap_scan_tol)
    for a, b in zip(qual_sorted[:-1], qual_sorted[1:]):
        if np.isfinite(a.g) and abs(a.g) <= tol:
            return a, a
        if np.isfinite(a.g) and np.isfinite(b.g) and (a.g * b.g < 0.0):
            return a, b
    if qual_sorted:
        last = qual_sorted[-1]
        if np.isfinite(last.g) and abs(last.g) <= tol:
            return last, last
    return None


def _bracket_on_qualified_nd(
    qual_sorted: list[HeteroclinicMissResultND],
    gap_scan_tol: float,
) -> tuple[HeteroclinicMissResultND, HeteroclinicMissResultND] | None:
    tol = float(gap_scan_tol)
    for a, b in zip(qual_sorted[:-1], qual_sorted[1:]):
        if np.isfinite(a.g) and abs(a.g) <= tol:
            return a, a
        if np.isfinite(a.g) and np.isfinite(b.g) and (a.g * b.g < 0.0):
            return a, b
    if qual_sorted:
        last = qual_sorted[-1]
        if np.isfinite(last.g) and abs(last.g) <= tol:
            return last, last
    return None


def _find_heteroclinic_mu_2d(
    ctx: _HeteroclinicContext,
    *,
    param_min: float,
    param_max: float,
    param_init: float,
    source_eq_guess: np.ndarray,
    target_eq_guess: np.ndarray,
    cfg: HeteroclinicFinderConfig2D,
) -> tuple[bool, float, HeteroclinicMissResult2D | None, dict[str, object]]:
    param_min = float(param_min)
    param_max = float(param_max)
    param_init = float(param_init)

    param_grid, idx0 = _scan_grid_centered(param_min, param_max, param_init, int(cfg.scan_n))

    if cfg.branch_mode == "fixed":
        combos = [(+1 if int(cfg.sign_u) >= 0 else -1, +1 if int(cfg.sign_s) >= 0 else -1)]
    else:
        combos = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

    best_overall: tuple[float, HeteroclinicMissResult2D, dict[str, object]] | None = None

    for (sign_u, sign_s) in combos:
        results: list[HeteroclinicMissResult2D | None] = [None] * len(param_grid)

        info: dict[str, object] = {
            "param_min": param_min,
            "param_max": param_max,
            "param_init": param_init,
            "param_used": float(param_grid[idx0]),
            "scan_n": int(cfg.scan_n),
            "sign_u": int(sign_u),
            "sign_s": int(sign_s),
            "eq_track_max_dist": (_track_max_dist_default(cfg.trace_u.r_leave) if cfg.eq_track_max_dist is None else float(cfg.eq_track_max_dist)),
            "qualified_count": 0,
            "eq_jump_count": 0,
            "eq_fail_count": 0,
            "best_by_abs_gap": None,
            "best_by_q": None,
            "fail": None,
            "circle_r_sec": float(_compute_circle_radius(cfg)),
        }

        params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_grid[idx0]))
        r0 = _heteroclinic_miss_circleB_2d(
            ctx,
            params_vec,
            float(param_grid[idx0]),
            source_eq_guess=np.array(source_eq_guess, float),
            target_eq_guess=np.array(target_eq_guess, float),
            source_eq_prev=None,
            target_eq_prev=None,
            sign_u=sign_u,
            sign_s=sign_s,
            cfg=cfg,
        )
        results[idx0] = r0

        if not r0.qualified and r0.status in ("eq_fail", "eq_jump"):
            info["fail"] = "anchor_equilibria_failed"
            continue

        A_prev = r0.source_eq.copy()
        B_prev = r0.target_eq.copy()

        for i in range(idx0 + 1, len(param_grid)):
            param_val = float(param_grid[i])
            params_vec = _params_with_override(ctx.params_base, ctx.param_index, param_val)
            r = _heteroclinic_miss_circleB_2d(
                ctx,
                params_vec,
                param_val,
                source_eq_guess=A_prev,
                target_eq_guess=B_prev,
                source_eq_prev=A_prev,
                target_eq_prev=B_prev,
                sign_u=sign_u,
                sign_s=sign_s,
                cfg=cfg,
            )
            results[i] = r
            if r.status == "eq_jump":
                info["eq_jump_count"] += 1
            elif r.status == "eq_fail":
                info["eq_fail_count"] += 1
            else:
                A_prev = r.source_eq.copy()
                B_prev = r.target_eq.copy()

        A_prev = r0.source_eq.copy()
        B_prev = r0.target_eq.copy()

        for i in range(idx0 - 1, -1, -1):
            param_val = float(param_grid[i])
            params_vec = _params_with_override(ctx.params_base, ctx.param_index, param_val)
            r = _heteroclinic_miss_circleB_2d(
                ctx,
                params_vec,
                param_val,
                source_eq_guess=A_prev,
                target_eq_guess=B_prev,
                source_eq_prev=A_prev,
                target_eq_prev=B_prev,
                sign_u=sign_u,
                sign_s=sign_s,
                cfg=cfg,
            )
            results[i] = r
            if r.status == "eq_jump":
                info["eq_jump_count"] += 1
            elif r.status == "eq_fail":
                info["eq_fail_count"] += 1
            else:
                A_prev = r.source_eq.copy()
                B_prev = r.target_eq.copy()

        qual = [r for r in results if (r is not None and r.qualified and np.isfinite(r.g))]
        info["qualified_count"] = int(len(qual))
        if not qual:
            info["fail"] = "no_qualified_points_in_scan"
            continue

        qual_sorted = sorted(qual, key=lambda rr: rr.param_value)
        _unwrap_qualified_sorted(qual_sorted)

        best_by_abs_g = min(qual_sorted, key=lambda rr: abs(rr.g))
        best_by_q = min(qual_sorted, key=lambda rr: rr.q)

        info["best_by_abs_gap"] = {"param_value": best_by_abs_g.param_value, "gap": best_by_abs_g.g, "q": best_by_abs_g.q, "status": best_by_abs_g.status}
        info["best_by_q"] = {
            "param_value": best_by_q.param_value,
            "gap": best_by_q.g,
            "q": best_by_q.q,
            "status": best_by_q.status,
        }

        if _is_success(cfg, best_by_q):
            info["param_found"] = float(best_by_q.param_value)
            info["gap_found"] = float(best_by_q.g)
            info["gap_tol_eff_found"] = float(_gap_tol_eff(cfg, best_by_q))
            info["q_found"] = float(best_by_q.q)
            info["x_u_cross"] = np.array(best_by_q.x_u_cross, float)
            info["x_s_cross"] = np.array(best_by_q.x_s_cross, float)
            info["source_eq"] = np.array(best_by_q.source_eq, float)
            info["target_eq"] = np.array(best_by_q.target_eq, float)
            info["bisection_iters"] = 0
            return True, float(best_by_q.param_value), best_by_q, info

        bracket = _bracket_on_qualified(qual_sorted, gap_scan_tol=float(cfg.gap_tol))
        if bracket is None:
            info["fail"] = "no_sign_change_in_qualified_scan"
            score = float(best_by_q.q + 0.1 * abs(best_by_q.g))
            if best_overall is None or score < best_overall[0]:
                best_overall = (score, best_by_q, info)
            continue

        a, b = bracket
        if a.param_value == b.param_value:
            r_star = a
            info["bisection_iters"] = 0
        else:
            param_lo, param_hi = float(a.param_value), float(b.param_value)
            r_lo, r_hi = a, b
            r_star = r_lo
            iters = 0

            for it in range(int(cfg.max_bisect)):
                iters = it + 1
                param_mid = 0.5 * (param_lo + param_hi)

                if abs(param_mid - param_lo) <= abs(param_mid - param_hi):
                    r_ref = r_lo
                    A_seed = r_lo.source_eq
                    B_seed = r_lo.target_eq
                    A_prev2 = r_lo.source_eq
                    B_prev2 = r_lo.target_eq
                else:
                    r_ref = r_hi
                    A_seed = r_hi.source_eq
                    B_seed = r_hi.target_eq
                    A_prev2 = r_hi.source_eq
                    B_prev2 = r_hi.target_eq

                params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_mid))
                r_mid = _heteroclinic_miss_circleB_2d(
                    ctx,
                    params_vec,
                    float(param_mid),
                    source_eq_guess=A_seed,
                    target_eq_guess=B_seed,
                    source_eq_prev=A_prev2,
                    target_eq_prev=B_prev2,
                    sign_u=sign_u,
                    sign_s=sign_s,
                    cfg=cfg,
                )
                _apply_unwrap_inplace(r_mid, r_ref)

                if _is_success(cfg, r_mid):
                    r_star = r_mid
                    break

                if (not r_mid.qualified) or (not np.isfinite(r_mid.g)):
                    if r_lo.q <= r_hi.q:
                        param_hi = float(param_mid)
                    else:
                        param_lo = float(param_mid)
                    continue

                if r_lo.g * r_mid.g <= 0.0:
                    param_hi, r_hi = float(param_mid), r_mid
                else:
                    param_lo, r_lo = float(param_mid), r_mid

            else:
                r_star = min([r_lo, r_hi], key=lambda rr: (rr.q, abs(rr.g)))

            info["bisection_iters"] = int(iters)

        if not _is_success(cfg, r_star):
            info["fail"] = "bisection_did_not_converge"
            cand = r_star if (r_star.qualified and np.isfinite(r_star.g)) else best_by_q
            score = float(cand.q + 0.1 * abs(cand.g))
            if best_overall is None or score < best_overall[0]:
                best_overall = (score, cand, info)
            continue

        info["param_found"] = float(r_star.param_value)
        info["gap_found"] = float(r_star.g)
        info["gap_tol_eff_found"] = float(_gap_tol_eff(cfg, r_star))
        info["q_found"] = float(r_star.q)
        info["x_u_cross"] = np.array(r_star.x_u_cross, float)
        info["x_s_cross"] = np.array(r_star.x_s_cross, float)
        info["source_eq"] = np.array(r_star.source_eq, float)
        info["target_eq"] = np.array(r_star.target_eq, float)

        return True, float(r_star.param_value), r_star, info

    if best_overall is not None:
        _, best_r, best_info = best_overall
        best_info = dict(best_info)
        best_info["fail"] = best_info.get("fail", "no_combo_converged")
        best_info["param_candidate"] = float(best_r.param_value)
        best_info["gap_candidate"] = float(best_r.g) if np.isfinite(best_r.g) else float("nan")
        best_info["q_candidate"] = float(best_r.q)
        best_info["gap_tol_eff_candidate"] = float(_gap_tol_eff(cfg, best_r)) if best_r.qualified else float("nan")
        return False, float("nan"), None, best_info

    return False, float("nan"), None, {"fail": "no_combo_had_qualified_points"}


def _find_heteroclinic_mu_nd(
    ctx: _HeteroclinicContext,
    *,
    param_min: float,
    param_max: float,
    param_init: float,
    source_eq_guess: np.ndarray,
    target_eq_guess: np.ndarray,
    cfg: HeteroclinicFinderConfigND,
) -> tuple[bool, float, HeteroclinicMissResultND | None, dict[str, object]]:
    param_min = float(param_min)
    param_max = float(param_max)
    param_init = float(param_init)

    param_grid, idx0 = _scan_grid_centered(param_min, param_max, param_init, int(cfg.scan_n))

    if cfg.branch_mode == "fixed":
        combos = [(+1 if int(cfg.sign_u) >= 0 else -1, +1 if int(cfg.sign_s) >= 0 else -1)]
    else:
        combos = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

    best_overall: tuple[float, HeteroclinicMissResultND, dict[str, object]] | None = None

    for (sign_u, sign_s) in combos:
        results: list[HeteroclinicMissResultND | None] = [None] * len(param_grid)

        info: dict[str, object] = {
            "param_min": param_min,
            "param_max": param_max,
            "param_init": param_init,
            "param_used": float(param_grid[idx0]),
            "scan_n": int(cfg.scan_n),
            "sign_u": int(sign_u),
            "sign_s": int(sign_s),
            "eq_track_max_dist": (_track_max_dist_default(cfg.trace_u.r_leave) if cfg.eq_track_max_dist is None else float(cfg.eq_track_max_dist)),
            "qualified_count": 0,
            "eq_jump_count": 0,
            "eq_fail_count": 0,
            "best_by_abs_gap": None,
            "best_by_q": None,
            "fail": None,
            "plane_r_sec": float(_compute_plane_r_sec(cfg)),
        }

        params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_grid[idx0]))
        r0 = _heteroclinic_miss_planeB_nd(
            ctx,
            params_vec,
            float(param_grid[idx0]),
            source_eq_guess=np.array(source_eq_guess, float),
            target_eq_guess=np.array(target_eq_guess, float),
            source_eq_prev=None,
            target_eq_prev=None,
            sign_u=sign_u,
            sign_s=sign_s,
            cfg=cfg,
            nB_ref=None,
            tau_ref=None,
        )
        results[idx0] = r0

        if not r0.qualified and r0.status in ("eq_fail", "eq_jump"):
            info["fail"] = "anchor_equilibria_failed"
            continue

        A_prev = r0.source_eq.copy()
        B_prev = r0.target_eq.copy()
        nB_prev = None if (r0.n_B is None or not np.all(np.isfinite(r0.n_B))) else r0.n_B.copy()
        tau_prev = None if (r0.tau is None or not np.all(np.isfinite(r0.tau))) else r0.tau.copy()

        for i in range(idx0 + 1, len(param_grid)):
            param_val = float(param_grid[i])
            params_vec = _params_with_override(ctx.params_base, ctx.param_index, param_val)
            r = _heteroclinic_miss_planeB_nd(
                ctx,
                params_vec,
                param_val,
                source_eq_guess=A_prev,
                target_eq_guess=B_prev,
                source_eq_prev=A_prev,
                target_eq_prev=B_prev,
                sign_u=sign_u,
                sign_s=sign_s,
                cfg=cfg,
                nB_ref=nB_prev,
                tau_ref=tau_prev,
            )
            results[i] = r
            if r.status == "eq_jump":
                info["eq_jump_count"] += 1
            elif r.status == "eq_fail":
                info["eq_fail_count"] += 1
            else:
                A_prev = r.source_eq.copy()
                B_prev = r.target_eq.copy()

            if np.all(np.isfinite(r.n_B)):
                nB_prev = r.n_B.copy()
            if np.all(np.isfinite(r.tau)):
                tau_prev = r.tau.copy()

        A_prev = r0.source_eq.copy()
        B_prev = r0.target_eq.copy()
        nB_prev = None if (r0.n_B is None or not np.all(np.isfinite(r0.n_B))) else r0.n_B.copy()
        tau_prev = None if (r0.tau is None or not np.all(np.isfinite(r0.tau))) else r0.tau.copy()

        for i in range(idx0 - 1, -1, -1):
            param_val = float(param_grid[i])
            params_vec = _params_with_override(ctx.params_base, ctx.param_index, param_val)
            r = _heteroclinic_miss_planeB_nd(
                ctx,
                params_vec,
                param_val,
                source_eq_guess=A_prev,
                target_eq_guess=B_prev,
                source_eq_prev=A_prev,
                target_eq_prev=B_prev,
                sign_u=sign_u,
                sign_s=sign_s,
                cfg=cfg,
                nB_ref=nB_prev,
                tau_ref=tau_prev,
            )
            results[i] = r
            if r.status == "eq_jump":
                info["eq_jump_count"] += 1
            elif r.status == "eq_fail":
                info["eq_fail_count"] += 1
            else:
                A_prev = r.source_eq.copy()
                B_prev = r.target_eq.copy()

            if np.all(np.isfinite(r.n_B)):
                nB_prev = r.n_B.copy()
            if np.all(np.isfinite(r.tau)):
                tau_prev = r.tau.copy()

        qual = [r for r in results if (r is not None and r.qualified and np.isfinite(r.g))]
        info["qualified_count"] = int(len(qual))
        if not qual:
            info["fail"] = "no_qualified_points_in_scan"
            continue

        qual_sorted = sorted(qual, key=lambda rr: rr.param_value)

        best_by_abs_g = min(qual_sorted, key=lambda rr: abs(rr.g))
        best_by_q = min(qual_sorted, key=lambda rr: rr.q)

        info["best_by_abs_gap"] = {"param_value": best_by_abs_g.param_value, "gap": best_by_abs_g.g, "q": best_by_abs_g.q, "status": best_by_abs_g.status}
        info["best_by_q"] = {
            "param_value": best_by_q.param_value,
            "gap": best_by_q.g,
            "q": best_by_q.q,
            "status": best_by_q.status,
        }

        if _is_success_nd(cfg, best_by_q):
            info["param_found"] = float(best_by_q.param_value)
            info["gap_found"] = float(best_by_q.g)
            info["gap_tol_eff_found"] = float(_gap_tol_eff_nd(cfg, best_by_q))
            info["q_found"] = float(best_by_q.q)
            info["x_u_cross"] = np.array(best_by_q.x_u_cross, float)
            info["x_s_cross"] = np.array(best_by_q.x_s_cross, float)
            info["source_eq"] = np.array(best_by_q.source_eq, float)
            info["target_eq"] = np.array(best_by_q.target_eq, float)
            info["bisection_iters"] = 0
            return True, float(best_by_q.param_value), best_by_q, info

        bracket = _bracket_on_qualified_nd(qual_sorted, gap_scan_tol=float(cfg.gap_tol))
        if bracket is None:
            info["fail"] = "no_sign_change_in_qualified_scan"
            score = float(best_by_q.q + 0.1 * abs(best_by_q.g))
            if best_overall is None or score < best_overall[0]:
                best_overall = (score, best_by_q, info)
            continue

        a, b = bracket
        if a.param_value == b.param_value:
            r_star = a
            info["bisection_iters"] = 0
        else:
            param_lo, param_hi = float(a.param_value), float(b.param_value)
            r_lo, r_hi = a, b
            r_star = r_lo
            iters = 0

            for it in range(int(cfg.max_bisect)):
                iters = it + 1
                param_mid = 0.5 * (param_lo + param_hi)

                if abs(param_mid - param_lo) <= abs(param_mid - param_hi):
                    r_ref = r_lo
                else:
                    r_ref = r_hi

                params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_mid))
                r_mid = _heteroclinic_miss_planeB_nd(
                    ctx,
                    params_vec,
                    float(param_mid),
                    source_eq_guess=r_ref.source_eq,
                    target_eq_guess=r_ref.target_eq,
                    source_eq_prev=r_ref.source_eq,
                    target_eq_prev=r_ref.target_eq,
                    sign_u=sign_u,
                    sign_s=sign_s,
                    cfg=cfg,
                    nB_ref=r_ref.n_B,
                    tau_ref=r_ref.tau,
                )

                if _is_success_nd(cfg, r_mid):
                    r_star = r_mid
                    break

                if (not r_mid.qualified) or (not np.isfinite(r_mid.g)):
                    if r_lo.q <= r_hi.q:
                        param_hi = float(param_mid)
                    else:
                        param_lo = float(param_mid)
                    continue

                if r_lo.g * r_mid.g <= 0.0:
                    param_hi, r_hi = float(param_mid), r_mid
                else:
                    param_lo, r_lo = float(param_mid), r_mid

            else:
                r_star = min([r_lo, r_hi], key=lambda rr: (rr.q, abs(rr.g)))

            info["bisection_iters"] = int(iters)

        if not _is_success_nd(cfg, r_star):
            info["fail"] = "bisection_did_not_converge"
            cand = r_star if (r_star.qualified and np.isfinite(r_star.g)) else best_by_q
            score = float(cand.q + 0.1 * abs(cand.g))
            if best_overall is None or score < best_overall[0]:
                best_overall = (score, cand, info)
            continue

        info["param_found"] = float(r_star.param_value)
        info["gap_found"] = float(r_star.g)
        info["gap_tol_eff_found"] = float(_gap_tol_eff_nd(cfg, r_star))
        info["q_found"] = float(r_star.q)
        info["x_u_cross"] = np.array(r_star.x_u_cross, float)
        info["x_s_cross"] = np.array(r_star.x_s_cross, float)
        info["source_eq"] = np.array(r_star.source_eq, float)
        info["target_eq"] = np.array(r_star.target_eq, float)

        return True, float(r_star.param_value), r_star, info

    if best_overall is not None:
        _, best_r, best_info = best_overall
        best_info = dict(best_info)
        best_info["fail"] = best_info.get("fail", "no_combo_converged")
        best_info["param_candidate"] = float(best_r.param_value)
        best_info["gap_candidate"] = float(best_r.g) if np.isfinite(best_r.g) else float("nan")
        best_info["q_candidate"] = float(best_r.q)
        best_info["gap_tol_eff_candidate"] = float(_gap_tol_eff_nd(cfg, best_r)) if best_r.qualified else float("nan")
        return False, float("nan"), None, best_info

    return False, float("nan"), None, {"fail": "no_combo_had_qualified_points"}


def _trace_heteroclinic_orbit(
    ctx: _HeteroclinicContext,
    params_vec: np.ndarray,
    param_value: float,
    *,
    source_eq: np.ndarray,
    target_eq: np.ndarray,
    sign_u: int,
    cfg_u: HeteroclinicBranchConfig,
    hit_radius: float = 1e-2,
) -> HeteroclinicTraceResult:
    tr = _trace_1d_manifold(
        ctx,
        params_vec,
        param_value=float(param_value),
        x_eq=np.array(source_eq, float),
        kind="unstable",
        sign=+1 if int(sign_u) >= 0 else -1,
        cfg=cfg_u,
        target=np.array(target_eq, float),
        target_r=float(hit_radius),
        stop_on_target=True,
    )
    meta = HeteroclinicTraceMeta(
        param_value=float(param_value),
        source_eq=np.array(source_eq, float),
        target_eq=np.array(target_eq, float),
        sign_u=int(sign_u),
        eps_used=float(tr.meta.get("eps", float("nan"))),
        status=str(tr.status),
        success=False,
        event=tr.event,
        diag=dict(tr.meta.get("diag", {})),
    )
    t = np.array(tr.t, float)
    X = np.array(tr.points, float)

    if meta.event is None and X.ndim == 2 and X.shape[0] >= 2:
        B = np.array(target_eq, float).reshape(-1,)
        try:
            d = np.linalg.norm(X - B[None, :], axis=1)
            i_min = int(np.argmin(d))
            if 0 <= i_min < (X.shape[0] - 1):
                meta.diag = dict(meta.diag)
                meta.diag["clip"] = {
                    "kind": "closest_approach_to_B",
                    "i_min": i_min,
                    "d_min": float(d[i_min]),
                    "d_end": float(d[-1]),
                    "status_before_clip": str(meta.status),
                }
                t = t[: i_min + 1]
                X = X[: i_min + 1, :]
        except Exception:
            pass

    success = False
    if meta.event is not None and meta.event.kind == "target_ball":
        success = True
    else:
        B = np.array(target_eq, float).reshape(-1,)
        d_min = float("inf")
        try:
            if X.ndim == 2 and X.shape[0] >= 1:
                d = np.linalg.norm(X - B[None, :], axis=1)
                d_min = float(np.min(d))
        except Exception:
            d_min = float("inf")
        if np.isfinite(d_min) and d_min <= float(hit_radius):
            success = True
    meta.success = bool(success)

    return HeteroclinicTraceResult(t=t, X=X, meta=meta)


def heteroclinic_finder(
    sim: "Sim",
    *,
    param: str | int | None,
    param_min: float,
    param_max: float,
    param_init: float,
    source_eq_guess: Mapping[str, float] | Sequence[float] | np.ndarray,
    target_eq_guess: Mapping[str, float] | Sequence[float] | np.ndarray,
    cfg: HeteroclinicFinderConfig2D | HeteroclinicFinderConfigND | None = None,
    mode: Literal["auto", "2d", "nd"] = "auto",
    # Simplified API: preset and flattened kwargs
    preset: str | HeteroclinicPreset | None = None,
    trace_cfg: HeteroclinicBranchConfig | None = None,
    window: Sequence[tuple[float, float]] | None = None,
    scan_n: int | None = None,
    max_bisect: int | None = None,
    gap_tol: float | None = None,
    x_tol: float | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
    # Base params
    params: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    t: float | None = None,
) -> HeteroclinicFinderResult:
    """
    Find a heteroclinic orbit connecting two equilibrium points.

    Parameters
    ----------
    sim : Sim
        The simulation object.
    param : str | int | None
        The parameter to vary (name or index). If None, model must have exactly one parameter.
    param_min, param_max : float
        Search bounds for the parameter.
    param_init : float
        Initial guess for the parameter value.
    source_eq_guess, target_eq_guess : Mapping | Sequence | np.ndarray
        Initial guesses for source and target equilibrium points. Can be:
        - dict: {"x": 0.0, "y": 0.0}
        - list/tuple: [0.0, 0.0]
        - np.ndarray: np.array([0.0, 0.0])
    cfg : HeteroclinicFinderConfig2D | HeteroclinicFinderConfigND | None
        Full configuration object (advanced). If provided, overrides preset/trace_cfg/kwargs.
    mode : {"auto", "2d", "nd"}
        Algorithm mode. "auto" selects based on system dimension.

    Simplified API (preferred for typical use):
    -------------------------------------------
    preset : str | HeteroclinicPreset | None
        Preset configuration: "fast", "default", or "precise".
    trace_cfg : HeteroclinicBranchConfig | None
        Unified trace configuration for both unstable and stable manifolds.
    window : Sequence[tuple[float, float]] | None
        State-space window as [(x_min, x_max), (y_min, y_max), ...].
    scan_n : int | None
        Number of parameter scan points.
    max_bisect : int | None
        Maximum bisection iterations.
    gap_tol : float | None
        Tolerance for gap distance.
    x_tol : float | None
        Tolerance for parameter convergence.
    t_max : float | None
        Maximum integration time for manifold tracing.
    r_blow : float | None
        Blow-up radius for manifold tracing.

    params : Mapping | Sequence | np.ndarray | None
        Fixed parameter values (other than the search parameter).
    t : float | None
        Evaluation time for auxiliary variables.

    Returns
    -------
    HeteroclinicFinderResult
        Result with fields ``success``, ``param_found``, ``miss``, and ``info``.

    Examples
    --------
    Simple usage with defaults:

        result = heteroclinic_finder(
            sim,
            param="c",
            param_min=-0.5,
            param_max=1.0,
            param_init=0.0,
            source_eq_guess=[0.0, 0.0],
            target_eq_guess=[1.0, 0.0],
        )

    With preset and window:

        result = heteroclinic_finder(
            sim,
            param="c",
            param_min=-0.5,
            param_max=1.0,
            param_init=0.0,
            source_eq_guess=[0.0, 0.0],
            target_eq_guess=[1.0, 0.0],
            preset="precise",
            window=[(-10, 10), (-10, 10)],
        )
    """
    if param_min >= param_max:
        raise ValueError("param_min must be < param_max")
    if mode not in ("auto", "2d", "nd"):
        raise ValueError("mode must be 'auto', '2d', or 'nd'")

    # Check for conflicting parameters
    has_simplified_kwargs = any(
        x is not None for x in [preset, trace_cfg, window, scan_n, max_bisect, gap_tol, x_tol, t_max, r_blow]
    )
    if cfg is not None and has_simplified_kwargs:
        raise ValueError(
            "Cannot use 'cfg' together with simplified kwargs (preset, trace_cfg, window, etc.). "
            "Use either 'cfg' for full control, or the simplified kwargs."
        )

    ctx = _build_heteroclinic_context(
        sim,
        params=params,
        param=param,
        t=t,
        caller="heteroclinic_finder",
    )
    n_state = len(ctx.model.spec.states)

    # Determine mode first
    if mode == "auto":
        if cfg is not None:
            mode = "2d" if isinstance(cfg, HeteroclinicFinderConfig2D) else "nd"
        else:
            mode = "2d" if n_state == 2 else "nd"

    # Build or use cfg
    if cfg is None:
        cfg = _build_finder_config_from_kwargs(
            mode=mode,
            preset=preset,
            trace_cfg=trace_cfg,
            window=window,
            scan_n=scan_n,
            max_bisect=max_bisect,
            gap_tol=gap_tol,
            x_tol=x_tol,
            t_max=t_max,
            r_blow=r_blow,
        )
    else:
        if not isinstance(cfg, (HeteroclinicFinderConfig2D, HeteroclinicFinderConfigND)):
            raise ValueError("cfg must be HeteroclinicFinderConfig2D or HeteroclinicFinderConfigND")
        cfg_mode = "2d" if isinstance(cfg, HeteroclinicFinderConfig2D) else "nd"
        if mode == "2d" and cfg_mode != "2d":
            raise ValueError("cfg must be HeteroclinicFinderConfig2D when mode='2d'")
        if mode == "nd" and cfg_mode != "nd":
            raise ValueError("cfg must be HeteroclinicFinderConfigND when mode='nd'")

    source_eq_guess_vec = _resolve_fixed_point(ctx.model, source_eq_guess)
    target_eq_guess_vec = _resolve_fixed_point(ctx.model, target_eq_guess)

    if mode == "2d":
        if n_state != 2:
            raise ValueError("mode='2d' requires a 2D system")
        cfg2d = cfg if isinstance(cfg, HeteroclinicFinderConfig2D) else HeteroclinicFinderConfig2D()
        _validate_finder_cfg_2d(cfg2d)
        success, param_found, miss, info = _find_heteroclinic_mu_2d(
            ctx,
            param_min=float(param_min),
            param_max=float(param_max),
            param_init=float(param_init),
            source_eq_guess=np.array(source_eq_guess_vec, float),
            target_eq_guess=np.array(target_eq_guess_vec, float),
            cfg=cfg2d,
        )
        return HeteroclinicFinderResult(
            success=bool(success),
            param_found=float(param_found),
            miss=miss,
            info=info,
        )

    cfg_nd = cfg if isinstance(cfg, HeteroclinicFinderConfigND) else HeteroclinicFinderConfigND()
    _validate_finder_cfg_nd(cfg_nd)
    success, param_found, miss, info = _find_heteroclinic_mu_nd(
        ctx,
        param_min=float(param_min),
        param_max=float(param_max),
        param_init=float(param_init),
        source_eq_guess=np.array(source_eq_guess_vec, float),
        target_eq_guess=np.array(target_eq_guess_vec, float),
        cfg=cfg_nd,
    )
    return HeteroclinicFinderResult(
        success=bool(success),
        param_found=float(param_found),
        miss=miss,
        info=info,
    )


def heteroclinic_tracer(
    sim: "Sim",
    *,
    param: str | int | None,
    param_value: float,
    source_eq: Mapping[str, float] | Sequence[float] | np.ndarray,
    target_eq: Mapping[str, float] | Sequence[float] | np.ndarray,
    sign_u: int,
    cfg_u: HeteroclinicBranchConfig | None = None,
    hit_radius: float = 1e-2,
    # Simplified API: preset and flattened kwargs
    preset: str | HeteroclinicPreset | None = None,
    trace_cfg: HeteroclinicBranchConfig | None = None,
    window: Sequence[tuple[float, float]] | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
    # Base params
    params: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    t: float | None = None,
) -> HeteroclinicTraceResult:
    """
    Trace a heteroclinic orbit from a source equilibrium towards a target equilibrium.

    Parameters
    ----------
    sim : Sim
        The simulation object.
    param : str | int | None
        The parameter name or index. If None, model must have exactly one parameter.
    param_value : float
        The parameter value at which to trace the orbit.
    source_eq, target_eq : Mapping | Sequence | np.ndarray
        Source and target equilibrium points. Can be:
        - dict: {"x": 0.0, "y": 0.0}
        - list/tuple: [0.0, 0.0]
        - np.ndarray: np.array([0.0, 0.0])
    sign_u : int
        Sign for unstable eigenvector direction (+1 or -1).
    cfg_u : HeteroclinicBranchConfig | None
        Full trace configuration (advanced). If provided, overrides preset/trace_cfg/kwargs.
    hit_radius : float
        Radius around B to stop tracing (default: 1e-2).

    Simplified API (preferred for typical use):
    -------------------------------------------
    preset : str | HeteroclinicPreset | None
        Preset configuration: "fast", "default", or "precise".
    trace_cfg : HeteroclinicBranchConfig | None
        Trace configuration (alternative to cfg_u).
    window : Sequence[tuple[float, float]] | None
        State-space window as [(x_min, x_max), (y_min, y_max), ...].
    t_max : float | None
        Maximum integration time for manifold tracing.
    r_blow : float | None
        Blow-up radius for manifold tracing.

    params : Mapping | Sequence | np.ndarray | None
        Fixed parameter values.
    t : float | None
        Evaluation time for auxiliary variables.

    Returns
    -------
    HeteroclinicTraceResult
        Tuple-like result with fields ``t``, ``X``, and ``meta``. It also
        exposes a ``success`` property (and ``meta.success``) that is True
        when the trace reaches the target ball, plus a ``branches`` attribute
        so it can be passed directly to ``dynlib.plot.manifold``.

    Examples
    --------
    Simple usage:

        trace = heteroclinic_tracer(
            sim,
            param="c",
            param_value=0.1234,
            source_eq=[0.0, 0.0],
            target_eq=[1.0, 0.0],
            sign_u=+1,
        )
        t, X, meta = trace

    With preset:

        trace = heteroclinic_tracer(
            sim,
            param="c",
            param_value=0.1234,
            source_eq=[0.0, 0.0],
            target_eq=[1.0, 0.0],
            sign_u=+1,
            preset="precise",
            window=[(-10, 10), (-10, 10)],
        )
        t, X, meta = trace
    """
    if hit_radius <= 0.0:
        raise ValueError("hit_radius must be positive")

    # Check for conflicting parameters
    has_simplified_kwargs = any(x is not None for x in [preset, trace_cfg, window, t_max, r_blow])
    if cfg_u is not None and has_simplified_kwargs:
        raise ValueError(
            "Cannot use 'cfg_u' together with simplified kwargs (preset, trace_cfg, window, etc.). "
            "Use either 'cfg_u' for full control, or the simplified kwargs."
        )

    ctx = _build_heteroclinic_context(
        sim,
        params=params,
        param=param,
        t=t,
        caller="heteroclinic_tracer",
    )

    # Build or use cfg_u
    if cfg_u is None:
        if trace_cfg is not None:
            cfg_u = trace_cfg
            # Apply overrides if any
            if window is not None or t_max is not None or r_blow is not None:
                window_min = None
                window_max = None
                if window is not None:
                    window_min = np.array([lo for lo, _ in window], dtype=float)
                    window_max = np.array([hi for _, hi in window], dtype=float)
                cfg_u = HeteroclinicBranchConfig(
                    eq_tol=trace_cfg.eq_tol,
                    eq_max_iter=trace_cfg.eq_max_iter,
                    eq_track_max_dist=trace_cfg.eq_track_max_dist,
                    eps_mode=trace_cfg.eps_mode,
                    eps=trace_cfg.eps,
                    eps_min=trace_cfg.eps_min,
                    eps_max=trace_cfg.eps_max,
                    r_leave=trace_cfg.r_leave,
                    t_leave_target=trace_cfg.t_leave_target,
                    t_max=t_max if t_max is not None else trace_cfg.t_max,
                    s_max=trace_cfg.s_max,
                    r_blow=r_blow if r_blow is not None else trace_cfg.r_blow,
                    window_min=window_min if window_min is not None else trace_cfg.window_min,
                    window_max=window_max if window_max is not None else trace_cfg.window_max,
                    t_min_event=trace_cfg.t_min_event,
                    require_leave_before_event=trace_cfg.require_leave_before_event,
                    eig_real_tol=trace_cfg.eig_real_tol,
                    eig_imag_tol=trace_cfg.eig_imag_tol,
                    strict_1d=trace_cfg.strict_1d,
                    jac=trace_cfg.jac,
                    fd_eps=trace_cfg.fd_eps,
                    rk=trace_cfg.rk,
                )
        else:
            pset = _get_heteroclinic_preset(preset if preset is not None else "default")
            cfg_u = _build_branch_config_from_preset(
                pset, window=window, t_max=t_max, r_blow=r_blow
            )

    _validate_branch_cfg(cfg_u)

    params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_value))
    source_eq_vec = _resolve_fixed_point(ctx.model, source_eq)
    target_eq_vec = _resolve_fixed_point(ctx.model, target_eq)

    return _trace_heteroclinic_orbit(
        ctx,
        params_vec,
        float(param_value),
        source_eq=np.array(source_eq_vec, float),
        target_eq=np.array(target_eq_vec, float),
        sign_u=int(sign_u),
        cfg_u=cfg_u,
        hit_radius=float(hit_radius),
    )


# =============================================================================
# Homoclinic finder/tracer (ODE)
# =============================================================================


@dataclass(frozen=True)
class HomoclinicRK45Config:
    dt0: float = 1e-3
    min_step: float = 1e-12
    dt_max: float = 1e-1
    atol: float = 1e-10
    rtol: float = 1e-7
    safety: float = 0.9
    max_steps: int = 2_000_000


@dataclass(frozen=True)
class HomoclinicBranchConfig:
    eq_tol: float = 1e-12
    eq_max_iter: int = 40
    eq_track_max_dist: float | None = None

    eps_mode: Literal["leave", "fixed"] = "leave"
    eps: float = 1e-6
    eps_min: float = 1e-10
    eps_max: float = 1e-2
    r_leave: float = 1e-2
    t_leave_target: float = 0.05

    r_sec: float = 1e-2
    t_min_event: float = 0.2

    t_max: float = 500.0
    s_max: float = 1e6
    r_blow: float = 200.0

    window_min: Sequence[float] | np.ndarray | None = None
    window_max: Sequence[float] | np.ndarray | None = None

    require_leave_before_event: bool = True

    eig_real_tol: float = 1e-10
    eig_imag_tol: float = 1e-12
    strict_1d: bool = True
    jac: Literal["auto", "fd", "analytic"] = "auto"
    fd_eps: float = 1e-6

    rk: HomoclinicRK45Config = field(default_factory=HomoclinicRK45Config)


@dataclass(frozen=True)
class HomoclinicFinderConfig:
    trace: HomoclinicBranchConfig = field(default_factory=HomoclinicBranchConfig)

    scan_n: int = 61
    max_bisect: int = 60

    gap_tol: float = 1e-4
    x_tol: float = 1e-4

    branch_mode: Literal["auto", "fixed"] = "auto"
    sign_u: int = +1


@dataclass
class HomoclinicMissResult:
    qualified: bool
    param_value: float

    eq: np.ndarray
    sign_u: int
    eps: float
    v_dir: np.ndarray
    x0: np.ndarray

    r_sec: float
    r_leave: float
    t_min: float

    t_cross: float
    x_cross: np.ndarray

    g: float
    q: float

    status: str
    diag: dict[str, object]


class HomoclinicFinderResult(NamedTuple):
    success: bool
    param_found: float
    miss: HomoclinicMissResult | None
    info: dict[str, object]


@dataclass
class HomoclinicTraceEvent:
    kind: str
    t: float
    x: np.ndarray
    info: dict[str, object] = field(default_factory=dict)


@dataclass
class HomoclinicTraceMeta:
    param_value: float
    eq: np.ndarray
    sign_u: int
    eps_used: float
    status: str
    success: bool
    event: HomoclinicTraceEvent | None
    t_cross: float
    x_cross: np.ndarray
    diag: dict[str, object]


class HomoclinicTraceResult(NamedTuple):
    t: np.ndarray
    X: np.ndarray
    meta: HomoclinicTraceMeta

    @property
    def branches(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return ([self.X], [])

    @property
    def branch_pos(self) -> list[np.ndarray]:
        return [self.X]

    @property
    def branch_neg(self) -> list[np.ndarray]:
        return []

    @property
    def kind(self) -> str:
        return "homoclinic"

    @property
    def success(self) -> bool:
        return bool(self.meta.success)


# =============================================================================
# Presets for homoclinic finder/tracer
# =============================================================================


@dataclass(frozen=True)
class HomoclinicPreset:
    """
    Preset configurations for homoclinic finder/tracer.

    Available presets:
    - "fast": Quick scan with lower accuracy, good for exploration.
    - "default": Balanced accuracy and speed for typical use.
    - "precise": High accuracy with more iterations, slower but robust.
    """
    name: str
    rk: HomoclinicRK45Config
    branch: HomoclinicBranchConfig
    scan_n: int
    max_bisect: int
    gap_tol: float
    x_tol: float


_HOMOCLINIC_PRESETS: dict[str, HomoclinicPreset] = {
    "fast": HomoclinicPreset(
        name="fast",
        rk=HomoclinicRK45Config(
            dt0=1e-2,
            dt_max=1e-1,
            atol=1e-8,
            rtol=1e-5,
            max_steps=500_000,
        ),
        branch=HomoclinicBranchConfig(
            r_leave=1e-2,
            t_leave_target=0.05,
            t_max=100.0,
            r_blow=200.0,
        ),
        scan_n=31,
        max_bisect=30,
        gap_tol=1e-3,
        x_tol=1e-3,
    ),
    "default": HomoclinicPreset(
        name="default",
        rk=HomoclinicRK45Config(
            dt0=1e-3,
            dt_max=5e-2,
            atol=1e-10,
            rtol=1e-7,
            max_steps=1_000_000,
        ),
        branch=HomoclinicBranchConfig(
            r_leave=1e-2,
            t_leave_target=0.05,
            t_max=200.0,
            r_blow=200.0,
        ),
        scan_n=61,
        max_bisect=60,
        gap_tol=1e-4,
        x_tol=1e-4,
    ),
    "precise": HomoclinicPreset(
        name="precise",
        rk=HomoclinicRK45Config(
            dt0=1e-4,
            dt_max=1e-2,
            atol=1e-12,
            rtol=1e-9,
            max_steps=3_000_000,
        ),
        branch=HomoclinicBranchConfig(
            r_leave=1e-3,
            t_leave_target=0.02,
            t_max=500.0,
            r_blow=500.0,
        ),
        scan_n=121,
        max_bisect=80,
        gap_tol=1e-6,
        x_tol=1e-6,
    ),
}


def _get_homoclinic_preset(preset: str | HomoclinicPreset) -> HomoclinicPreset:
    if isinstance(preset, HomoclinicPreset):
        return preset
    if preset not in _HOMOCLINIC_PRESETS:
        available = ", ".join(sorted(_HOMOCLINIC_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    return _HOMOCLINIC_PRESETS[preset]


def _build_homoclinic_branch_config_from_preset(
    preset: HomoclinicPreset,
    *,
    window: Sequence[tuple[float, float]] | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
    r_sec: float | None = None,
    t_min_event: float | None = None,
) -> HomoclinicBranchConfig:
    window_min = None
    window_max = None
    if window is not None:
        window_min = np.array([lo for lo, _ in window], dtype=float)
        window_max = np.array([hi for _, hi in window], dtype=float)

    return HomoclinicBranchConfig(
        eq_tol=preset.branch.eq_tol,
        eq_max_iter=preset.branch.eq_max_iter,
        eq_track_max_dist=preset.branch.eq_track_max_dist,
        eps_mode=preset.branch.eps_mode,
        eps=preset.branch.eps,
        eps_min=preset.branch.eps_min,
        eps_max=preset.branch.eps_max,
        r_leave=preset.branch.r_leave,
        t_leave_target=preset.branch.t_leave_target,
        r_sec=r_sec if r_sec is not None else preset.branch.r_sec,
        t_min_event=t_min_event if t_min_event is not None else preset.branch.t_min_event,
        t_max=t_max if t_max is not None else preset.branch.t_max,
        s_max=preset.branch.s_max,
        r_blow=r_blow if r_blow is not None else preset.branch.r_blow,
        window_min=window_min,
        window_max=window_max,
        require_leave_before_event=preset.branch.require_leave_before_event,
        eig_real_tol=preset.branch.eig_real_tol,
        eig_imag_tol=preset.branch.eig_imag_tol,
        strict_1d=preset.branch.strict_1d,
        jac=preset.branch.jac,
        fd_eps=preset.branch.fd_eps,
        rk=preset.rk,
    )


def _build_homoclinic_finder_config_from_kwargs(
    *,
    preset: str | HomoclinicPreset | None = None,
    trace_cfg: HomoclinicBranchConfig | None = None,
    window: Sequence[tuple[float, float]] | None = None,
    scan_n: int | None = None,
    max_bisect: int | None = None,
    gap_tol: float | None = None,
    x_tol: float | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
    r_sec: float | None = None,
    t_min_event: float | None = None,
) -> HomoclinicFinderConfig:
    pset = _get_homoclinic_preset(preset if preset is not None else "default")

    if trace_cfg is not None:
        if window is not None or t_max is not None or r_blow is not None or r_sec is not None or t_min_event is not None:
            window_min = None
            window_max = None
            if window is not None:
                window_min = np.array([lo for lo, _ in window], dtype=float)
                window_max = np.array([hi for _, hi in window], dtype=float)
            trace_cfg = HomoclinicBranchConfig(
                eq_tol=trace_cfg.eq_tol,
                eq_max_iter=trace_cfg.eq_max_iter,
                eq_track_max_dist=trace_cfg.eq_track_max_dist,
                eps_mode=trace_cfg.eps_mode,
                eps=trace_cfg.eps,
                eps_min=trace_cfg.eps_min,
                eps_max=trace_cfg.eps_max,
                r_leave=trace_cfg.r_leave,
                t_leave_target=trace_cfg.t_leave_target,
                r_sec=r_sec if r_sec is not None else trace_cfg.r_sec,
                t_min_event=t_min_event if t_min_event is not None else trace_cfg.t_min_event,
                t_max=t_max if t_max is not None else trace_cfg.t_max,
                s_max=trace_cfg.s_max,
                r_blow=r_blow if r_blow is not None else trace_cfg.r_blow,
                window_min=window_min if window_min is not None else trace_cfg.window_min,
                window_max=window_max if window_max is not None else trace_cfg.window_max,
                require_leave_before_event=trace_cfg.require_leave_before_event,
                eig_real_tol=trace_cfg.eig_real_tol,
                eig_imag_tol=trace_cfg.eig_imag_tol,
                strict_1d=trace_cfg.strict_1d,
                jac=trace_cfg.jac,
                fd_eps=trace_cfg.fd_eps,
                rk=trace_cfg.rk,
            )
    else:
        trace_cfg = _build_homoclinic_branch_config_from_preset(
            pset,
            window=window,
            t_max=t_max,
            r_blow=r_blow,
            r_sec=r_sec,
            t_min_event=t_min_event,
        )

    final_scan_n = scan_n if scan_n is not None else pset.scan_n
    final_max_bisect = max_bisect if max_bisect is not None else pset.max_bisect
    final_gap_tol = gap_tol if gap_tol is not None else pset.gap_tol
    final_x_tol = x_tol if x_tol is not None else pset.x_tol

    return HomoclinicFinderConfig(
        trace=trace_cfg,
        scan_n=final_scan_n,
        max_bisect=final_max_bisect,
        gap_tol=final_gap_tol,
        x_tol=final_x_tol,
    )


def _validate_homoclinic_rk45_cfg(cfg: HomoclinicRK45Config) -> None:
    if cfg.dt0 <= 0.0:
        raise ValueError("rk.dt0 must be positive")
    if cfg.min_step <= 0.0:
        raise ValueError("rk.min_step must be positive")
    if cfg.dt_max <= 0.0:
        raise ValueError("rk.dt_max must be positive")
    if cfg.min_step > cfg.dt_max:
        raise ValueError("rk.min_step must be <= rk.dt_max")
    if cfg.atol <= 0.0:
        raise ValueError("rk.atol must be positive")
    if cfg.rtol <= 0.0:
        raise ValueError("rk.rtol must be positive")
    if cfg.safety <= 0.0:
        raise ValueError("rk.safety must be positive")
    if cfg.max_steps <= 0:
        raise ValueError("rk.max_steps must be positive")


def _validate_homoclinic_branch_cfg(cfg: HomoclinicBranchConfig) -> None:
    if cfg.eq_tol <= 0.0:
        raise ValueError("eq_tol must be positive")
    if cfg.eq_max_iter <= 0:
        raise ValueError("eq_max_iter must be positive")
    if cfg.eps <= 0.0:
        raise ValueError("eps must be positive")
    if cfg.eps_min <= 0.0:
        raise ValueError("eps_min must be positive")
    if cfg.eps_max <= 0.0:
        raise ValueError("eps_max must be positive")
    if cfg.eps_min > cfg.eps_max:
        raise ValueError("eps_min must be <= eps_max")
    if cfg.r_leave <= 0.0:
        raise ValueError("r_leave must be positive")
    if cfg.t_leave_target <= 0.0:
        raise ValueError("t_leave_target must be positive")
    if cfg.r_sec <= 0.0:
        raise ValueError("r_sec must be positive")
    if cfg.t_min_event < 0.0:
        raise ValueError("t_min_event must be non-negative")
    if cfg.t_max <= 0.0:
        raise ValueError("t_max must be positive")
    if cfg.s_max <= 0.0:
        raise ValueError("s_max must be positive")
    if cfg.r_blow <= 0.0:
        raise ValueError("r_blow must be positive")
    if cfg.eig_real_tol < 0.0:
        raise ValueError("eig_real_tol must be non-negative")
    if cfg.eig_imag_tol < 0.0:
        raise ValueError("eig_imag_tol must be non-negative")
    if cfg.fd_eps <= 0.0:
        raise ValueError("fd_eps must be positive")
    if cfg.jac not in ("auto", "fd", "analytic"):
        raise ValueError("jac must be 'auto', 'fd', or 'analytic'")
    _validate_homoclinic_rk45_cfg(cfg.rk)


def _validate_homoclinic_finder_cfg(cfg: HomoclinicFinderConfig) -> None:
    if cfg.scan_n < 2:
        raise ValueError("scan_n must be >= 2")
    if cfg.max_bisect < 0:
        raise ValueError("max_bisect must be >= 0")
    if cfg.gap_tol < 0.0:
        raise ValueError("gap_tol must be non-negative")
    if cfg.x_tol < 0.0:
        raise ValueError("x_tol must be non-negative")
    if cfg.branch_mode not in ("auto", "fixed"):
        raise ValueError("branch_mode must be 'auto' or 'fixed'")
    _validate_homoclinic_branch_cfg(cfg.trace)


def _eig_index1_saddle_data(
    J: np.ndarray,
    *,
    real_tol: float,
    imag_tol: float,
    strict_1d: bool,
) -> tuple[bool, tuple[complex, np.ndarray, np.ndarray, int] | None, dict[str, object]]:
    w, V = np.linalg.eig(J)
    re = np.real(w)

    unstable = np.where(re > real_tol)[0]
    stable = np.where(re < -real_tol)[0]

    info: dict[str, object] = {
        "eigvals": w,
        "unstable_count": int(unstable.size),
        "stable_count": int(stable.size),
    }

    if strict_1d and unstable.size != 1:
        info["fail"] = "not_index1_saddle"
        return False, None, info
    if unstable.size == 0:
        info["fail"] = "no_unstable"
        return False, None, info
    if stable.size < 1:
        info["fail"] = "no_stable"
        return False, None, info

    idx = int(unstable[0])
    lam = w[idx]
    v_u = V[:, idx]

    if np.max(np.abs(np.imag(lam))) > imag_tol or np.max(np.abs(np.imag(v_u))) > imag_tol:
        info["fail"] = "complex_unstable_mode_not_supported"
        return False, None, info
    v_u = np.real(v_u)

    wt, Lt = np.linalg.eig(J.T)
    jmatch = int(np.argmin(np.abs(wt - lam)))
    l_u = Lt[:, jmatch]
    if np.max(np.abs(np.imag(wt[jmatch]))) > imag_tol or np.max(np.abs(np.imag(l_u))) > imag_tol:
        info["fail"] = "complex_left_mode_not_supported"
        return False, None, info
    l_u = np.real(l_u)

    denom = float(l_u @ v_u)
    if abs(denom) < 1e-14:
        info["fail"] = "left_right_normalization_failed"
        return False, None, info
    l_u = l_u / denom

    info["unstable_index"] = idx
    return True, (lam, v_u, l_u, idx), info


def _integrate_homoclinic_branch(
    ctx: _HeteroclinicContext,
    params_vec: np.ndarray,
    *,
    x0: np.ndarray,
    x_eq: np.ndarray,
    cfg: HomoclinicBranchConfig,
) -> tuple[str, np.ndarray, np.ndarray]:
    wmin = None if cfg.window_min is None else np.array(cfg.window_min, float).reshape(-1,)
    wmax = None if cfg.window_max is None else np.array(cfg.window_max, float).reshape(-1,)
    if wmin is not None and wmin.shape != x_eq.shape:
        raise ValueError("window_min must have shape (n_state,)")
    if wmax is not None and wmax.shape != x_eq.shape:
        raise ValueError("window_max must have shape (n_state,)")

    def blow_fn(_t: float, x: np.ndarray) -> bool:
        if not np.all(np.isfinite(x)):
            return True
        if float(np.linalg.norm(np.array(x, float) - x_eq)) > float(cfg.r_blow):
            return True
        if not _in_window(np.array(x, float), wmin, wmax):
            return True
        return False

    use_numba = ctx.use_jit and _rk45_integrate_numba is not None
    update_aux_fn = ctx.update_aux_fn
    if use_numba:
        if ctx.do_aux_pre or ctx.do_aux_post:
            if not _is_numba_dispatcher(ctx.update_aux_fn):
                use_numba = False
        else:
            if _update_aux_noop_numba is not None:
                update_aux_fn = _update_aux_noop_numba
            else:
                use_numba = False

    n_state = int(x_eq.size)
    has_window = wmin is not None and wmax is not None
    wmin_arr = np.array(wmin if wmin is not None else np.zeros(n_state), float).reshape(-1,)
    wmax_arr = np.array(wmax if wmax is not None else np.zeros(n_state), float).reshape(-1,)
    if use_numba and has_window:
        if wmin_arr.size != n_state or wmax_arr.size != n_state:
            use_numba = False

    ctx.prep_ws(x0)
    if use_numba:
        status, ts, xs, _ = _rk45_integrate_numba_wrapper(
            ctx.rhs_fn,
            params_vec,
            ctx.runtime_ws,
            update_aux_fn,
            ctx.do_aux_pre,
            ctx.do_aux_post,
            x0,
            float(ctx.t_eval),
            float(ctx.t_eval) + float(cfg.t_max),
            cfg.rk,
            sign_factor=1.0,
            x_eq=x_eq,
            r_blow=float(cfg.r_blow),
            wmin=wmin_arr,
            wmax=wmax_arr,
            has_window=bool(has_window),
            t_min_event=float(cfg.t_min_event),
            require_leave_before_event=bool(cfg.require_leave_before_event),
            r_leave=float(cfg.r_leave),
            target_center=np.zeros(n_state, float),
            target_r=0.0,
            enable_target=False,
            section_kind=0,
            section_center=np.zeros(n_state, float),
            section_normal=np.zeros(n_state, float),
            section_tau=np.zeros(n_state, float),
            section_radius=0.0,
            section_xsec=np.zeros(n_state, float),
            rho_max=-1.0,
            enable_section=False,
        )
    else:
        def rhs_eval(t: float, x: np.ndarray, out: np.ndarray) -> None:
            ctx.rhs_fn(t, x, out, params_vec, ctx.runtime_ws)

        status, ts, xs, _ = _rk45_integrate(
            rhs_eval,
            params_vec,
            ctx.runtime_ws,
            ctx.update_aux_fn,
            ctx.do_aux_pre,
            ctx.do_aux_post,
            x0,
            float(ctx.t_eval),
            float(ctx.t_eval) + float(cfg.t_max),
            cfg.rk,
            blow_fn=blow_fn,
            event_fn=None,
        )

    if status == "blow":
        x_last = xs[-1]
        if (wmin is not None and wmax is not None) and (not _in_window(x_last, wmin, wmax)):
            status = "window"
        else:
            status = "blow"

    s_len = _polyline_arclength(xs)
    if s_len > float(cfg.s_max) and status in ("ok", "max_steps"):
        status = "smax"

    return str(status), np.array(ts, float), np.array(xs, float)


def _scan_homoclinic_section(
    ts: np.ndarray,
    xs: np.ndarray,
    *,
    x_eq: np.ndarray,
    l_u: np.ndarray,
    v_proj: np.ndarray,
    cfg: HomoclinicBranchConfig,
) -> tuple[bool, float, np.ndarray, float, float]:
    if xs.ndim != 2 or xs.shape[0] == 0:
        return False, float("nan"), np.array(x_eq, float), float("nan"), float("inf")

    def a_of(x: np.ndarray) -> float:
        return float(l_u @ (x - x_eq))

    def b_of(x: np.ndarray) -> float:
        xi = x - x_eq
        a = float(l_u @ xi)
        stable = xi - v_proj * a
        return float(np.linalg.norm(stable))

    left = False
    best_q = float("inf")
    best_t = float("nan")
    best_x = xs[-1].copy()

    b_prev = b_of(xs[0])
    s_prev = b_prev - float(cfg.r_sec)

    for i in range(1, len(ts)):
        t0, x_prev = float(ts[i - 1]), xs[i - 1]
        t1, x_curr = float(ts[i]), xs[i]

        if t1 < float(cfg.t_min_event):
            b_prev = b_of(x_curr)
            s_prev = b_prev - float(cfg.r_sec)
            continue

        r_curr = float(np.linalg.norm(x_curr - x_eq))
        if (not left) and (r_curr >= float(cfg.r_leave)):
            left = True

        b_curr = b_of(x_curr)
        s_curr = b_curr - float(cfg.r_sec)

        q_here = abs(s_curr)
        if q_here < best_q:
            best_q = float(q_here)
            best_t = float(t1)
            best_x = x_curr.copy()

        if left and (s_prev > 0.0) and (s_curr <= 0.0):
            denom = (s_curr - s_prev)
            if denom == 0.0:
                alpha = 0.0
            else:
                alpha = (0.0 - s_prev) / denom
                alpha = min(1.0, max(0.0, alpha))
            t_cross = t0 + alpha * (t1 - t0)
            x_cross = x_prev + alpha * (x_curr - x_prev)
            g = a_of(x_cross)
            return True, float(t_cross), np.array(x_cross, float), float(g), 0.0

        b_prev = b_curr
        s_prev = s_curr

    return False, float(best_t), np.array(best_x, float), float("nan"), float(best_q)


def _homoclinic_miss_index1(
    ctx: _HeteroclinicContext,
    params_vec: np.ndarray,
    param_value: float,
    *,
    eq_guess: np.ndarray,
    eq_prev: np.ndarray | None,
    sign_u: int,
    cfg: HomoclinicFinderConfig,
) -> HomoclinicMissResult:
    diag: dict[str, object] = {"param_value": float(param_value)}

    def rhs_eval_reset(x: np.ndarray, out: np.ndarray) -> None:
        ctx.prep_ws(x)
        ctx.rhs_fn(ctx.t_eval, x, out, params_vec, ctx.runtime_ws)

    def jac_eval(x: np.ndarray) -> np.ndarray:
        cfg_branch = cfg.trace
        if cfg_branch.jac == "fd" or (cfg_branch.jac == "auto" and ctx.jac_fn is None):
            fx = np.empty((x.size,), dtype=ctx.model.dtype)
            rhs_eval_reset(x, fx)
            J = np.zeros((x.size, x.size), dtype=float)
            for j in range(x.size):
                step = cfg_branch.fd_eps * (1.0 + abs(float(x[j])))
                if step == 0.0:
                    step = cfg_branch.fd_eps
                x_step = np.array(x, copy=True)
                x_step[j] += step
                f_step = np.empty((x.size,), dtype=ctx.model.dtype)
                rhs_eval_reset(x_step, f_step)
                J[:, j] = (f_step - fx) / step
            return J

        if cfg_branch.jac == "analytic" and ctx.jac_fn is None:
            raise ValueError("jac='analytic' requires a model Jacobian.")
        if ctx.jac_fn is None:
            raise ValueError("Jacobian is not available (jac='auto' found none).")

        jac_out = np.zeros((x.size, x.size), dtype=ctx.model.dtype)
        ctx.prep_ws(x)
        ctx.jac_fn(ctx.t_eval, x, params_vec, jac_out, ctx.runtime_ws)
        return np.array(jac_out, copy=True)

    ok_eq, x_eq, eqinfo = _solve_equilibrium_locked(
        rhs_eval_reset,
        jac_eval,
        np.array(eq_guess, float),
        x_prev=(None if eq_prev is None else np.array(eq_prev, float)),
        eq_tol=float(cfg.trace.eq_tol),
        eq_max_iter=int(cfg.trace.eq_max_iter),
        eq_track_max_dist=cfg.trace.eq_track_max_dist,
        r_leave_for_default=float(cfg.trace.r_leave),
    )
    diag["eq"] = eqinfo
    if not ok_eq:
        status = "eq_jump" if eqinfo.get("fail_mode") == "eq_jump" else "eq_fail"
        z = np.array(x_eq, float, copy=True)
        return HomoclinicMissResult(
            qualified=False,
            param_value=float(param_value),
            eq=z,
            sign_u=int(sign_u),
            eps=float(cfg.trace.eps),
            v_dir=np.zeros_like(z),
            x0=z.copy(),
            r_sec=float(cfg.trace.r_sec),
            r_leave=float(cfg.trace.r_leave),
            t_min=float(cfg.trace.t_min_event),
            t_cross=float("nan"),
            x_cross=z.copy(),
            g=float("nan"),
            q=float("inf"),
            status=status,
            diag=diag,
        )

    J = jac_eval(np.array(x_eq, float))
    ok_e, ed, einfo = _eig_index1_saddle_data(
        J,
        real_tol=float(cfg.trace.eig_real_tol),
        imag_tol=float(cfg.trace.eig_imag_tol),
        strict_1d=bool(cfg.trace.strict_1d),
    )
    diag["saddle"] = einfo
    if not ok_e or ed is None:
        z = np.array(x_eq, float)
        return HomoclinicMissResult(
            qualified=False,
            param_value=float(param_value),
            eq=z,
            sign_u=int(sign_u),
            eps=float(cfg.trace.eps),
            v_dir=np.zeros_like(z),
            x0=z.copy(),
            r_sec=float(cfg.trace.r_sec),
            r_leave=float(cfg.trace.r_leave),
            t_min=float(cfg.trace.t_min_event),
            t_cross=float("nan"),
            x_cross=z.copy(),
            g=float("nan"),
            q=float("inf"),
            status="saddle_fail",
            diag=diag,
        )

    lam, v_raw, l_u, _ = ed
    v_norm = float(np.linalg.norm(v_raw))
    if not np.isfinite(v_norm) or v_norm <= 1e-300:
        diag["fail"] = "unstable_vector_norm_bad"
        z = np.array(x_eq, float)
        return HomoclinicMissResult(
            qualified=False,
            param_value=float(param_value),
            eq=z,
            sign_u=int(sign_u),
            eps=float(cfg.trace.eps),
            v_dir=np.zeros_like(z),
            x0=z.copy(),
            r_sec=float(cfg.trace.r_sec),
            r_leave=float(cfg.trace.r_leave),
            t_min=float(cfg.trace.t_min_event),
            t_cross=float("nan"),
            x_cross=z.copy(),
            g=float("nan"),
            q=float("inf"),
            status="saddle_fail",
            diag=diag,
        )
    v_dir = np.array(v_raw, float) / v_norm

    denom = float(l_u @ v_dir)
    if abs(denom) < 1e-14:
        diag["fail"] = "projection_normalization_failed"
        z = np.array(x_eq, float)
        return HomoclinicMissResult(
            qualified=False,
            param_value=float(param_value),
            eq=z,
            sign_u=int(sign_u),
            eps=float(cfg.trace.eps),
            v_dir=np.zeros_like(z),
            x0=z.copy(),
            r_sec=float(cfg.trace.r_sec),
            r_leave=float(cfg.trace.r_leave),
            t_min=float(cfg.trace.t_min_event),
            t_cross=float("nan"),
            x_cross=z.copy(),
            g=float("nan"),
            q=float("inf"),
            status="saddle_fail",
            diag=diag,
        )
    v_proj = v_dir / denom

    sign = +1.0 if int(sign_u) >= 0 else -1.0
    growth_rate = float(np.real(lam))
    eps = float(cfg.trace.eps) if cfg.trace.eps_mode == "fixed" else _choose_eps_leave(
        r_leave=float(cfg.trace.r_leave),
        rate=growth_rate,
        t_leave_target=float(cfg.trace.t_leave_target),
        eps_min=float(cfg.trace.eps_min),
        eps_max=float(cfg.trace.eps_max),
        fallback=float(cfg.trace.eps),
    )
    x0 = np.array(x_eq, float) + sign * eps * v_dir

    status_int, ts, xs = _integrate_homoclinic_branch(
        ctx,
        params_vec,
        x0=np.array(x0, ctx.model.dtype),
        x_eq=np.array(x_eq, ctx.model.dtype),
        cfg=cfg.trace,
    )
    diag["integrate_status"] = status_int

    qualified, t_cross, x_cross, g_val, q_val = _scan_homoclinic_section(
        ts,
        xs,
        x_eq=np.array(x_eq, float),
        l_u=np.array(l_u, float),
        v_proj=np.array(v_proj, float),
        cfg=cfg.trace,
    )

    status = str(status_int)
    if not qualified and status in ("ok", "max_steps"):
        status = "no_cross"

    return HomoclinicMissResult(
        qualified=bool(qualified),
        param_value=float(param_value),
        eq=np.array(x_eq, float),
        sign_u=int(sign_u),
        eps=float(eps),
        v_dir=np.array(v_dir, float),
        x0=np.array(x0, float),
        r_sec=float(cfg.trace.r_sec),
        r_leave=float(cfg.trace.r_leave),
        t_min=float(cfg.trace.t_min_event),
        t_cross=float(t_cross),
        x_cross=np.array(x_cross, float),
        g=float(g_val),
        q=float(q_val),
        status=status,
        diag=diag,
    )


def _homoclinic_is_success(cfg: HomoclinicFinderConfig, r: HomoclinicMissResult) -> bool:
    if not (r.qualified and np.isfinite(r.g) and np.isfinite(r.q)):
        return False
    if abs(float(r.g)) > float(cfg.gap_tol):
        return False
    if float(r.q) > float(cfg.x_tol):
        return False
    return True


def _find_homoclinic_param(
    ctx: _HeteroclinicContext,
    *,
    param_min: float,
    param_max: float,
    param_init: float,
    eq_guess: np.ndarray,
    cfg: HomoclinicFinderConfig,
) -> tuple[bool, float, HomoclinicMissResult | None, dict[str, object]]:
    param_min = float(param_min)
    param_max = float(param_max)
    param_init = float(param_init)

    param_grid, idx0 = _scan_grid_centered(param_min, param_max, param_init, int(cfg.scan_n))

    if cfg.branch_mode == "fixed":
        combos = [+1 if int(cfg.sign_u) >= 0 else -1]
    else:
        combos = [+1, -1]

    best_overall: tuple[float, HomoclinicMissResult, dict[str, object]] | None = None

    for sign_u in combos:
        results: list[HomoclinicMissResult | None] = [None] * len(param_grid)

        info: dict[str, object] = {
            "param_min": param_min,
            "param_max": param_max,
            "param_init": param_init,
            "param_used": float(param_grid[idx0]),
            "scan_n": int(cfg.scan_n),
            "sign_u": int(sign_u),
            "branch_mode": str(cfg.branch_mode),
            "eq_track_max_dist": (
                _track_max_dist_default(cfg.trace.r_leave)
                if cfg.trace.eq_track_max_dist is None
                else float(cfg.trace.eq_track_max_dist)
            ),
            "qualified_count": 0,
            "eq_jump_count": 0,
            "eq_fail_count": 0,
            "best_by_abs_gap": None,
            "best_by_q": None,
            "fail": None,
        }

        params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_grid[idx0]))
        r0 = _homoclinic_miss_index1(
            ctx,
            params_vec,
            float(param_grid[idx0]),
            eq_guess=np.array(eq_guess, float),
            eq_prev=None,
            sign_u=sign_u,
            cfg=cfg,
        )
        results[idx0] = r0

        if not r0.qualified and r0.status in ("eq_fail", "eq_jump"):
            info["fail"] = "anchor_equilibrium_failed"
            continue

        eq_prev_right = r0.eq.copy()
        for i in range(idx0 + 1, len(param_grid)):
            param_val = float(param_grid[i])
            params_vec = _params_with_override(ctx.params_base, ctx.param_index, param_val)
            r = _homoclinic_miss_index1(
                ctx,
                params_vec,
                param_val,
                eq_guess=eq_prev_right,
                eq_prev=eq_prev_right,
                sign_u=sign_u,
                cfg=cfg,
            )
            results[i] = r
            if r.status == "eq_jump":
                info["eq_jump_count"] += 1
            elif r.status == "eq_fail":
                info["eq_fail_count"] += 1
            else:
                eq_prev_right = r.eq.copy()

        eq_prev_left = r0.eq.copy()
        for i in range(idx0 - 1, -1, -1):
            param_val = float(param_grid[i])
            params_vec = _params_with_override(ctx.params_base, ctx.param_index, param_val)
            r = _homoclinic_miss_index1(
                ctx,
                params_vec,
                param_val,
                eq_guess=eq_prev_left,
                eq_prev=eq_prev_left,
                sign_u=sign_u,
                cfg=cfg,
            )
            results[i] = r
            if r.status == "eq_jump":
                info["eq_jump_count"] += 1
            elif r.status == "eq_fail":
                info["eq_fail_count"] += 1
            else:
                eq_prev_left = r.eq.copy()

        qual = [r for r in results if (r is not None and r.qualified and np.isfinite(r.g))]
        info["qualified_count"] = int(len(qual))
        if not qual:
            info["fail"] = "no_qualified_points_in_scan"
            allr = [r for r in results if r is not None]
            best_any = min(allr, key=lambda rr: rr.q if np.isfinite(rr.q) else float("inf"))
            info["best_by_q"] = {
                "param_value": best_any.param_value,
                "g": best_any.g,
                "q": best_any.q,
                "status": best_any.status,
            }
            continue

        qual_sorted = sorted(qual, key=lambda rr: rr.param_value)
        best_by_q = min(qual_sorted, key=lambda rr: rr.q)
        best_by_abs_g = min(qual_sorted, key=lambda rr: abs(rr.g))

        info["best_by_abs_gap"] = {
            "param_value": best_by_abs_g.param_value,
            "gap": best_by_abs_g.g,
            "q": best_by_abs_g.q,
            "status": best_by_abs_g.status,
        }
        info["best_by_q"] = {
            "param_value": best_by_q.param_value,
            "gap": best_by_q.g,
            "q": best_by_q.q,
            "status": best_by_q.status,
        }

        if _homoclinic_is_success(cfg, best_by_q):
            info["param_found"] = float(best_by_q.param_value)
            info["gap_found"] = float(best_by_q.g)
            info["q_found"] = float(best_by_q.q)
            info["t_cross"] = float(best_by_q.t_cross)
            info["section"] = {
                "r_sec": float(best_by_q.r_sec),
                "r_leave": float(best_by_q.r_leave),
                "t_min_event": float(best_by_q.t_min),
            }
            info["departure"] = {"eps": float(best_by_q.eps), "sign_u": int(best_by_q.sign_u)}
            info["bisection_iters"] = 0
            return True, float(best_by_q.param_value), best_by_q, info

        bracket: tuple[HomoclinicMissResult, HomoclinicMissResult] | None = None
        for a, b in zip(qual_sorted[:-1], qual_sorted[1:]):
            if a.g == 0.0:
                bracket = (a, a)
                break
            if a.g * b.g < 0.0:
                bracket = (a, b)
                break
        if bracket is None:
            last = qual_sorted[-1]
            if np.isfinite(last.g) and abs(last.g) <= float(cfg.gap_tol):
                bracket = (last, last)

        if bracket is None:
            info["fail"] = "no_sign_change_in_qualified_scan"
            score = float(best_by_q.q + 0.1 * abs(best_by_q.g))
            if best_overall is None or score < best_overall[0]:
                best_overall = (score, best_by_q, info)
            continue

        a, b = bracket
        if a.param_value == b.param_value:
            r_star = a
            info["bisection_iters"] = 0
        else:
            param_lo, param_hi = float(a.param_value), float(b.param_value)
            r_lo, r_hi = a, b
            r_star = r_lo
            iters = 0

            for it in range(int(cfg.max_bisect)):
                iters = it + 1
                param_mid = 0.5 * (param_lo + param_hi)

                if abs(param_mid - param_lo) <= abs(param_mid - param_hi):
                    eq_seed = r_lo.eq
                    eq_prev = r_lo.eq
                else:
                    eq_seed = r_hi.eq
                    eq_prev = r_hi.eq

                params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_mid))
                r_mid = _homoclinic_miss_index1(
                    ctx,
                    params_vec,
                    float(param_mid),
                    eq_guess=eq_seed,
                    eq_prev=eq_prev,
                    sign_u=sign_u,
                    cfg=cfg,
                )

                if _homoclinic_is_success(cfg, r_mid):
                    r_star = r_mid
                    break

                if (not r_mid.qualified) or (not np.isfinite(r_mid.g)):
                    if r_lo.q <= r_hi.q:
                        param_hi = float(param_mid)
                    else:
                        param_lo = float(param_mid)
                    continue

                if r_lo.g * r_mid.g <= 0.0:
                    param_hi, r_hi = float(param_mid), r_mid
                else:
                    param_lo, r_lo = float(param_mid), r_mid

            else:
                r_star = min([r_lo, r_hi], key=lambda rr: (rr.q, abs(rr.g)))

            info["bisection_iters"] = int(iters)

        if not _homoclinic_is_success(cfg, r_star):
            info["fail"] = "bisection_did_not_converge"
            cand = r_star if (r_star.qualified and np.isfinite(r_star.g)) else best_by_q
            score = float(cand.q + 0.1 * abs(cand.g))
            if best_overall is None or score < best_overall[0]:
                best_overall = (score, cand, info)
            continue

        info["param_found"] = float(r_star.param_value)
        info["gap_found"] = float(r_star.g)
        info["q_found"] = float(r_star.q)
        info["t_cross"] = float(r_star.t_cross)
        info["section"] = {
            "r_sec": float(r_star.r_sec),
            "r_leave": float(r_star.r_leave),
            "t_min_event": float(r_star.t_min),
        }
        info["departure"] = {"eps": float(r_star.eps), "sign_u": int(r_star.sign_u)}

        return True, float(r_star.param_value), r_star, info

    if best_overall is not None:
        _, best_r, best_info = best_overall
        best_info = dict(best_info)
        best_info["fail"] = best_info.get("fail", "no_branch_converged")
        best_info["param_candidate"] = float(best_r.param_value)
        best_info["gap_candidate"] = float(best_r.g) if np.isfinite(best_r.g) else float("nan")
        best_info["q_candidate"] = float(best_r.q)
        return False, float("nan"), None, best_info

    return False, float("nan"), None, {"fail": "no_branch_had_qualified_points"}


def homoclinic_finder(
    sim: "Sim",
    *,
    param: str | int | None,
    param_min: float,
    param_max: float,
    param_init: float,
    eq_guess: Mapping[str, float] | Sequence[float] | np.ndarray,
    cfg: HomoclinicFinderConfig | None = None,
    # Simplified API: preset and flattened kwargs
    preset: str | HomoclinicPreset | None = None,
    trace_cfg: HomoclinicBranchConfig | None = None,
    window: Sequence[tuple[float, float]] | None = None,
    scan_n: int | None = None,
    max_bisect: int | None = None,
    gap_tol: float | None = None,
    x_tol: float | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
    r_sec: float | None = None,
    t_min_event: float | None = None,
    # Base params
    params: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    t: float | None = None,
) -> HomoclinicFinderResult:
    """
    Find a homoclinic orbit associated with a saddle equilibrium.

    Parameters
    ----------
    sim : Sim
        The simulation object.
    param : str | int | None
        The parameter to vary (name or index). If None, model must have exactly one parameter.
    param_min, param_max : float
        Search bounds for the parameter.
    param_init : float
        Initial guess for the parameter value.
    eq_guess : Mapping | Sequence | np.ndarray
        Initial equilibrium guess near the homoclinic orbit.

    cfg : HomoclinicFinderConfig | None
        Full configuration object (advanced). If provided, overrides preset/trace_cfg/kwargs.

    Simplified API (preferred for typical use):
    -------------------------------------------
    preset : str | HomoclinicPreset | None
        Preset configuration: "fast", "default", or "precise".
    trace_cfg : HomoclinicBranchConfig | None
        Trace configuration for the unstable branch.
    window : Sequence[tuple[float, float]] | None
        State-space window as [(x_min, x_max), (y_min, y_max), ...].
    scan_n : int | None
        Number of parameter scan points.
    max_bisect : int | None
        Maximum bisection iterations.
    gap_tol : float | None
        Tolerance for the signed return value g.
    x_tol : float | None
        Tolerance for section miss distance.
    t_max : float | None
        Maximum integration time for manifold tracing.
    r_blow : float | None
        Blow-up radius for manifold tracing.
    r_sec : float | None
        Section radius in the stable complement.
    t_min_event : float | None
        Minimum time before checking the return section.

    params : Mapping | Sequence | np.ndarray | None
        Fixed parameter values (other than the search parameter).
    t : float | None
        Evaluation time for auxiliary variables.
    """
    if param_min >= param_max:
        raise ValueError("param_min must be < param_max")

    has_simplified_kwargs = any(
        x is not None
        for x in [
            preset,
            trace_cfg,
            window,
            scan_n,
            max_bisect,
            gap_tol,
            x_tol,
            t_max,
            r_blow,
            r_sec,
            t_min_event,
        ]
    )
    if cfg is not None and has_simplified_kwargs:
        raise ValueError(
            "Cannot use 'cfg' together with simplified kwargs (preset, trace_cfg, window, etc.). "
            "Use either 'cfg' for full control, or the simplified kwargs."
        )

    ctx = _build_heteroclinic_context(
        sim,
        params=params,
        param=param,
        t=t,
        caller="homoclinic_finder",
    )

    if cfg is None:
        cfg = _build_homoclinic_finder_config_from_kwargs(
            preset=preset,
            trace_cfg=trace_cfg,
            window=window,
            scan_n=scan_n,
            max_bisect=max_bisect,
            gap_tol=gap_tol,
            x_tol=x_tol,
            t_max=t_max,
            r_blow=r_blow,
            r_sec=r_sec,
            t_min_event=t_min_event,
        )
    else:
        if not isinstance(cfg, HomoclinicFinderConfig):
            raise ValueError("cfg must be a HomoclinicFinderConfig")

    _validate_homoclinic_finder_cfg(cfg)

    eq_guess_vec = _resolve_fixed_point(ctx.model, eq_guess)

    success, param_found, miss, info = _find_homoclinic_param(
        ctx,
        param_min=float(param_min),
        param_max=float(param_max),
        param_init=float(param_init),
        eq_guess=np.array(eq_guess_vec, float),
        cfg=cfg,
    )

    return HomoclinicFinderResult(
        success=bool(success),
        param_found=float(param_found),
        miss=miss,
        info=info,
    )


def homoclinic_tracer(
    sim: "Sim",
    *,
    param: str | int | None,
    param_value: float,
    eq: Mapping[str, float] | Sequence[float] | np.ndarray,
    sign_u: int,
    cfg_u: HomoclinicBranchConfig | None = None,
    # Simplified API: preset and flattened kwargs
    preset: str | HomoclinicPreset | None = None,
    trace_cfg: HomoclinicBranchConfig | None = None,
    window: Sequence[tuple[float, float]] | None = None,
    t_max: float | None = None,
    r_blow: float | None = None,
    r_sec: float | None = None,
    t_min_event: float | None = None,
    # Base params
    params: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    t: float | None = None,
) -> HomoclinicTraceResult:
    """
    Trace a homoclinic excursion from a saddle equilibrium.

    Parameters
    ----------
    sim : Sim
        The simulation object.
    param : str | int | None
        The parameter name or index. If None, model must have exactly one parameter.
    param_value : float
        The parameter value at which to trace the orbit.
    eq : Mapping | Sequence | np.ndarray
        Equilibrium point (saddle) to trace from.
    sign_u : int
        Sign for unstable eigenvector direction (+1 or -1).
    cfg_u : HomoclinicBranchConfig | None
        Full trace configuration (advanced). If provided, overrides preset/trace_cfg/kwargs.

    Simplified API (preferred for typical use):
    -------------------------------------------
    preset : str | HomoclinicPreset | None
        Preset configuration: "fast", "default", or "precise".
    trace_cfg : HomoclinicBranchConfig | None
        Trace configuration (alternative to cfg_u).
    window : Sequence[tuple[float, float]] | None
        State-space window as [(x_min, x_max), (y_min, y_max), ...].
    t_max : float | None
        Maximum integration time for manifold tracing.
    r_blow : float | None
        Blow-up radius for manifold tracing.
    r_sec : float | None
        Section radius in the stable complement.
    t_min_event : float | None
        Minimum time before checking the return section.
    """
    has_simplified_kwargs = any(
        x is not None for x in [preset, trace_cfg, window, t_max, r_blow, r_sec, t_min_event]
    )
    if cfg_u is not None and has_simplified_kwargs:
        raise ValueError(
            "Cannot use 'cfg_u' together with simplified kwargs (preset, trace_cfg, window, etc.). "
            "Use either 'cfg_u' for full control, or the simplified kwargs."
        )

    ctx = _build_heteroclinic_context(
        sim,
        params=params,
        param=param,
        t=t,
        caller="homoclinic_tracer",
    )

    if cfg_u is None:
        if trace_cfg is not None:
            cfg_u = trace_cfg
            if window is not None or t_max is not None or r_blow is not None or r_sec is not None or t_min_event is not None:
                window_min = None
                window_max = None
                if window is not None:
                    window_min = np.array([lo for lo, _ in window], dtype=float)
                    window_max = np.array([hi for _, hi in window], dtype=float)
                cfg_u = HomoclinicBranchConfig(
                    eq_tol=trace_cfg.eq_tol,
                    eq_max_iter=trace_cfg.eq_max_iter,
                    eq_track_max_dist=trace_cfg.eq_track_max_dist,
                    eps_mode=trace_cfg.eps_mode,
                    eps=trace_cfg.eps,
                    eps_min=trace_cfg.eps_min,
                    eps_max=trace_cfg.eps_max,
                    r_leave=trace_cfg.r_leave,
                    t_leave_target=trace_cfg.t_leave_target,
                    r_sec=r_sec if r_sec is not None else trace_cfg.r_sec,
                    t_min_event=t_min_event if t_min_event is not None else trace_cfg.t_min_event,
                    t_max=t_max if t_max is not None else trace_cfg.t_max,
                    s_max=trace_cfg.s_max,
                    r_blow=r_blow if r_blow is not None else trace_cfg.r_blow,
                    window_min=window_min if window_min is not None else trace_cfg.window_min,
                    window_max=window_max if window_max is not None else trace_cfg.window_max,
                    require_leave_before_event=trace_cfg.require_leave_before_event,
                    eig_real_tol=trace_cfg.eig_real_tol,
                    eig_imag_tol=trace_cfg.eig_imag_tol,
                    strict_1d=trace_cfg.strict_1d,
                    jac=trace_cfg.jac,
                    fd_eps=trace_cfg.fd_eps,
                    rk=trace_cfg.rk,
                )
        else:
            pset = _get_homoclinic_preset(preset if preset is not None else "default")
            cfg_u = _build_homoclinic_branch_config_from_preset(
                pset,
                window=window,
                t_max=t_max,
                r_blow=r_blow,
                r_sec=r_sec,
                t_min_event=t_min_event,
            )

    _validate_homoclinic_branch_cfg(cfg_u)

    params_vec = _params_with_override(ctx.params_base, ctx.param_index, float(param_value))
    eq_vec = _resolve_fixed_point(ctx.model, eq)

    def rhs_eval_reset(x: np.ndarray, out: np.ndarray) -> None:
        ctx.prep_ws(x)
        ctx.rhs_fn(ctx.t_eval, x, out, params_vec, ctx.runtime_ws)

    def jac_eval(x: np.ndarray) -> np.ndarray:
        if cfg_u.jac == "fd" or (cfg_u.jac == "auto" and ctx.jac_fn is None):
            fx = np.empty((x.size,), dtype=ctx.model.dtype)
            rhs_eval_reset(x, fx)
            J = np.zeros((x.size, x.size), dtype=float)
            for j in range(x.size):
                step = cfg_u.fd_eps * (1.0 + abs(float(x[j])))
                if step == 0.0:
                    step = cfg_u.fd_eps
                x_step = np.array(x, copy=True)
                x_step[j] += step
                f_step = np.empty((x.size,), dtype=ctx.model.dtype)
                rhs_eval_reset(x_step, f_step)
                J[:, j] = (f_step - fx) / step
            return J

        if cfg_u.jac == "analytic" and ctx.jac_fn is None:
            raise ValueError("jac='analytic' requires a model Jacobian.")
        if ctx.jac_fn is None:
            raise ValueError("Jacobian is not available (jac='auto' found none).")

        jac_out = np.zeros((x.size, x.size), dtype=ctx.model.dtype)
        ctx.prep_ws(x)
        ctx.jac_fn(ctx.t_eval, x, params_vec, jac_out, ctx.runtime_ws)
        return np.array(jac_out, copy=True)

    J = jac_eval(np.array(eq_vec, float))
    ok_e, ed, einfo = _eig_index1_saddle_data(
        J,
        real_tol=float(cfg_u.eig_real_tol),
        imag_tol=float(cfg_u.eig_imag_tol),
        strict_1d=bool(cfg_u.strict_1d),
    )
    diag: dict[str, object] = {"saddle": einfo}
    if not ok_e or ed is None:
        meta = HomoclinicTraceMeta(
            param_value=float(param_value),
            eq=np.array(eq_vec, float),
            sign_u=int(sign_u),
            eps_used=float("nan"),
            status="saddle_fail",
            success=False,
            event=None,
            t_cross=float("nan"),
            x_cross=np.array(eq_vec, float),
            diag=diag,
        )
        return HomoclinicTraceResult(
            t=np.array([0.0], float),
            X=np.array([eq_vec], float),
            meta=meta,
        )

    lam, v_raw, l_u, _ = ed
    v_norm = float(np.linalg.norm(v_raw))
    if not np.isfinite(v_norm) or v_norm <= 1e-300:
        diag["fail"] = "unstable_vector_norm_bad"
        meta = HomoclinicTraceMeta(
            param_value=float(param_value),
            eq=np.array(eq_vec, float),
            sign_u=int(sign_u),
            eps_used=float("nan"),
            status="saddle_fail",
            success=False,
            event=None,
            t_cross=float("nan"),
            x_cross=np.array(eq_vec, float),
            diag=diag,
        )
        return HomoclinicTraceResult(
            t=np.array([0.0], float),
            X=np.array([eq_vec], float),
            meta=meta,
        )
    v_dir = np.array(v_raw, float) / v_norm

    denom = float(l_u @ v_dir)
    if abs(denom) < 1e-14:
        diag["fail"] = "projection_normalization_failed"
        meta = HomoclinicTraceMeta(
            param_value=float(param_value),
            eq=np.array(eq_vec, float),
            sign_u=int(sign_u),
            eps_used=float("nan"),
            status="saddle_fail",
            success=False,
            event=None,
            t_cross=float("nan"),
            x_cross=np.array(eq_vec, float),
            diag=diag,
        )
        return HomoclinicTraceResult(
            t=np.array([0.0], float),
            X=np.array([eq_vec], float),
            meta=meta,
        )
    v_proj = v_dir / denom

    sign = +1.0 if int(sign_u) >= 0 else -1.0
    growth_rate = float(np.real(lam))
    eps = float(cfg_u.eps) if cfg_u.eps_mode == "fixed" else _choose_eps_leave(
        r_leave=float(cfg_u.r_leave),
        rate=growth_rate,
        t_leave_target=float(cfg_u.t_leave_target),
        eps_min=float(cfg_u.eps_min),
        eps_max=float(cfg_u.eps_max),
        fallback=float(cfg_u.eps),
    )
    x0 = np.array(eq_vec, float) + sign * eps * v_dir

    status_int, ts, xs = _integrate_homoclinic_branch(
        ctx,
        params_vec,
        x0=np.array(x0, ctx.model.dtype),
        x_eq=np.array(eq_vec, ctx.model.dtype),
        cfg=cfg_u,
    )
    diag["integrate_status"] = status_int

    qualified, t_cross, x_cross, g_val, q_val = _scan_homoclinic_section(
        ts,
        xs,
        x_eq=np.array(eq_vec, float),
        l_u=np.array(l_u, float),
        v_proj=np.array(v_proj, float),
        cfg=cfg_u,
    )

    status = str(status_int)
    if not qualified and status in ("ok", "max_steps"):
        status = "no_cross"

    if qualified:
        idx = int(np.searchsorted(ts, t_cross, side="right"))
        if idx < 1:
            idx = 1
        ts_out = np.concatenate([ts[:idx], np.array([t_cross], float)])
        xs_out = np.vstack([xs[:idx], np.array(x_cross, float)])
    else:
        ts_out = np.array(ts, float)
        xs_out = np.array(xs, float)

    diag["section"] = {
        "g": float(g_val) if np.isfinite(g_val) else float("nan"),
        "q": float(q_val),
    }

    event = None
    if qualified:
        event = HomoclinicTraceEvent(
            kind="section_cross",
            t=float(t_cross),
            x=np.array(x_cross, float),
            info={"g": float(g_val), "q": float(q_val)},
        )

    meta = HomoclinicTraceMeta(
        param_value=float(param_value),
        eq=np.array(eq_vec, float),
        sign_u=int(sign_u),
        eps_used=float(eps),
        status=status,
        success=bool(qualified),
        event=event,
        t_cross=float(t_cross),
        x_cross=np.array(x_cross, float),
        diag=diag,
    )

    return HomoclinicTraceResult(t=ts_out, X=xs_out, meta=meta)
