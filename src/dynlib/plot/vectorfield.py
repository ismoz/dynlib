from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import animation

from dynlib import build
from dynlib import Sim
from dynlib.dsl.spec import ModelSpec
from dynlib.runtime.workspace import (
    initialize_lag_runtime_workspace,
    make_runtime_workspace,
)

from ._primitives import _get_ax, _apply_limits, _apply_labels
from . import _theme, _fig

__all__ = [
    "eval_vectorfield",
    "vectorfield",
    "VectorFieldHandle",
    "vectorfield_sweep",
    "VectorFieldSweep",
    "vectorfield_animate",
    "VectorFieldAnimation",
]


def _clone_runtime_workspace(template):
    """Return an independent runtime workspace cloned from the template."""
    if template is None:
        return None
    return type(template)(
        np.array(template.lag_ring, copy=True),
        np.array(template.lag_head, copy=True),
        template.lag_info,
        np.array(template.aux_values, copy=True),
        np.array(template.stop_flag, copy=True),
        np.array(template.stop_phase_mask, copy=True),
    )


def _resolve_model(model_or_sim, *, stepper: str | None, jit: bool, disk_cache: bool):
    """
    Accept Sim, compiled model, ModelSpec, or URI and return
    (model, base_state, base_params, t0, lag_state_info).
    """
    # Sim path (session-aware defaults)
    sim = getattr(model_or_sim, "model", None)
    if sim is not None and hasattr(model_or_sim, "state_vector"):
        model = sim
        if stepper is not None and getattr(model.spec.sim, "stepper", None) != stepper:
            warnings.warn("stepper override ignored when passing an existing Sim.", stacklevel=2)
        base_state = np.asarray(model_or_sim.state_vector(copy=True), dtype=model.dtype)
        base_params = np.asarray(model_or_sim.param_vector(copy=True), dtype=model.dtype)
        t0 = float(getattr(getattr(model_or_sim, "_session_state", None), "t_curr", model.spec.sim.t0))
        lag_state_info = getattr(model, "lag_state_info", None)
        return model, base_state, base_params, t0, lag_state_info

    # Already-compiled model
    if hasattr(model_or_sim, "spec") and hasattr(model_or_sim, "rhs"):
        model = model_or_sim
        if stepper is not None and getattr(model.spec.sim, "stepper", None) != stepper:
            warnings.warn("stepper override ignored for an already-compiled model.", stacklevel=2)
        base_state = np.asarray(model.spec.state_ic, dtype=model.dtype)
        base_params = np.asarray(model.spec.param_vals, dtype=model.dtype)
        t0 = float(model.spec.sim.t0)
        lag_state_info = getattr(model, "lag_state_info", None)
        return model, base_state, base_params, t0, lag_state_info

    # URI / ModelSpec
    if isinstance(model_or_sim, (str, ModelSpec)):
        model = build(model_or_sim, stepper=stepper, jit=jit, disk_cache=disk_cache)
        base_state = np.asarray(model.spec.state_ic, dtype=model.dtype)
        base_params = np.asarray(model.spec.param_vals, dtype=model.dtype)
        t0 = float(model.spec.sim.t0)
        lag_state_info = getattr(model, "lag_state_info", None)
        return model, base_state, base_params, t0, lag_state_info

    raise TypeError("model_or_sim must be a Sim, compiled model, ModelSpec, or URI.")


def _lag_state_info_from_spec(model, lag_state_info):
    if lag_state_info is not None:
        return tuple(lag_state_info)
    lag_map = getattr(getattr(model, "spec", None), "lag_map", None) or {}
    state_index = {name: idx for idx, name in enumerate(getattr(model.spec, "states", ()))}
    return tuple(
        (state_index[name], int(depth), int(offset), int(head_index))
        for name, (depth, offset, head_index) in lag_map.items()
        if name in state_index
    )


def _apply_state_overrides(state: np.ndarray, fixed: Mapping[str, float] | None, *, state_index: Mapping[str, int], skip: Sequence[int]) -> None:
    if not fixed:
        return
    for key, val in fixed.items():
        if key not in state_index:
            raise KeyError(f"Unknown state '{key}'.")
        idx = state_index[key]
        if idx in skip:
            # Plane variables are set per grid point; ignore fixed override silently
            continue
        state[idx] = float(val)


def _apply_param_overrides(params: np.ndarray, updates: Mapping[str, float] | None, *, param_index: Mapping[str, int]) -> None:
    if not updates:
        return
    for key, val in updates.items():
        if key not in param_index:
            raise KeyError(f"Unknown param '{key}'.")
        params[param_index[key]] = float(val)


def _make_meshgrid(xlim, ylim, grid, *, dtype=float) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = grid
    xs = np.linspace(xlim[0], xlim[1], int(gx), dtype=dtype)
    ys = np.linspace(ylim[0], ylim[1], int(gy), dtype=dtype)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X, Y


def _coerce_grid(grid: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(grid, int):
        return (int(grid), int(grid))
    gx, gy = grid
    return (int(gx), int(gy))


def _normalize_overrides(mapping: Mapping[str, float] | None) -> dict[str, float]:
    if mapping is None:
        return {}
    return {k: float(v) for k, v in mapping.items()}


@dataclass(frozen=True)
class _FacetSlice:
    key: Any
    params: dict[str, float]
    fixed: dict[str, float]
    value: Any | None = None


@dataclass(frozen=True)
class _FrameSpec:
    idx: int
    value: Any
    params: dict[str, float]
    fixed: dict[str, float]
    title: str | None


def _build_param_value_slices(param: str, values, *, target: str) -> list[_FacetSlice]:
    entries: list[_FacetSlice] = []
    if isinstance(values, Mapping):
        items = list(values.items())
    else:
        items = [(v, v) for v in values]

    for key, val in items:
        norm_val = float(val)
        if target == "params":
            entries.append(_FacetSlice(key=key, value=norm_val, params={param: norm_val}, fixed={}))
        else:
            entries.append(_FacetSlice(key=key, value=norm_val, params={}, fixed={param: norm_val}))
    if not entries:
        raise ValueError("values cannot be empty for a sweep.")
    return entries


def _build_custom_slices(sweep) -> list[_FacetSlice]:
    if isinstance(sweep, Mapping):
        items = list(sweep.items())
    else:
        if isinstance(sweep, (str, bytes)):
            raise TypeError("sweep must be a mapping or sequence of (key, overrides) pairs.")
        items = list(sweep)
    entries: list[_FacetSlice] = []
    for pair in items:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("Each sweep entry must be a (key, overrides) pair.")
        key, overrides = pair
        if not isinstance(overrides, Mapping):
            raise TypeError("sweep overrides must be a mapping with optional 'params'/'fixed' keys.")
        params_override = _normalize_overrides(overrides.get("params"))
        fixed_override = _normalize_overrides(overrides.get("fixed"))
        entries.append(_FacetSlice(key=key, params=params_override, fixed=fixed_override))
    if not entries:
        raise ValueError("sweep must define at least one facet entry.")
    return entries


def _resolve_sweep_entries(
    *,
    sweep,
    param: str | None,
    values,
    target: str,
) -> list[_FacetSlice]:
    target_norm = str(target or "params").lower()
    if target_norm not in {"params", "fixed"}:
        raise ValueError("target must be 'params' or 'fixed'.")

    if sweep is not None and (param is not None or values is not None):
        raise ValueError("Pass either sweep or param/values, not both.")

    if sweep is not None:
        return _build_custom_slices(sweep)

    if param is None or values is None:
        raise ValueError("param and values are required when sweep is not provided.")
    return _build_param_value_slices(str(param), values, target=target_norm)


def _resolve_titles(facet_titles, *, default_template: str | None, key: Any, value: Any, idx: int) -> str | None:
    if facet_titles is None:
        if default_template is None:
            return None
        try:
            return default_template.format(key=key, value=value, idx=idx)
        except Exception:
            return default_template

    if callable(facet_titles):
        return str(facet_titles(key))

    if isinstance(facet_titles, Mapping):
        if key in facet_titles:
            return str(facet_titles[key])
        return None

    if isinstance(facet_titles, (list, tuple)):
        if idx < len(facet_titles):
            return str(facet_titles[idx])
        return None

    if isinstance(facet_titles, str):
        try:
            return facet_titles.format(key=key, value=value, idx=idx)
        except Exception:
            return facet_titles

    return None


def _flatten_axes(grid_axes) -> list[Any]:
    return [ax for row in grid_axes for ax in row]


def _compute_shared_speed_norm(
    slices: Sequence[_FacetSlice],
    *,
    speed_norm,
    share_speed_norm: bool,
    speed_color: bool,
    model_or_sim,
    vars,
    xlim,
    ylim,
    grid,
    normalize,
    stepper,
    jit: bool,
    disk_cache: bool,
) -> Any | None:
    if not speed_color:
        return None
    if speed_norm is not None:
        return speed_norm
    if not share_speed_norm:
        return None

    max_speed = 0.0
    has_speed = False
    for sl in slices:
        _, _, _, _, speed = eval_vectorfield(
            model_or_sim,
            vars=vars,
            fixed=sl.fixed,
            params=sl.params,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            normalize=normalize,
            return_speed=True,
            stepper=stepper,
            jit=jit,
            disk_cache=disk_cache,
        )
        if speed is None or speed.size == 0:
            continue
        local_max = float(np.nanmax(speed))
        if np.isfinite(local_max) and local_max > max_speed:
            max_speed = local_max
            has_speed = True
    if not has_speed or max_speed <= 0:
        return None
    return mcolors.Normalize(vmin=0.0, vmax=max_speed)


def _call_frame_fn(fn: Callable, frame_value: Any, idx: int):
    try:
        return fn(frame_value, idx)
    except TypeError:
        return fn(frame_value)


def _speed_mappable(handle: "VectorFieldHandle"):
    if handle.mode == "quiver":
        return handle.quiver
    if handle.mode == "stream" and handle.stream is not None:
        return getattr(handle.stream, "lines", None)
    return None


def _default_nullcline_grid(grid: tuple[int, int]) -> tuple[int, int]:
    """
    Use a denser grid for nullcline computation to reduce numerical wobble.
    Caps growth to avoid runaway cost while ensuring at least a moderate density.
    """
    gx, gy = _coerce_grid(grid)
    dense_x = max(gx, min(max(gx * 2, 40), 120))
    dense_y = max(gy, min(max(gy * 2, 40), 120))
    return dense_x, dense_y


def _build_state_and_params(
    *,
    base_state_template: np.ndarray,
    base_params_template: np.ndarray,
    fixed: Mapping[str, float],
    params: Mapping[str, float],
    state_index: Mapping[str, int],
    param_index: Mapping[str, int],
    var_indices: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    state = np.array(base_state_template, copy=True)
    params_vec = np.array(base_params_template, copy=True)
    _apply_state_overrides(state, fixed, state_index=state_index, skip=var_indices)
    _apply_param_overrides(params_vec, params, param_index=param_index)
    return state, params_vec


def _evaluate_field(
    *,
    rhs: Callable,
    t0: float,
    X: np.ndarray,
    Y: np.ndarray,
    base_state: np.ndarray,
    base_params: np.ndarray,
    var_indices: tuple[int, int],
    runtime_ws_template,
    lag_state_info,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = X.shape
    U = np.zeros_like(X, dtype=base_state.dtype)
    V = np.zeros_like(Y, dtype=base_state.dtype)

    idx_x, idx_y = var_indices
    lag_state_info_tuple = tuple(lag_state_info or ())

    for j in range(ny):
        for i in range(nx):
            y_vec = np.array(base_state, copy=True)
            y_vec[idx_x] = X[j, i]
            y_vec[idx_y] = Y[j, i]
            params = np.array(base_params, copy=True)
            dy = np.empty_like(y_vec)
            runtime_ws = _clone_runtime_workspace(runtime_ws_template)
            if lag_state_info_tuple:
                initialize_lag_runtime_workspace(runtime_ws, lag_state_info=lag_state_info_tuple, y_curr=y_vec)
            rhs(t0, y_vec, dy, params, runtime_ws)
            U[j, i] = dy[idx_x]
            V[j, i] = dy[idx_y]
    return U, V


def eval_vectorfield(
    model_or_sim,
    *,
    vars: tuple[str, str] | None = None,
    fixed: Mapping[str, float] | None = None,
    params: Mapping[str, float] | None = None,
    xlim: tuple[float, float] = (-1, 1),
    ylim: tuple[float, float] = (-1, 1),
    grid: tuple[int, int] = (20, 20),
    normalize: bool = False,
    return_speed: bool = False,
    stepper: str | None = None,
    jit: bool = False,
    disk_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a 2D vector field on a grid and return X, Y, U, V arrays (and optionally speed).
    """
    model, base_state, base_params, t0, lag_state_info = _resolve_model(model_or_sim, stepper=stepper, jit=jit, disk_cache=disk_cache)
    spec = getattr(model, "spec", None)
    if spec is None or getattr(spec, "kind", None) != "ode":
        raise TypeError("vectorfield requires an ODE model (spec.kind == 'ode').")

    state_names = tuple(spec.states)
    param_names = tuple(spec.params)
    state_index = {name: idx for idx, name in enumerate(state_names)}
    param_index = {name: idx for idx, name in enumerate(param_names)}

    if vars is None:
        if len(state_names) < 2:
            raise ValueError("Model must have at least two states or specify vars explicitly.")
        vars = (state_names[0], state_names[1])
    var_x, var_y = vars
    if var_x not in state_index or var_y not in state_index:
        raise KeyError(f"Unknown vars {vars!r}; available states: {state_names}.")
    var_indices = (state_index[var_x], state_index[var_y])

    # Apply overrides to fresh templates
    base_state_template = np.array(base_state, copy=True)
    base_params_template = np.array(base_params, copy=True)
    _apply_state_overrides(base_state_template, fixed, state_index=state_index, skip=var_indices)
    _apply_param_overrides(base_params_template, params, param_index=param_index)

    # Grid
    grid = _coerce_grid(grid)
    X, Y = _make_meshgrid(xlim, ylim, grid, dtype=model.dtype)

    lag_state_info_tuple = _lag_state_info_from_spec(model, lag_state_info)
    runtime_ws_template = make_runtime_workspace(
        lag_state_info=lag_state_info_tuple,
        dtype=model.dtype,
        n_aux=len(spec.aux or {}),
    )

    U, V = _evaluate_field(
        rhs=model.rhs,
        t0=t0,
        X=X,
        Y=Y,
        base_state=base_state_template,
        base_params=base_params_template,
        var_indices=var_indices,
        runtime_ws_template=runtime_ws_template,
        lag_state_info=lag_state_info_tuple,
    )

    speed = None
    norm = None
    if normalize or return_speed:
        norm = np.hypot(U, V)
    if normalize:
        mask = norm > 0
        U = U.copy()
        V = V.copy()
        U[mask] /= norm[mask]
        V[mask] /= norm[mask]

    if return_speed:
        speed = norm

    if return_speed:
        return X, Y, U, V, speed
    return X, Y, U, V


@dataclass
class VectorFieldHandle:
    ax: Any
    model: Any
    rhs: Callable
    runtime_ws_template: Any
    lag_state_info: tuple
    var_names: tuple[str, str]
    var_indices: tuple[int, int]
    state_names: tuple[str, ...]
    param_names: tuple[str, ...]
    state_index: Mapping[str, int]
    param_index: Mapping[str, int]
    base_state_template: np.ndarray
    base_params_template: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    U: np.ndarray
    V: np.ndarray
    speed: np.ndarray | None
    mode: str
    quiver: Any | None
    stream: Any | None
    quiver_kwargs: dict[str, Any]
    stream_kwargs: dict[str, Any]
    normalize: bool
    speed_color: bool
    nullclines_enabled: bool
    nullcline_artists: list[Any]
    nullcline_style: dict
    t0: float
    nullcline_X: np.ndarray | None
    nullcline_Y: np.ndarray | None
    nullcline_U: np.ndarray | None
    nullcline_V: np.ndarray | None
    nullcline_cache_valid: bool
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    grid: tuple[int, int]
    nullcline_grid: tuple[int, int]
    last_fixed: dict[str, float]
    last_params: dict[str, float]
    traj_lines: list[Any]
    traj_style: dict[str, Any]
    interactive: bool
    run_T: float
    run_dt: float | None
    sim_record_vars: tuple[str, str]
    sim: Any | None
    _cid_click: Any
    _cid_keypress: Any

    def update(
        self,
        *,
        params: Mapping[str, float] | None = None,
        fixed: Mapping[str, float] | None = None,
        normalize: bool | None = None,
        redraw: bool = True,
    ) -> None:
        """Re-evaluate U,V on cached X,Y and update artists in-place."""
        new_fixed, new_params, overrides_changed = self._resolve_overrides(fixed, params)
        base_state, base_params = _build_state_and_params(
            base_state_template=self.base_state_template,
            base_params_template=self.base_params_template,
            fixed=new_fixed,
            params=new_params,
            state_index=self.state_index,
            param_index=self.param_index,
            var_indices=self.var_indices,
        )

        U_new, V_new = _evaluate_field(
            rhs=self.rhs,
            t0=self.t0,
            X=self.X,
            Y=self.Y,
            base_state=base_state,
            base_params=base_params,
            var_indices=self.var_indices,
            runtime_ws_template=self.runtime_ws_template,
            lag_state_info=self.lag_state_info,
        )

        speed_new = np.hypot(U_new, V_new) if self.speed_color else None
        new_normalize = self.normalize if normalize is None else bool(normalize)
        if new_normalize:
            norm = speed_new if speed_new is not None else np.hypot(U_new, V_new)
            mask = norm > 0
            U_new[mask] /= norm[mask]
            V_new[mask] /= norm[mask]

        self.U[:, :] = U_new
        self.V[:, :] = V_new
        if self.speed_color and speed_new is not None:
            if self.speed is None or self.speed.shape != speed_new.shape:
                self.speed = np.array(speed_new, copy=True)
            else:
                self.speed[:, :] = speed_new
        self.normalize = new_normalize

        if redraw:
            self._redraw_field()
            if self.nullclines_enabled:
                self._ensure_nullclines(base_state, base_params, force=overrides_changed or not self.nullcline_cache_valid)
                self._redraw_nullclines()
            elif overrides_changed:
                self.nullcline_cache_valid = False
            self.ax.figure.canvas.draw_idle()

    def _redraw_field(self) -> None:
        if self.mode == "quiver":
            if self.quiver is None:
                if self.speed_color:
                    self.quiver = self.ax.quiver(
                        self.X,
                        self.Y,
                        self.U,
                        self.V,
                        self.speed,
                        pivot="mid",
                        angles="xy",
                        **self.quiver_kwargs,
                    )
                else:
                    self.quiver = self.ax.quiver(self.X, self.Y, self.U, self.V, pivot="mid", angles="xy", **self.quiver_kwargs)
            else:
                if self.speed_color:
                    self.quiver.set_UVC(self.U, self.V, self.speed)
                else:
                    self.quiver.set_UVC(self.U, self.V)
            return

        if self.mode == "stream":
            self._redraw_streamplot()
            return

        raise ValueError(f"Unknown vectorfield mode '{self.mode}'.")

    def _redraw_streamplot(self) -> None:
        if self.speed_color:
            self.stream_kwargs["color"] = self.speed
        if self.stream is not None:
            for artist in (getattr(self.stream, "lines", None), getattr(self.stream, "arrows", None)):
                try:
                    if artist is not None:
                        artist.remove()
                except (ValueError, NotImplementedError):
                    try:
                        artist.set_visible(False)
                    except AttributeError:
                        pass
        self.stream = self.ax.streamplot(self.X, self.Y, self.U, self.V, **self.stream_kwargs)

    def _resolve_overrides(
        self,
        fixed: Mapping[str, float] | None,
        params: Mapping[str, float] | None,
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        fixed_norm = _normalize_overrides(self.last_fixed if fixed is None else fixed)
        params_norm = _normalize_overrides(self.last_params if params is None else params)
        changed = fixed_norm != self.last_fixed or params_norm != self.last_params
        self.last_fixed = fixed_norm
        self.last_params = params_norm
        return fixed_norm, params_norm, changed

    def _ensure_nullcline_grid(self) -> None:
        if self.nullcline_X is not None and self.nullcline_Y is not None:
            return
        resolved_nc_grid = self.nullcline_grid
        use_same_grid = resolved_nc_grid == self.grid and not self.normalize
        if use_same_grid:
            self.nullcline_X, self.nullcline_Y = self.X, self.Y
            self.nullcline_U = np.array(self.U, copy=True)
            self.nullcline_V = np.array(self.V, copy=True)
        else:
            self.nullcline_X, self.nullcline_Y = _make_meshgrid(
                self.xlim,
                self.ylim,
                resolved_nc_grid,
                dtype=self.model.dtype,
            )
            self.nullcline_U = np.zeros_like(self.nullcline_X, dtype=self.model.dtype)
            self.nullcline_V = np.zeros_like(self.nullcline_Y, dtype=self.model.dtype)

    def _ensure_nullclines(self, base_state: np.ndarray, base_params: np.ndarray, *, force: bool = False) -> None:
        if self.nullcline_cache_valid and not force and self.nullcline_X is not None and self.nullcline_Y is not None:
            return
        self._ensure_nullcline_grid()
        self.nullcline_U[:, :], self.nullcline_V[:, :] = _evaluate_field(
            rhs=self.rhs,
            t0=self.t0,
            X=self.nullcline_X,
            Y=self.nullcline_Y,
            base_state=base_state,
            base_params=base_params,
            var_indices=self.var_indices,
            runtime_ws_template=self.runtime_ws_template,
            lag_state_info=self.lag_state_info,
        )
        self.nullcline_cache_valid = True

    def _redraw_nullclines(self) -> None:
        for artist in self.nullcline_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.nullcline_artists.clear()
        X_nc = self.nullcline_X if self.nullcline_X is not None else self.X
        Y_nc = self.nullcline_Y if self.nullcline_Y is not None else self.Y
        U_nc = self.nullcline_U if self.nullcline_U is not None else self.U
        V_nc = self.nullcline_V if self.nullcline_V is not None else self.V
        self.nullcline_artists.extend(_draw_nullclines(self.ax, X_nc, Y_nc, U_nc, V_nc, self.nullcline_style))
        self.nullclines_enabled = True

    def toggle_nullclines(self) -> None:
        if self.nullclines_enabled:
            for artist in self.nullcline_artists:
                try:
                    artist.remove()
                except ValueError:
                    pass
            self.nullcline_artists.clear()
            self.nullclines_enabled = False
            self.ax.figure.canvas.draw_idle()
            return

        base_state, base_params = _build_state_and_params(
            base_state_template=self.base_state_template,
            base_params_template=self.base_params_template,
            fixed=self.last_fixed,
            params=self.last_params,
            state_index=self.state_index,
            param_index=self.param_index,
            var_indices=self.var_indices,
        )
        self._ensure_nullclines(base_state, base_params, force=not self.nullcline_cache_valid)
        self._redraw_nullclines()
        self.ax.figure.canvas.draw_idle()

    def clear_trajectories(self) -> None:
        for line in self.traj_lines:
            try:
                line.remove()
            except ValueError:
                pass
        self.traj_lines.clear()
        self.ax.figure.canvas.draw_idle()

    def _ensure_sim(self):
        if self.sim is None:
            raise RuntimeError("Interactive simulation is not enabled for this vector field.")
        self.sim.reset()
        return self.sim

    def simulate_at(self, x0: float, y0: float, *, T: float | None = None) -> Any:
        if not self.interactive:
            raise RuntimeError("Interactive simulation is disabled; pass interactive=True to vectorfield().")
        sim = self._ensure_sim()
        base_state, base_params = _build_state_and_params(
            base_state_template=self.base_state_template,
            base_params_template=self.base_params_template,
            fixed=self.last_fixed,
            params=self.last_params,
            state_index=self.state_index,
            param_index=self.param_index,
            var_indices=self.var_indices,
        )
        base_state[self.var_indices[0]] = float(x0)
        base_state[self.var_indices[1]] = float(y0)
        run_kwargs = {
            "T": self.run_T if T is None else float(T),
            "ic": base_state,
            "params": base_params,
            "record": True,
            "record_vars": list(self.sim_record_vars),
        }
        if self.run_dt is not None:
            run_kwargs["dt"] = float(self.run_dt)
        try:
            sim.run(**run_kwargs)
            res = sim.results()
            traj_x = res[self.var_names[0]]
            traj_y = res[self.var_names[1]]
            line, = self.ax.plot(traj_x, traj_y, **self.traj_style)
            self.traj_lines.append(line)
            self.ax.figure.canvas.draw_idle()
            return line
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to simulate trajectory from ({x0:.3f}, {y0:.3f}): {exc}", stacklevel=2)
            return None


def _draw_nullclines(ax, X, Y, U, V, style: Mapping[str, Any] | None):
    style = {} if style is None else dict(style)
    artists = []
    if U.size:
        cs_u = ax.contour(X, Y, U, levels=[0], **style)
        artists.append(cs_u)
    if V.size:
        cs_v = ax.contour(X, Y, V, levels=[0], **style)
        artists.append(cs_v)
    return artists


def vectorfield(
    model_or_sim,
    *,
    ax=None,
    vars: tuple[str, str] | None = None,
    fixed: Mapping[str, float] | None = None,
    params: Mapping[str, float] | None = None,
    xlim=(-1, 1),
    ylim=(-1, 1),
    grid=(20, 20),
    normalize: bool = False,
    color: str | None = None,
    speed_color: bool = False,
    speed_cmap: Any | None = None,
    speed_norm: Any | None = None,
    mode: str = "quiver",
    stream_kwargs: Mapping[str, Any] | None = None,
    nullclines: bool = False,
    nullcline_grid: tuple[int, int] | int | None = None,
    nullcline_style: Mapping[str, Any] | None = None,
    interactive: bool = True,
    T: float | None = None,
    dt: float | None = None,
    trajectory_style: Mapping[str, Any] | None = None,
    scale: float | None = None,
    stepper: str | None = None,
    jit: bool = False,
    disk_cache: bool = False,
) -> "VectorFieldHandle":
    """
    Draw a quiver or streamline plot (and optional numerical nullclines) and return a handle with .update().

    Nullclines are evaluated on a denser, always-un-normalized grid by default to avoid
    visual wobble; override with nullcline_grid to control the density.

    Interactive controls:
      - Click anywhere on the axes to launch a trajectory from that point (uses the model's
        sim.t_end by default, override with T).
      - Press "N" to toggle nullclines on/off (cached values are reused after disabling).
      - Press "C" to clear trajectories drawn via clicks.

    Args:
        stepper: Optional stepper override used when compiling from a URI/ModelSpec/string.
                 Ignored when an existing Sim or compiled model is passed.
        T: Trajectory duration for interactive runs (defaults to model sim.t_end).
        dt: Optional fixed dt override for interactive runs.
        mode: "quiver" (default) for arrows, "stream" for matplotlib.streamplot().
        stream_kwargs: Extra keyword arguments forwarded to matplotlib.streamplot() when mode="stream".
        speed_color: If True, color arrows/streamlines by speed magnitude instead of a single color.
        speed_cmap: Optional colormap used when speed_color=True (forwarded to quiver/streamplot).
        speed_norm: Optional matplotlib norm used when speed_color=True.
        scale: Optional scale factor for arrow lengths when normalize=True. If None (default),
               matplotlib's auto-scaling is used. Higher values make arrows shorter.
    """
    mode_norm = str(mode or "quiver").lower()
    if mode_norm in ("quiver", "arrow", "arrows"):
        mode_norm = "quiver"
    elif mode_norm in ("stream", "streamplot", "streamline", "streamlines"):
        mode_norm = "stream"
    else:
        raise ValueError("mode must be 'quiver' or 'stream'.")

    if speed_color:
        X, Y, U, V, speed = eval_vectorfield(
            model_or_sim,
            vars=vars,
            fixed=fixed,
            params=params,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            normalize=normalize,
            return_speed=True,
            stepper=stepper,
            jit=jit,
            disk_cache=disk_cache,
        )
    else:
        X, Y, U, V = eval_vectorfield(
            model_or_sim,
            vars=vars,
            fixed=fixed,
            params=params,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            normalize=normalize,
            stepper=stepper,
            jit=jit,
            disk_cache=disk_cache,
        )
        speed = None

    grid = _coerce_grid(grid)
    resolved_nc_grid = _coerce_grid(nullcline_grid) if nullcline_grid is not None else _default_nullcline_grid(grid)
    model, base_state, base_params, t0, lag_state_info = _resolve_model(model_or_sim, stepper=stepper, jit=jit, disk_cache=disk_cache)
    spec = model.spec
    state_names = tuple(spec.states)
    param_names = tuple(spec.params)
    state_index = {name: idx for idx, name in enumerate(state_names)}
    param_index = {name: idx for idx, name in enumerate(param_names)}

    if vars is None:
        vars = (state_names[0], state_names[1])
    var_indices = (state_index[vars[0]], state_index[vars[1]])
    vars = (str(vars[0]), str(vars[1]))

    base_state_template = np.array(base_state, copy=True)
    base_params_template = np.array(base_params, copy=True)
    last_fixed = _normalize_overrides(fixed)
    last_params = _normalize_overrides(params)
    _apply_state_overrides(base_state_template, last_fixed, state_index=state_index, skip=var_indices)
    _apply_param_overrides(base_params_template, last_params, param_index=param_index)

    runtime_ws_template = make_runtime_workspace(
        lag_state_info=_lag_state_info_from_spec(model, lag_state_info),
        dtype=model.dtype,
        n_aux=len(spec.aux or {}),
    )

    plot_ax = _get_ax(ax)
    color_kw: dict[str, Any] = {}
    if speed_color and color is not None:
        warnings.warn("color is ignored when speed_color=True.", stacklevel=2)
    elif color is not None:
        color_kw = {"color": color}

    quiver_kwargs = dict(color_kw)
    if speed_color:
        if speed_cmap is not None:
            quiver_kwargs.setdefault("cmap", speed_cmap)
        if speed_norm is not None:
            quiver_kwargs.setdefault("norm", speed_norm)
    # When normalize=True and scale is provided, ensure quiver respects data-units so the unit
    # vectors are visible as-is instead of being auto-rescaled relative to the axes width.
    if normalize and scale is not None:
        quiver_kwargs.setdefault("scale_units", "xy")
        quiver_kwargs.setdefault("scale", scale)

    stream_kwargs_resolved = dict(stream_kwargs or {})
    if speed_color:
        stream_kwargs_resolved["color"] = speed
        if speed_cmap is not None:
            stream_kwargs_resolved.setdefault("cmap", speed_cmap)
        if speed_norm is not None:
            stream_kwargs_resolved.setdefault("norm", speed_norm)
    elif color is not None and "color" not in stream_kwargs_resolved:
        stream_kwargs_resolved["color"] = color

    quiver = None
    stream = None
    if mode_norm == "quiver":
        if speed_color:
            quiver = plot_ax.quiver(X, Y, U, V, speed, pivot="mid", angles="xy", **quiver_kwargs)
        else:
            quiver = plot_ax.quiver(X, Y, U, V, pivot="mid", angles="xy", **quiver_kwargs)
    else:
        stream = plot_ax.streamplot(X, Y, U, V, **stream_kwargs_resolved)

    _apply_limits(plot_ax, xlim=xlim, ylim=ylim)
    _apply_labels(
        plot_ax,
        xlabel=vars[0],
        ylabel=vars[1],
        title=None,
        xpad=_theme.get("label_pad"),
        ypad=_theme.get("label_pad"),
        xlabel_fs=_theme.get("fontsize_label"),
        ylabel_fs=_theme.get("fontsize_label"),
    )

    nullcline_artists = []
    nullcline_X = None
    nullcline_Y = None
    nullcline_U = None
    nullcline_V = None
    nullcline_cache_valid = False
    if nullclines:
        use_same_grid = resolved_nc_grid == grid and not normalize
        if use_same_grid:
            nullcline_X, nullcline_Y = X, Y
            nullcline_U, nullcline_V = np.array(U, copy=True), np.array(V, copy=True)
        else:
            nullcline_X, nullcline_Y, nullcline_U, nullcline_V = eval_vectorfield(
                model_or_sim,
                vars=vars,
                fixed=last_fixed,
                params=last_params,
                xlim=xlim,
                ylim=ylim,
                grid=resolved_nc_grid,
                normalize=False,
                jit=jit,
                disk_cache=disk_cache,
            )
        nullcline_artists = _draw_nullclines(plot_ax, nullcline_X, nullcline_Y, nullcline_U, nullcline_V, nullcline_style)
        nullcline_cache_valid = True

    traj_style = {"lw": 1.6, "alpha": 0.85}
    if trajectory_style:
        traj_style.update(dict(trajectory_style))
    run_T = float(T) if T is not None else float(spec.sim.t_end)
    run_dt_resolved = float(dt) if dt is not None else None
    sim_instance = Sim(model) if interactive else None

    handle = VectorFieldHandle(
        ax=plot_ax,
        model=model,
        rhs=model.rhs,
        runtime_ws_template=runtime_ws_template,
        lag_state_info=_lag_state_info_from_spec(model, lag_state_info),
        var_names=vars,
        var_indices=var_indices,
        state_names=state_names,
        param_names=param_names,
        state_index=state_index,
        param_index=param_index,
        base_state_template=base_state_template,
        base_params_template=base_params_template,
        X=X,
        Y=Y,
        U=U,
        V=V,
        speed=speed if speed_color else None,
        mode=mode_norm,
        quiver=quiver,
        stream=stream,
        quiver_kwargs=quiver_kwargs,
        stream_kwargs=stream_kwargs_resolved,
        normalize=normalize,
        speed_color=bool(speed_color),
        nullclines_enabled=bool(nullclines),
        nullcline_artists=list(nullcline_artists),
        nullcline_style=dict(nullcline_style or {}),
        t0=t0,
        nullcline_X=nullcline_X,
        nullcline_Y=nullcline_Y,
        nullcline_U=nullcline_U,
        nullcline_V=nullcline_V,
        nullcline_cache_valid=nullcline_cache_valid,
        xlim=(float(xlim[0]), float(xlim[1])),
        ylim=(float(ylim[0]), float(ylim[1])),
        grid=grid,
        nullcline_grid=resolved_nc_grid,
        last_fixed=last_fixed,
        last_params=last_params,
        traj_lines=[],
        traj_style=traj_style,
        interactive=bool(interactive),
        run_T=run_T,
        run_dt=run_dt_resolved,
        sim_record_vars=vars,
        sim=sim_instance,
        _cid_click=None,
        _cid_keypress=None,
    )

    if interactive:
        canvas = plot_ax.figure.canvas

        def _on_click(event):
            if event.inaxes is not plot_ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            handle.simulate_at(event.xdata, event.ydata)

        def _on_key(event):
            key = (event.key or "").lower()
            if key == "n":
                handle.toggle_nullclines()
            elif key == "c":
                handle.clear_trajectories()

        handle._cid_click = canvas.mpl_connect("button_press_event", _on_click)
        handle._cid_keypress = canvas.mpl_connect("key_press_event", _on_key)
    return handle


@dataclass
class VectorFieldSweep:
    handles: tuple[VectorFieldHandle, ...]
    axes: tuple[Any, ...]
    sweep_keys: tuple[Any, ...]
    colorbar: Any | None
    speed_norm: Any | None

    @property
    def figure(self):
        if not self.axes:
            return None
        return self.axes[0].figure


@dataclass
class VectorFieldAnimation:
    handle: VectorFieldHandle
    animation: Any
    frames: tuple[_FrameSpec, ...]

    @property
    def ax(self):
        return self.handle.ax

    @property
    def figure(self):
        return self.handle.ax.figure

    def save(self, path, **kwargs):
        """
        Save the underlying matplotlib animation.

        Examples:
            anim.save(\"out.gif\", writer=\"pillow\")  # respects fps/interval from construction
        """
        self.animation.save(path, **kwargs)
        return path


def vectorfield_sweep(
    model_or_sim,
    *,
    param: str | None = None,
    values=None,
    sweep=None,
    target: str = "params",
    vars: tuple[str, str] | None = None,
    fixed: Mapping[str, float] | None = None,
    params: Mapping[str, float] | None = None,
    xlim: tuple[float, float] = (-1, 1),
    ylim: tuple[float, float] = (-1, 1),
    grid: tuple[int, int] | int = (20, 20),
    normalize: bool = False,
    color: str | None = None,
    speed_color: bool = False,
    speed_cmap: Any | None = None,
    speed_norm: Any | None = None,
    share_speed_norm: bool = True,
    mode: str = "quiver",
    stream_kwargs: Mapping[str, Any] | None = None,
    nullclines: bool = False,
    nullcline_grid: tuple[int, int] | int | None = None,
    nullcline_style: Mapping[str, Any] | None = None,
    interactive: bool = False,
    T: float | None = None,
    dt: float | None = None,
    trajectory_style: Mapping[str, Any] | None = None,
    cols: int = 3,
    sharex: bool | str = True,
    sharey: bool | str = True,
    title: str | None = None,
    facet_titles=None,
    size: tuple[float, float] | None = None,
    scale: float | None = None,
    add_colorbar: bool = True,
    stepper: str | None = None,
    jit: bool = False,
    disk_cache: bool = False,
) -> VectorFieldSweep:
    """
    Draw a grid of vector fields for a 1D sweep (parameter or state overrides).

    You can either provide ``param`` + ``values`` for a simple sweep, or pass
    ``sweep`` as a mapping/sequence of ``(key, {"params": {...}, "fixed": {...}})``
    to fully control overrides per facet.

    Args:
        param: Name of the parameter/state being swept when ``values`` is used.
        values: Iterable or mapping of sweep values (keys in the mapping become facet keys).
        sweep: Explicit facet definitions as ``{key: {"params": {...}, "fixed": {...}}}``.
        target: ``"params"`` (default) or ``"fixed"`` when using ``param``/``values``.
        cols: Number of columns in the wrapped grid.
        share_speed_norm: If True, compute a shared Normalize from all facets when coloring by speed.
        facet_titles: Optional mapping/sequence/formatter for per-facet titles.
        add_colorbar: Add a shared colorbar when speed coloring is enabled and a shared norm is used.
    """

    base_params = _normalize_overrides(params)
    base_fixed = _normalize_overrides(fixed)

    raw_slices = _resolve_sweep_entries(sweep=sweep, param=param, values=values, target=target)
    resolved_slices: list[_FacetSlice] = []
    for sl in raw_slices:
        merged_params = dict(base_params)
        merged_params.update(sl.params)
        merged_fixed = dict(base_fixed)
        merged_fixed.update(sl.fixed)
        resolved_slices.append(_FacetSlice(key=sl.key, params=merged_params, fixed=merged_fixed, value=sl.value))

    if not resolved_slices:
        raise ValueError("vectorfield_sweep needs at least one facet entry.")

    axes_grid = _fig.wrap(
        n=len(resolved_slices),
        cols=int(cols),
        title=title,
        size=size,
        scale=scale,
        sharex=sharex,
        sharey=sharey,
    )
    flat_axes = _flatten_axes(axes_grid)

    shared_norm = _compute_shared_speed_norm(
        resolved_slices,
        speed_norm=speed_norm,
        share_speed_norm=share_speed_norm,
        speed_color=speed_color,
        model_or_sim=model_or_sim,
        vars=vars,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        normalize=normalize,
        stepper=stepper,
        jit=jit,
        disk_cache=disk_cache,
    )
    norm_to_use = shared_norm if shared_norm is not None else speed_norm

    default_template = None
    if facet_titles is None:
        if param is not None:
            default_template = f"{param}={{value}}"
        else:
            default_template = "{key}"

    handles: list[VectorFieldHandle] = []
    for idx, (ax, sl) in enumerate(zip(flat_axes, resolved_slices)):
        handle = vectorfield(
            model_or_sim,
            ax=ax,
            vars=vars,
            fixed=sl.fixed,
            params=sl.params,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            normalize=normalize,
            color=color,
            speed_color=speed_color,
            speed_cmap=speed_cmap,
            speed_norm=norm_to_use,
            mode=mode,
            stream_kwargs=stream_kwargs,
            nullclines=nullclines,
            nullcline_grid=nullcline_grid,
            nullcline_style=nullcline_style,
            interactive=interactive,
            T=T,
            dt=dt,
            trajectory_style=trajectory_style,
            stepper=stepper,
            jit=jit,
            disk_cache=disk_cache,
        )
        title_text = _resolve_titles(
            facet_titles,
            default_template=default_template,
            key=sl.key,
            value=sl.value if sl.value is not None else sl.key,
            idx=idx,
        )
        if title_text:
            ax.set_title(title_text)
        handles.append(handle)

    colorbar = None
    if speed_color and add_colorbar and norm_to_use is not None:
        mappable = _speed_mappable(handles[0]) if handles else None
        if mappable is not None:
            fig = handles[0].ax.figure
            colorbar = fig.colorbar(mappable, ax=flat_axes)

    return VectorFieldSweep(
        handles=tuple(handles),
        axes=tuple(flat_axes),
        sweep_keys=tuple(sl.key for sl in resolved_slices),
        colorbar=colorbar,
        speed_norm=norm_to_use,
    )


# Backward compatibility shim (not exported in __all__).
vectorfield_facet_sweep = vectorfield_sweep


def _prepare_frame_specs(
    *,
    frames,
    param: str | None,
    values,
    params_func: Callable | None,
    fixed_func: Callable | None,
    title_func: Callable | None,
    base_params: Mapping[str, float],
    base_fixed: Mapping[str, float],
) -> list[_FrameSpec]:
    values_list = None
    if values is not None:
        values_list = list(values)

    if values is not None and params_func is not None:
        raise ValueError("Cannot mix values with params_func; choose one animation mode.")

    if frames is None:
        if values_list is not None:
            frames_list = list(values_list)
        else:
            raise ValueError("frames, values, or duration must be provided for animation.")
    elif isinstance(frames, int):
        if frames <= 0:
            raise ValueError("frames must be a positive integer.")
        frames_list = list(range(int(frames)))
        if values_list is not None and len(values_list) != len(frames_list):
            raise ValueError("Length of values must match frames.")
    else:
        frames_list = list(frames)
        if not frames_list:
            raise ValueError("frames iterable is empty.")
        if values_list is not None:
            raise ValueError("Cannot supply both frames iterable and values.")

    if (params_func or fixed_func or title_func) and frames is None and values is None:
        raise ValueError("frames or values are required when using custom frame functions.")

    if values_list is not None and param is None:
        raise ValueError("param must be provided when supplying values.")

    base_params_norm = _normalize_overrides(base_params)
    base_fixed_norm = _normalize_overrides(base_fixed)

    specs: list[_FrameSpec] = []
    for idx, frame_val in enumerate(frames_list):
        params_override = dict(base_params_norm)
        fixed_override = dict(base_fixed_norm)

        if param is not None:
            if values_list is not None:
                params_override[param] = float(frame_val)
            elif not isinstance(frame_val, Mapping):
                params_override[param] = float(frame_val)

        if isinstance(frame_val, Mapping):
            params_override.update(_normalize_overrides(frame_val.get("params")))
            fixed_override.update(_normalize_overrides(frame_val.get("fixed")))

        if params_func is not None:
            extra_params = _call_frame_fn(params_func, frame_val, idx)
            params_override.update(_normalize_overrides(extra_params))
        if fixed_func is not None:
            extra_fixed = _call_frame_fn(fixed_func, frame_val, idx)
            fixed_override.update(_normalize_overrides(extra_fixed))

        title_val = None
        if title_func is not None:
            res = _call_frame_fn(title_func, frame_val, idx)
            if res is not None:
                title_val = str(res)

        specs.append(_FrameSpec(idx=idx, value=frame_val, params=params_override, fixed=fixed_override, title=title_val))

    if not specs:
        raise ValueError("No frames prepared for animation.")
    return specs


def vectorfield_animate(
    model_or_sim,
    *,
    ax=None,
    frames=None,
    duration: float | None = None,
    fps: float = 15.0,
    interval: float | None = None,
    repeat: bool = True,
    repeat_delay: float | None = None,
    param: str | None = None,
    values=None,
    params_func: Callable | None = None,
    fixed_func: Callable | None = None,
    title: str | None = None,
    title_func: Callable | None = None,
    vars: tuple[str, str] | None = None,
    fixed: Mapping[str, float] | None = None,
    params: Mapping[str, float] | None = None,
    xlim=(-1, 1),
    ylim=(-1, 1),
    grid=(20, 20),
    normalize: bool = False,
    color: str | None = None,
    speed_color: bool = False,
    speed_cmap: Any | None = None,
    speed_norm: Any | None = None,
    share_speed_norm: bool = True,
    mode: str = "quiver",
    stream_kwargs: Mapping[str, Any] | None = None,
    nullclines: bool = False,
    nullcline_grid: tuple[int, int] | int | None = None,
    nullcline_style: Mapping[str, Any] | None = None,
    interactive: bool = False,
    T: float | None = None,
    dt: float | None = None,
    trajectory_style: Mapping[str, Any] | None = None,
    scale: float | None = None,
    stepper: str | None = None,
    jit: bool = False,
    disk_cache: bool = False,
    blit: bool = False,
) -> VectorFieldAnimation:
    """
    Animate a vector field by updating a single VectorFieldHandle across frames.

    Provide either ``values`` (and ``param``) for a simple sweep, or ``frames`` with
    optional ``params_func``/``fixed_func`` to customize per-frame overrides. When
    ``duration`` is given without frames/values, frames are derived from ``fps``.
    """

    if frames is None and values is None and duration is not None:
        if fps <= 0:
            raise ValueError("fps must be positive when using duration.")
        frames = int(round(float(duration) * float(fps)))

    frame_specs = _prepare_frame_specs(
        frames=frames,
        param=param,
        values=values,
        params_func=params_func,
        fixed_func=fixed_func,
        title_func=title_func,
        base_params=params or {},
        base_fixed=fixed or {},
    )

    # Shared speed normalization for consistent coloring.
    slices = [_FacetSlice(key=fs.idx, params=fs.params, fixed=fs.fixed, value=fs.value) for fs in frame_specs]
    shared_norm = _compute_shared_speed_norm(
        slices,
        speed_norm=speed_norm,
        share_speed_norm=share_speed_norm,
        speed_color=speed_color,
        model_or_sim=model_or_sim,
        vars=vars,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        normalize=normalize,
        stepper=stepper,
        jit=jit,
        disk_cache=disk_cache,
    )
    norm_to_use = shared_norm if shared_norm is not None else speed_norm

    first = frame_specs[0]
    handle = vectorfield(
        model_or_sim,
        ax=ax,
        vars=vars,
        fixed=first.fixed,
        params=first.params,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        normalize=normalize,
        color=color,
        speed_color=speed_color,
        speed_cmap=speed_cmap,
        speed_norm=norm_to_use,
        mode=mode,
        stream_kwargs=stream_kwargs,
        nullclines=nullclines,
        nullcline_grid=nullcline_grid,
        nullcline_style=nullcline_style,
        interactive=interactive,
        T=T,
        dt=dt,
        trajectory_style=trajectory_style,
        scale=scale,
        stepper=stepper,
        jit=jit,
        disk_cache=disk_cache,
    )

    if title:
        handle.ax.set_title(title)
    elif first.title:
        handle.ax.set_title(first.title)

    _in_update = False

    def _update(frame_spec: _FrameSpec):
        nonlocal _in_update
        if _in_update:
            return []
        _in_update = True
        try:
            handle.update(params=frame_spec.params, fixed=frame_spec.fixed, redraw=True)
        finally:
            _in_update = False
        if frame_spec.title is not None:
            handle.ax.set_title(frame_spec.title)
        elif title:
            handle.ax.set_title(title)
        return []

    interval_ms = interval if interval is not None else 1000.0 / float(fps)
    repeat_delay_ms = None if repeat_delay is None else float(repeat_delay)

    anim = animation.FuncAnimation(
        handle.ax.figure,
        _update,
        frames=frame_specs,
        interval=interval_ms,
        repeat=repeat,
        repeat_delay=repeat_delay_ms,
        blit=blit,
    )
    # We intentionally manage the animation object ourselves (e.g. in tests) without
    # ever calling `save`/`show`, so mark it as started to suppress Matplotlib's
    # "Animation was deleted without rendering" warning when it is GC'd.
    anim._draw_was_started = True
    anim.save_count = len(frame_specs)
    anim.repeat = bool(repeat)
    if repeat_delay_ms is not None:
        anim.repeat_delay = repeat_delay_ms

    return VectorFieldAnimation(handle=handle, animation=anim, frames=tuple(frame_specs))
