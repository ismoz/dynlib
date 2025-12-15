from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np

from dynlib import build
from dynlib import Sim
from dynlib.dsl.spec import ModelSpec
from dynlib.runtime.workspace import (
    initialize_lag_runtime_workspace,
    make_runtime_workspace,
)

from ._primitives import _get_ax, _apply_limits, _apply_labels
from . import _theme

__all__ = ["eval_vectorfield", "vectorfield", "VectorFieldHandle"]


def _clone_runtime_workspace(template):
    """Return an independent runtime workspace cloned from the template."""
    if template is None:
        return None
    return type(template)(
        np.array(template.lag_ring, copy=True),
        np.array(template.lag_head, copy=True),
        template.lag_info,
        np.array(template.aux_values, copy=True),
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
    stepper: str | None = None,
    jit: bool = False,
    disk_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a 2D vector field on a grid and return X, Y, U, V arrays.
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

    if normalize:
        norm = np.hypot(U, V)
        mask = norm > 0
        U = U.copy()
        V = V.copy()
        U[mask] /= norm[mask]
        V[mask] /= norm[mask]

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
    mode: str
    quiver: Any | None
    stream: Any | None
    quiver_kwargs: dict[str, Any]
    stream_kwargs: dict[str, Any]
    normalize: bool
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

        new_normalize = self.normalize if normalize is None else bool(normalize)
        if new_normalize:
            norm = np.hypot(U_new, V_new)
            mask = norm > 0
            U_new[mask] /= norm[mask]
            V_new[mask] /= norm[mask]

        self.U[:, :] = U_new
        self.V[:, :] = V_new
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
                self.quiver = self.ax.quiver(self.X, self.Y, self.U, self.V, pivot="mid", angles="xy", **self.quiver_kwargs)
            else:
                self.quiver.set_UVC(self.U, self.V)
            return

        if self.mode == "stream":
            self._redraw_streamplot()
            return

        raise ValueError(f"Unknown vectorfield mode '{self.mode}'.")

    def _redraw_streamplot(self) -> None:
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
    mode: str = "quiver",
    stream_kwargs: Mapping[str, Any] | None = None,
    nullclines: bool = False,
    nullcline_grid: tuple[int, int] | int | None = None,
    nullcline_style: Mapping[str, Any] | None = None,
    interactive: bool = True,
    T: float | None = None,
    dt: float | None = None,
    trajectory_style: Mapping[str, Any] | None = None,
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
    """
    mode_norm = str(mode or "quiver").lower()
    if mode_norm in ("quiver", "arrow", "arrows"):
        mode_norm = "quiver"
    elif mode_norm in ("stream", "streamplot", "streamline", "streamlines"):
        mode_norm = "stream"
    else:
        raise ValueError("mode must be 'quiver' or 'stream'.")

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
    color_kw = {} if color is None else {"color": color}
    quiver_kwargs = dict(color_kw)
    stream_kwargs_resolved = dict(stream_kwargs or {})
    if color is not None and "color" not in stream_kwargs_resolved:
        stream_kwargs_resolved["color"] = color

    quiver = None
    stream = None
    if mode_norm == "quiver":
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
        mode=mode_norm,
        quiver=quiver,
        stream=stream,
        quiver_kwargs=quiver_kwargs,
        stream_kwargs=stream_kwargs_resolved,
        normalize=normalize,
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
