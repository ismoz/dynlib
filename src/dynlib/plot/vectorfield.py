from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dynlib import build
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


def _resolve_model(model_or_sim, *, jit: bool, disk_cache: bool):
    """
    Accept Sim, compiled model, ModelSpec, or URI and return
    (model, base_state, base_params, t0, lag_state_info).
    """
    # Sim path (session-aware defaults)
    sim = getattr(model_or_sim, "model", None)
    if sim is not None and hasattr(model_or_sim, "state_vector"):
        model = sim
        base_state = np.asarray(model_or_sim.state_vector(copy=True), dtype=model.dtype)
        base_params = np.asarray(model_or_sim.param_vector(copy=True), dtype=model.dtype)
        t0 = float(getattr(getattr(model_or_sim, "_session_state", None), "t_curr", model.spec.sim.t0))
        lag_state_info = getattr(model, "lag_state_info", None)
        return model, base_state, base_params, t0, lag_state_info

    # Already-compiled model
    if hasattr(model_or_sim, "spec") and hasattr(model_or_sim, "rhs"):
        model = model_or_sim
        base_state = np.asarray(model.spec.state_ic, dtype=model.dtype)
        base_params = np.asarray(model.spec.param_vals, dtype=model.dtype)
        t0 = float(model.spec.sim.t0)
        lag_state_info = getattr(model, "lag_state_info", None)
        return model, base_state, base_params, t0, lag_state_info

    # URI / ModelSpec
    if isinstance(model_or_sim, (str, ModelSpec)):
        model = build(model_or_sim, jit=jit, disk_cache=disk_cache)
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


def _default_nullcline_grid(grid: tuple[int, int]) -> tuple[int, int]:
    """
    Use a denser grid for nullcline computation to reduce numerical wobble.
    Caps growth to avoid runaway cost while ensuring at least a moderate density.
    """
    gx, gy = _coerce_grid(grid)
    dense_x = max(gx, min(max(gx * 2, 40), 120))
    dense_y = max(gy, min(max(gy * 2, 40), 120))
    return dense_x, dense_y


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
    jit: bool = False,
    disk_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a 2D vector field on a grid and return X, Y, U, V arrays.
    """
    model, base_state, base_params, t0, lag_state_info = _resolve_model(model_or_sim, jit=jit, disk_cache=disk_cache)
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
    base_state_template: np.ndarray
    base_params_template: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    U: np.ndarray
    V: np.ndarray
    quiver: Any
    normalize: bool
    nullclines_enabled: bool
    nullcline_artists: list[Any]
    nullcline_style: dict
    t0: float
    nullcline_X: np.ndarray | None
    nullcline_Y: np.ndarray | None
    nullcline_U: np.ndarray | None
    nullcline_V: np.ndarray | None

    def update(
        self,
        *,
        params: Mapping[str, float] | None = None,
        fixed: Mapping[str, float] | None = None,
        normalize: bool | None = None,
        redraw: bool = True,
    ) -> None:
        """Re-evaluate U,V on cached X,Y and update artists in-place."""
        state_index = {n: i for i, n in enumerate(self.state_names)}
        param_index = {n: i for i, n in enumerate(self.param_names)}
        base_state = np.array(self.base_state_template, copy=True)
        base_params = np.array(self.base_params_template, copy=True)
        _apply_state_overrides(base_state, fixed, state_index=state_index, skip=self.var_indices)
        _apply_param_overrides(base_params, params, param_index=param_index)

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
            self.quiver.set_UVC(self.U, self.V)
            if self.nullclines_enabled:
                if self.nullcline_X is not None and self.nullcline_Y is not None and self.nullcline_U is not None and self.nullcline_V is not None:
                    U_nc, V_nc = _evaluate_field(
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
                    self.nullcline_U[:, :] = U_nc
                    self.nullcline_V[:, :] = V_nc
                for artist in self.nullcline_artists:
                    artist.remove()
                self.nullcline_artists.clear()
                X_nc = self.nullcline_X if self.nullcline_X is not None else self.X
                Y_nc = self.nullcline_Y if self.nullcline_Y is not None else self.Y
                U_nc = self.nullcline_U if self.nullcline_U is not None else self.U
                V_nc = self.nullcline_V if self.nullcline_V is not None else self.V
                self.nullcline_artists.extend(_draw_nullclines(self.ax, X_nc, Y_nc, U_nc, V_nc, self.nullcline_style))
            self.ax.figure.canvas.draw_idle()


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
    nullclines: bool = False,
    nullcline_grid: tuple[int, int] | int | None = None,
    nullcline_style: Mapping[str, Any] | None = None,
    jit: bool = False,
    disk_cache: bool = False,
) -> "VectorFieldHandle":
    """
    Draw a quiver plot (and optional numerical nullclines) and return a handle with .update().

    Nullclines are evaluated on a denser, always-un-normalized grid by default to avoid
    visual wobble; override with nullcline_grid to control the density.
    """
    X, Y, U, V = eval_vectorfield(
        model_or_sim,
        vars=vars,
        fixed=fixed,
        params=params,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        normalize=normalize,
        jit=jit,
        disk_cache=disk_cache,
    )

    grid = _coerce_grid(grid)
    model, base_state, base_params, t0, lag_state_info = _resolve_model(model_or_sim, jit=jit, disk_cache=disk_cache)
    spec = model.spec
    state_names = tuple(spec.states)
    param_names = tuple(spec.params)
    state_index = {name: idx for idx, name in enumerate(state_names)}

    if vars is None:
        vars = (state_names[0], state_names[1])
    var_indices = (state_index[vars[0]], state_index[vars[1]])

    base_state_template = np.array(base_state, copy=True)
    base_params_template = np.array(base_params, copy=True)
    _apply_state_overrides(base_state_template, fixed, state_index=state_index, skip=var_indices)
    _apply_param_overrides(base_params_template, params, param_index={n: i for i, n in enumerate(param_names)})

    runtime_ws_template = make_runtime_workspace(
        lag_state_info=_lag_state_info_from_spec(model, lag_state_info),
        dtype=model.dtype,
        n_aux=len(spec.aux or {}),
    )

    plot_ax = _get_ax(ax)
    color_kw = {} if color is None else {"color": color}
    quiver = plot_ax.quiver(X, Y, U, V, pivot="mid", angles="xy", **color_kw)
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
    if nullclines:
        resolved_nc_grid = _coerce_grid(nullcline_grid) if nullcline_grid is not None else _default_nullcline_grid(grid)
        use_same_grid = resolved_nc_grid == grid and not normalize
        if use_same_grid:
            nullcline_X, nullcline_Y = X, Y
            nullcline_U, nullcline_V = np.array(U, copy=True), np.array(V, copy=True)
        else:
            nullcline_X, nullcline_Y, nullcline_U, nullcline_V = eval_vectorfield(
                model_or_sim,
                vars=vars,
                fixed=fixed,
                params=params,
                xlim=xlim,
                ylim=ylim,
                grid=resolved_nc_grid,
                normalize=False,
                jit=jit,
                disk_cache=disk_cache,
            )
        nullcline_artists = _draw_nullclines(plot_ax, nullcline_X, nullcline_Y, nullcline_U, nullcline_V, nullcline_style)

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
        base_state_template=base_state_template,
        base_params_template=base_params_template,
        X=X,
        Y=Y,
        U=U,
        V=V,
        quiver=quiver,
        normalize=normalize,
        nullclines_enabled=bool(nullclines),
        nullcline_artists=list(nullcline_artists),
        nullcline_style=dict(nullcline_style or {}),
        t0=t0,
        nullcline_X=nullcline_X,
        nullcline_Y=nullcline_Y,
        nullcline_U=nullcline_U,
        nullcline_V=nullcline_V,
    )
    return handle
