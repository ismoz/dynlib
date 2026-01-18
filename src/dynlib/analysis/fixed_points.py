# src/dynlib/analysis/fixed_points.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import numpy as np

__all__ = [
    "FixedPointConfig",
    "FixedPointResult",
    "find_fixed_points",
]


@dataclass(frozen=True)
class FixedPointConfig:
    method: Literal["newton"] = "newton"
    tol: float = 1e-10
    max_iter: int = 50
    unique_tol: float = 1e-6
    jac: Literal["auto", "fd", "provided"] = "auto"
    fd_eps: float = 1e-6
    classify: bool = True
    kind: Literal["ode", "map"] = "ode"
    stability_tol: float = 1e-6


@dataclass
class FixedPointResult:
    points: np.ndarray
    residuals: np.ndarray
    jacobians: list[np.ndarray] | None
    eigvals: list[np.ndarray] | None
    stability: list[str] | None
    meta: dict[str, object] = field(default_factory=dict)


def _coerce_seeds(seeds: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    seed_arr = np.asarray(seeds, dtype=float)
    if seed_arr.ndim == 1:
        seed_arr = seed_arr[None, :]
    if seed_arr.ndim != 2 or seed_arr.shape[1] == 0:
        raise ValueError("seeds must be an array-like of shape (n_seeds, n_state)")
    return seed_arr


def _coerce_params(params: np.ndarray | None) -> np.ndarray:
    if params is None:
        return np.zeros((0,), dtype=float)
    params_arr = np.asarray(params, dtype=float)
    if params_arr.ndim != 1:
        raise ValueError("params must be a 1-D array")
    return params_arr


def _eval_f(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    out = np.asarray(f(x, params), dtype=float)
    if out.shape != x.shape:
        raise ValueError(f"f(x, params) must return shape {x.shape}, got {out.shape}")
    return out


def _finite_diff_jac(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    fx: np.ndarray,
    params: np.ndarray,
    eps: float,
) -> np.ndarray:
    n = x.size
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        step = eps * (1.0 + abs(float(x[j])))
        if step == 0.0:
            step = eps
        x_step = np.array(x, copy=True)
        x_step[j] += step
        f_step = _eval_f(f, x_step, params)
        J[:, j] = (f_step - fx) / step
    return J


def _solve_newton(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    params: np.ndarray,
    cfg: FixedPointConfig,
) -> tuple[np.ndarray, bool, float, int, np.ndarray | None]:
    x = np.array(x0, copy=True, dtype=float)
    jac_last: np.ndarray | None = None

    for it in range(int(cfg.max_iter)):
        fx = _eval_f(f, x, params)
        resid = float(np.linalg.norm(fx))
        if not np.isfinite(resid):
            return x, False, resid, it, jac_last
        if resid <= cfg.tol:
            return x, True, resid, it, jac_last

        if jac is None:
            J = _finite_diff_jac(f, x, fx, params, cfg.fd_eps)
        else:
            J = np.asarray(jac(x, params), dtype=float)
            if J.shape != (x.size, x.size):
                raise ValueError(f"jac(x, params) must return shape {(x.size, x.size)}, got {J.shape}")

        jac_last = J
        try:
            step = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -fx, rcond=None)[0]

        x = x + step

    fx = _eval_f(f, x, params)
    resid = float(np.linalg.norm(fx))
    return x, False, resid, int(cfg.max_iter), jac_last


def _dedupe_points(
    points: np.ndarray,
    residuals: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        return points, residuals, np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    unique_points: list[np.ndarray] = []
    unique_residuals: list[float] = []
    unique_sources: list[int] = []
    point_to_unique = np.full((points.shape[0],), -1, dtype=int)

    for i, point in enumerate(points):
        assigned = False
        for j, u in enumerate(unique_points):
            if np.linalg.norm(point - u) <= tol:
                point_to_unique[i] = j
                assigned = True
                if residuals[i] < unique_residuals[j]:
                    unique_points[j] = point.copy()
                    unique_residuals[j] = float(residuals[i])
                    unique_sources[j] = i
                break
        if not assigned:
            point_to_unique[i] = len(unique_points)
            unique_points.append(point.copy())
            unique_residuals.append(float(residuals[i]))
            unique_sources.append(i)

    return (
        np.asarray(unique_points, dtype=float),
        np.asarray(unique_residuals, dtype=float),
        point_to_unique,
        np.asarray(unique_sources, dtype=int),
    )


def _classify_eigs(kind: str, eigvals: np.ndarray, tol: float) -> str:
    if kind == "map":
        mags = np.abs(eigvals)
        if np.all(mags < 1.0 - tol):
            return "stable"
        if np.all(mags > 1.0 + tol):
            return "unstable"
        if np.any(np.abs(mags - 1.0) <= tol):
            return "neutral"
        return "saddle"

    real_parts = np.real(eigvals)
    if np.all(real_parts < -tol):
        return "stable"
    if np.all(real_parts > tol):
        return "unstable"
    if np.any(np.abs(real_parts) <= tol):
        return "neutral"
    return "saddle"


def find_fixed_points(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    *,
    seeds: Iterable[Iterable[float]] | np.ndarray,
    params: np.ndarray | None = None,
    cfg: FixedPointConfig | None = None,
) -> FixedPointResult:
    cfg = cfg or FixedPointConfig()
    if cfg.kind not in ("ode", "map"):
        raise ValueError("cfg.kind must be 'ode' or 'map'")
    if cfg.tol <= 0.0:
        raise ValueError("cfg.tol must be positive")
    if cfg.max_iter <= 0:
        raise ValueError("cfg.max_iter must be positive")
    if cfg.fd_eps <= 0.0:
        raise ValueError("cfg.fd_eps must be positive")
    if cfg.method != "newton":
        raise ValueError(f"Unknown method: {cfg.method!r}")

    seed_arr = _coerce_seeds(seeds)
    params_arr = _coerce_params(params)

    if cfg.jac == "fd":
        jac_fn = None
    elif cfg.jac == "provided":
        if jac is None:
            raise ValueError("jac is required when cfg.jac='provided'")
        jac_fn = jac
    else:
        jac_fn = jac

    n_seeds = seed_arr.shape[0]
    seed_points = np.zeros_like(seed_arr, dtype=float)
    seed_residuals = np.full((n_seeds,), np.nan, dtype=float)
    seed_converged = np.zeros((n_seeds,), dtype=bool)
    seed_iterations = np.zeros((n_seeds,), dtype=int)
    seed_jacobians: list[np.ndarray | None] = [None] * n_seeds

    for i, seed in enumerate(seed_arr):
        x, ok, resid, iters, jac_last = _solve_newton(f, jac_fn, seed, params_arr, cfg)
        seed_points[i] = x
        seed_residuals[i] = resid
        seed_converged[i] = ok
        seed_iterations[i] = iters
        seed_jacobians[i] = jac_last if ok else None

    conv_idx = np.where(seed_converged)[0]
    conv_points = seed_points[conv_idx]
    conv_residuals = seed_residuals[conv_idx]

    if cfg.unique_tol is None or cfg.unique_tol <= 0.0:
        unique_points = conv_points
        unique_residuals = conv_residuals
        conv_to_unique = np.arange(conv_points.shape[0], dtype=int)
        unique_sources = np.arange(conv_points.shape[0], dtype=int)
    else:
        unique_points, unique_residuals, conv_to_unique, unique_sources = _dedupe_points(
            conv_points,
            conv_residuals,
            float(cfg.unique_tol),
        )

    seed_to_unique = np.full((n_seeds,), -1, dtype=int)
    seed_to_unique[conv_idx] = conv_to_unique
    unique_seed_idx = conv_idx[unique_sources] if unique_sources.size else np.zeros((0,), dtype=int)

    jacobians: list[np.ndarray] | None = None
    eigvals: list[np.ndarray] | None = None
    stability: list[str] | None = None

    if unique_points.size and (cfg.classify or jac_fn is not None or cfg.jac == "fd"):
        jacobians = []
        eigvals = []
        stability = []
        for src_idx, point in zip(unique_seed_idx, unique_points):
            J = seed_jacobians[src_idx]
            if J is None:
                fx = _eval_f(f, point, params_arr)
                J = _finite_diff_jac(f, point, fx, params_arr, cfg.fd_eps)
            jacobians.append(J)

            if cfg.classify:
                if cfg.kind == "map":
                    J_eval = J + np.eye(J.shape[0])
                else:
                    J_eval = J
                vals = np.linalg.eigvals(J_eval)
                eigvals.append(vals)
                stability.append(_classify_eigs(cfg.kind, vals, cfg.stability_tol))
            else:
                eigvals = None
                stability = None

    meta = {
        "seed_points": seed_points,
        "seed_residuals": seed_residuals,
        "seed_converged": seed_converged,
        "seed_iterations": seed_iterations,
        "seed_to_unique": seed_to_unique,
        "unique_seed_indices": unique_seed_idx,
        "params": params_arr,
        "config": cfg,
    }

    return FixedPointResult(
        points=unique_points,
        residuals=unique_residuals,
        jacobians=jacobians,
        eigvals=eigvals,
        stability=stability,
        meta=meta,
    )
