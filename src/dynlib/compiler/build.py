# src/dynlib/compiler/build.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Tuple, Callable, Dict, Any, Union, List, Optional, Mapping, Sequence, TYPE_CHECKING, TextIO, ClassVar
from pathlib import Path
import inspect
import sys
import numpy as np
import tomllib

from dynlib.dsl.spec import ModelSpec, compute_spec_hash, build_spec, choose_default_stepper
from dynlib.dsl.parser import parse_model_v2
from dynlib.steppers.registry import get_stepper
from dynlib.compiler.codegen.emitter import (
    emit_rhs_and_events,
    emit_jacobian,
    CompiledCallables,
)
from dynlib.compiler.codegen import runner_cache as runner_cache_codegen
from dynlib.compiler.codegen.runner_variants import RunnerVariant, get_runner
from dynlib.compiler.codegen.cache_importer import register_cache_root
from dynlib.compiler.codegen.validate import validate_stepper_function, report_validation_issues
from dynlib.compiler.jit.compile import maybe_jit_triplet, jit_compile
from dynlib.compiler.jit.cache import JITCache, CacheKey
from dynlib.compiler.paths import resolve_uri, load_config, PathConfig, resolve_cache_root
from dynlib.compiler.mods import apply_mods_v2, ModSpec
from dynlib.compiler.guards import get_guards, configure_guards_disk_cache
from dynlib.errors import ModelLoadError, StepperKindMismatchError
from dynlib.runtime.workspace import (
    make_runtime_workspace,
    initialize_lag_runtime_workspace,
    workspace_structsig,
)
from dynlib.runtime.softdeps import softdeps
from dynlib.runtime.stepper_checks import check_stepper

if TYPE_CHECKING:
    from dynlib.analysis.fixed_points import FixedPointResult, FixedPointConfig


def _format_toml_parse_error(toml_content: str, error: Exception, source_desc: str = "inline model") -> str:
    """
    Format a TOML parsing error with context showing the problematic line.
    
    Args:
        toml_content: The TOML content that failed to parse
        error: The tomllib exception
        source_desc: Description of what's being parsed (e.g., "inline model", "mod file")
    
    Returns:
        A formatted error message with context
    """
    error_msg = str(error)
    
    # Try to extract line and column from the error message
    # Format: "... (at line X, column Y)"
    import re
    match = re.search(r'at line (\d+)(?:, column (\d+))?', error_msg)
    
    if not match:
        # Couldn't parse line/column, return basic error
        return f"Failed to parse {source_desc}: {error_msg}"
    
    line_num = int(match.group(1))
    col_num = int(match.group(2)) if match.group(2) else None
    
    # Split content into lines
    lines = toml_content.split('\n')
    
    # Build context message
    context_lines = []
    context_lines.append(f"Failed to parse {source_desc}:")
    context_lines.append(f"  {error_msg}")
    context_lines.append("")
    
    # Show a few lines of context around the error
    start_line = max(0, line_num - 3)
    end_line = min(len(lines), line_num + 2)
    
    context_lines.append("Context:")
    for i in range(start_line, end_line):
        line_marker = ">>>" if i == line_num - 1 else "   "
        context_lines.append(f"  {line_marker} {i+1:3d} | {lines[i]}")
        
        # Add column pointer if available and this is the error line
        if i == line_num - 1 and col_num is not None:
            # Calculate spacing: "  >>> NNN | " takes up 13 chars + col_num - 1
            pointer = " " * (13 + col_num - 1) + "^"
            context_lines.append(f"  {pointer}")
    
    context_lines.append("")
    
    # Add helpful hints based on common errors
    if "after a statement" in error_msg:
        context_lines.append("Hint: Check for invalid characters or syntax in TOML values.")
        context_lines.append("      Common issues:")
        context_lines.append("      - Division (/) in numeric values: use parentheses or quotes")
        context_lines.append("        WRONG: beta=8/3    CORRECT: beta=2.667 or beta='8/3' (as string)")
        context_lines.append("      - Missing quotes around string values")
        context_lines.append("      - Unclosed quotes or brackets")
    elif "Expected" in error_msg and "=" in error_msg:
        context_lines.append("Hint: Check that all key-value pairs use the format: key = value")
    elif "Invalid" in error_msg:
        context_lines.append("Hint: Check for typos or invalid TOML syntax")
    
    return "\n".join(context_lines)


__all__ = ["CompiledPieces", "build_callables", "FullModel", "build", "load_model_from_uri", "export_model_sources"]

@dataclass(frozen=True)
class CompiledPieces:
    spec: ModelSpec
    stepper_name: str
    rhs: callable
    events_pre: callable
    events_post: callable
    update_aux: callable
    spec_hash: str
    inv_rhs: Optional[Callable] = None
    triplet_digest: Optional[str] = None
    triplet_from_disk: bool = False
    rhs_source: Optional[str] = None
    events_pre_source: Optional[str] = None
    events_post_source: Optional[str] = None
    update_aux_source: Optional[str] = None
    inv_rhs_source: Optional[str] = None

@dataclass(frozen=True)
class FullModel:
    """Complete compiled model: includes runner + stepper + callables."""
    spec: ModelSpec
    stepper_name: str
    workspace_sig: Tuple[int, ...]
    rhs: Callable
    inv_rhs: Optional[Callable]
    events_pre: Callable
    events_post: Callable
    update_aux: Callable
    stepper: Callable
    runner: Callable
    spec_hash: str
    dtype: np.dtype
    jvp: Optional[Callable] = None
    jacobian: Optional[Callable] = None
    guards: Optional[Tuple[Callable, Callable]] = None  # (allfinite1d, allfinite_scalar)
    rhs_source: Optional[str] = None
    inv_rhs_source: Optional[str] = None
    events_pre_source: Optional[str] = None
    events_post_source: Optional[str] = None
    update_aux_source: Optional[str] = None
    stepper_source: Optional[str] = None
    jvp_source: Optional[str] = None
    jacobian_source: Optional[str] = None
    lag_state_info: Optional[Tuple[Tuple[int, int, int, int], ...]] = None
    uses_lag: bool = False
    equations_use_lag: bool = False
    make_stepper_workspace: Optional[Callable[[], object]] = None
    stepper_spec: Optional[object] = None  # StepperSpec for accessing capabilities
    _equation_tables: ClassVar[Dict[str, Tuple[str, Callable[[ModelSpec], List[str]]]]] = {}

    @classmethod
    def register_equation_table(
        cls,
        key: str,
        label: str,
        render: Callable[[ModelSpec], List[str]],
    ) -> None:
        cls._equation_tables[key] = (label, render)

    @classmethod
    def available_equation_tables(cls) -> Tuple[str, ...]:
        return tuple(cls._equation_tables.keys())

    def export_sources(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Export all source code files to a directory for inspection.
        
        Args:
            output_dir: Directory path where source files will be written
            
        Returns:
            Dictionary mapping component names to their file paths
            
        Example:
            >>> model = build("decay.toml", stepper="euler")
            >>> files = model.export_sources("./compiled_sources")
            >>> print(files)
            {'rhs': Path('./compiled_sources/rhs.py'), 
             'events_pre': Path('./compiled_sources/events_pre.py'), ...}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # Export each component's source if available
        components = [
            ("rhs", self.rhs_source),
            ("events_pre", self.events_pre_source),
            ("events_post", self.events_post_source),
            ("update_aux", self.update_aux_source),
            ("stepper", self.stepper_source),
            ("jvp", self.jvp_source),
            ("jacobian", self.jacobian_source),
        ]
        
        for name, source in components:
            if source is not None:
                file_path = output_path / f"{name}.py"
                file_path.write_text(source, encoding="utf-8")
                exported[name] = file_path
        
        return exported

    def print_equations(
        self,
        *,
        tables: Sequence[str] | str | None = ("equations",),
        include_headers: bool = True,
        file: Optional[TextIO] = None,
    ) -> None:
        """
        Print model equations from the DSL spec (not generated code).

        Args:
            tables: TOML table keys to print (e.g., "equations", "equations.inverse",
                "equations.jacobian"). Use "all" or None for all registered tables.
            include_headers: Include section headers.
            file: File-like stream to write to (defaults to stdout).
        """
        spec = self.spec
        lines: List[str] = []

        if tables is None or tables == "all":
            selected = list(self._equation_tables.keys())
        elif isinstance(tables, str):
            selected = [tables]
        else:
            selected = list(tables)

        for key in selected:
            if key not in self._equation_tables:
                known = ", ".join(self._equation_tables.keys())
                raise ValueError(f"Unknown equations table: {key!r}. Known tables: {known}")
            label, render = self._equation_tables[key]
            rendered = render(spec)
            if not rendered:
                continue
            if include_headers:
                if lines:
                    lines.append("")
                header = label
                if key == "equations" and spec.name:
                    header = f"{label} ({spec.name})"
                lines.append(f"{header}:")
            lines.extend(rendered)

        text = "\n".join(lines).rstrip()
        print(text, file=file or sys.stdout)

    @staticmethod
    def _ordered_rhs_lines(rhs: Dict[str, str], states: Tuple[str, ...]) -> List[str]:
        ordered: List[Tuple[str, str]] = []
        for state in states:
            if state in rhs:
                ordered.append((state, rhs[state]))
        for state, expr in rhs.items():
            if state not in states:
                ordered.append((state, expr))
        return [f"{state} = {expr}" for state, expr in ordered]

    @classmethod
    def _render_equations_table(cls, spec: ModelSpec) -> List[str]:
        if spec.equations_rhs:
            return cls._ordered_rhs_lines(spec.equations_rhs, spec.states)
        if spec.equations_block:
            return [spec.equations_block.strip()]
        return ["<no equations found>"]

    @classmethod
    def _render_inverse_table(cls, spec: ModelSpec) -> List[str]:
        if spec.inverse_rhs:
            return cls._ordered_rhs_lines(spec.inverse_rhs, spec.states)
        if spec.inverse_block:
            return [spec.inverse_block.strip()]
        return []

    @staticmethod
    def _render_jacobian_table(spec: ModelSpec) -> List[str]:
        if not spec.jacobian_exprs:
            return []
        lines: List[str] = []
        for row in spec.jacobian_exprs:
            row_items = ", ".join(str(expr) for expr in row)
            lines.append(f"[{row_items}]")
        return lines

    def fixed_points(
        self,
        *,
        params: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        seeds: Mapping[str, float] | Sequence[Sequence[float]] | np.ndarray | None = None,
        method: str | None = None,
        jac: str | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
        unique_tol: float | None = None,
        classify: bool | None = None,
        cfg: "FixedPointConfig | None" = None,
        t: float | None = None,
    ) -> "FixedPointResult":
        """
        Find fixed points / equilibria using a numerical root solver.

        Args:
            params: Parameter vector or dict overrides (default: model defaults).
            seeds: Initial guesses (array-like) or dict overrides for a single seed.
            method: Solver method (currently "newton").
            jac: "auto" (default), "fd" (finite-diff), or "analytic".
            tol: Residual tolerance for convergence.
            max_iter: Maximum solver iterations per seed.
            unique_tol: De-duplication tolerance for solutions.
            classify: Compute eigenvalues and stability labels.
            cfg: Optional FixedPointConfig override.
            t: Evaluation time for non-autonomous systems (default: spec.sim.t0).
        """
        from dynlib.analysis.fixed_points import FixedPointConfig, find_fixed_points

        n_state = len(self.spec.states)
        n_params = len(self.spec.params)
        dtype = self.dtype

        base_params = np.asarray(self.spec.param_vals, dtype=dtype)
        if params is None:
            params_vec = np.array(base_params, copy=True)
        elif isinstance(params, Mapping):
            params_vec = np.array(base_params, copy=True)
            param_index = {name: i for i, name in enumerate(self.spec.params)}
            for key, val in params.items():
                if key not in param_index:
                    raise KeyError(f"Unknown param '{key}'.")
                params_vec[param_index[key]] = float(val)
        else:
            params_arr = np.asarray(params, dtype=dtype)
            if params_arr.ndim != 1 or params_arr.shape[0] != n_params:
                raise ValueError(f"params must have shape ({n_params},)")
            params_vec = params_arr

        base_state = np.asarray(self.spec.state_ic, dtype=dtype)
        if seeds is None:
            seed_arr = base_state[None, :]
        elif isinstance(seeds, Mapping):
            seed_vec = np.array(base_state, copy=True)
            state_index = {name: i for i, name in enumerate(self.spec.states)}
            for key, val in seeds.items():
                if key not in state_index:
                    raise KeyError(f"Unknown state '{key}'.")
                seed_vec[state_index[key]] = float(val)
            seed_arr = seed_vec[None, :]
        else:
            seed_arr = np.asarray(seeds, dtype=dtype)
            if seed_arr.ndim == 1:
                seed_arr = seed_arr[None, :]
            if seed_arr.ndim != 2 or seed_arr.shape[1] != n_state:
                raise ValueError(f"seeds must have shape (n_seeds, {n_state})")

        stop_phase_mask = 0
        if self.spec.sim.stop is not None:
            phase = self.spec.sim.stop.phase
            if phase in ("pre", "both"):
                stop_phase_mask |= 1
            if phase in ("post", "both"):
                stop_phase_mask |= 2
        runtime_ws = make_runtime_workspace(
            lag_state_info=self.lag_state_info or (),
            dtype=dtype,
            n_aux=len(self.spec.aux or {}),
            stop_enabled=stop_phase_mask != 0,
            stop_phase_mask=stop_phase_mask,
        )

        t_eval = float(self.spec.sim.t0 if t is None else t)
        rhs_out = np.zeros((n_state,), dtype=dtype)
        jac_out = np.zeros((n_state, n_state), dtype=dtype)
        kind = self.spec.kind
        map_eye = np.eye(n_state, dtype=dtype) if kind == "map" else None

        def _prep_ws(y_vec: np.ndarray) -> None:
            if self.lag_state_info:
                initialize_lag_runtime_workspace(
                    runtime_ws,
                    lag_state_info=self.lag_state_info,
                    y_curr=y_vec,
                )

        def f_root(x: np.ndarray, params_in: np.ndarray) -> np.ndarray:
            y_vec = np.asarray(x, dtype=dtype)
            if y_vec.shape != (n_state,):
                raise ValueError(f"state must have shape ({n_state},)")
            if params_in.shape != (n_params,):
                raise ValueError(f"params must have shape ({n_params},)")
            p_vec = np.asarray(params_in, dtype=dtype)
            _prep_ws(y_vec)
            self.rhs(t_eval, y_vec, rhs_out, p_vec, runtime_ws)
            out = np.array(rhs_out, copy=True)
            if kind == "map":
                out -= y_vec
            return out

        def jac_root(x: np.ndarray, params_in: np.ndarray) -> np.ndarray:
            if self.jacobian is None:
                raise ValueError("Analytic Jacobian not available for this model.")
            y_vec = np.asarray(x, dtype=dtype)
            if y_vec.shape != (n_state,):
                raise ValueError(f"state must have shape ({n_state},)")
            if params_in.shape != (n_params,):
                raise ValueError(f"params must have shape ({n_params},)")
            p_vec = np.asarray(params_in, dtype=dtype)
            _prep_ws(y_vec)
            self.jacobian(t_eval, y_vec, p_vec, jac_out, runtime_ws)
            out = np.array(jac_out, copy=True)
            if kind == "map":
                out -= map_eye
            return out

        cfg_obj = cfg or FixedPointConfig()
        if method is not None:
            cfg_obj = replace(cfg_obj, method=method)
        if tol is not None:
            cfg_obj = replace(cfg_obj, tol=float(tol))
        if max_iter is not None:
            cfg_obj = replace(cfg_obj, max_iter=int(max_iter))
        if unique_tol is not None:
            cfg_obj = replace(cfg_obj, unique_tol=float(unique_tol))
        if classify is not None:
            cfg_obj = replace(cfg_obj, classify=bool(classify))
        cfg_obj = replace(cfg_obj, kind=kind)

        jac_mode = cfg_obj.jac
        if jac is not None:
            jac_norm = jac.lower()
            if jac_norm in ("auto",):
                jac_mode = "auto"
            elif jac_norm in ("fd", "finite-diff", "finite_diff"):
                jac_mode = "fd"
            elif jac_norm in ("analytic", "model", "provided"):
                jac_mode = "provided"
            else:
                raise ValueError("jac must be 'auto', 'fd', or 'analytic'")

        if jac_mode == "fd":
            jac_fn = None
        elif jac_mode == "provided":
            if self.jacobian is None:
                raise ValueError("jac='analytic' requires a model Jacobian.")
            jac_fn = jac_root
        else:
            jac_fn = jac_root if self.jacobian is not None else None

        cfg_obj = replace(cfg_obj, jac=jac_mode)
        return find_fixed_points(
            f_root,
            jac_fn,
            seeds=seed_arr,
            params=params_vec,
            cfg=cfg_obj,
        )


FullModel.register_equation_table("equations", "Equations", FullModel._render_equations_table)
FullModel.register_equation_table("equations.inverse", "Inverse equations", FullModel._render_inverse_table)
FullModel.register_equation_table("equations.jacobian", "Jacobian", FullModel._render_jacobian_table)


@dataclass
class _StepperCacheEntry:
    fn: Callable
    digest: Optional[str]
    from_disk: bool
    source: Optional[str] = None


_cache = JITCache()
_jac_cache = JITCache()
_stepper_cache: Dict[str, _StepperCacheEntry] = {}


def _dispatcher_compiled(fn: Callable) -> bool:
    signatures = getattr(fn, "signatures", None)
    return bool(signatures)


def _combined_workspace_sig(*workspaces: object | None) -> Tuple[int, ...]:
    sig: list[int] = []
    for ws in workspaces:
        sig.extend(workspace_structsig(ws))
    return tuple(sig)


def _compute_lag_state_info(spec: ModelSpec) -> Tuple[Tuple[int, int, int, int], ...]:
    if not spec.lag_map:
        return ()
    state_to_idx = {name: idx for idx, name in enumerate(spec.states)}
    return tuple(
        (state_to_idx[state_name], int(depth), int(offset), int(head_index))
        for state_name, (depth, offset, head_index) in spec.lag_map.items()
    )


class _TripletCacheContext:
    def __init__(
        self,
        *,
        spec_hash: str,
        stepper_name: str,
        structsig: Tuple[int, ...],
        dtype: str,
        cache_root: Path,
        sources: Dict[str, str],
    ):
        self.spec_hash = spec_hash
        self.stepper_name = stepper_name
        self.structsig = structsig
        self.dtype = dtype
        self.cache_root = cache_root
        self.sources = sources

    def configure(self, component: str) -> None:
        source = self.sources.get(component)
        if source is None:
            raise RuntimeError(f"No source available for component '{component}'")
        runner_cache_codegen.configure_triplet_disk_cache(
            component=component,
            spec_hash=self.spec_hash,
            stepper_name=self.stepper_name,
            structsig=self.structsig,
            dtype=self.dtype,
            cache_root=self.cache_root,
            source=source,
            function_name=component,
        )


def _render_stepper_source(stepper_fn: Callable) -> str:
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(stepper_fn)).strip()
    freevars = stepper_fn.__code__.co_freevars
    closure = stepper_fn.__closure__ or ()
    assignments = []
    for name, cell in zip(freevars, closure):
        value = cell.cell_contents
        assignments.append(f"{name} = {repr(value)}")
    prefix = "\n".join(assignments)
    if prefix:
        return f"{prefix}\n\n{source}\n"
    return f"{source}\n"


def _render_stepper_source_plain(stepper_fn: Callable) -> str:
    """Render stepper source for inspection/export (not for disk cache)."""
    import inspect
    import textwrap

    return textwrap.dedent(inspect.getsource(stepper_fn)).strip() + "\n"


def build_callables(
    spec: ModelSpec,
    *,
    stepper_name: str,
    jit: bool,
    dtype: str = "float64",
    cache_root: Optional[Path] = None,
    disk_cache: bool = True,
) -> CompiledPieces:
    """
    produce (rhs, events_pre, events_post, update_aux) with optional JIT.
    Also caches the stepper if jit=True to avoid recompilation.
    """
    # Guards are configured in build() and passed to runner via FullModel

    s_hash = compute_spec_hash(spec)
    lag_state_info = _compute_lag_state_info(spec)
    n_aux = len(spec.aux or {})

    stop_phase_mask = 0
    if spec.sim.stop is not None:
        phase = spec.sim.stop.phase
        if phase in ("pre", "both"):
            stop_phase_mask |= 1
        if phase in ("post", "both"):
            stop_phase_mask |= 2
    runtime_sig = workspace_structsig(
        make_runtime_workspace(
            lag_state_info=lag_state_info,
            dtype=np.dtype(dtype),
            n_aux=n_aux,
            stop_enabled=stop_phase_mask != 0,
            stop_phase_mask=stop_phase_mask,
        )
    )
    structsig = tuple(runtime_sig)
    key = CacheKey(
        spec_hash=s_hash,
        stepper=stepper_name,
        structsig=structsig,
        dtype=dtype,
        version_pins=("dynlib=2",),
    )

    cached = _cache.get(key)
    if cached is not None and cached.get("jit") == bool(jit):
        quad = cached.get("quadruplet")  # NEW: may have 4-tuple with update_aux
        tri = cached.get("triplet")  # OLD: 3-tuple without update_aux
        meta = cached.get("triplet_meta", {})
        sources = cached.get("sources", {})
        inv_rhs_cached = cached.get("inv_rhs")
        
        # Handle old caches (3-tuple) vs new caches (4-tuple)
        if quad and len(quad) == 4:
            return CompiledPieces(
                spec=spec,
                stepper_name=stepper_name,
                rhs=quad[0],
                events_pre=quad[1],
                events_post=quad[2],
                update_aux=quad[3],
                inv_rhs=inv_rhs_cached,
                spec_hash=s_hash,
                triplet_digest=meta.get("digest"),
                triplet_from_disk=meta.get("from_disk", False),
                rhs_source=sources.get("rhs"),
                inv_rhs_source=sources.get("inv_rhs"),
                events_pre_source=sources.get("events_pre"),
                events_post_source=sources.get("events_post"),
                update_aux_source=sources.get("update_aux"),
            )
        elif tri and len(tri) == 3:
            # Old cache without update_aux - regenerate by falling through
            pass
        else:
            # Unknown format - regenerate
            pass

    cc: CompiledCallables = emit_rhs_and_events(spec)
    use_disk_cache = bool(jit and disk_cache and cache_root is not None)

    cache_context = None
    if use_disk_cache:
        assert cache_root is not None
        cache_context = _TripletCacheContext(
            spec_hash=s_hash,
            stepper_name=stepper_name,
            structsig=structsig,
            dtype=dtype,
            cache_root=cache_root,
            sources={
                "rhs": cc.rhs_source,
                "events_pre": cc.events_pre_source,
                "events_post": cc.events_post_source,
                "update_aux": cc.update_aux_source,
                "inv_rhs": cc.inv_rhs_source,
            },
        )

    rhs_art, pre_art, post_art, aux_art = maybe_jit_triplet(
        cc.rhs,
        cc.events_pre,
        cc.events_post,
        cc.update_aux,
        jit=jit,
        cache=use_disk_cache,
        cache_setup=cache_context.configure if cache_context else None,
    )

    inv_rhs_fn = None
    inv_rhs_digest = None
    inv_rhs_from_disk = False
    if cc.inv_rhs is not None:
        if use_disk_cache and cache_context is not None:
            cache_context.configure("inv_rhs")
        inv_rhs_art = jit_compile(cc.inv_rhs, jit=jit, cache=use_disk_cache)
        inv_rhs_fn = inv_rhs_art.fn
        inv_rhs_digest = inv_rhs_art.cache_digest
        inv_rhs_from_disk = bool(inv_rhs_art.cache_hit)

    triplet_digest = rhs_art.cache_digest or pre_art.cache_digest or post_art.cache_digest or aux_art.cache_digest
    triplet_from_disk = rhs_art.cache_hit and pre_art.cache_hit and post_art.cache_hit and aux_art.cache_hit

    rhs_fn, pre_fn, post_fn, aux_fn = rhs_art.fn, pre_art.fn, post_art.fn, aux_art.fn

    _cache.put(
        key,
        {
            "quadruplet": (rhs_fn, pre_fn, post_fn, aux_fn),
            "inv_rhs": inv_rhs_fn,
            "jit": bool(jit),
            "triplet_meta": {"digest": triplet_digest, "from_disk": triplet_from_disk},
            "inv_rhs_meta": {"digest": inv_rhs_digest, "from_disk": inv_rhs_from_disk},
            "sources": {
                "rhs": cc.rhs_source,
                "events_pre": cc.events_pre_source,
                "events_post": cc.events_post_source,
                "update_aux": cc.update_aux_source,
                "inv_rhs": cc.inv_rhs_source,
            },
        },
    )

    return CompiledPieces(
        spec=spec,
        stepper_name=stepper_name,
        rhs=rhs_fn,
        events_pre=pre_fn,
        events_post=post_fn,
        update_aux=aux_fn,
        inv_rhs=inv_rhs_fn,
        spec_hash=s_hash,
        triplet_digest=triplet_digest,
        triplet_from_disk=triplet_from_disk,
        rhs_source=cc.rhs_source,
        inv_rhs_source=cc.inv_rhs_source,
        events_pre_source=cc.events_pre_source,
        events_post_source=cc.events_post_source,
        update_aux_source=cc.update_aux_source,
    )


def _warmup_jit_runner(
    runner: Callable,
    stepper: Callable,
    rhs: Callable,
    events_pre: Callable,
    events_post: Callable,
    update_aux: Callable,
    spec: ModelSpec,
    dtype: str,
    stepper_spec=None,  # NEW: optional stepper spec for config
) -> None:
    """
    Warm up JIT-compiled runner by calling it once with minimal inputs.
    This triggers Numba compilation at build time instead of at first runtime call.
    
    This warmup ALSO compiles the stepper, rhs, and events functions when they
    are called by the runner, ensuring everything is warmed up.
    """
    from dynlib.runtime.buffers import allocate_pools
    from dynlib.runtime.observers import observer_noop_variational_step
    
    dtype_np = np.dtype(dtype)
    n_state = len(spec.states)
    n_aux = len(spec.aux or {})

    stop_phase_mask = 0
    if spec.sim.stop is not None:
        phase = spec.sim.stop.phase
        if phase in ("pre", "both"):
            stop_phase_mask |= 1
        if phase in ("post", "both"):
            stop_phase_mask |= 2
    max_log_width = 0
    for evt in spec.events:
        max_log_width = max(max_log_width, len(getattr(evt, "log", ()) or ()))
    
    # Allocate minimal workspaces and buffers for warmup
    runtime_ws = make_runtime_workspace(
        lag_state_info=_compute_lag_state_info(spec),
        dtype=dtype_np,
        n_aux=n_aux,
        stop_enabled=stop_phase_mask != 0,
        stop_phase_mask=stop_phase_mask,
    )
    stepper_ws = (
        stepper_spec.make_workspace(n_state, dtype_np, model_spec=spec)
        if stepper_spec is not None
        else None
    )
    rec, ev = allocate_pools(
        n_state=n_state,
        dtype=dtype_np,
        cap_rec=2,      # Minimal capacity
        cap_evt=1,
        max_log_width=max(1, max_log_width),
    )
    n_rec_states = min(1, n_state) if n_state > 0 else 0
    n_rec_aux = min(1, n_aux) if n_aux > 0 else 0
    state_rec_indices = np.arange(n_rec_states, dtype=np.int32)
    aux_rec_indices = np.arange(n_rec_aux, dtype=np.int32)
    aux_rec = np.zeros((n_rec_aux, rec.cap_rec), dtype=dtype_np) if n_rec_aux else np.zeros((0, rec.cap_rec), dtype=dtype_np)
    
    # Create minimal arrays for warmup call
    y_curr = np.array(list(spec.state_ic), dtype=dtype_np)
    y_prev = np.array(list(spec.state_ic), dtype=dtype_np)
    params = np.array(list(spec.param_vals), dtype=dtype_np)
    
    y_prop = np.zeros((n_state,), dtype=dtype_np)
    # Stepper control values must be float64 (not model dtype) to match runtime
    t_prop = np.zeros((1,), dtype=np.float64)
    dt_next = np.zeros((1,), dtype=np.float64)
    err_est = np.zeros((1,), dtype=np.float64)
    evt_log_scratch = np.zeros((max(1, max_log_width),), dtype=dtype_np)
    
    # Analysis placeholders (warm-up runs without analysis)
    analysis_ws = np.zeros((0,), dtype=dtype_np)
    analysis_out = np.zeros((0,), dtype=dtype_np)
    analysis_trace = np.zeros((0, 0), dtype=dtype_np)
    analysis_trace_count = np.zeros((1,), dtype=np.int64)
    analysis_trace_cap = np.int64(0)
    analysis_trace_stride = np.int64(0)
    variational_step_enabled = np.int32(0)
    variational_step_fn = observer_noop_variational_step()
    
    user_break_flag = np.zeros((1,), dtype=np.int32)
    status_out = np.zeros((1,), dtype=np.int32)
    hint_out = np.zeros((1,), dtype=np.int32)
    i_out = np.zeros((1,), dtype=np.int64)
    step_out = np.zeros((1,), dtype=np.int64)
    t_out = np.zeros((1,), dtype=np.float64)  # Always float64, regardless of model dtype
    
    # Create stepper config (use defaults from model_spec)
    # NOTE: stepper_config is always float64, regardless of model dtype
    if stepper_spec is not None:
        default_config = stepper_spec.default_config(spec)
        stepper_config = stepper_spec.pack_config(default_config)
    else:
        stepper_config = np.array([], dtype=np.float64)
    
    # Warmup call: run for a single tiny step
    try:
        runner(
            0.0, 0.1, 0.1,  # t0, t_end, dt_init - use Python floats (float64), not model dtype
            100, n_state, 0,  # max_steps, n_state, record_interval (0 = no recording)
            y_curr, y_prev, params,
            runtime_ws,
            stepper_ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est,
            rec.T, rec.Y, aux_rec, rec.STEP, rec.FLAGS,
            ev.EVT_CODE, ev.EVT_INDEX, ev.EVT_LOG_DATA,
            evt_log_scratch,
            analysis_ws, analysis_out, analysis_trace,
            analysis_trace_count, int(analysis_trace_cap), int(analysis_trace_stride),
            int(variational_step_enabled), variational_step_fn,
            np.int64(0), np.int64(0), int(rec.cap_rec), int(ev.cap_evt),
            user_break_flag, status_out, hint_out,
            i_out, step_out, t_out,
            stepper, rhs, events_pre, events_post, update_aux,
            state_rec_indices, aux_rec_indices, n_rec_states, n_rec_aux,
        )
    except Exception:
        # Warmup failure is not critical - the JIT will compile on first real call
        # This might happen if the model has issues, but those will be caught later
        # Silently continue - errors will surface during actual usage
        pass


def load_model_from_uri(
    model_uri: str,
    *,
    mods: Optional[List[str]] = None,
    config: Optional[PathConfig] = None,
) -> ModelSpec:
    """
    Load and build a ModelSpec from a URI, applying mods if specified.
    
    Args:
        model_uri: URI for the base model:
            - Inline (same line): "inline: [model]\\ntype='ode'\\n..."
            - Inline (cleaner): "inline:\\n    [model]\\n    type='ode'\\n..."
            - Absolute path: "/abs/path/model.toml"
            - Relative path: "relative/model.toml" (from cwd)
            - TAG resolution: "TAG://model.toml" (from config)
            - With mod selector: Any of above + "#mod=NAME"
        mods: List of mod URIs to apply (same URI schemes).
            Each can be:
            - A full mod TOML file: "path/to/mods.toml"
            - A mod within a file: "path/to/file.toml#mod=NAME"
            - Inline mod (same line): "inline: [mod]\\nname='drive'\\n..."
            - Inline mod (cleaner): "inline:\\n    [mod]\\n    name='drive'\\n..."
        config: PathConfig for resolution (loads default if None)
    
    Returns:
        Validated ModelSpec with mods applied
    
    Raises:
        ModelNotFoundError: If any URI cannot be resolved
        ModelLoadError: If parsing/validation fails
        ConfigError: If TAG is unknown
    """
    if config is None:
        config = load_config()
    
    # Resolve base model URI
    resolved_model, model_fragment = resolve_uri(model_uri, config=config)
    
    # Load base model TOML
    # Check if it's inline by looking at the stripped/normalized URI
    if model_uri.strip().startswith("inline:"):
        try:
            model_data = tomllib.loads(resolved_model)
        except Exception as e:
            error_msg = _format_toml_parse_error(resolved_model, e, "inline model")
            raise ModelLoadError(error_msg)
    else:
        try:
            with open(resolved_model, "rb") as f:
                content = f.read().decode('utf-8')
                model_data = tomllib.loads(content)
        except tomllib.TOMLDecodeError as e:
            # For TOML parse errors, show context
            with open(resolved_model, "rb") as f:
                content = f.read().decode('utf-8')
            error_msg = _format_toml_parse_error(content, e, f"model file '{resolved_model}'")
            raise ModelLoadError(error_msg)
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {resolved_model}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {resolved_model}: {e}")
    
    # Parse to normalized form
    normal = parse_model_v2(model_data)
    
    # If model fragment specifies a mod, extract and apply it first
    if model_fragment and model_fragment.startswith("mod="):
        mod_name = model_fragment[4:].strip()
        if "mods" not in model_data or mod_name not in model_data["mods"]:
            raise ModelLoadError(
                f"Mod '{mod_name}' not found in {model_uri}. "
                f"Available mods: {list(model_data.get('mods', {}).keys())}"
            )
        
        # Extract and apply the specified mod
        mod_data = model_data["mods"][mod_name]
        mod_spec = _parse_mod_spec(mod_name, mod_data)
        normal = apply_mods_v2(normal, [mod_spec])
    
    # Apply additional mods if specified
    if mods:
        mod_specs = []
        for mod_uri in mods:
            mod_spec = _load_mod_from_uri(mod_uri, config=config)
            mod_specs.append(mod_spec)
        
        if mod_specs:
            normal = apply_mods_v2(normal, mod_specs)
    
    # Build and validate final spec
    spec = build_spec(normal)
    return spec


def _parse_mod_spec(name: str, mod_data: Dict[str, Any]) -> ModSpec:
    """Parse a mod table into a ModSpec."""
    return ModSpec(
        name=name,
        group=mod_data.get("group"),
        exclusive=mod_data.get("exclusive", False),
        remove=mod_data.get("remove"),
        replace=mod_data.get("replace"),
        add=mod_data.get("add"),
        set=mod_data.get("set"),
    )


def _load_mod_from_uri(mod_uri: str, config: PathConfig) -> ModSpec:
    """
    Load a ModSpec from a URI.
    
    Supports:
        - "path.toml#mod=NAME" -> load specific mod from file
        - "path.toml" -> load entire file as a mod collection (use first/only mod)
        - "inline: [mod]\\n..." -> parse inline mod
    """
    resolved, fragment = resolve_uri(mod_uri, config=config)
    
    # Load TOML
    # Check if it's inline by looking at the stripped/normalized URI
    if mod_uri.strip().startswith("inline:"):
        try:
            mod_data = tomllib.loads(resolved)
        except Exception as e:
            error_msg = _format_toml_parse_error(resolved, e, "inline mod")
            raise ModelLoadError(error_msg)
    else:
        try:
            with open(resolved, "rb") as f:
                content = f.read().decode('utf-8')
                mod_data = tomllib.loads(content)
        except tomllib.TOMLDecodeError as e:
            # For TOML parse errors, show context
            with open(resolved, "rb") as f:
                content = f.read().decode('utf-8')
            error_msg = _format_toml_parse_error(content, e, f"mod file '{resolved}'")
            raise ModelLoadError(error_msg)
        except FileNotFoundError as e:
            raise ModelLoadError(f"Mod file not found: {resolved}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load mod from {resolved}: {e}")
    
    # Extract mod
    if fragment and fragment.startswith("mod="):
        mod_name = fragment[4:].strip()
        if "mods" not in mod_data or mod_name not in mod_data["mods"]:
            raise ModelLoadError(
                f"Mod '{mod_name}' not found in {mod_uri}. "
                f"Available mods: {list(mod_data.get('mods', {}).keys())}"
            )
        return _parse_mod_spec(mod_name, mod_data["mods"][mod_name])
    else:
        # No fragment specified - check if this is a [mod] table or has [mods.*]
        if "mod" in mod_data and isinstance(mod_data["mod"], dict):
            # Single [mod] table
            name = mod_data["mod"].get("name", "unnamed")
            return _parse_mod_spec(name, mod_data["mod"])
        elif "mods" in mod_data:
            # Multiple [mods.*] - error, need to specify which one
            available = list(mod_data["mods"].keys())
            raise ModelLoadError(
                f"Mod file {mod_uri} contains multiple mods: {available}. "
                f"Please specify which one using #mod=NAME"
            )
        else:
            raise ModelLoadError(f"No [mod] or [mods.*] table found in {mod_uri}")


def build(
    model: Union[ModelSpec, str],
    *,
    stepper: Optional[str] = None,
    mods: Optional[List[str]] = None,
    jit: bool = False,
    dtype: Optional[str] = None,
    disk_cache: bool = False,
    config: Optional[PathConfig] = None,
    validate_stepper: bool = True,
) -> FullModel:
    """
    Build a complete compiled model with runner + stepper.
    
    Supports both direct ModelSpec and URI-based loading.
    
    Args:
        model: Either a validated ModelSpec or a URI string:
            - "inline: [model]\\ntype='ode'\\n..." -> parse directly
            - "/abs/path/model.toml" -> load from absolute path
            - "relative/model.toml" -> load relative to cwd
            - "TAG://model.toml" -> resolve using config tags
            - Any of above with "#mod=NAME" fragment for mod selection
        stepper_name: Name of the registered stepper (e.g., "euler").
            If None, uses the model's sim.stepper default.
        mods: List of mod URIs to apply (same URI schemes as model).
            Mods are applied in order after loading the base model.
        jit: Enable JIT compilation (default False)
        disk_cache: Enable persistent runner cache on disk (default False)
        dtype: Model dtype string. If None (default), uses the dtype from the model spec.
        config: PathConfig for URI resolution (loads default if None)
        validate_stepper: Enable build-time stepper validation (default True)
    
    Returns:
        FullModel with all compiled components
    
    Raises:
        ModelNotFoundError: If URI cannot be resolved
        ModelLoadError: If parsing/validation fails
        ConfigError: If config is invalid or TAG is unknown
        StepperKindMismatchError: If stepper kind doesn't match model kind
        StepperValidationError: If stepper validation fails
    """
    # Always resolve a config so cache_root resolution stays in sync with path lookup
    config_in_use = config or load_config()

    # If model is already a ModelSpec, use it directly
    if isinstance(model, ModelSpec):
        spec = model
    else:
        # Load from URI
        spec = load_model_from_uri(model, mods=mods, config=config_in_use)
    
    # Configure finiteness guard according to jit toggle
    # TODO: Place Inf / NaN check
    # guards.configure_allfinite_guard(bool(jit))

    # Use spec's default stepper if not specified
    stepper_name = stepper
    if stepper_name is None:
        stepper_name = spec.sim.stepper or choose_default_stepper(spec.kind)
    
    # Use spec's dtype if not specified
    dtype_str = dtype if dtype is not None else spec.dtype
    dtype_np = np.dtype(dtype_str)
    n_state = len(spec.states)
    
    # Get stepper spec
    stepper_spec = get_stepper(stepper_name)
    
    # Validate stepper kind matches model kind (guardrails check)
    if stepper_spec.meta.kind != spec.kind:
        raise StepperKindMismatchError(
            stepper_name=stepper_name,
            stepper_kind=stepper_spec.meta.kind,
            model_kind=spec.kind
        )

    deps = softdeps()
    check_stepper(
        stepper_name=stepper_name,
        jit=bool(jit),
        dtype=dtype_np,
        stepper_spec=stepper_spec,
        deps=deps,
    )
    
    lag_state_info_list = _compute_lag_state_info(spec)
    n_aux = len(spec.aux or {})

    stop_phase_mask = 0
    if spec.sim.stop is not None:
        phase = spec.sim.stop.phase
        if phase in ("pre", "both"):
            stop_phase_mask |= 1
        if phase in ("post", "both"):
            stop_phase_mask |= 2
    runtime_ws_template = make_runtime_workspace(
        lag_state_info=lag_state_info_list,
        dtype=dtype_np,
        n_aux=n_aux,
        stop_enabled=stop_phase_mask != 0,
        stop_phase_mask=stop_phase_mask,
    )

    def _make_stepper_workspace() -> object | None:
        return stepper_spec.make_workspace(n_state, dtype_np, model_spec=spec)

    stepper_ws_sample = _make_stepper_workspace()
    workspace_sig = _combined_workspace_sig(stepper_ws_sample, runtime_ws_template)
    stepper_sig = workspace_structsig(stepper_ws_sample)

    cache_root_path: Optional[Path] = None
    if jit and disk_cache:
        cache_root_path = resolve_cache_root(config_in_use)
        register_cache_root(cache_root_path)
    
    # Build RHS and events
    pieces = build_callables(
        spec,
        stepper_name=stepper_name,
        jit=jit,
        dtype=dtype_str,
        cache_root=cache_root_path,
        disk_cache=disk_cache,
    )

    # Jacobian-derived operators (optional)
    jvp_fn = None
    jacobian_fn = None
    jvp_source = None
    jacobian_source = None
    jac_digest = None
    jac_from_disk = False
    if spec.jacobian_exprs:
        jac_structsig = workspace_structsig(runtime_ws_template)
        jac_key = CacheKey(
            spec_hash=pieces.spec_hash,
            stepper="jacobian",
            structsig=jac_structsig,
            dtype=dtype_str,
            version_pins=("dynlib=2",),
        )
        cached_jac = _jac_cache.get(jac_key)
        if cached_jac is not None and cached_jac.get("jit") == bool(jit):
            jvp_fn = cached_jac.get("jvp")
            jacobian_fn = cached_jac.get("jacobian")
            jvp_source = cached_jac.get("jvp_source")
            jacobian_source = cached_jac.get("jacobian_source")
            meta = cached_jac.get("meta", {}) or {}
            jac_digest = meta.get("digest")
            jac_from_disk = meta.get("from_disk", False)
        else:
            compiled_jac = emit_jacobian(spec)
            if compiled_jac is not None:
                use_disk_cache = bool(jit and disk_cache and cache_root_path is not None)
                cache_digest = None
                cache_from_disk = True if use_disk_cache else False

                if use_disk_cache:
                    runner_cache_codegen.configure_triplet_disk_cache(
                        component="jvp",
                        spec_hash=pieces.spec_hash,
                        stepper_name="jacobian",
                        structsig=jac_structsig,
                        dtype=dtype_str,
                        cache_root=cache_root_path,
                        source=compiled_jac.jvp_source,
                        function_name="jvp",
                    )
                jvp_art = jit_compile(compiled_jac.jvp, jit=jit, cache=use_disk_cache)
                jvp_fn = jvp_art.fn
                jvp_source = compiled_jac.jvp_source
                cache_digest = cache_digest or jvp_art.cache_digest
                cache_from_disk = cache_from_disk and bool(jvp_art.cache_hit)

                if compiled_jac.jacobian is not None and compiled_jac.jacobian_source is not None:
                    if use_disk_cache:
                        runner_cache_codegen.configure_triplet_disk_cache(
                            component="jacobian",
                            spec_hash=pieces.spec_hash,
                            stepper_name="jacobian",
                            structsig=jac_structsig,
                            dtype=dtype_str,
                            cache_root=cache_root_path,
                            source=compiled_jac.jacobian_source,
                            function_name="jacobian",
                        )
                    jac_art = jit_compile(compiled_jac.jacobian, jit=jit, cache=use_disk_cache)
                    jacobian_fn = jac_art.fn
                    jacobian_source = compiled_jac.jacobian_source
                    cache_digest = cache_digest or jac_art.cache_digest
                    cache_from_disk = cache_from_disk and bool(jac_art.cache_hit)

                jac_digest = cache_digest
                jac_from_disk = cache_from_disk
                _jac_cache.put(
                    jac_key,
                    {
                        "jvp": jvp_fn,
                        "jacobian": jacobian_fn,
                        "jit": bool(jit),
                        "meta": {"digest": jac_digest, "from_disk": jac_from_disk},
                        "jvp_source": compiled_jac.jvp_source,
                        "jacobian_source": jacobian_source,
                    },
                )

    # Guards must be ready before JIT compilation so steppers see the updated symbols.
    guards_tuple = get_guards(jit=jit, disk_cache=False)  # disk cache unnecessary (inline functions)
    
    stepper_cache_key = f"{pieces.spec_hash}:{stepper_name}:{jit}"
    stepper_entry = _stepper_cache.get(stepper_cache_key)
    stepper_from_disk = False
    
    if stepper_entry is None:
        emit_sig = inspect.signature(stepper_spec.emit)
        emit_kwargs = {"model_spec": spec}
        if "jacobian_fn" in emit_sig.parameters:
            emit_kwargs["jacobian_fn"] = jacobian_fn
        if "jvp_fn" in emit_sig.parameters:
            emit_kwargs["jvp_fn"] = jvp_fn
        stepper_py = stepper_spec.emit(pieces.rhs, **emit_kwargs)
        
        if validate_stepper:
            issues = validate_stepper_function(stepper_py, stepper_name)
            report_validation_issues(issues, stepper_name, strict=False)
        
        freevars = stepper_py.__code__.co_freevars
        has_closure = bool(freevars)
        # Always keep source for export/inspection regardless of disk_cache.
        # (Disk cache below is more restrictive.)
        stepper_source = _render_stepper_source_plain(stepper_py)

        # Disable disk cache when the stepper closes over non-serializable objects
        stepper_disk_cache = bool(jit and disk_cache and cache_root_path is not None and not has_closure)
        if stepper_disk_cache:
            # For disk cache we can keep the richer rendering (closure-safe path is already gated).
            stepper_source_disk = _render_stepper_source(stepper_py)
            runner_cache_codegen.configure_stepper_disk_cache(
                spec_hash=pieces.spec_hash,
                stepper_name=stepper_name,
                structsig=stepper_sig,
                dtype=dtype_str,
                cache_root=cache_root_path,
                source=stepper_source_disk,
                function_name=stepper_py.__name__,
            )
        compiled = jit_compile(stepper_py, jit=jit, cache=stepper_disk_cache)
        stepper_entry = _StepperCacheEntry(
            fn=compiled.fn,
            digest=compiled.cache_digest,
            from_disk=compiled.cache_hit,
            source=stepper_source,
        )
        _stepper_cache[stepper_cache_key] = stepper_entry
    else:
        # Retrieved from cache, get source if available
        stepper_source = stepper_entry.source
    
    stepper_fn = stepper_entry.fn
    stepper_from_disk = stepper_entry.from_disk
    
    is_discrete = stepper_spec.meta.kind == "map"
    if jit and disk_cache and cache_root_path is not None:
        runner_cache_codegen.configure_runner_disk_cache(
            model_hash=pieces.spec_hash,
            stepper_name=stepper_name,
            structsig=workspace_sig,
            dtype=dtype_str,
            cache_root=cache_root_path,
        )
    else:
        runner_cache_codegen.disable_runner_disk_cache(
            model_hash=pieces.spec_hash,
            stepper_name=stepper_name,
        )

    runner_fn = get_runner(
        RunnerVariant.BASE,
        model_hash=pieces.spec_hash,
        stepper_name=stepper_name,
        analysis=None,
        dtype=np.dtype(dtype_str),
        jit=jit,
        discrete=is_discrete,
    )

    def _all_compiled() -> bool:
        return all(
            _dispatcher_compiled(obj)
            for obj in (
                runner_fn,
                stepper_fn,
                pieces.rhs,
                pieces.events_pre,
                pieces.events_post,
                pieces.update_aux,
            )
        )

    if jit and not _all_compiled():
        _warmup_jit_runner(
            runner_fn,
            stepper_fn,
            pieces.rhs,
            pieces.events_pre,
            pieces.events_post,
            pieces.update_aux,
            spec,
            dtype_str,
            stepper_spec,  # NEW: pass stepper_spec
        )
    
    dtype_np = np.dtype(dtype_str)
    
    return FullModel(
        spec=spec,
        stepper_name=stepper_name,
        workspace_sig=workspace_sig,
        rhs=pieces.rhs,
        inv_rhs=pieces.inv_rhs,
        events_pre=pieces.events_pre,
        events_post=pieces.events_post,
        update_aux=pieces.update_aux,
        stepper=stepper_fn,
        runner=runner_fn,
        spec_hash=pieces.spec_hash,
        dtype=dtype_np,
        jvp=jvp_fn,
        jacobian=jacobian_fn,
        guards=guards_tuple,
        rhs_source=pieces.rhs_source,
        inv_rhs_source=pieces.inv_rhs_source,
        events_pre_source=pieces.events_pre_source,
        events_post_source=pieces.events_post_source,
        update_aux_source=pieces.update_aux_source,
        stepper_source=stepper_source if 'stepper_source' in locals() and stepper_source else None,
        jvp_source=jvp_source,
        jacobian_source=jacobian_source,
        lag_state_info=lag_state_info_list,
        uses_lag=spec.uses_lag,
        equations_use_lag=spec.equations_use_lag,
        make_stepper_workspace=_make_stepper_workspace,
        stepper_spec=stepper_spec,
    )


def export_model_sources(model: FullModel, output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Export all source code files from a compiled model to a directory for inspection.
    
    Args:
        model: The compiled FullModel instance
        output_dir: Directory path where source files will be written
        
    Returns:
        Dictionary mapping component names to their file paths
        
    Example:
        >>> from dynlib import build
        >>> from dynlib.compiler.build import export_model_sources
        >>> model = build("decay.toml", stepper="euler")
        >>> files = export_model_sources(model, "./compiled_sources")
        >>> print(files)
        {'rhs': Path('./compiled_sources/rhs.py'), 
         'events_pre': Path('./compiled_sources/events_pre.py'), ...}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exported = {}
    
    # Export each component's source if available
    components = [
        ("rhs", model.rhs_source),
        ("inv_rhs", getattr(model, "inv_rhs_source", None)),
        ("events_pre", model.events_pre_source),
        ("events_post", model.events_post_source),
        ("update_aux", getattr(model, "update_aux_source", None)),
        ("stepper", model.stepper_source),
        ("jvp", getattr(model, "jvp_source", None)),
        ("jacobian", getattr(model, "jacobian_source", None)),
    ]
    
    for name, source in components:
        if source is not None:
            file_path = output_path / f"{name}.py"
            file_path.write_text(source, encoding="utf-8")
            exported[name] = file_path
    
    # Also export model spec summary as text file
    spec_path = output_path / "model_info.txt"
    info_lines = [
        f"Model Information",
        f"=" * 60,
        f"Spec Hash: {model.spec_hash}",
        f"Kind: {model.spec.kind}",
        f"Stepper: {model.stepper_name}",
        f"Dtype: {model.dtype}",
        f"",
        f"States: {', '.join(model.spec.states)}",
        f"Parameters: {', '.join(model.spec.params)}",
        f"",
    ]
    
    if model.spec.equations_rhs:
        info_lines.append("Equations (RHS):")
        for state, expr in model.spec.equations_rhs.items():
            info_lines.append(f"  {state} = {expr}")
        info_lines.append("")
    
    if model.spec.events:
        info_lines.append(f"Events ({len(model.spec.events)}):")
        for i, event in enumerate(model.spec.events, 1):
            info_lines.append(f"  [{i}] phase={event.phase}, cond={event.cond}")
            if event.action_block:
                action_preview = event.action_block[:50] + "..." if len(event.action_block) > 50 else event.action_block
                info_lines.append(f"      action={action_preview}")
            elif event.action_keyed:
                info_lines.append(f"      action={event.action_keyed}")
        info_lines.append("")
    
    spec_path.write_text("\n".join(info_lines), encoding="utf-8")
    exported["info"] = spec_path
    
    return exported
