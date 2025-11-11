# src/dynlib/compiler/build.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any, Union, List, Optional
from pathlib import Path
import numpy as np

from dynlib.dsl.spec import ModelSpec, compute_spec_hash, build_spec
from dynlib.dsl.parser import parse_model_v2
from dynlib.steppers.registry import get_stepper
from dynlib.steppers.base import StructSpec
from dynlib.compiler.codegen.emitter import emit_rhs_and_events, CompiledCallables
from dynlib.compiler.codegen import runner as runner_codegen
from dynlib.compiler.codegen import runner_discrete as runner_discrete_codegen
from dynlib.compiler.codegen.validate import validate_stepper_function, report_validation_issues
from dynlib.compiler.jit.compile import maybe_jit_triplet, jit_compile
from dynlib.compiler.jit.cache import JITCache, CacheKey
from dynlib.compiler.paths import resolve_uri, load_config, PathConfig, resolve_cache_root
from dynlib.compiler.mods import apply_mods_v2, ModSpec
from dynlib.errors import ModelLoadError, StepperKindMismatchError

__all__ = ["CompiledPieces", "build_callables", "FullModel", "build", "load_model_from_uri", "export_model_sources"]

@dataclass(frozen=True)
class CompiledPieces:
    spec: ModelSpec
    stepper_name: str
    rhs: callable
    events_pre: callable
    events_post: callable
    spec_hash: str
    triplet_digest: Optional[str] = None
    triplet_from_disk: bool = False
    rhs_source: Optional[str] = None
    events_pre_source: Optional[str] = None
    events_post_source: Optional[str] = None

@dataclass(frozen=True)
class FullModel:
    """Complete compiled model: includes runner + stepper + callables."""
    spec: ModelSpec
    stepper_name: str
    struct: StructSpec
    rhs: Callable
    events_pre: Callable
    events_post: Callable
    stepper: Callable
    runner: Callable
    spec_hash: str
    dtype: np.dtype
    rhs_source: Optional[str] = None
    events_pre_source: Optional[str] = None
    events_post_source: Optional[str] = None
    stepper_source: Optional[str] = None

@dataclass
class _StepperCacheEntry:
    fn: Callable
    digest: Optional[str]
    from_disk: bool
    source: Optional[str] = None


_cache = JITCache()
_stepper_cache: Dict[str, _StepperCacheEntry] = {}


def _dispatcher_compiled(fn: Callable) -> bool:
    signatures = getattr(fn, "signatures", None)
    return bool(signatures)


def _structsig_from_struct(struct: StructSpec) -> Tuple[int, ...]:
    return (
        struct.sp_size, struct.ss_size,
        struct.sw0_size, struct.sw1_size, struct.sw2_size, struct.sw3_size,
        struct.iw0_size, struct.bw0_size,
        int(bool(struct.use_history)), int(bool(struct.use_f_history)),
        int(bool(struct.dense_output)), int(bool(struct.needs_jacobian)),
        -1 if struct.embedded_order is None else int(struct.embedded_order),
        int(bool(struct.stiff_ok)),
    )


def _structsig_from_stepper(stepper_name: str) -> Tuple[int, ...]:
    # StructSpec signature is a tuple of sizes; we only need it for the cache key.
    spec = get_stepper(stepper_name).struct_spec()
    return _structsig_from_struct(spec)


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
        runner_codegen.configure_triplet_disk_cache(
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
    produce (rhs, events_pre, events_post) with optional JIT.
    Also caches the stepper if jit=True to avoid recompilation.
    """
    s_hash = compute_spec_hash(spec)
    structsig = _structsig_from_stepper(stepper_name)
    key = CacheKey(
        spec_hash=s_hash,
        stepper=stepper_name,
        structsig=structsig,
        dtype=dtype,
        version_pins=("dynlib=2",),
    )

    cached = _cache.get(key)
    if cached is not None and cached.get("jit") == bool(jit):
        tri = cached["triplet"]
        meta = cached.get("triplet_meta", {})
        sources = cached.get("sources", {})
        return CompiledPieces(
            spec,
            stepper_name,
            tri[0],
            tri[1],
            tri[2],
            s_hash,
            triplet_digest=meta.get("digest"),
            triplet_from_disk=meta.get("from_disk", False),
            rhs_source=sources.get("rhs"),
            events_pre_source=sources.get("events_pre"),
            events_post_source=sources.get("events_post"),
        )

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
            },
        )

    rhs_art, pre_art, post_art = maybe_jit_triplet(
        cc.rhs,
        cc.events_pre,
        cc.events_post,
        jit=jit,
        cache=use_disk_cache,
        cache_setup=cache_context.configure if cache_context else None,
    )

    triplet_digest = rhs_art.cache_digest or pre_art.cache_digest or post_art.cache_digest
    triplet_from_disk = rhs_art.cache_hit and pre_art.cache_hit and post_art.cache_hit

    rhs_fn, pre_fn, post_fn = rhs_art.fn, pre_art.fn, post_art.fn

    _cache.put(
        key,
        {
            "triplet": (rhs_fn, pre_fn, post_fn),
            "jit": bool(jit),
            "triplet_meta": {"digest": triplet_digest, "from_disk": triplet_from_disk},
            "sources": {
                "rhs": cc.rhs_source,
                "events_pre": cc.events_pre_source,
                "events_post": cc.events_post_source,
            },
        },
    )

    return CompiledPieces(
        spec,
        stepper_name,
        rhs_fn,
        pre_fn,
        post_fn,
        s_hash,
        triplet_digest=triplet_digest,
        triplet_from_disk=triplet_from_disk,
        rhs_source=cc.rhs_source,
        events_pre_source=cc.events_pre_source,
        events_post_source=cc.events_post_source,
    )


def _warmup_jit_runner(
    runner: Callable,
    stepper: Callable,
    rhs: Callable,
    events_pre: Callable,
    events_post: Callable,
    struct: StructSpec,
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
    
    dtype_np = np.dtype(dtype)
    n_state = len(spec.states)
    
    # Allocate minimal banks and buffers for warmup
    banks, rec, ev = allocate_pools(
        n_state=n_state,
        struct=struct,
        dtype=dtype_np,
        cap_rec=2,      # Minimal capacity
        cap_evt=1,
        max_log_width=1,
    )
    
    # Create minimal arrays for warmup call
    y_curr = np.array(list(spec.state_ic), dtype=dtype_np)
    y_prev = np.array(list(spec.state_ic), dtype=dtype_np)
    params = np.array(list(spec.param_vals), dtype=dtype_np)
    
    y_prop = np.zeros((n_state,), dtype=dtype_np)
    t_prop = np.zeros((1,), dtype=dtype_np)
    dt_next = np.zeros((1,), dtype=dtype_np)
    err_est = np.zeros((1,), dtype=dtype_np)
    evt_log_scratch = np.zeros((1,), dtype=dtype_np)
    
    user_break_flag = np.zeros((1,), dtype=np.int32)
    status_out = np.zeros((1,), dtype=np.int32)
    hint_out = np.zeros((1,), dtype=np.int32)
    i_out = np.zeros((1,), dtype=np.int64)
    step_out = np.zeros((1,), dtype=np.int64)
    t_out = np.zeros((1,), dtype=np.float64)
    
    # Create stepper config (use defaults from model_spec)
    if stepper_spec is not None:
        default_config = stepper_spec.default_config(spec)
        stepper_config = stepper_spec.pack_config(default_config)
    else:
        stepper_config = np.array([], dtype=np.float64)
    
    # Warmup call: run for a single tiny step
    try:
        runner(
            0.0, 0.1, 0.1,  # t0, t_end, dt_init
            100, n_state, 0,  # max_steps, n_state, record_interval (0 = no recording)
            y_curr, y_prev, params,
            banks.sp, banks.ss, banks.sw0, banks.sw1, banks.sw2, banks.sw3,
            banks.iw0, banks.bw0,
            stepper_config,  # NEW: stepper config
            y_prop, t_prop, dt_next, err_est,
            rec.T, rec.Y, rec.STEP, rec.FLAGS,
            ev.EVT_CODE, ev.EVT_INDEX, ev.EVT_LOG_DATA,
            evt_log_scratch,
            np.int64(0), np.int64(0), int(rec.cap_rec), int(ev.cap_evt),
            user_break_flag, status_out, hint_out,
            i_out, step_out, t_out,
            stepper, rhs, events_pre, events_post,
        )
    except Exception:
        # Warmup failure is not critical - the JIT will compile on first real call
        # This might happen if the model has issues, but those will be caught later
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
        # Parse inline content
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # Python < 3.11
        
        try:
            model_data = tomllib.loads(resolved_model)
        except Exception as e:
            raise ModelLoadError(f"Failed to parse inline model: {e}")
    else:
        # Load from file
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # Python < 3.11
        
        try:
            with open(resolved_model, "rb") as f:
                model_data = tomllib.load(f)
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
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        try:
            mod_data = tomllib.loads(resolved)
        except Exception as e:
            raise ModelLoadError(f"Failed to parse inline mod: {e}")
    else:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        try:
            with open(resolved, "rb") as f:
                mod_data = tomllib.load(f)
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
    jit: bool = True,
    dtype: str = "float64",
    disk_cache: bool = True,
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
        jit: Enable JIT compilation (default True)
        disk_cache: Enable persistent runner cache on disk (default True)
        dtype: Model dtype string (default "float64")
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
    
    # Use spec's default stepper if not specified
    stepper_name = stepper
    if stepper_name is None:
        stepper_name = spec.sim.stepper
    
    # Get stepper spec
    stepper_spec = get_stepper(stepper_name)
    
    # Validate stepper kind matches model kind (guardrails check)
    if stepper_spec.meta.kind != spec.kind:
        raise StepperKindMismatchError(
            stepper_name=stepper_name,
            stepper_kind=stepper_spec.meta.kind,
            model_kind=spec.kind
        )
    
    struct = stepper_spec.struct_spec()
    structsig = _structsig_from_struct(struct)

    cache_root_path: Optional[Path] = None
    if jit and disk_cache:
        cache_root_path = resolve_cache_root(config_in_use)
    
    # Build RHS and events
    pieces = build_callables(
        spec,
        stepper_name=stepper_name,
        jit=jit,
        dtype=dtype,
        cache_root=cache_root_path,
        disk_cache=disk_cache,
    )
    
    stepper_cache_key = f"{pieces.spec_hash}:{stepper_name}:{jit}"
    stepper_entry = _stepper_cache.get(stepper_cache_key)
    stepper_from_disk = False
    
    if stepper_entry is None:
        stepper_py = stepper_spec.emit(pieces.rhs, struct, model_spec=spec)
        
        if validate_stepper:
            issues = validate_stepper_function(stepper_py, stepper_name, struct_spec=struct)
            report_validation_issues(issues, stepper_name, strict=False)
        
        stepper_source = _render_stepper_source(stepper_py)
        stepper_disk_cache = bool(jit and disk_cache and cache_root_path is not None)
        if stepper_disk_cache and cache_root_path is not None:
            runner_codegen.configure_stepper_disk_cache(
                spec_hash=pieces.spec_hash,
                stepper_name=stepper_name,
                structsig=structsig,
                dtype=dtype,
                cache_root=cache_root_path,
                source=stepper_source,
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
    
    # Select runner based on stepper kind (discrete vs continuous)
    # Discrete systems (maps, difference equations): use runner_discrete with N-based termination
    # Continuous systems (ODEs, SDEs, DAEs): use runner with T-based termination
    if stepper_spec.meta.kind == "map":
        # Use discrete runner
        if jit and disk_cache and cache_root_path is not None:
            runner_discrete_codegen.configure_runner_disk_cache_discrete(
                spec_hash=pieces.spec_hash,
                stepper_name=stepper_name,
                structsig=structsig,
                dtype=dtype,
                cache_root=cache_root_path,
            )
        runner_fn = runner_discrete_codegen.get_runner_discrete(jit=jit, disk_cache=disk_cache)
    else:
        # Use continuous runner (default)
        if jit and disk_cache and cache_root_path is not None:
            runner_codegen.configure_runner_disk_cache(
                spec_hash=pieces.spec_hash,
                stepper_name=stepper_name,
                structsig=structsig,
                dtype=dtype,
                cache_root=cache_root_path,
            )
        runner_fn = runner_codegen.get_runner(jit=jit, disk_cache=disk_cache)

    
    def _all_compiled() -> bool:
        return all(
            _dispatcher_compiled(obj)
            for obj in (runner_fn, stepper_fn, pieces.rhs, pieces.events_pre, pieces.events_post)
        )

    if jit and not _all_compiled():
        _warmup_jit_runner(
            runner_fn, stepper_fn, pieces.rhs, pieces.events_pre, pieces.events_post,
            struct, spec, dtype, stepper_spec  # NEW: pass stepper_spec
        )
    
    dtype_np = np.dtype(dtype)
    
    return FullModel(
        spec=spec,
        stepper_name=stepper_name,
        struct=struct,
        rhs=pieces.rhs,
        events_pre=pieces.events_pre,
        events_post=pieces.events_post,
        stepper=stepper_fn,
        runner=runner_fn,
        spec_hash=pieces.spec_hash,
        dtype=dtype_np,
        rhs_source=pieces.rhs_source,
        events_pre_source=pieces.events_pre_source,
        events_post_source=pieces.events_post_source,
        stepper_source=stepper_source if 'stepper_source' in locals() and stepper_source else None,
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
        ("events_pre", model.events_pre_source),
        ("events_post", model.events_post_source),
        ("stepper", model.stepper_source),
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
