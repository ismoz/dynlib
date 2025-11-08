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
from dynlib.compiler.codegen.validate import validate_stepper_function, report_validation_issues
from dynlib.compiler.jit.compile import maybe_jit_triplet, jit_compile
from dynlib.compiler.jit.cache import JITCache, CacheKey
from dynlib.compiler.paths import resolve_uri, load_config, PathConfig, resolve_cache_root
from dynlib.compiler.mods import apply_mods_v2, ModSpec
from dynlib.errors import ModelLoadError, StepperKindMismatchError

__all__ = ["CompiledPieces", "build_callables", "FullModel", "build", "load_model_from_uri"]

@dataclass(frozen=True)
class CompiledPieces:
    spec: ModelSpec
    stepper_name: str
    rhs: callable
    events_pre: callable
    events_post: callable
    spec_hash: str

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
    model_dtype: np.dtype

_cache = JITCache()
_stepper_cache: Dict[str, Callable] = {}  # Cache compiled steppers by cache key hash


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

def build_callables(spec: ModelSpec, *, stepper_name: str, jit: bool, model_dtype: str = "float64") -> CompiledPieces:
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
        model_dtype=model_dtype,
        version_pins=("dynlib=2",),
    )

    cached = _cache.get(key)
    if cached is not None and cached.get("jit") == bool(jit):
        tri = cached["triplet"]
        # Also return cached stepper if available
        stepper_cached = cached.get("stepper")
        return CompiledPieces(spec, stepper_name, tri[0], tri[1], tri[2], s_hash)

    cc: CompiledCallables = emit_rhs_and_events(spec)
    rhs_j, pre_j, post_j = maybe_jit_triplet(cc.rhs, cc.events_pre, cc.events_post, jit=jit)

    _cache.put(key, {"triplet": (rhs_j, pre_j, post_j), "jit": bool(jit)})

    return CompiledPieces(spec, stepper_name, rhs_j, pre_j, post_j, s_hash)


def _warmup_jit_runner(
    runner: Callable,
    stepper: Callable,
    rhs: Callable,
    events_pre: Callable,
    events_post: Callable,
    struct: StructSpec,
    spec: ModelSpec,
    model_dtype: str,
) -> None:
    """
    Warm up JIT-compiled runner by calling it once with minimal inputs.
    This triggers Numba compilation at build time instead of at first runtime call.
    
    This warmup ALSO compiles the stepper, rhs, and events functions when they
    are called by the runner, ensuring everything is warmed up.
    """
    from dynlib.runtime.buffers import allocate_pools
    
    dtype_np = np.dtype(model_dtype)
    n_state = len(spec.states)
    
    # Allocate minimal banks and buffers for warmup
    banks, rec, ev = allocate_pools(
        n_state=n_state,
        struct=struct,
        model_dtype=dtype_np,
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
    
    # Warmup call: run for a single tiny step
    try:
        runner(
            0.0, 0.1, 0.1,  # t0, t_end, dt_init
            100, n_state, 0,  # max_steps, n_state, record_every_step (0 = no recording)
            y_curr, y_prev, params,
            banks.sp, banks.ss, banks.sw0, banks.sw1, banks.sw2, banks.sw3,
            banks.iw0, banks.bw0,
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
    stepper_name: Optional[str] = None,
    mods: Optional[List[str]] = None,
    jit: bool = True,
    model_dtype: str = "float64",
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
        model_dtype: Model dtype string (default "float64")
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
    
    # Build RHS and events
    pieces = build_callables(spec, stepper_name=stepper_name, jit=jit, model_dtype=model_dtype)
    
    # Check if stepper is cached
    stepper_cache_key = f"{pieces.spec_hash}:{stepper_name}:{jit}"
    stepper_fn = _stepper_cache.get(stepper_cache_key)
    needs_warmup = stepper_fn is None  # Track if we need to warm up
    
    if stepper_fn is None:
        # Generate stepper function (returns a callable)
        stepper_fn = stepper_spec.emit(pieces.rhs, struct, model_spec=spec)
        
        # Validate stepper if requested (build-time guardrails check)
        if validate_stepper:
            issues = validate_stepper_function(stepper_fn, stepper_name, struct_spec=struct)
            report_validation_issues(issues, stepper_name, strict=False)
        
        # Apply JIT to stepper if enabled (using centralized helper)
        stepper_fn = jit_compile(stepper_fn, jit=jit)
        
        # Cache the compiled stepper
        _stepper_cache[stepper_cache_key] = stepper_fn
    
    # Get runner function (optionally JIT-compiled)
    if jit and disk_cache:
        cache_root = resolve_cache_root(config_in_use)
        runner_codegen.configure_runner_disk_cache(
            spec_hash=pieces.spec_hash,
            stepper_name=stepper_name,
            structsig=structsig,
            model_dtype=model_dtype,
            cache_root=cache_root,
        )
    runner_fn = runner_codegen.get_runner(jit=jit, disk_cache=disk_cache)
    
    # Warm up JIT-compiled functions to trigger compilation at build time
    # Always warmup when jit=True to ensure stepper/rhs/events are compiled
    if jit:
        _warmup_jit_runner(
            runner_fn, stepper_fn, pieces.rhs, pieces.events_pre, pieces.events_post,
            struct, spec, model_dtype
        )
    
    dtype_np = np.dtype(model_dtype)
    
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
        model_dtype=dtype_np,
    )
