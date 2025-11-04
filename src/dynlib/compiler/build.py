# src/dynlib/compiler/build.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any
import numpy as np

from dynlib.dsl.spec import ModelSpec, compute_spec_hash
from dynlib.steppers.registry import get_stepper
from dynlib.steppers.base import StructSpec
from dynlib.compiler.codegen.emitter import emit_rhs_and_events, CompiledCallables
from dynlib.compiler.codegen.runner import get_runner
from dynlib.compiler.jit.compile import maybe_jit_triplet
from dynlib.compiler.jit.cache import JITCache, CacheKey

__all__ = ["CompiledPieces", "build_callables", "FullModel", "build"]

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
    """Complete compiled model (Slice 4): includes runner + stepper + callables."""
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

def _structsig_from_stepper(stepper_name: str) -> Tuple[int, ...]:
    # StructSpec signature is a tuple of sizes; we only need it for the cache key.
    spec = get_stepper(stepper_name).struct_spec()
    return (
        spec.sp_size, spec.ss_size,
        spec.sw0_size, spec.sw1_size, spec.sw2_size, spec.sw3_size,
        spec.iw0_size, spec.bw0_size,
        int(bool(spec.use_history)), int(bool(spec.use_f_history)),
        int(bool(spec.dense_output)), int(bool(spec.needs_jacobian)),
        -1 if spec.embedded_order is None else int(spec.embedded_order),
        int(bool(spec.stiff_ok)),
    )

def build_callables(spec: ModelSpec, *, stepper_name: str, jit: bool, model_dtype: str = "float64") -> CompiledPieces:
    """
    Slice-3 target: produce (rhs, events_pre, events_post) with optional JIT.
    No runner/stepper glue here yet.
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
        return CompiledPieces(spec, stepper_name, tri[0], tri[1], tri[2], s_hash)

    cc: CompiledCallables = emit_rhs_and_events(spec)
    rhs_j, pre_j, post_j = maybe_jit_triplet(cc.rhs, cc.events_pre, cc.events_post, jit=jit)

    _cache.put(key, {"triplet": (rhs_j, pre_j, post_j), "jit": bool(jit)})

    return CompiledPieces(spec, stepper_name, rhs_j, pre_j, post_j, s_hash)


def build(spec: ModelSpec, *, stepper_name: str, jit: bool = True, model_dtype: str = "float64") -> FullModel:
    """
    Slice 4: Build a complete compiled model with runner + stepper.
    
    Args:
        spec: Validated ModelSpec
        stepper_name: Name of the registered stepper (e.g., "euler")
        jit: Enable JIT compilation (default True)
        model_dtype: Model dtype string (default "float64")
    
    Returns:
        FullModel with all compiled components
    """
    # Get stepper spec
    stepper_spec = get_stepper(stepper_name)
    struct = stepper_spec.struct_spec()
    
    # Build RHS and events
    pieces = build_callables(spec, stepper_name=stepper_name, jit=jit, model_dtype=model_dtype)
    
    # Generate stepper function (returns a callable)
    stepper_fn = stepper_spec.emit(pieces.rhs, struct)
    
    # Get runner function (optionally JIT-compiled)
    runner_fn = get_runner(jit=jit)
    
    # Apply JIT to stepper if enabled
    if jit:
        try:
            from numba import njit
            stepper_fn = njit(cache=False)(stepper_fn)
        except ImportError:
            pass  # Fall back to non-JIT
    
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
