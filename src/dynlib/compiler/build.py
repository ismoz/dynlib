# src/dynlib/compiler/build.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from dynlib.dsl.spec import ModelSpec, compute_spec_hash
from dynlib.steppers.registry import get_stepper
from dynlib.compiler.codegen.emitter import emit_rhs_and_events, CompiledCallables
from dynlib.compiler.jit.compile import maybe_jit_triplet
from dynlib.compiler.jit.cache import JITCache, CacheKey

__all__ = ["CompiledPieces", "build_callables"]

@dataclass(frozen=True)
class CompiledPieces:
    spec: ModelSpec
    stepper_name: str
    rhs: callable
    events_pre: callable
    events_post: callable
    spec_hash: str

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
