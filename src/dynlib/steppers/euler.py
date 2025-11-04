# src/dynlib/steppers/euler.py
"""
Euler (explicit, fixed-step) stepper implementation.

Slice 4: first real stepper. Minimal StructSpec (mostly size-1 banks).
"""
from __future__ import annotations
import ast
from typing import TYPE_CHECKING

from .base import StepperMeta, StructSpec

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["EulerSpec"]


class EulerSpec:
    """
    Explicit Euler stepper: y_{n+1} = y_n + dt * f(t_n, y_n)

    Fixed-step, order 1, explicit scheme for ODEs.
    """
    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="euler",
                kind="ode",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="euler",
                order=1,
                embedded_order=None,
                dense_output=False,
                stiff_ok=False,
                aliases=("fwd_euler", "forward_euler"),
            )
        self.meta = meta

    def struct_spec(self) -> StructSpec:
        """
        Euler requires minimal workspace:
          - sw0: scratch for RHS evaluations (n_state floats)
          - All other banks are size 1 (unused but must exist per ABI)
        """
        # We'll allocate sw0 dynamically based on n_state at build time.
        # For the spec itself, we declare sizes. The builder will use n_state.
        # Since n_state is not known here, we use a placeholder. The actual allocation
        # happens in the wrapper with the concrete n_state from ModelSpec.
        # 
        # Per guardrails: StructSpec sizes are counts in elements.
        # For Euler, we'll say sw0_size is a marker (0 here), and the actual 
        # allocation will be max(sw0_size, n_state) by convention in the wrapper.
        # 
        # Actually, let's follow the design more carefully: the StructSpec should
        # declare the *additional* workspace beyond y_curr/y_prop. For Euler,
        # we need one n_state-sized scratch for dy. We'll use sw0 for that.
        # 
        # But n_state is model-specific. The solution: StructSpec sizes can be
        # expressed as multipliers of n_state or as fixed sizes. For simplicity,
        # we'll declare sw0_size=0 and document that the emitted stepper uses
        # sw0 as scratch (which wrapper already allocates with at least n_state).
        # 
        # Looking at test_numba_probe.py: sw0 = np.zeros(n_state, dtype=np.float64)
        # So the wrapper allocates sw0 with size >= n_state if needed.
        # 
        # Let's be explicit: Euler's emit() will use sw0[:n] as scratch. The
        # StructSpec declares minimum sizes. For Euler, we need sw0 >= n_state.
        # We'll use a convention: sw0_size=0 means "at least n_state".
        # Or better: we declare sw0_size=1 as a flag, and document that the
        # builder interprets this as "allocate n_state elements".
        # 
        # Actually, re-reading the design: StructSpec sizes are in elements.
        # The builder knows n_state. For Euler, we need sw0 of size n_state.
        # We'll declare sw0_size as a symbolic value. Let's use -1 to mean
        # "n_state" or just document it. Or simpler: declare sw0_size=0 and
        # let the builder/wrapper ensure sw0.size >= n_state (which it already does
        # in test_numba_probe.py).
        # 
        # For now, let's keep it simple per the design doc: mostly size-1 banks,
        # with the understanding that the wrapper allocates sw0 with size n_state
        # as needed (it's already doing that in the test).
        
        return StructSpec(
            sp_size=1,    # unused
            ss_size=1,    # unused
            sw0_size=0,   # actual size will be max(0, n_state) by wrapper convention
            sw1_size=1,   # unused
            sw2_size=1,   # unused
            sw3_size=1,   # unused
            iw0_size=1,   # unused
            bw0_size=1,   # unused
            use_history=False,
            use_f_history=False,
            dense_output=False,
            needs_jacobian=False,
            embedded_order=None,
            stiff_ok=False,
        )

    def emit(self, rhs_fn: Callable, struct: StructSpec) -> str:
        """
        Generate source code for the Euler stepper function.
        
        Signature (per ABI):
            status = stepper(
                t: float, dt: float,
                y_curr: float[:], rhs,
                params: float[:] | int[:],
                sp: float[:], ss: float[:],
                sw0: float[:], sw1: float[:], sw2: float[:], sw3: float[:],
                iw0: int32[:], bw0: uint8[:],
                y_prop: float[:], t_prop: float[:], dt_next: float[:], err_est: float[:]
            ) -> int32
        
        Returns:
            Python source code as a string that defines the stepper function.
        """
        # We'll generate an AST and convert it to source, or just return source directly.
        # For simplicity in Slice 4, let's return source as a string (less AST wrangling).
        # The compile layer will exec() it.
        
        source = """
def euler_stepper(
    t, dt,
    y_curr, rhs,
    params,
    sp, ss,
    sw0, sw1, sw2, sw3,
    iw0, bw0,
    y_prop, t_prop, dt_next, err_est
):
    # Euler: y_prop = y_curr + dt * f(t, y_curr)
    n = y_curr.size
    
    # Use sw0 as scratch for dy (RHS evaluation)
    # Wrapper ensures sw0.size >= n
    dy = sw0[:n]
    
    # Evaluate RHS
    rhs(t, y_curr, dy, params)
    
    # Propose next state
    for i in range(n):
        y_prop[i] = y_curr[i] + dt * dy[i]
    
    # Fixed step: dt_next = dt
    t_prop[0] = t + dt
    dt_next[0] = dt
    err_est[0] = 0.0
    
    # Return OK status (import at top of generated module)
    return OK
"""
        return source


# Auto-register on module import (optional; can also register explicitly in __init__.py)
def _auto_register():
    from .registry import register
    spec = EulerSpec()
    register(spec)

_auto_register()
