# src/dynlib/steppers/ode/rk4.py
"""
RK4 (Runge-Kutta 4th order, explicit, fixed-step) stepper implementation.

Classic fixed-step RK4 without touching wrapper/results/ABI.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import numpy as np

from ..base import StepperMeta, StepperCaps
from ..config_base import ConfigMixin
from dynlib.runtime.runner_api import OK

if TYPE_CHECKING:
    from typing import Callable

__all__ = ["RK4Spec"]


class RK4Spec(ConfigMixin):
    """
    Classic 4th-order Runge-Kutta stepper (explicit, fixed-step).
    
    Formula:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    Fixed-step, order 4, explicit scheme for ODEs.
    """
    Config = None  # No runtime configuration
    
    def __init__(self, meta: StepperMeta | None = None):
        if meta is None:
            meta = StepperMeta(
                name="rk4",
                kind="ode",
                time_control="fixed",
                scheme="explicit",
                geometry=frozenset(),
                family="runge-kutta",
                order=4,
                embedded_order=None,
                stiff=False,
                aliases=("rk4_classic", "classical_rk4"),
                caps=StepperCaps(variational_stepping=True),
            )
        self.meta = meta

    class Workspace(NamedTuple):
        y_stage: np.ndarray
        k1: np.ndarray
        k2: np.ndarray
        k3: np.ndarray
        k4: np.ndarray
        # Variational stepping buffers (for Lyapunov analysis)
        v_stage: np.ndarray
        kv1: np.ndarray
        kv2: np.ndarray
        kv3: np.ndarray
        kv4: np.ndarray

    class VariationalWorkspace(NamedTuple):
        v_stage: np.ndarray
        kv1: np.ndarray
        kv2: np.ndarray
        kv3: np.ndarray
        kv4: np.ndarray

    def workspace_type(self) -> type | None:
        return RK4Spec.Workspace

    def variational_workspace(self, n_state: int, model_spec=None):
        """
        Return the size and factory for the variational workspace.
        
        Returns:
            (size, factory_fn)
            factory_fn(analysis_ws, start, n_state) -> VariationalWorkspace
        """
        size = 5 * n_state

        def factory(analysis_ws, start, n_state):
            v_stage = analysis_ws[start : start + n_state]
            kv1 = analysis_ws[start + n_state : start + 2 * n_state]
            kv2 = analysis_ws[start + 2 * n_state : start + 3 * n_state]
            kv3 = analysis_ws[start + 3 * n_state : start + 4 * n_state]
            kv4 = analysis_ws[start + 4 * n_state : start + 5 * n_state]
            return RK4Spec.VariationalWorkspace(v_stage, kv1, kv2, kv3, kv4)

        return size, factory

    def make_workspace(
        self,
        n_state: int,
        dtype: np.dtype,
        model_spec=None,
    ) -> Workspace:
        zeros = lambda: np.zeros((n_state,), dtype=dtype)
        return RK4Spec.Workspace(
            y_stage=zeros(),
            k1=zeros(),
            k2=zeros(),
            k3=zeros(),
            k4=zeros(),
            v_stage=zeros(),
            kv1=zeros(),
            kv2=zeros(),
            kv3=zeros(),
            kv4=zeros(),
        )

    def emit(self, rhs_fn: Callable, model_spec=None) -> Callable:
        """
        Generate a jittable RK4 stepper function.
        
        Returns:
            A callable Python function implementing the RK4 stepper.
        """
        def rk4_stepper(
            t, dt,
            y_curr, rhs,
            params,
            runtime_ws,
            ws,
            stepper_config,
            y_prop, t_prop, dt_next, err_est
        ):
            # RK4: classic 4-stage explicit method
            # stepper_config is ignored (RK4 has no runtime config)
            n = y_curr.size
            
            k1 = ws.k1
            k2 = ws.k2
            k3 = ws.k3
            k4 = ws.k4
            y_stage = ws.y_stage
            
            # Stage 1: k1 = f(t, y)
            rhs(t, y_curr, k1, params, runtime_ws)
            
            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k1[i]
            rhs(t + 0.5 * dt, y_stage, k2, params, runtime_ws)
            
            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k2[i]
            rhs(t + 0.5 * dt, y_stage, k3, params, runtime_ws)
            
            # Stage 4: k4 = f(t + dt, y + dt * k3)
            for i in range(n):
                y_stage[i] = y_curr[i] + dt * k3[i]
            rhs(t + dt, y_stage, k4, params, runtime_ws)
            
            # Combine: y_prop = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            for i in range(n):
                y_prop[i] = y_curr[i] + (dt / 6.0) * (
                    k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
                )
            
            # Fixed step: dt_next = dt
            t_prop[0] = t + dt
            dt_next[0] = dt
            err_est[0] = 0.0
            
            return OK
        
        return rk4_stepper

    def emit_tangent_step(
        self,
        jvp_fn: Callable,
        model_spec=None
    ) -> Callable:
        """
        Generate a tangent-only RK4-style stepping function for Lyapunov analysis.
        
        LIMITATION: This evaluates the Jacobian at y_curr for all stages (not at
        intermediate RK stage states), so it's not fully consistent with RK4 state
        integration. It uses RK4 weights which is better than Euler, but for full
        consistency, state and tangent must be integrated together.
        
        Args:
            jvp_fn: Compiled JVP function jvp(t, y, params, v_in, v_out, runtime_ws)
            model_spec: Optional model specification
            
        Returns:
            Callable with signature:
                rk4_tangent_step(
                    t, dt,
                    y_curr,      # Current state (for Jacobian evaluation)
                    v_curr,      # Current tangent vector (input)
                    v_prop,      # Propagated tangent vector (output)
                    params, runtime_ws, ws
                ) -> None
                
        Note:
            Requires workspace with: v_stage, kv1, kv2, kv3, kv4
        """
        def rk4_tangent_step(
            t, dt,
            y_curr,      # State for Jacobian evaluation
            v_curr,      # Input tangent
            v_prop,      # Output tangent
            params, runtime_ws, ws
        ):
            n = y_curr.size
            
            # Get tangent buffers
            if hasattr(ws, "v_stage"):
                v_stage = ws.v_stage
                kv1 = ws.kv1
                kv2 = ws.kv2
                kv3 = ws.kv3
                kv4 = ws.kv4
            else:
                v_stage = ws[0]
                kv1 = ws[1]
                kv2 = ws[2]
                kv3 = ws[3]
                kv4 = ws[4]
            
            # Approximate RK4 on variational equation
            # NOTE: This is not fully consistent - we evaluate J at y_curr for all stages
            # rather than at the correct RK intermediate states
            
            # Stage 1: kv1 = J(y) * v
            jvp_fn(t, y_curr, params, v_curr, kv1, runtime_ws)
            
            # Stage 2: kv2 = J(y) * (v + dt/2*kv1)
            for i in range(n):
                v_stage[i] = v_curr[i] + 0.5 * dt * kv1[i]
            jvp_fn(t, y_curr, params, v_stage, kv2, runtime_ws)
            
            # Stage 3: kv3 = J(y) * (v + dt/2*kv2)
            for i in range(n):
                v_stage[i] = v_curr[i] + 0.5 * dt * kv2[i]
            jvp_fn(t, y_curr, params, v_stage, kv3, runtime_ws)
            
            # Stage 4: kv4 = J(y) * (v + dt*kv3)
            for i in range(n):
                v_stage[i] = v_curr[i] + dt * kv3[i]
            jvp_fn(t, y_curr, params, v_stage, kv4, runtime_ws)
            
            # RK4 combination for tangent
            for i in range(n):
                v_prop[i] = v_curr[i] + (dt/6.0) * (kv1[i] + 2.0*kv2[i] + 2.0*kv3[i] + kv4[i])
        
        return rk4_tangent_step

    def emit_step_with_variational(
        self, 
        rhs_fn: Callable,
        jvp_fn: Callable,
        model_spec=None
    ) -> Callable:
        """
        Generate a combined state + variational stepping function for Lyapunov analysis.
        
        This evaluates the Jacobian at the correct RK4 stage states, ensuring
        mathematical consistency between state and tangent space integration.
        
        Args:
            rhs_fn: Compiled RHS function f(t, y, dy_out, params, runtime_ws)
            jvp_fn: Compiled JVP function jvp(t, y, params, v_in, v_out, runtime_ws)
            model_spec: Optional model specification
            
        Returns:
            Callable with signature:
                rk4_step_with_variational(
                    t, dt,
                    y_curr, v_curr,
                    y_prop, v_prop,
                    params, runtime_ws, ws
                ) -> None
                
        Note:
            The Jacobian J(y) is evaluated at the same intermediate stage states
            used for state integration:
                Stage 1: J(y)
                Stage 2: J(y + dt/2·k1)
                Stage 3: J(y + dt/2·k2)
                Stage 4: J(y + dt·k3)
            This ensures the variational equation is integrated with the same
            accuracy as the state equation.
        """
        def rk4_step_with_variational(
            t, dt,
            y_curr, v_curr,
            y_prop, v_prop,
            params, runtime_ws, ws
        ):
            n = y_curr.size
            
            # Unpack workspace buffers
            # ws can be either RK4Spec.Workspace or a simple object with the required fields
            k1 = getattr(ws, 'k1', None)
            k2 = getattr(ws, 'k2', None)
            k3 = getattr(ws, 'k3', None)
            k4 = getattr(ws, 'k4', None)
            kv1 = ws.kv1
            kv2 = ws.kv2
            kv3 = ws.kv3
            kv4 = ws.kv4
            y_stage = getattr(ws, 'y_stage', None)
            v_stage = ws.v_stage
            
            # Allocate temporary buffers if not provided (for standalone variational stepping)
            if k1 is None:
                k1 = np.zeros((n,), dtype=y_curr.dtype)
            if k2 is None:
                k2 = np.zeros((n,), dtype=y_curr.dtype)
            if k3 is None:
                k3 = np.zeros((n,), dtype=y_curr.dtype)
            if k4 is None:
                k4 = np.zeros((n,), dtype=y_curr.dtype)
            if y_stage is None:
                y_stage = np.zeros((n,), dtype=y_curr.dtype)
            
            # === Stage 1: at (t, y) ===
            rhs_fn(t, y_curr, k1, params, runtime_ws)
            jvp_fn(t, y_curr, params, v_curr, kv1, runtime_ws)  # J(y)·v
            
            # === Stage 2: at (t + dt/2, y + dt/2·k1) ===
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k1[i]
                v_stage[i] = v_curr[i] + 0.5 * dt * kv1[i]
            rhs_fn(t + 0.5*dt, y_stage, k2, params, runtime_ws)
            jvp_fn(t + 0.5*dt, y_stage, params, v_stage, kv2, runtime_ws)  # J(y₂)·v₂
            
            # === Stage 3: at (t + dt/2, y + dt/2·k2) ===
            for i in range(n):
                y_stage[i] = y_curr[i] + 0.5 * dt * k2[i]
                v_stage[i] = v_curr[i] + 0.5 * dt * kv2[i]
            rhs_fn(t + 0.5*dt, y_stage, k3, params, runtime_ws)
            jvp_fn(t + 0.5*dt, y_stage, params, v_stage, kv3, runtime_ws)  # J(y₃)·v₃
            
            # === Stage 4: at (t + dt, y + dt·k3) ===
            for i in range(n):
                y_stage[i] = y_curr[i] + dt * k3[i]
                v_stage[i] = v_curr[i] + dt * kv3[i]
            rhs_fn(t + dt, y_stage, k4, params, runtime_ws)
            jvp_fn(t + dt, y_stage, params, v_stage, kv4, runtime_ws)  # J(y₄)·v₄
            
            # === Final combination (RK4 formula) ===
            for i in range(n):
                y_prop[i] = y_curr[i] + (dt/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
                v_prop[i] = v_curr[i] + (dt/6.0) * (kv1[i] + 2.0*kv2[i] + 2.0*kv3[i] + kv4[i])
        
        return rk4_step_with_variational



# Auto-register on module import
def _auto_register():
    from ..registry import register
    spec = RK4Spec()
    register(spec)

_auto_register()
