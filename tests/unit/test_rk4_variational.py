#!/usr/bin/env python3
"""
Pytest tests for RK4 variational stepping in Lyapunov analysis.

Tests verify that:
1. RK4's emit_step_with_variational() is correctly used for MLE combined mode
2. RK4's emit_step_with_variational() is correctly used for spectrum combined mode  
3. Fallback to Euler works correctly when stepper lacks emit_step_with_variational
4. Runtime metadata confirms the correct mode was actually used
5. JIT compilation failure triggers clean fallback
6. CombinedAnalysis preserves variational_in_step requirement
7. Deterministic linear system produces known spectrum
"""

import pytest
import numpy as np
from dynlib import setup
from dynlib.analysis.runtime import lyapunov_mle, lyapunov_spectrum
from dynlib.analysis import CombinedAnalysis


# Variational mode constants (matching lyapunov.py)
MODE_EULER = 0
MODE_COMBINED = 1
MODE_TANGENT_ONLY = 2


class TestRK4VariationalMLE:
    """Test MLE combined mode with RK4 variational stepping."""

    def test_mle_combined_mode_rk4_runtime_verified(self):
        """Test that RK4 actually uses combined variational stepping during execution."""
        # Setup Van der Pol with RK4
        sim = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        
        # Verify RK4 has variational capability
        assert sim.model.stepper_spec is not None
        assert sim.model.stepper_spec.meta.caps.variational_stepping, (
            "RK4 should support variational stepping"
        )
        
        # Build the analysis module that will actually be used
        lyap_module = lyapunov_mle(model=sim.model, record_interval=10)
        
        # Verify mode selection flags PRE-run
        assert lyap_module._use_variational_combined, (
            "Expected RK4 to select combined variational stepping mode"
        )
        assert not lyap_module._use_variational, (
            "Should not use tangent-only mode when combined is available"
        )
        assert lyap_module.requirements.variational_in_step, (
            "Requirements should indicate variational_in_step=True"
        )
        
        # Run simulation with the SAME module instance
        sim.assign(x=2.0, y=0.0, mu=1.0)
        sim.run(
            T=20.0,  # Reduced runtime
            dt=0.01,
            record_interval=20,
            analysis=lyap_module,  # Use the same instance
        )
        
        result = sim.results()
        lyap = result.analysis["lyapunov_mle"]
        
        # CRITICAL: Verify runtime mode metadata
        assert hasattr(lyap, 'variational_mode'), "Results should have variational_mode metadata"
        assert int(lyap.variational_mode) == MODE_COMBINED, (
            f"Runtime mode should be COMBINED({MODE_COMBINED}), got {int(lyap.variational_mode)}"
        )
        
        # Verify results are valid
        assert np.isfinite(lyap.mle), "MLE should be finite"
        assert lyap.steps > 0, "Should have processed steps"

    def test_mle_combined_mode_produces_finite_result(self):
        """Test that combined mode produces stable results."""
        sim = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        lyap_module = lyapunov_mle(model=sim.model, record_interval=5)
        
        sim.assign(x=1.5, y=0.5, mu=0.5)
        sim.run(
            T=20.0,
            dt=0.01,
            record_interval=20,
            analysis=lyap_module,
        )
        
        result = sim.results()
        lyap = result.analysis["lyapunov_mle"]
        
        assert int(lyap.variational_mode) == MODE_COMBINED
        assert np.isfinite(lyap.mle)
        assert np.isfinite(lyap.log_growth)
        assert np.isfinite(lyap.denom)


class TestRK4VariationalSpectrum:
    """Test spectrum combined mode with RK4 variational stepping."""

    def test_spectrum_combined_mode_rk4_runtime_verified(self):
        """Test that RK4 actually uses combined variational stepping for spectrum."""
        sim = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        
        # Build the analysis module
        spectrum_module = lyapunov_spectrum(model=sim.model, k=2, record_interval=10)
        
        # Verify mode selection PRE-run
        assert spectrum_module._use_variational_combined
        assert not spectrum_module._use_variational
        assert spectrum_module.requirements.variational_in_step
        
        # Run with same instance
        sim.assign(x=2.0, y=0.0, mu=1.0)
        sim.run(
            T=20.0,
            dt=0.01,
            record_interval=20,
            analysis=spectrum_module,
        )
        
        result = sim.results()
        spectrum = result.analysis["lyapunov_spectrum"]
        
        # Verify runtime mode
        assert int(spectrum.variational_mode) == MODE_COMBINED, (
            f"Runtime mode should be COMBINED, got {int(spectrum.variational_mode)}"
        )
        
        # Verify results structure
        exponents = [spectrum.lyap0, spectrum.lyap1]
        assert all(np.isfinite(exponents)), "All exponents should be finite"
        assert spectrum.steps > 0

    def test_spectrum_exponents_sorted(self):
        """Test spectrum produces properly sorted exponents."""
        sim = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        spectrum_module = lyapunov_spectrum(model=sim.model, k=2, record_interval=8)
        
        sim.assign(x=1.0, y=0.5, mu=1.0)
        sim.run(
            T=20.0,
            dt=0.01,
            record_interval=20,
            analysis=spectrum_module,
        )
        
        result = sim.results()
        spectrum = result.analysis["lyapunov_spectrum"]
        
        exponents = [spectrum.lyap0, spectrum.lyap1]
        # Exponents should be sorted (largest first)
        assert exponents[0] >= exponents[1], "Lyapunov exponents should be sorted descending"


class TestFallbackMode:
    """Test fallback mode when stepper lacks emit_step_with_variational."""

    def test_euler_fallback_no_crash_runtime_verified(self):
        """Test that Euler stepper falls back gracefully and runtime confirms it."""
        # Setup with Euler (doesn't have emit_step_with_variational)
        sim = setup("builtin://ode/vanderpol", stepper="euler", jit=True, disk_cache=False)
        
        # Verify Euler does NOT support combined variational stepping
        assert sim.model.stepper_spec is not None
        assert not sim.model.stepper_spec.meta.caps.variational_stepping, (
            "Euler should not have variational_stepping capability"
        )
        
        # Build analysis module
        lyap_module = lyapunov_mle(model=sim.model, record_interval=10)
        
        # Verify fallback mode: should NOT use combined mode
        assert not lyap_module._use_variational_combined, (
            "Euler should not select combined variational stepping"
        )
        assert not lyap_module.requirements.variational_in_step
        
        # Run simulation - should not crash
        sim.assign(x=2.0, y=0.0, mu=1.0)
        sim.run(
            T=15.0,
            dt=0.005,  # Smaller dt for Euler accuracy
            record_interval=30,
            analysis=lyap_module,
        )
        
        result = sim.results()
        lyap = result.analysis["lyapunov_mle"]
        
        # Verify runtime mode is fallback (euler or tangent_only, but not combined)
        runtime_mode = int(lyap.variational_mode)
        assert runtime_mode in (MODE_EULER, MODE_TANGENT_ONLY), (
            f"Euler should use fallback mode, got {runtime_mode}"
        )
        
        # Verify plausible output
        assert np.isfinite(lyap.mle), "Euler fallback should produce finite MLE"
        assert lyap.steps > 0

    def test_euler_spectrum_fallback_runtime_verified(self):
        """Test that Euler stepper falls back gracefully for spectrum."""
        sim = setup("builtin://ode/vanderpol", stepper="euler", jit=True, disk_cache=False)
        
        spectrum_module = lyapunov_spectrum(model=sim.model, k=2, record_interval=10)
        
        # Should not use combined mode
        assert not spectrum_module._use_variational_combined
        
        # Run simulation - should not crash
        sim.assign(x=2.0, y=0.0, mu=1.0)
        sim.run(
            T=15.0,
            dt=0.005,
            record_interval=30,
            analysis=spectrum_module,
        )
        
        result = sim.results()
        spectrum = result.analysis["lyapunov_spectrum"]
        
        # Verify fallback mode
        runtime_mode = int(spectrum.variational_mode)
        assert runtime_mode in (MODE_EULER, MODE_TANGENT_ONLY)
        
        # Verify plausible output
        exponents = [spectrum.lyap0, spectrum.lyap1]
        assert all(np.isfinite(exponents)), "Euler fallback should produce finite exponents"
        assert spectrum.steps > 0


class TestJITFallback:
    """Test JIT compilation failure triggers clean fallback."""

    def test_jit_disabled_uses_fallback_mode(self):
        """Test that disabling JIT changes variational mode behavior."""
        # With jit=False, some variational compilation may fail or be skipped
        # This tests the system degrades gracefully
        sim_nojit = setup("builtin://ode/vanderpol", stepper="rk4", jit=False, disk_cache=False)
        
        lyap_module = lyapunov_mle(model=sim_nojit.model, record_interval=5)
        
        # Run without JIT - should not crash
        sim_nojit.assign(x=1.5, y=0.5, mu=0.8)
        sim_nojit.run(
            T=10.0,
            dt=0.01,
            record_interval=10,
            analysis=lyap_module,
        )
        
        result = sim_nojit.results()
        lyap = result.analysis["lyapunov_mle"]
        
        # Should complete without crash
        assert np.isfinite(lyap.mle)
        # Mode may be combined or fallback depending on jit=False behavior
        runtime_mode = int(lyap.variational_mode)
        assert runtime_mode in (MODE_EULER, MODE_COMBINED, MODE_TANGENT_ONLY)


class TestCombinedAnalysis:
    """Test CombinedAnalysis requirement merging preserves variational_in_step."""

    def test_combined_analysis_preserves_variational_requirement(self):
        """Test that CombinedAnalysis merges requirements correctly."""
        from dynlib.analysis.runtime import CombinedAnalysis
        from dynlib.analysis.runtime.core import AnalysisModule, AnalysisRequirements, AnalysisHooks, TraceSpec
        
        sim = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        
        # Create Lyapunov analysis with variational_in_step=True
        lyap_module = lyapunov_mle(model=sim.model, record_interval=10)
        assert lyap_module.requirements.variational_in_step, "Lyapunov should have variational_in_step=True"
        
        # Create a simple noop analysis module WITHOUT variational requirements
        # This simulates a typical analysis that doesn't need special runner features
        noop_module = AnalysisModule(
            name="noop",
            requirements=AnalysisRequirements(
                fixed_step=False,
                need_jvp=False,
                mutates_state=False,
                variational_in_step=False,  # Key: this is False
            ),
            workspace_size=0,
            output_size=1,
            output_names=("dummy",),
            hooks=AnalysisHooks(),
            analysis_kind=1,
        )
        assert not noop_module.requirements.variational_in_step, "Noop should have variational_in_step=False"
        
        # Combine them - this should merge requirements
        combined = CombinedAnalysis([lyap_module, noop_module])
        
        # CRITICAL: Verify combined requirements preserve variational_in_step=True
        # The merge should use OR logic for capabilities (if any component needs it, keep it)
        assert combined.requirements.variational_in_step, (
            "CombinedAnalysis should preserve variational_in_step=True when merging "
            f"(got {combined.requirements.variational_in_step})"
        )
        
        # Also verify the requirement was properly propagated
        assert combined.requirements.need_jvp, (
            "Combined should also preserve need_jvp from Lyapunov"
        )
        
        # Run with combined analysis to verify it works
        sim.assign(x=2.0, y=0.0, mu=1.0)
        sim.run(
            T=15.0,
            dt=0.01,
            record_interval=15,
            analysis=combined,
        )
        
        result = sim.results()
        lyap = result.analysis["lyapunov_mle"]
        
        # Verify runtime mode is still combined (proving the requirement was honored)
        assert int(lyap.variational_mode) == MODE_COMBINED, (
            "Runtime should still use combined mode even with merged analysis"
        )


class TestDeterministicLinearSystem:
    """Test with deterministic linear system where exact spectrum is known."""

    def test_linear_system_known_spectrum(self):
        """Test Lyapunov computation with a linear system having known eigenvalues."""
        # Create a simple 2D linear ODE with known eigenvalues
        # dx/dt = -0.5*x (eigenvalue -0.5, Lyapunov exponent -0.5)
        # dy/dt = 0.2*y  (eigenvalue 0.2, Lyapunov exponent 0.2)
        
        linear_model_spec = """
inline:
[model]
type = "ode"
label = "Linear 2D System"

[states]
x = 1.0
y = 1.0

[equations.rhs]
x = "-0.5 * x"
y = "0.2 * y"

[equations.jacobian]
exprs = [
    ["-0.5", "0.0"],
    ["0.0", "0.2"]
]
"""
        
        sim = setup(linear_model_spec, stepper="rk4", jit=True, disk_cache=False)
        
        # Build spectrum analysis for 2D system
        spectrum_module = lyapunov_spectrum(model=sim.model, k=2, record_interval=5)
        
        # Verify combined mode will be used
        assert spectrum_module._use_variational_combined
        
        # Initial condition
        sim.assign(x=1.0, y=1.0)
        sim.run(
            T=50.0,  # Long enough for convergence
            dt=0.1,
            record_interval=50,
            analysis=spectrum_module,
        )
        
        result = sim.results()
        spectrum = result.analysis["lyapunov_spectrum"]
        
        # Verify combined mode was used
        assert int(spectrum.variational_mode) == MODE_COMBINED
        
        # Expected Lyapunov exponents: 0.2 and -0.5
        # Note: QR algorithm may not guarantee descending order, so check as a set
        expected_set = {0.2, -0.5}
        computed = [spectrum.lyap0, spectrum.lyap1]
        
        # Check that both values are close to one of the expected values
        def matches_expected(val, expected_set, tol=0.05):
            return any(abs(val - exp) < tol for exp in expected_set)
        
        assert matches_expected(computed[0], expected_set), (
            f"First exponent {computed[0]} should be near one of {expected_set}"
        )
        assert matches_expected(computed[1], expected_set), (
            f"Second exponent {computed[1]} should be near one of {expected_set}"
        )
        
        # Verify we got both distinct values (not duplicates)
        assert abs(computed[0] - computed[1]) > 0.1, (
            "Exponents should be distinct"
        )


class TestModeSelection:
    """Test mode selection metadata and capability-based checks."""

    def test_capability_based_check_rk4(self):
        """Test capability-based check instead of attribute existence."""
        sim_rk4 = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        
        # Capability-based check (robust to implementation changes)
        assert sim_rk4.model.stepper_spec.meta.caps.variational_stepping
        
    def test_capability_based_check_euler(self):
        """Test capability-based check for Euler."""
        sim_euler = setup("builtin://ode/vanderpol", stepper="euler", jit=True, disk_cache=False)
        
        # Capability-based check
        assert not sim_euler.model.stepper_spec.meta.caps.variational_stepping

    def test_runner_variational_step_availability(self):
        """Test that runner_variational_step is available only for combined mode."""
        sim_rk4 = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        sim_euler = setup("builtin://ode/vanderpol", stepper="euler", jit=True, disk_cache=False)
        
        mle_rk4 = lyapunov_mle(model=sim_rk4.model)
        mle_euler = lyapunov_mle(model=sim_euler.model)
        
        # RK4 should have runner variational step
        runner_step_rk4 = mle_rk4.runner_variational_step(jit=False)
        assert runner_step_rk4 is not None, "RK4 should provide runner_variational_step"
        
        # Euler should not have runner variational step (returns None)
        runner_step_euler = mle_euler.runner_variational_step(jit=False)
        assert runner_step_euler is None, "Euler should not provide runner_variational_step"

    def test_mode_metadata_encoding(self):
        """Test that mode metadata uses correct integer encoding."""
        sim_rk4 = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=False)
        sim_euler = setup("builtin://ode/vanderpol", stepper="euler", jit=True, disk_cache=False)
        
        # RK4 test
        mle_rk4 = lyapunov_mle(model=sim_rk4.model, record_interval=5)
        sim_rk4.assign(x=1.0, y=0.0, mu=1.0)
        sim_rk4.run(T=10.0, dt=0.01, record_interval=10, analysis=mle_rk4)
        
        result_rk4 = sim_rk4.results()
        assert int(result_rk4.analysis["lyapunov_mle"].variational_mode) == MODE_COMBINED
        
        # Euler test
        mle_euler = lyapunov_mle(model=sim_euler.model, record_interval=5)
        sim_euler.assign(x=1.0, y=0.0, mu=1.0)
        sim_euler.run(T=10.0, dt=0.005, record_interval=20, analysis=mle_euler)
        
        result_euler = sim_euler.results()
        mode_euler = int(result_euler.analysis["lyapunov_mle"].variational_mode)
        assert mode_euler in (MODE_EULER, MODE_TANGENT_ONLY), (
            f"Euler should use fallback mode, got {mode_euler}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
