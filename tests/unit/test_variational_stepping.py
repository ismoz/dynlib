#!/usr/bin/env python3
"""
Variational stepping tests for Lyapunov analyses.

Coverage:
- All steppers advertising variational_stepping run Lyapunov MLE and spectrum with consistent runtime metadata.
- Steppers without variational_stepping raise with a clear, supported-stepper list (no silent fallback).
- Runner helpers and CombinedObserver preserve variational_in_step requirements.
- Deterministic linear system stays accurate under a variational-capable stepper.
"""

import pytest
import numpy as np

from dynlib import setup
from dynlib.runtime.observers import (
    CombinedObserver,
    lyapunov_mle_observer,
    lyapunov_spectrum_observer,
)
from dynlib.steppers.registry import list_steppers, get_stepper

# Variational mode constants (matching lyapunov.py)
MODE_COMBINED = 1
MODE_TANGENT_ONLY = 2


def _partition_steppers_by_variational_capability():
    """Return (supported, unsupported) stepper name lists."""
    supported: list[str] = []
    unsupported: list[str] = []
    for name in list_steppers(kind="ode"):
        caps = get_stepper(name).meta.caps
        (supported if caps.variational_stepping else unsupported).append(name)
    return supported, unsupported


def test_unsupported_stepper_raises_and_lists_supported():
    supported, unsupported = _partition_steppers_by_variational_capability()
    if not unsupported:
        pytest.skip("All steppers advertise variational_stepping; nothing unsupported to test.")

    bad = unsupported[0]
    sim = setup("builtin://ode/vanderpol", stepper=bad, jit=True, disk_cache=False)

    with pytest.raises(ValueError) as excinfo:
        lyapunov_mle_observer(model=sim.model, record_interval=10)

    msg = str(excinfo.value).lower()
    assert "selected stepper does not support variational stepping" in msg
    for s in supported:
        assert s.lower() in msg


@pytest.mark.parametrize("stepper", _partition_steppers_by_variational_capability()[0])
def test_mle_variational_supported_stepper_runtime_verified(stepper):
    sim = setup("builtin://ode/vanderpol", stepper=stepper, jit=True, disk_cache=False)
    mod = lyapunov_mle_observer(model=sim.model, record_interval=10)

    sim.assign(x=2.0, y=0.0, mu=1.0)
    sim.run(T=10.0, dt=0.01, record_interval=20, observers=mod)

    lyap = sim.results().observers["lyapunov_mle"]
    assert lyap.steps > 0
    assert np.isfinite(lyap.mle)

    if mod.requirements.variational_in_step:
        assert int(lyap.variational_mode) == MODE_COMBINED
    else:
        assert int(lyap.variational_mode) == MODE_TANGENT_ONLY


@pytest.mark.parametrize("stepper", _partition_steppers_by_variational_capability()[0])
def test_spectrum_variational_supported_stepper_runtime_verified(stepper):
    sim = setup("builtin://ode/vanderpol", stepper=stepper, jit=True, disk_cache=False)
    mod = lyapunov_spectrum_observer(model=sim.model, k=2, record_interval=10)

    sim.assign(x=2.0, y=0.0, mu=1.0)
    sim.run(T=10.0, dt=0.01, record_interval=20, observers=mod)

    spectrum = sim.results().observers["lyapunov_spectrum"]
    assert spectrum.steps > 0
    exponents = [spectrum.lyap0, spectrum.lyap1]
    assert all(np.isfinite(exponents))

    if mod.requirements.variational_in_step:
        assert int(spectrum.variational_mode) == MODE_COMBINED
    else:
        assert int(spectrum.variational_mode) == MODE_TANGENT_ONLY


def test_runner_variational_step_matches_requirement():
    supported, _ = _partition_steppers_by_variational_capability()
    if not supported:
        pytest.skip("No variational steppers available.")
    stepper = supported[0]

    sim = setup("builtin://ode/vanderpol", stepper=stepper, jit=True, disk_cache=False)
    mod = lyapunov_mle_observer(model=sim.model)

    runner_step = mod.runner_variational_step(jit=False)
    if mod.requirements.variational_in_step:
        assert runner_step is not None
    else:
        assert runner_step is None


def test_jit_disabled_respects_strategy():
    supported, _ = _partition_steppers_by_variational_capability()
    if not supported:
        pytest.skip("No variational steppers available.")
    stepper = supported[0]

    sim = setup("builtin://ode/vanderpol", stepper=stepper, jit=False, disk_cache=False)
    mod = lyapunov_mle_observer(model=sim.model, record_interval=5)

    sim.assign(x=1.5, y=0.5, mu=0.8)
    sim.run(T=8.0, dt=0.01, record_interval=10, observers=mod)

    lyap = sim.results().observers["lyapunov_mle"]
    assert np.isfinite(lyap.mle)
    if mod.requirements.variational_in_step:
        assert int(lyap.variational_mode) == MODE_COMBINED
    else:
        assert int(lyap.variational_mode) == MODE_TANGENT_ONLY


def test_combined_analysis_preserves_variational_requirement():
    from dynlib.runtime.observers import ObserverModule, ObserverRequirements, ObserverHooks

    supported, _ = _partition_steppers_by_variational_capability()
    if not supported:
        pytest.skip("No variational steppers available.")
    stepper = supported[0]

    sim = setup("builtin://ode/vanderpol", stepper=stepper, jit=True, disk_cache=False)
    lyap_module = lyapunov_mle_observer(model=sim.model, record_interval=10)

    noop_module = ObserverModule(
        key="noop",
        name="noop",
        requirements=ObserverRequirements(
            fixed_step=False,
            need_jvp=False,
            mutates_state=False,
            variational_in_step=False,
        ),
        workspace_size=0,
        output_size=1,
        output_names=("dummy",),
        hooks=ObserverHooks(),
        analysis_kind=1,
    )

    combined = CombinedObserver([lyap_module, noop_module])
    assert combined.requirements.variational_in_step == lyap_module.requirements.variational_in_step
    assert combined.requirements.need_jvp

    sim.assign(x=2.0, y=0.0, mu=1.0)
    sim.run(T=12.0, dt=0.01, record_interval=15, observers=combined)

    lyap = sim.results().observers["lyapunov_mle"]
    if lyap_module.requirements.variational_in_step:
        assert int(lyap.variational_mode) == MODE_COMBINED
    else:
        assert int(lyap.variational_mode) == MODE_TANGENT_ONLY


def test_linear_system_known_spectrum():
    supported, _ = _partition_steppers_by_variational_capability()
    if not supported:
        pytest.skip("No variational steppers available.")
    stepper = supported[0]

    linear_model_spec = """
inline:
[model]
type = "ode"
name = "Linear 2D System"

[states]
x = 1.0
y = 1.0

[equations.rhs]
x = "-0.5 * x"
y = "0.2 * y"

[equations.jacobian]
expr = [
    ["-0.5", "0.0"],
    ["0.0", "0.2"]
]
"""

    sim = setup(linear_model_spec, stepper=stepper, jit=True, disk_cache=False)
    spectrum_module = lyapunov_spectrum_observer(model=sim.model, k=2, record_interval=5)

    sim.assign(x=1.0, y=1.0)
    sim.run(T=40.0, dt=0.1, record_interval=50, observers=spectrum_module)

    spectrum = sim.results().observers["lyapunov_spectrum"]
    assert int(spectrum.variational_mode) in (MODE_COMBINED, MODE_TANGENT_ONLY)

    expected_set = {0.2, -0.5}
    computed = [spectrum.lyap0, spectrum.lyap1]

    def matches_expected(val, expected, tol=0.05):
        return any(abs(val - exp) < tol for exp in expected)

    assert matches_expected(computed[0], expected_set)
    assert matches_expected(computed[1], expected_set)
    assert abs(computed[0] - computed[1]) > 0.1


def test_combined_analysis_rejects_mutating_child():
    from dynlib.runtime.observers import ObserverModule, ObserverRequirements, ObserverHooks

    mutating = ObserverModule(
        key="mutator",
        name="mutator",
        requirements=ObserverRequirements(mutates_state=True),
        workspace_size=0,
        output_size=0,
        hooks=ObserverHooks(),
        analysis_kind=1,
    )
    with pytest.raises(ValueError) as excinfo:
        CombinedObserver([mutating])
    assert "mutate state" in str(excinfo.value).lower()


def test_combined_analysis_rejects_multiple_variational_runner_children():
    from dynlib.runtime.observers import ObserverModule, ObserverRequirements, ObserverHooks

    class VarAnalysis(ObserverModule):
        def __init__(self, name: str):
            super().__init__(
                key=name,
                name=name,
                requirements=ObserverRequirements(variational_in_step=True),
                workspace_size=0,
                output_size=0,
                hooks=ObserverHooks(),
                analysis_kind=1,
            )

        def runner_variational_step(self, *, jit: bool):
            return lambda *args, **kwargs: None

    a1 = VarAnalysis("var1")
    a2 = VarAnalysis("var2")

    with pytest.raises(ValueError) as excinfo:
        CombinedObserver([a1, a2])

    msg = str(excinfo.value)
    assert "runner-level variational stepping" in msg
    assert "only one variational integrator" in msg


def test_capability_partitioning_matches_caps():
    supported, unsupported = _partition_steppers_by_variational_capability()
    for name in supported:
        assert get_stepper(name).meta.caps.variational_stepping
    for name in unsupported:
        assert not get_stepper(name).meta.caps.variational_stepping


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
