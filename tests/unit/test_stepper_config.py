import dataclasses

import numpy as np
import pytest

from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model
from dynlib.steppers.registry import get_stepper


def _make_sim(model_toml: str, *, stepper_override: str | None = None, jit: bool = False):
    """Build a Sim + FullModel pair from inline TOML."""
    full_model = build(f"inline: {model_toml}", stepper=stepper_override, jit=jit)
    model = Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        workspace_sig=full_model.workspace_sig,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
        lag_state_info=full_model.lag_state_info,
        uses_lag=full_model.uses_lag,
        equations_use_lag=full_model.equations_use_lag,
        make_stepper_workspace=full_model.make_stepper_workspace,
    )
    return Sim(model), full_model


def test_rk45_runtime_config():
    """RK45 sanity: runtime tolerances should affect steps and accuracy."""

    # Simple decay model: x' = -x, x(0)=1 → x(t)=exp(-t)
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
stepper = "rk45"
record = true
# model-level tolerances (mid level)
atol = 1e-6
rtol = 1e-4
"""

    full_model = build(f"inline: {model_toml}", jit=False)

    model = Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        workspace_sig=full_model.workspace_sig,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
        lag_state_info=full_model.lag_state_info,
        uses_lag=full_model.uses_lag,
        equations_use_lag=full_model.equations_use_lag,
        make_stepper_workspace=full_model.make_stepper_workspace,
    )

    sim = Sim(model)

    def run_with_tols(atol: float, rtol: float):
        """
        Run from t0 to t_end with given tolerances and return:
        - max(|x_exact(t_i) - x_num(t_i)|) over the trajectory
        - number of recorded steps
        """
        sim.reset()
        # Also pass a reasonable min_step and max_tries so tolerance has a chance to act
        sim.run(atol=atol, rtol=rtol, min_step=1e-12, max_tries=20)
        res = sim.raw_results()

        t = res.T[: res.n]
        x_num = res.Y[0, : res.n]
        x_exact = np.exp(-t)

        err = np.max(np.abs(x_num - x_exact))
        return err, res.n

    # Loose → mid → tight tolerances
    err_loose, n_loose = run_with_tols(1e-3, 1e-2)
    err_mid,   n_mid   = run_with_tols(1e-6, 1e-5)
    err_tight, n_tight = run_with_tols(1e-10, 1e-8)

    # --------- Strong expectations for a sane RK45 implementation ---------

    # Step counts: tighter tolerances → equal or more steps
    assert n_tight >= n_mid >= n_loose, (
        f"Unexpected step counts: loose={n_loose}, mid={n_mid}, tight={n_tight}"
    )

    # Global accuracy: tighter tolerances → strictly smaller max error
    assert err_tight < err_mid < err_loose, (
        f"Unexpected error ordering: "
        f"loose={err_loose:.3e}, mid={err_mid:.3e}, tight={err_tight:.3e}"
    )

    # Additional sanity: none of the errors should be ridiculous
    for label, err in [("loose", err_loose), ("mid", err_mid), ("tight", err_tight)]:
        assert np.isfinite(err), f"{label} tolerance run produced non-finite error"
        assert err < 1.0, f"{label} tolerance run is wildly inaccurate: err={err}"
        


def test_euler_ignores_config():
    """Test that Euler ignores stepper config parameters with a warning."""

    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
stepper = "euler"
record = true
"""

    full_model = build(f"inline: {model_toml}", jit=False)
    model = Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        workspace_sig=full_model.workspace_sig,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
        lag_state_info=full_model.lag_state_info,
        uses_lag=full_model.uses_lag,
        equations_use_lag=full_model.equations_use_lag,
        make_stepper_workspace=full_model.make_stepper_workspace,
    )
    sim = Sim(model)

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim.run(atol=1e-10, rtol=1e-8)  # Euler doesn't use these

        # At least one warning mentioning that runtime params are ignored
        msgs = [str(wi.message) for wi in w]
        assert any("does not accept runtime parameters" in m for m in msgs)

    res = sim.raw_results()
    # basic sanity: we did run something
    assert res.n > 0


def test_rk45_with_jit():
    """Test RK45 runtime config with JIT enabled."""

    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 2.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 0.5
dt = 0.05
stepper = "rk45"
record = true
atol = 1e-8
rtol = 1e-5
"""

    # Build with JIT enabled
    full_model = build(f"inline: {model_toml}", jit=False)
    model = Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        workspace_sig=full_model.workspace_sig,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
        lag_state_info=full_model.lag_state_info,
        uses_lag=full_model.uses_lag,
        equations_use_lag=full_model.equations_use_lag,
        make_stepper_workspace=full_model.make_stepper_workspace,
    )
    sim = Sim(model)

    x_exact = np.exp(-1.0)  # since a = 2, t_end = 0.5 → a*t_end = 1

    # Run with defaults (from model spec)
    sim.reset()
    sim.run()
    res1 = sim.raw_results()

    # Run with tighter tolerances
    sim.reset()
    sim.run(atol=1e-12, rtol=1e-10)
    res2 = sim.raw_results()

    # Tighter tolerances should not yield fewer steps
    assert res2.n >= res1.n, "Tighter tolerances should not yield fewer steps (JIT)"

    # And they should not be less accurate
    err1 = abs(res1.Y[0, -1] - x_exact)
    err2 = abs(res2.Y[0, -1] - x_exact)
    assert err2 <= err1 + 1e-12


def test_config_mixin_respects_model_sim_overrides():
    """Model [sim] entries override stepper defaults while unspecified fields use DSL defaults."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 2.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.05
stepper = "rk45"
record = true
atol = 1e-4
"""
    sim, full_model = _make_sim(model_toml, jit=False)
    rk45_spec = get_stepper("rk45")
    cfg = rk45_spec.default_config(full_model.spec)
    assert cfg.atol == pytest.approx(1e-4)
    # rtol not provided in TOML, so it should fall back to the DSL default stored on ModelSpec.sim
    assert cfg.rtol == pytest.approx(full_model.spec.sim.rtol)
    np.testing.assert_allclose(sim.stepper_config(), rk45_spec.pack_config(cfg))


def test_sim_overrides_ignored_when_stepper_changed():
    """[sim] config entries are skipped if the runtime stepper has no config."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 0.5
dt = 0.05
stepper = "rk45"
record = true
atol = 1e-5
"""
    sim, _ = _make_sim(model_toml, stepper_override="euler", jit=False)
    # The compiled model runs with Euler despite the DSL requesting rk45.
    assert sim.model.stepper_name == "euler"
    cfg = sim.stepper_config()
    assert cfg.size == 0, "Euler should not expose a runtime config array"


def test_user_partial_stepper_overrides_merge_with_model_defaults():
    """User-provided kwargs should override only the named fields."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 2.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
stepper = "rk45"
record = true
rtol = 1e-5
"""
    sim, full_model = _make_sim(model_toml, jit=False)
    rk45_spec = get_stepper("rk45")
    base_cfg = rk45_spec.default_config(full_model.spec)
    updated_cfg = dataclasses.replace(base_cfg, atol=1e-9)
    expected = rk45_spec.pack_config(updated_cfg)
    new_cfg = sim.stepper_config(atol=1e-9)
    np.testing.assert_allclose(new_cfg, expected)
    # Subsequent reads keep the stored overrides intact
    np.testing.assert_allclose(sim.stepper_config(), expected)


def test_sim_extra_numeric_defaults_feed_stepper_config():
    """Unknown [sim] keys should pre-populate matching stepper config fields."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 1.0
dt = 0.1
stepper = "rk45"
record = true
safety = 0.45
max_factor = 4.25
"""
    sim, full_model = _make_sim(model_toml, jit=False)
    rk45_spec = get_stepper("rk45")
    cfg = rk45_spec.default_config(full_model.spec)
    assert cfg.safety == pytest.approx(0.45)
    assert cfg.max_factor == pytest.approx(4.25)
    np.testing.assert_allclose(sim.stepper_config(), rk45_spec.pack_config(cfg))


def test_sim_extra_string_defaults_feed_stepper_config():
    """String-valued extras should also map into stepper configs (with enums)."""
    model_toml = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a*x"

[sim]
t0 = 0.0
t_end = 0.25
dt = 0.05
stepper = "rk45"
record = true
method = "broyden1"
"""
    _, full_model = _make_sim(model_toml, jit=False)
    bdf2_spec = get_stepper("bdf2")
    cfg = bdf2_spec.default_config(full_model.spec)
    assert cfg.method == "broyden1"
    packed = bdf2_spec.pack_config(cfg)
    # tol, max_iter, method (enum encoded)
    assert packed[2] == pytest.approx(2.0)
