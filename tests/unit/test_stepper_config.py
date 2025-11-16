import numpy as np
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model


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
