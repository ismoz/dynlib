# tests/unit/test_stepper_guardrails.py
from dynlib.compiler.codegen.validate import validate_stepper_function


def _baseline_stepper(t, dt, y_curr, rhs, params, runtime_ws, ws, cfg, y_prop, t_prop, dt_next, err_est):
    y_prop[:] = y_curr
    t_prop[0] = t + dt
    dt_next[0] = dt
    err_est[0] = 0.0
    return 0


def test_forbidden_write_to_runner_state_detected():
    def bad_stepper(t, dt, y_curr, rhs, params, runtime_ws, ws, cfg, y_prop, t_prop, dt_next, err_est):
        y_curr[0] = 0.0  # runner-owned buffer must be read-only
        return 0

    issues = validate_stepper_function(bad_stepper, "bad")
    assert any("y_curr" in issue.message and issue.severity == "error" for issue in issues), issues


def test_workspace_rebinding_is_flagged():
    def bad_stepper(t, dt, y_curr, rhs, params, runtime_ws, ws, cfg, y_prop, t_prop, dt_next, err_est):
        ws = None  # workspace tuple must not be rebound
        return 0

    issues = validate_stepper_function(bad_stepper, "bad_ws")
    assert any("Workspace tuples are immutable" in issue.message for issue in issues), issues


def test_valid_stepper_has_no_issues():
    issues = validate_stepper_function(_baseline_stepper, "good")
    assert issues == []
