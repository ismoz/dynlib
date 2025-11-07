# tests/unit/test_stepper_guardrails.py
from dynlib.compiler.codegen.validate import validate_stepper_function
from dynlib.steppers.base import StructSpec


def _noop_stepper(t, dt, y_curr, rhs, params,
                  sp, ss, sw0, sw1, sw2, sw3,
                  iw0, bw0, y_prop, t_prop, dt_next, err_est):
    """Stepper that touches nothing (used for struct-spec tests)."""
    return 0


def test_struct_spec_sizes_must_be_non_negative():
    struct = StructSpec(sp_size=-1)
    issues = validate_stepper_function(_noop_stepper, "noop", struct_spec=struct)
    assert any(
        iss.severity == "error" and "sp_size" in iss.message
        for iss in issues
    ), issues


def test_iw0_rejects_float_assignments():
    def stepper(t, dt, y_curr, rhs, params,
                sp, ss, sw0, sw1, sw2, sw3,
                iw0, bw0, y_prop, t_prop, dt_next, err_est):
        iw0[0] = t  # storing float timestamp into integer bank should fail
        return 0

    issues = validate_stepper_function(stepper, "float_iw0", struct_spec=StructSpec(iw0_size=1))
    assert any(
        iss.severity == "error" and "iw0" in iss.message
        for iss in issues
    ), issues


def test_persistence_warning_when_stage_bank_read_before_write():
    def stepper(t, dt, y_curr, rhs, params,
                sp, ss, sw0, sw1, sw2, sw3,
                iw0, bw0, y_prop, t_prop, dt_next, err_est):
        sw_snapshot = sw0[:]  # read before any writes → persistence warning
        sw0[:] = y_curr
        return sw_snapshot.shape[0]

    issues = validate_stepper_function(stepper, "persist_sw0", struct_spec=StructSpec(sw0_size=1))
    assert any(
        iss.severity == "warning" and "read before any writes" in iss.message
        for iss in issues
    ), issues
