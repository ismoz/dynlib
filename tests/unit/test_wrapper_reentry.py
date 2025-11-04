# tests/unit/test_wrapper_reentry.py
import numpy as np
from dynlib.runtime.wrapper import run_with_wrapper
from dynlib.runtime.runner_api import GROW_REC, GROW_EVT, DONE
from dynlib.steppers.base import StepperMeta, StepperSpec, StructSpec

# ---- Fake stepper/spec (no real stepping) ------------------------------------
class _DummyStepper(StepperSpec):
    def __init__(self):
        super().__init__(StepperMeta(name="dummy", kind="ode"))
    def struct_spec(self) -> StructSpec:
        # No work memory needed for the fake runner
        return StructSpec(0,0,0,0,0,0, 0,0)
    def emit(self, *args, **kwargs):
        raise NotImplementedError

# ---- Fake callables -----------------------------------------------------------
def _rhs(t, y_vec, dy_out, params):  # pragma: no cover
    pass
def _events_pre(t, y, params):  # pragma: no cover
    pass
def _events_post(t, y, params):  # pragma: no cover
    pass
def _stepper(*args, **kwargs):  # pragma: no cover
    return 0

class _RunnerScript:
    """
    Simulate three entries:
      1) GROW_REC (i_out=1)
      2) GROW_EVT (i_out stays 1, hint_out=1)
      3) DONE     (i_out=2, hint_out=2)
    """
    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        # Unpack outs (last group before function symbols)
        # ... args[?] layout mirrors wrapper call order; we read by position:
        # [ ... rec.T, rec.Y, rec.STEP, rec.FLAGS,
        #   ev.EVT_TIME, ev.EVT_CODE, ev.EVT_INDEX,
        #   i_start, step_start, cap_rec, cap_evt,
        #   user_break_flag, status_out, hint_out, i_out, step_out, t_out, ...]
        # Last 4 args are (stepper, rhs, events_pre, events_post)
        # Outs just before them: t_out, step_out, i_out, hint_out, status_out, user_break_flag
        user_break_flag = args[-10]
        status_out      = args[-9]
        hint_out        = args[-8]
        i_out           = args[-7]
        step_out        = args[-6]
        t_out           = args[-5]

        if self.calls == 1:
            i_out[0] = 1
            step_out[0] = 1
            hint_out[0] = 0
            t_out[0] = 0.1
            status_out[0] = GROW_REC
            return GROW_REC
        elif self.calls == 2:
            i_out[0] = 1
            step_out[0] = 1
            hint_out[0] = 1
            t_out[0] = 0.1
            status_out[0] = GROW_EVT
            return GROW_EVT
        else:
            i_out[0] = 2
            step_out[0] = 2
            hint_out[0] = 2
            t_out[0] = 0.2
            status_out[0] = DONE
            return DONE

def test_wrapper_reentry_calls_and_cursors():
    struct = _DummyStepper()
    y0 = np.array([1.0], dtype=np.float64)
    params = np.array([2.0], dtype=np.float64)
    fake_runner = _RunnerScript()

    res = run_with_wrapper(
        runner=fake_runner,
        stepper=_stepper, rhs=_rhs, events_pre=_events_pre, events_post=_events_post,
        struct=struct, model_dtype=np.float64, n_state=1,
        t0=0.0, t_end=1.0, dt_init=0.1, max_steps=100,
        record=True, record_every_step=1,
        y0=y0, params=params,
        cap_rec=1, cap_evt=1,
    )

    # 3 calls: initial + 2 re-entries
    assert fake_runner.calls == 3
    # Final cursors from fake status_out/hint_out
    assert res.n == 2
    assert res.m == 2
