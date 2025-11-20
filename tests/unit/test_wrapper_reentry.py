# tests/unit/test_wrapper_reentry.py
import numpy as np
import pytest

from dynlib.runtime.runner_api import DONE, GROW_EVT, GROW_REC, STEPFAIL
from dynlib.runtime.wrapper import run_with_wrapper
from dynlib.runtime.initial_step import WRMSConfig

# ---- Fake callables -----------------------------------------------------------
def _rhs(t, y_vec, dy_out, params, runtime_ws):  # pragma: no cover
    pass


def _events_pre(t, y_vec, params, evt_log_scratch, runtime_ws):  # pragma: no cover
    return -1, 0


def _events_post(t, y_vec, params, evt_log_scratch, runtime_ws):  # pragma: no cover
    return -1, 0


def _stepper(t, dt, y_curr, rhs, params, runtime_ws, ws, cfg, y_prop, t_prop, dt_next, err_est):  # pragma: no cover
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
        #   ev.EVT_CODE, ev.EVT_INDEX, ev.EVT_LOG_DATA, evt_log_scratch,
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
    ic = np.array([1.0], dtype=np.float64)
    params = np.array([2.0], dtype=np.float64)
    fake_runner = _RunnerScript()

    res = run_with_wrapper(
        runner=fake_runner,
        stepper=_stepper, rhs=_rhs, events_pre=_events_pre, events_post=_events_post,
        dtype=np.float64, n_state=1,
        t0=0.0, t_end=1.0, dt_init=0.1, max_steps=100,
        record=True, record_interval=1,
        ic=ic, params=params,
        cap_rec=1, cap_evt=1,
    )

    # 3 calls: initial + 2 re-entries
    assert fake_runner.calls == 3
    # Final cursors from fake status_out/hint_out
    assert res.n == 2
    assert res.m == 2
    assert res.status == DONE
    assert res.ok


class _FailingRunner:
    def __call__(self, *args):
        user_break_flag = args[-10]
        status_out = args[-9]
        hint_out = args[-8]
        i_out = args[-7]
        step_out = args[-6]
        t_out = args[-5]

        i_out[0] = 0
        step_out[0] = 0
        hint_out[0] = 0
        t_out[0] = 0.0
        status_out[0] = STEPFAIL
        return STEPFAIL


def test_wrapper_warns_and_sets_status_on_failure():
    ic = np.array([1.0], dtype=np.float64)
    params = np.array([2.0], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="STEPFAIL"):
        res = run_with_wrapper(
            runner=_FailingRunner(),
            stepper=_stepper, rhs=_rhs, events_pre=_events_pre, events_post=_events_post,
            dtype=np.float64, n_state=1,
            t0=0.0, t_end=1.0, dt_init=0.1, max_steps=10,
            record=True, record_interval=1,
            ic=ic, params=params,
            cap_rec=1, cap_evt=1,
        )

    assert res.status == STEPFAIL
    assert not res.ok
    assert res.n == 0
    assert res.m == 0


class _RecordingRunner:
    def __init__(self):
        self.dt_arg = None

    def __call__(self, *args):
        # dt is the third scalar argument (t0, t_end, dt_init, ...)
        self.dt_arg = float(args[2])
        user_break_flag = args[-10]
        status_out = args[-9]
        hint_out = args[-8]
        i_out = args[-7]
        step_out = args[-6]
        t_out = args[-5]

        user_break_flag[0] = 0
        status_out[0] = DONE
        hint_out[0] = 0
        i_out[0] = 0
        step_out[0] = 0
        t_out[0] = float(args[0])
        return DONE


def _rhs_wrms(t, y_vec, dy_out, params, runtime_ws):  # pragma: no cover - simple helper
    dy_out[:] = 1.0


def test_wrapper_uses_wrms_dt_when_available():
    ic = np.array([0.0, 0.0], dtype=np.float64)
    params = np.array([0.0], dtype=np.float64)
    runner = _RecordingRunner()
    cfg = WRMSConfig(atol=1e-6, rtol=1e-3, order=1, safety=1.0, min_dt=0.05, max_dt=0.05)

    res = run_with_wrapper(
        runner=runner,
        stepper=_stepper,
        rhs=_rhs_wrms,
        events_pre=_events_pre,
        events_post=_events_post,
        dtype=np.float64,
        n_state=2,
        t0=0.0,
        t_end=1.0,
        dt_init=None,
        max_steps=1,
        record=False,
        record_interval=1,
        ic=ic,
        params=params,
        cap_rec=1,
        cap_evt=1,
        wrms_cfg=cfg,
        adaptive=True,
    )

    assert res.status == DONE
    assert pytest.approx(runner.dt_arg) == 0.05
