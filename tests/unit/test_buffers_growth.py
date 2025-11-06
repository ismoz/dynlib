# tests/unit/test_buffers_growth.py
import numpy as np
from dynlib.runtime.buffers import allocate_pools, grow_rec_arrays, grow_evt_arrays
from dynlib.steppers.base import StepperMeta, StepperSpec, StructSpec

class _DummyStepper(StepperSpec):
    def __init__(self):
        super().__init__(StepperMeta(name="dummy", kind="ode"))
    def struct_spec(self) -> StructSpec:
        return StructSpec(0,0,0,0,0,0, 0,0)
    def emit(self, *args, **kwargs):
        raise NotImplementedError

def test_grow_rec_copies_filled_region():
    n_state = 3
    struct_spec = StructSpec(0,0,0,0,0,0, 0,0)
    banks, rec, ev = allocate_pools(n_state=n_state, struct=struct_spec, model_dtype=np.float64, cap_rec=2, cap_evt=1, max_log_width=0)

    # Fill two records
    rec.T[:2] = [0.0, 0.1]
    rec.Y[:, :2] = np.array([[1,2],[3,4],[5,6]], dtype=np.float64)
    rec.STEP[:2] = [0, 1]
    rec.FLAGS[:2] = [0, 0]

    # Ask for growth to 3
    rec2 = grow_rec_arrays(rec, filled=2, min_needed=3)
    assert rec2.cap_rec >= 3 and rec2.cap_rec == 4  # ×2
    np.testing.assert_allclose(rec2.T[:2], [0.0, 0.1])
    np.testing.assert_allclose(rec2.Y[:, :2], rec.Y[:, :2])
    assert rec2.T[2] == 0.0 and rec2.STEP[2] == 0  # zeroed

def test_grow_evt_copies_filled_region():
    # cap_evt=1 -> grow to 2 then 4
    class V: pass
    v = V()
    v.EVT_TIME = None  # just to signal attribute names in the test
    n_state = 1
    struct_spec = StructSpec(0,0,0,0,0,0, 0,0)
    _, rec, ev = allocate_pools(n_state=n_state, struct=struct_spec, model_dtype=np.float64, cap_rec=1, cap_evt=2, max_log_width=0)
    ev.EVT_TIME[:2] = [0.2, 0.3]
    ev2 = grow_evt_arrays(ev, filled=2, min_needed=3, model_dtype=np.float64)
    assert ev2.cap_evt == 4
    np.testing.assert_allclose(ev2.EVT_TIME[:2], [0.2, 0.3])
