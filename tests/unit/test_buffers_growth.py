# tests/unit/test_buffers_growth.py
import numpy as np

from dynlib.runtime.buffers import allocate_pools, grow_evt_arrays, grow_rec_arrays

def test_grow_rec_copies_filled_region():
    n_state = 3
    rec, ev = allocate_pools(
        n_state=n_state,
        dtype=np.float64,
        cap_rec=2,
        cap_evt=1,
        max_log_width=0,
    )

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
    n_state = 1
    rec, ev = allocate_pools(
        n_state=n_state,
        dtype=np.float64,
        cap_rec=1,
        cap_evt=2,
        max_log_width=0,
    )
    ev.EVT_CODE[:2] = [7, 8]
    ev.EVT_INDEX[:2] = [1, 2]
    ev2 = grow_evt_arrays(ev, filled=2, min_needed=3, dtype=np.float64)
    assert ev2.cap_evt == 4
    np.testing.assert_array_equal(ev2.EVT_CODE[:2], [7, 8])
    np.testing.assert_array_equal(ev2.EVT_INDEX[:2], [1, 2])
