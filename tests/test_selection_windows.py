import numpy as np

from ego_video_camera.selection import _fixed_window_starts


def test_fixed_window_starts_rejects_short_sequences():
    assert _fixed_window_starts(19.99, 20.0, 10.0).size == 0


def test_fixed_window_starts_never_emits_partial_tail():
    np.testing.assert_allclose(
        _fixed_window_starts(45.0, 20.0, 10.0),
        np.asarray([0.0, 10.0, 20.0]),
    )
