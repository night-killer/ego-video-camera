import numpy as np
import pytest

from ego_video_camera.da3_adapter import (
    DA3_STREAMING_TO_EGOBODY_PV_CAMERA,
    _effective_streaming_overlap,
    da3_streaming_c2w_to_egobody_pv,
)


def test_one_chunk_da3_disables_overlap_to_avoid_official_tail_drop():
    assert _effective_streaming_overlap(25, 60, 30) == 0
    assert _effective_streaming_overlap(160, 60, 30) == 30


def test_invalid_streaming_window_is_rejected():
    with pytest.raises(ValueError):
        _effective_streaming_overlap(100, 30, 30)


def test_streaming_camera_basis_is_changed_to_egobody_pv_without_moving_center():
    official = np.eye(4)[None]
    official[0, :3, 3] = [1.0, 2.0, 3.0]
    converted = da3_streaming_c2w_to_egobody_pv(official)
    np.testing.assert_allclose(converted[0, :3, 3], official[0, :3, 3])
    np.testing.assert_allclose(
        converted[0, :3, :3], DA3_STREAMING_TO_EGOBODY_PV_CAMERA[:3, :3]
    )
    assert np.linalg.det(converted[0, :3, :3]) == pytest.approx(1.0)
