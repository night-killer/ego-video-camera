from __future__ import annotations

import numpy as np
import pytest

from ego_video_camera.camera import (
    PLY_TO_SPZ,
    c2w_to_look_at,
    fov_y_to_intrinsics,
    look_at_c2w,
    ply_rdf_c2w_to_three_display_matrix,
    three_display_matrix_to_ply_rdf_c2w,
)
from ego_video_camera.schema import CameraFrame


def test_three_spz_camera_round_trip() -> None:
    c2w = look_at_c2w([2.0, 1.0, -3.0], [0.2, 0.4, 1.5], [0.0, 1.0, 0.0])
    three = ply_rdf_c2w_to_three_display_matrix(c2w, PLY_TO_SPZ)
    recovered = three_display_matrix_to_ply_rdf_c2w(three, PLY_TO_SPZ)
    np.testing.assert_allclose(recovered, c2w, atol=1e-10)


def test_look_at_uses_rdf_axes() -> None:
    c2w = look_at_c2w([0, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(c2w[:3, 0], [-1, 0, 0], atol=1e-8)
    np.testing.assert_allclose(c2w[:3, 1], [0, -1, 0], atol=1e-8)
    np.testing.assert_allclose(c2w[:3, 2], [0, 0, 1], atol=1e-8)
    assert np.linalg.det(c2w[:3, :3]) == pytest.approx(1.0)
    look = c2w_to_look_at(c2w)
    np.testing.assert_allclose(look["target"], [0, 0, 1])
    np.testing.assert_allclose(look["up"], [0, 1, 0])


def test_fov_intrinsics() -> None:
    K = fov_y_to_intrinsics(896, 504, 90.0)
    assert K[1, 1] == pytest.approx(252.0)
    assert K[0, 0] == pytest.approx(252.0)
    assert K[0, 2] == pytest.approx(448.0)


def test_schema_rejects_non_rotation() -> None:
    pose = np.eye(4)
    pose[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        CameraFrame(
            frame_index=0,
            timestamp_seconds=0,
            camera_to_world=pose.tolist(),
            K=fov_y_to_intrinsics(896, 504, 65).tolist(),
        )
