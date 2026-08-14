import numpy as np

from ego_video_camera.visualization import (
    compose_triptych,
    letterbox,
    semantic_pose_directions,
)


def test_letterbox_preserves_aspect_ratio_and_layout():
    source = np.full((100, 200, 3), 255, dtype=np.uint8)
    result, transform = letterbox(source, 300, 300)
    assert result.shape == (300, 300, 3)
    assert transform.width == 300
    assert transform.height == 150
    assert transform.y == 75


def test_triptych_has_requested_panel_geometry():
    ego = np.zeros((480, 640, 3), dtype=np.uint8)
    exo = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas = compose_triptych(
        ego,
        exo,
        exo.copy(),
        gt_title="Exo + GT Head Pose",
        da3_title="Exo + ActiMind Ego Estimation Head Pose",
        sequence_label="sequence",
        timestamp_label="timestamp",
        alignment_label="Calibration-prefix alignment",
    )
    assert canvas.shape == (1080, 1920, 3)


def test_semantic_axes_use_head_and_egobody_pv_camera_conventions():
    rotation = np.eye(3)
    head_right, head_up, head_gaze = semantic_pose_directions(rotation, "head")
    camera_right, camera_up, camera_gaze = semantic_pose_directions(
        rotation, "camera"
    )
    np.testing.assert_allclose(head_right, [1, 0, 0])
    np.testing.assert_allclose(head_up, [0, 1, 0])
    np.testing.assert_allclose(head_gaze, [0, 0, -1])
    np.testing.assert_allclose(camera_right, [1, 0, 0])
    np.testing.assert_allclose(camera_up, [0, 1, 0])
    np.testing.assert_allclose(camera_gaze, [0, 0, -1])
