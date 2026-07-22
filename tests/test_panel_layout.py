import numpy as np

from ego_video_camera.visualization import compose_triptych, letterbox


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
        da3_title="Exo + DA3 Head Pose",
        sequence_label="sequence",
        timestamp_label="timestamp",
        alignment_label="Calibration-prefix alignment",
    )
    assert canvas.shape == (1080, 1920, 3)
