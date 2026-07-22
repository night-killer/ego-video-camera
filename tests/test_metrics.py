import numpy as np

from ego_video_camera.metrics import trajectory_metrics


def test_perfect_trajectory_metrics_are_zero():
    poses = np.repeat(np.eye(4)[None], 20, axis=0)
    poses[:, 0, 3] = np.linspace(0, 2, 20)
    metrics = trajectory_metrics(poses, poses.copy(), np.linspace(0, 4, 20))
    assert metrics["ate_rmse_m"] == 0.0
    assert metrics["rotation_mean_deg"] == 0.0
    assert metrics["rpe_translation_rmse_m"] == 0.0
    assert metrics["rpe_rotation_rmse_deg"] == 0.0
