from __future__ import annotations

import numpy as np

from conftest import make_trajectory
from ego_video_camera.trajectory_visualization import (
    build_observer_cameras,
    iter_overlay_frames,
    tone_gaussian_background,
)


def test_observers_are_orthogonal_and_overlay_is_cumulative(scene_spec) -> None:
    gt = make_trajectory(scene_spec, count=3)
    predicted = make_trajectory(scene_spec, count=3)
    predicted = predicted.model_copy(update={"trajectory_type": "da3_aligned"})
    observers = build_observer_cameras(
        gt, predicted, np.asarray([-2, -2, -2]), np.asarray([4, 3, 5]), width=160, height=90
    )
    forward_0 = observers[0].camera_to_world[:3, 2]
    forward_1 = observers[1].camera_to_world[:3, 2]
    assert abs(float(np.dot(forward_0, forward_1))) < 1e-8
    background = np.full((90, 160, 3), 180, np.uint8)
    toned = tone_gaussian_background(background)
    assert toned.mean() < background.mean()
    frames = list(iter_overlay_frames(toned, observers[0], gt, predicted, frustum_depth=0.2))
    assert len(frames) == 3
    assert frames[-1].shape == background.shape
    assert np.any(frames[-1] != toned)

