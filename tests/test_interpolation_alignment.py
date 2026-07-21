from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego_video_camera.alignment import align_predicted_trajectory, apply_similarity, estimate_similarity
from ego_video_camera.interpolation import interpolate_keyframes
from ego_video_camera.schema import CameraFrame, CameraTrajectory
from conftest import make_trajectory


def test_interpolation_is_dense_and_preserves_knots(scene_spec) -> None:
    keyframes = make_trajectory(scene_spec, trajectory_type="keyframes", count=3)
    poses, _ = keyframes.matrices()
    dense = interpolate_keyframes(keyframes)
    assert len(dense.frames) == 31
    assert [frame.frame_index for frame in dense.frames] == list(range(31))
    dense_poses, _ = dense.matrices()
    np.testing.assert_allclose(dense_poses[0], poses[0])
    np.testing.assert_allclose(dense_poses[15], poses[1])
    np.testing.assert_allclose(dense_poses[30], poses[2])


def test_rotation_spline_is_continuous_across_angle_wrap(scene_spec) -> None:
    keyframes = make_trajectory(scene_spec, trajectory_type="keyframes", count=3)
    angles = [170.0, 180.0, 190.0]
    frames = [
        CameraFrame(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            camera_to_world=np.block(
                [
                    [Rotation.from_euler("y", angle, degrees=True).as_matrix(), np.zeros((3, 1))],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            ).tolist(),
            K=frame.K,
        )
        for frame, angle in zip(keyframes.frames, angles, strict=True)
    ]
    keyframes = keyframes.model_copy(update={"frames": frames})
    dense = interpolate_keyframes(keyframes)
    dense_poses, _ = dense.matrices()
    relative = Rotation.from_matrix(
        np.transpose(dense_poses[:-1, :3, :3], (0, 2, 1)) @ dense_poses[1:, :3, :3]
    )
    assert np.degrees(relative.magnitude()).max() < 2.0


def test_umeyama_recovers_known_similarity() -> None:
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, :3, 3] = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0.5], [3, 1.5, 0.8], [4, 2, 1.3]]
    )
    poses[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 0.4, len(poses))).as_matrix()
    rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.5]).as_matrix()
    scale = 2.7
    translation = np.asarray([4.0, -2.0, 1.5])
    gt = poses.copy()
    gt[:, :3, :3] = rotation[None] @ poses[:, :3, :3]
    gt[:, :3, 3] = scale * (rotation @ poses[:, :3, 3].T).T + translation
    transform = estimate_similarity(poses, gt)
    assert transform.method == "umeyama_all_camera_centers"
    assert transform.scale == pytest.approx(scale, rel=1e-8)
    np.testing.assert_allclose(transform.rotation, rotation, atol=1e-8)
    np.testing.assert_allclose(apply_similarity(poses, transform), gt, atol=1e-8)


def test_straight_path_uses_orientation_fallback() -> None:
    predicted = np.repeat(np.eye(4)[None], 4, axis=0)
    predicted[:, 2, 3] = np.arange(4)
    rotation = Rotation.from_euler("y", 0.7).as_matrix()
    gt = predicted.copy()
    gt[:, :3, :3] = rotation[None] @ predicted[:, :3, :3]
    gt[:, :3, 3] = 3.0 * (rotation @ predicted[:, :3, 3].T).T + [2, 1, -4]
    transform = estimate_similarity(predicted, gt)
    assert transform.method == "orientation_mean_least_squares_scale"
    np.testing.assert_allclose(apply_similarity(predicted, transform), gt, atol=1e-8)


def test_nearly_straight_path_uses_robust_degeneracy_fallback() -> None:
    predicted = np.repeat(np.eye(4)[None], 6, axis=0)
    predicted[:, 2, 3] = np.arange(6)
    predicted[:, 0, 3] = np.asarray([0, 1, -1, 1, -1, 0]) * 1e-5
    rotation = Rotation.from_euler("xyz", [0.1, 0.4, -0.2]).as_matrix()
    gt = predicted.copy()
    gt[:, :3, :3] = rotation[None]
    gt[:, :3, 3] = 2.0 * (rotation @ predicted[:, :3, 3].T).T + [1, 2, 3]
    transform = estimate_similarity(predicted, gt)
    assert transform.position_rank == 1
    assert transform.method == "orientation_mean_least_squares_scale"


def test_align_trajectory_keeps_predicted_intrinsics(scene_spec) -> None:
    gt = make_trajectory(scene_spec, count=4)
    pred = make_trajectory(
        scene_spec,
        trajectory_type="dense",
        count=4,
        coordinate_system="da3_raw_world",
        translation_step=(0.0, 0.0, 1.0),
    )
    pred = pred.model_copy(update={"trajectory_type": "da3_raw"})
    aligned, report = align_predicted_trajectory(pred, gt)
    assert aligned.trajectory_type == "da3_aligned"
    assert report["frame_count"] == 4
