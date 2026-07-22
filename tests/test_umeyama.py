import numpy as np

from ego_video_camera.trajectory_alignment import umeyama


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(50, 3))
    angle = np.radians(27)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    scale = 2.3
    translation = np.asarray([0.8, -1.1, 3.2])
    target = scale * (source @ rotation.T) + translation
    estimated = umeyama(source, target, with_scale=True)
    assert np.isclose(estimated.scale, scale, atol=1e-10)
    assert np.allclose(estimated.rotation, rotation, atol=1e-10)
    assert np.allclose(estimated.translation, translation, atol=1e-10)
    assert np.allclose(estimated.apply_points(source), target, atol=1e-10)


def test_se3_alignment_does_not_apply_scale_to_rotation():
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    target = source + [2, 3, 4]
    estimated = umeyama(source, target, with_scale=False)
    assert estimated.scale == 1.0
    assert np.allclose(estimated.rotation.T @ estimated.rotation, np.eye(3))


def test_sim3_pose_rotation_is_left_aligned_without_scale():
    angle = np.radians(35)
    alignment_rotation = np.asarray(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    from ego_video_camera.transforms import Sim3

    transform = Sim3(4.2, alignment_rotation, np.asarray([1.0, 2.0, 3.0]))
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 1.0
    aligned = transform.apply_c2w_poses(poses)
    assert np.allclose(aligned[:, :3, :3], alignment_rotation)
    assert np.allclose(aligned[1, :3, 3], 4.2 * alignment_rotation[:, 0] + [1, 2, 3])
