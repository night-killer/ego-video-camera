import numpy as np

from ego_video_camera.transforms import invert_pose, transform_points


def make_pose(translation):
    pose = np.eye(4)
    pose[:3, 3] = translation
    return pose


def test_pose_inverse_identity():
    pose = make_pose([1.2, -0.4, 2.5])
    angle = np.radians(31)
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    assert np.allclose(pose @ invert_pose(pose), np.eye(4), atol=1e-10)


def test_transform_chain_T_A_B_convention():
    T_K_W = make_pose([1, 0, 0])
    T_W_E = make_pose([0, 2, 0])
    T_K_E = T_K_W @ T_W_E
    assert np.allclose(transform_points(T_K_E, [[0, 0, 0]]), [[1, 2, 0]])
