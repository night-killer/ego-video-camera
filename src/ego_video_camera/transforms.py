from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


EPS = 1e-12


def as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3] = matrix
        return result
    raise ValueError(f"Expected a 3x4 or 4x4 matrix, got {matrix.shape}")


def validate_pose(matrix: np.ndarray, atol: float = 1e-4) -> None:
    matrix = as_homogeneous(matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("Pose contains non-finite values")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=atol):
        raise ValueError(f"Invalid homogeneous bottom row: {matrix[3]}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("Rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("Rotation has negative determinant")


def project_to_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = as_homogeneous(matrix)
    result = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def transform_points(T_A_B: np.ndarray, points_B: np.ndarray) -> np.ndarray:
    points = np.asarray(points_B, dtype=np.float64)
    original_shape = points.shape
    points = points.reshape(-1, 3)
    transformed = points @ T_A_B[:3, :3].T + T_A_B[:3, 3]
    return transformed.reshape(original_shape)


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = project_to_so3(rotation_a).T @ project_to_so3(rotation_b)
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_rotation_errors_deg(poses_a: np.ndarray, poses_b: np.ndarray) -> np.ndarray:
    return np.asarray(
        [rotation_error_deg(a[:3, :3], b[:3, :3]) for a, b in zip(poses_a, poses_b)],
        dtype=np.float64,
    )


def camera_centers_from_w2c(extrinsics: np.ndarray) -> np.ndarray:
    return np.asarray([invert_pose(ext)[:3, 3] for ext in extrinsics])


@dataclass(frozen=True)
class Sim3:
    """Similarity transform p_target = scale * rotation @ p_source + translation."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    def apply_c2w_poses(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses, dtype=np.float64)
        output = poses.copy()
        output[:, :3, 3] = self.apply_points(poses[:, :3, 3])
        output[:, :3, :3] = np.einsum("ij,njk->nik", self.rotation, poses[:, :3, :3])
        return output

    def inverse(self) -> "Sim3":
        rotation = self.rotation.T
        scale = 1.0 / self.scale
        translation = -scale * (rotation @ self.translation)
        return Sim3(scale=scale, rotation=rotation, translation=translation)


def average_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) == 0:
        raise ValueError("Cannot average an empty pose set")
    result = np.eye(4)
    result[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    result[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    return result
