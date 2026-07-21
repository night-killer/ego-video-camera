"""Camera convention conversion and projection helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .io_utils import PipelineInputError, finite_float


PLY_TO_SPZ = np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float64)
RDF_TO_THREE_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def fov_y_to_intrinsics(width: int, height: int, fov_y_degrees: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if not 0.0 < fov_y_degrees < 180.0:
        raise ValueError("vertical FOV must be between 0 and 180 degrees")
    focal = 0.5 * float(height) / math.tan(math.radians(fov_y_degrees) * 0.5)
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def intrinsics_to_fov_y(K: np.ndarray, height: int) -> float:
    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3) or matrix[1, 1] <= 0 or height <= 0:
        raise ValueError("invalid intrinsics or image height")
    return math.degrees(2.0 * math.atan(float(height) / (2.0 * float(matrix[1, 1]))))


def look_at_c2w(
    position: np.ndarray | list[float],
    target: np.ndarray | list[float],
    up: np.ndarray | list[float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Build an RDF camera-to-world matrix from a look-at description."""

    center = np.asarray(position, dtype=np.float64)
    look = np.asarray(target, dtype=np.float64)
    up_hint = np.asarray(up, dtype=np.float64)
    if center.shape != (3,) or look.shape != (3,) or up_hint.shape != (3,):
        raise ValueError("position, target, and up must each contain three values")
    forward = look - center
    forward_norm = np.linalg.norm(forward)
    if not np.isfinite(forward_norm) or forward_norm <= 1e-9:
        raise ValueError("camera position and target must differ")
    forward /= forward_norm
    up_hint_norm = np.linalg.norm(up_hint)
    if not np.isfinite(up_hint_norm) or up_hint_norm <= 1e-9:
        raise ValueError("camera up vector is degenerate")
    up_hint /= up_hint_norm
    # RDF is right-handed: right x down = forward. With an upward world hint,
    # this means right = forward x up and down = -up.
    right = np.cross(forward, up_hint)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-8:
        fallback = np.asarray([0.0, 0.0, 1.0] if abs(forward[1]) > 0.9 else [0.0, 1.0, 0.0])
        right = np.cross(forward, fallback)
        right_norm = np.linalg.norm(right)
    right /= right_norm
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = -true_up  # RDF camera +Y is down.
    matrix[:3, 2] = forward
    matrix[:3, 3] = center
    return matrix


def c2w_to_look_at(camera_to_world: np.ndarray, target_distance: float = 1.0) -> dict[str, list[float]]:
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    position = matrix[:3, 3]
    forward = matrix[:3, 2]
    up = -matrix[:3, 1]
    return {
        "position": position.tolist(),
        "target": (position + forward * target_distance).tolist(),
        "up": up.tolist(),
    }


def three_display_matrix_to_ply_rdf_c2w(
    three_camera_matrix_world: np.ndarray,
    display_from_ply: np.ndarray,
) -> np.ndarray:
    """Convert a Three.js RUB camera world matrix to canonical PLY/RDF c2w."""

    three_matrix = np.asarray(three_camera_matrix_world, dtype=np.float64)
    display = np.asarray(display_from_ply, dtype=np.float64)
    if three_matrix.shape != (4, 4) or display.shape != (4, 4):
        raise ValueError("camera and display transforms must be 4x4")
    return np.linalg.inv(display) @ three_matrix @ RDF_TO_THREE_CAMERA


def ply_rdf_c2w_to_three_display_matrix(
    camera_to_world: np.ndarray,
    display_from_ply: np.ndarray,
) -> np.ndarray:
    camera = np.asarray(camera_to_world, dtype=np.float64)
    display = np.asarray(display_from_ply, dtype=np.float64)
    if camera.shape != (4, 4) or display.shape != (4, 4):
        raise ValueError("camera and display transforms must be 4x4")
    return display @ camera @ RDF_TO_THREE_CAMERA


def parse_supersplat_camera(payload: dict[str, Any], *, asset_kind: str) -> dict[str, Any]:
    key = "spz_camera" if asset_kind == "spz" else "supersplat_camera"
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PipelineInputError(f"camera JSON does not contain an object named {key!r}")
    try:
        position = [finite_float(item, f"{key}.position") for item in value["position"]]
        target = [finite_float(item, f"{key}.target") for item in value["target"]]
    except (KeyError, TypeError) as exc:
        raise PipelineInputError(f"{key} must contain three-value position and target") from exc
    if len(position) != 3 or len(target) != 3:
        raise PipelineInputError(f"{key} position and target must contain three values")
    fov = finite_float(value.get("fov"), f"{key}.fov")
    if not 0.0 < fov < 180.0:
        raise PipelineInputError(f"{key}.fov must be between 0 and 180 degrees")
    if np.linalg.norm(np.asarray(target) - np.asarray(position)) <= 1e-8:
        raise PipelineInputError(f"{key} position and target must differ")
    return {
        "position": position,
        "target": target,
        "up": [0.0, 1.0, 0.0],
        "fov_y_degrees": fov,
    }


def scale_intrinsics(K: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(K, dtype=np.float64)
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    if matrix.shape != (3, 3) or min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("invalid intrinsics or image size")
    scale = np.diag([target_w / source_w, target_h / source_h, 1.0])
    return scale @ matrix


def project_points(
    points_world: np.ndarray,
    observer_c2w: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (np.linalg.inv(observer_c2w) @ homogeneous.T).T[:, :3]
    pixels_h = (np.asarray(K, dtype=np.float64) @ camera.T).T
    valid = camera[:, 2] > 1e-6
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid] = pixels_h[valid, :2] / pixels_h[valid, 2:3]
    return pixels, camera[:, 2]


def frustum_vertices(camera_to_world: np.ndarray, K: np.ndarray, width: int, height: int, depth: float) -> np.ndarray:
    if depth <= 0:
        raise ValueError("frustum depth must be positive")
    matrix = np.asarray(K, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    pixels = np.asarray(
        [[0.0, 0.0, 1.0], [width, 0.0, 1.0], [width, height, 1.0], [0.0, height, 1.0]],
        dtype=np.float64,
    )
    corners = (inverse @ pixels.T).T
    corners *= depth / corners[:, 2:3]
    local = np.concatenate([np.zeros((1, 3)), corners], axis=0)
    homogeneous = np.concatenate([local, np.ones((5, 1))], axis=1)
    return (np.asarray(camera_to_world, dtype=np.float64) @ homogeneous.T).T[:, :3]


FRUSTUM_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
)
