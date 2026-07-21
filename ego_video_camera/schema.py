"""Versioned camera-trajectory interchange schema.

The canonical camera convention is OpenCV RDF: +X right, +Y down, +Z forward.
Every ``camera_to_world`` matrix maps those local camera coordinates into the
coordinate frame named by ``coordinate_system``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .io_utils import atomic_write_json, read_json_object


SCHEMA_VERSION = "camera_trajectory.v1"
CAMERA_AXES = "opencv_rdf_x_right_y_down_z_forward"
PLY_WORLD_FRAME = "supersplat_source_ply_world"


def _matrix(value: Any, shape: tuple[int, int], label: str) -> list[list[float]]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric {shape[0]}x{shape[1]} matrix") from exc
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array.tolist()


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1)
    ply_path: str = Field(min_length=1)
    camera_json_path: str | None = None
    display_asset_path: str | None = None
    display_asset_kind: Literal["ply", "spz"] | None = None
    display_from_ply: list[list[float]] = Field(
        default_factory=lambda: np.eye(4, dtype=np.float64).tolist()
    )

    @field_validator("display_from_ply", mode="before")
    @classmethod
    def validate_display_transform(cls, value: Any) -> list[list[float]]:
        matrix = np.asarray(_matrix(value, (4, 4), "display_from_ply"), dtype=np.float64)
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError("display_from_ply must be an affine transform")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("display_from_ply rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
            raise ValueError("display_from_ply rotation must have determinant +1")
        return matrix.tolist()


class VideoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=896, ge=2)
    height: int = Field(default=504, ge=2)
    fps: float = Field(default=15.0, gt=0.0, le=240.0)
    fov_y_degrees: float | None = Field(default=65.0, gt=0.0, lt=180.0)

    @model_validator(mode="after")
    def require_even_video_size(self) -> "VideoSpec":
        if self.width % 2 or self.height % 2:
            raise ValueError("video width and height must be even for yuv420p encoding")
        return self


class CameraFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    camera_to_world: list[list[float]]
    K: list[list[float]]

    @field_validator("camera_to_world", mode="before")
    @classmethod
    def validate_pose(cls, value: Any) -> list[list[float]]:
        matrix = np.asarray(_matrix(value, (4, 4), "camera_to_world"), dtype=np.float64)
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("camera_to_world must end with [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError("camera_to_world rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3):
            raise ValueError("camera_to_world rotation must have determinant +1")
        matrix[:3, :3] = _nearest_rotation(rotation)
        return matrix.tolist()

    @field_validator("K", mode="before")
    @classmethod
    def validate_intrinsics(cls, value: Any) -> list[list[float]]:
        matrix = np.asarray(_matrix(value, (3, 3), "K"), dtype=np.float64)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("K focal lengths must be positive")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("K must end with [0, 0, 1]")
        return matrix.tolist()


def _nearest_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


class CameraTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    trajectory_type: Literal["keyframes", "dense", "da3_raw", "da3_aligned"]
    coordinate_system: str = Field(min_length=1)
    camera_axes: Literal[CAMERA_AXES] = CAMERA_AXES
    scene: SceneSpec
    video: VideoSpec
    frames: list[CameraFrame] = Field(min_length=1)
    source: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sequence(self) -> "CameraTrajectory":
        previous_index = -1
        previous_time = -math.inf
        for position, frame in enumerate(self.frames):
            if frame.frame_index <= previous_index:
                raise ValueError("frame_index values must be strictly increasing")
            if frame.timestamp_seconds <= previous_time:
                raise ValueError("timestamp_seconds values must be strictly increasing")
            expected_time = frame.frame_index / self.video.fps
            if not math.isclose(frame.timestamp_seconds, expected_time, abs_tol=5e-6):
                raise ValueError(
                    f"frame {frame.frame_index} timestamp must equal frame_index/fps "
                    f"({expected_time:.9f}), got {frame.timestamp_seconds:.9f}"
                )
            if self.trajectory_type in {"dense", "da3_raw", "da3_aligned"} and position > 0:
                if frame.frame_index != self.frames[position - 1].frame_index + 1:
                    raise ValueError("dense trajectory frame indexes must be contiguous")
            previous_index = frame.frame_index
            previous_time = frame.timestamp_seconds
        if self.trajectory_type == "keyframes":
            if len(self.frames) < 2:
                raise ValueError("a keyframe trajectory requires at least two frames")
            if self.frames[0].frame_index != 0:
                raise ValueError("the first keyframe must use frame_index 0")
        return self

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        poses = np.asarray([frame.camera_to_world for frame in self.frames], dtype=np.float64)
        intrinsics = np.asarray([frame.K for frame in self.frames], dtype=np.float64)
        return poses, intrinsics


def load_trajectory(path: Path) -> CameraTrajectory:
    return CameraTrajectory.model_validate(read_json_object(path))


def save_trajectory(path: Path, trajectory: CameraTrajectory) -> None:
    atomic_write_json(path, trajectory.model_dump(mode="json"))

