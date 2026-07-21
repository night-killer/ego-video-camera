"""Egocentric camera authoring and evaluation pipeline."""

from .schema import (
    CAMERA_AXES,
    PLY_WORLD_FRAME,
    SCHEMA_VERSION,
    CameraFrame,
    CameraTrajectory,
    SceneSpec,
    VideoSpec,
    load_trajectory,
    save_trajectory,
)

__all__ = [
    "CAMERA_AXES",
    "PLY_WORLD_FRAME",
    "SCHEMA_VERSION",
    "CameraFrame",
    "CameraTrajectory",
    "SceneSpec",
    "VideoSpec",
    "load_trajectory",
    "save_trajectory",
]

