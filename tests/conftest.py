from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego_video_camera.camera import fov_y_to_intrinsics
from ego_video_camera.schema import CameraFrame, CameraTrajectory, SceneSpec, VideoSpec


@pytest.fixture
def scene_spec(tmp_path: Path) -> SceneSpec:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\n")
    return SceneSpec(scene_id="scene", ply_path=str(ply))


def make_trajectory(
    scene: SceneSpec,
    *,
    trajectory_type: str = "dense",
    count: int = 4,
    fps: float = 15.0,
    coordinate_system: str = "supersplat_source_ply_world",
    translation_step: tuple[float, float, float] = (1.0, 0.1, 0.2),
) -> CameraTrajectory:
    video = VideoSpec(width=896, height=504, fps=fps, fov_y_degrees=65.0)
    K = fov_y_to_intrinsics(video.width, video.height, 65.0)
    frames = []
    indexes = list(range(count))
    if trajectory_type == "keyframes":
        indexes = [0, 15, 30][:count]
    for position, frame_index in enumerate(indexes):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(translation_step) * position
        frames.append(
            CameraFrame(
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
                camera_to_world=pose.tolist(),
                K=K.tolist(),
            )
        )
    return CameraTrajectory(
        trajectory_type=trajectory_type,
        coordinate_system=coordinate_system,
        scene=scene,
        video=video,
        frames=frames,
    )

