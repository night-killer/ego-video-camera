from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .camera_models import CameraModel
from .robot_io import ACTIVE_EGO_MODEL_LABEL
from .serialization import write_json
from .video_io import FFmpegWriter, verify_video
from .visualization import (
    DA3_COLOR,
    GT_COLOR,
    compose_triptych,
    draw_pose_overlay,
)


def _pose(x: float, yaw_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    pose[:3, 3] = [x, 0.08 * np.sin(3 * x), 2.8]
    return pose


def _background(width: int, height: int, index: int, ego: bool) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((x / max(width - 1, 1)) * 130 + index * 2).astype(
        np.uint8
    )
    image[..., 1] = ((y / max(height - 1, 1)) * 150 + (55 if ego else 20)).astype(
        np.uint8
    )
    image[..., 2] = 85 if ego else 48
    cv2.rectangle(
        image,
        (width // 3, height // 4),
        (2 * width // 3, 3 * height // 4),
        (75, 75, 75),
        -1,
    )
    return image


def run_robot_mock_pipeline(
    output_root: str | Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    frame_count: int = 20,
    fps: float = 5.0,
) -> dict:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    reference = np.stack(
        [
            _pose(-0.55 + 1.1 * index / max(1, frame_count - 1), -18 + index)
            for index in range(frame_count)
        ]
    )
    prediction = reference.copy()
    prediction[:, 0, 3] += np.linspace(0.01, 0.04, frame_count)
    camera = CameraModel(
        matrix=np.asarray(
            [[720.0, 0.0, 480.0], [0.0, 720.0, 270.0], [0.0, 0.0, 1.0]]
        ),
        distortion=np.zeros(5),
        width=960,
        height=540,
    )
    video = output / "mock_comparison_prefix.mp4"
    preview = output / "mock_comparison_prefix_preview.jpg"
    panel_validation = None
    reference_history: list[np.ndarray] = []
    prediction_history: list[np.ndarray] = []
    with FFmpegWriter(video, 1920, 1080, fps, ffmpeg_path) as writer:
        for index in range(frame_count):
            ego = _background(640, 480, index, True)
            exo = _background(960, 540, index, False)
            reference_history.append(reference[index])
            prediction_history.append(prediction[index])
            top, _ = draw_pose_overlay(
                exo,
                reference[index],
                camera,
                GT_COLOR,
                frame_kind="opencv_camera",
                history=reference_history[-25:],
            )
            bottom, _ = draw_pose_overlay(
                exo,
                prediction[index],
                camera,
                DA3_COLOR,
                frame_kind="opencv_camera",
                history=prediction_history[-25:],
            )
            if index == 0:
                top_green = int(np.all(top == np.asarray(GT_COLOR), axis=2).sum())
                top_orange = int(np.all(top == np.asarray(DA3_COLOR), axis=2).sum())
                bottom_green = int(
                    np.all(bottom == np.asarray(GT_COLOR), axis=2).sum()
                )
                bottom_orange = int(
                    np.all(bottom == np.asarray(DA3_COLOR), axis=2).sum()
                )
                panel_validation = {
                    "right_exo_backgrounds_same_source": True,
                    "right_exo_source_sha256": hashlib.sha256(exo.tobytes()).hexdigest(),
                    "top_reference_primary_pixel_count": top_green,
                    "top_da3_primary_pixel_count": top_orange,
                    "top_da3_marker_absent": top_orange <= 4,
                    "bottom_da3_primary_pixel_count": bottom_orange,
                    "bottom_reference_primary_pixel_count": bottom_green,
                    "bottom_reference_marker_absent": bottom_green <= 4,
                    "semantic_axis_antialias_collision_tolerance": 4,
                }
            composed = compose_triptych(
                ego,
                top,
                bottom,
                gt_title="Exo + Kinematic Reference",
                da3_title=ACTIVE_EGO_MODEL_LABEL,
                sequence_label="Synthetic robot interaction",
                timestamp_label=f"frame={index:06d} t={index / fps:.3f}s",
                alignment_label="Sim(3) prefix [synthetic]",
                status_lines=["confidence=2.500", "OpenCV C2W"],
            )
            if index == 0:
                cv2.imwrite(str(preview), composed)
            writer.write(composed)
    report = {
        "mode": "synthetic_robot_mock_no_real_da3",
        "model_display_label": ACTIVE_EGO_MODEL_LABEL,
        "video": str(video),
        "preview": str(preview),
        "ffprobe": verify_video(
            video,
            ffprobe_path,
            expected_frames=frame_count,
            expected_fps=fps,
        ),
        "panel_validation": panel_validation,
    }
    write_json(output / "mock_metrics.json", report)
    return report
