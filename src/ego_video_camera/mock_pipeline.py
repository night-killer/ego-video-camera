from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .camera_models import CameraModel
from .metrics import trajectory_metrics
from .serialization import write_json
from .trajectory_alignment import umeyama
from .transforms import Sim3
from .video_io import FFmpegWriter, verify_video
from .visualization import (
    ACTIMIND_EGO_ESTIMATION_LABEL,
    DA3_COLOR,
    GT_COLOR,
    compose_triptych,
    draw_pose_overlay,
)


def _pose(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    parent_yaw = np.asarray(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    # The synthetic parent is an OpenCV exo camera (+Y down, +Z forward),
    # while the semantic head frame is +Y up and -Z gaze. This base rotation
    # makes UP project upward and GAZE point away from the head toward +Z.
    rotation = parent_yaw @ np.diag([1.0, -1.0, -1.0])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, 0.1 * np.sin(x * 2), 3.0 + 0.05 * np.cos(x)]
    return pose


def _background(width: int, height: int, index: int, ego: bool) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((x / max(width - 1, 1)) * 160 + index * 3).astype(np.uint8)
    image[..., 1] = ((y / max(height - 1, 1)) * 160 + (60 if ego else 20)).astype(np.uint8)
    image[..., 2] = 80 if ego else 45
    cv2.rectangle(image, (width // 3, height // 4), (2 * width // 3, 3 * height // 4), (70, 70, 70), -1)
    return image


def run_mock_pipeline(
    output_root: str | Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    frame_count: int = 20,
    fps: float = 5.0,
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gt = np.stack(
        [
            _pose(
                -0.6 + 1.2 * i / max(frame_count - 1, 1),
                -25.0 + i * 2.0,
            )
            for i in range(frame_count)
        ]
    )
    source = Sim3(
        1.7,
        np.asarray([[0.94, 0, -0.342], [0, 1, 0], [0.342, 0, 0.94]]),
        np.asarray([1.5, -0.4, 0.8]),
    ).inverse().apply_c2w_poses(gt)
    source[:, :3, 3] += np.linspace(0, 0.015, frame_count)[:, None] * np.asarray([1, 0.2, -0.1])
    alignment = umeyama(source[:, :3, 3], gt[:, :3, 3], with_scale=True)
    estimate = alignment.apply_c2w_poses(source)
    timestamps = np.arange(frame_count) / fps
    metrics = trajectory_metrics(estimate, gt, timestamps, np.linspace(3.0, 1.8, frame_count))
    camera = CameraModel(
        matrix=np.asarray([[720.0, 0, 480.0], [0, 720.0, 270.0], [0, 0, 1.0]]),
        distortion=np.zeros(5),
        width=960,
        height=540,
    )
    video_path = output_root / "mock_comparison_prefix.mp4"
    preview_path = output_root / "mock_comparison_prefix_preview.jpg"
    gt_history: list[np.ndarray] = []
    estimate_history: list[np.ndarray] = []
    panel_validation = None
    with FFmpegWriter(video_path, 1920, 1080, fps, ffmpeg_path) as writer:
        for index in range(frame_count):
            ego = _background(640, 480, index, True)
            exo = _background(960, 540, index, False)
            gt_history.append(gt[index])
            estimate_history.append(estimate[index])
            gt_overlay, _ = draw_pose_overlay(exo, gt[index], camera, GT_COLOR, history=gt_history[-25:])
            da3_overlay, _ = draw_pose_overlay(
                exo, estimate[index], camera, DA3_COLOR, history=estimate_history[-25:]
            )
            if index == 0:
                gt_pixels = np.all(gt_overlay == np.asarray(GT_COLOR), axis=2)
                da3_pixels = np.all(da3_overlay == np.asarray(DA3_COLOR), axis=2)
                top_da3_pixels = int(
                    np.all(gt_overlay == np.asarray(DA3_COLOR), axis=2).sum()
                )
                bottom_gt_pixels = int(
                    np.all(da3_overlay == np.asarray(GT_COLOR), axis=2).sum()
                )
                # Semantic R/UP/GAZE axes use fixed shared colors. A handful
                # of anti-aliased pixels can numerically equal a primary
                # marker color without representing a cross-panel marker.
                collision_tolerance = 4
                panel_validation = {
                    "right_exo_backgrounds_same_source": True,
                    "right_exo_source_sha256": hashlib.sha256(exo.tobytes()).hexdigest(),
                    "top_gt_primary_pixel_count": int(gt_pixels.sum()),
                    "top_da3_primary_pixel_count": top_da3_pixels,
                    "top_da3_marker_absent": top_da3_pixels <= collision_tolerance,
                    "bottom_da3_primary_pixel_count": int(da3_pixels.sum()),
                    "bottom_gt_primary_pixel_count": bottom_gt_pixels,
                    "bottom_gt_marker_absent": bottom_gt_pixels <= collision_tolerance,
                    "semantic_axis_antialias_collision_tolerance": collision_tolerance,
                }
            frame = compose_triptych(
                ego,
                gt_overlay,
                da3_overlay,
                gt_title="Exo + GT Head Pose",
                da3_title=f"Exo + {ACTIMIND_EGO_ESTIMATION_LABEL} Head Pose",
                sequence_label="Synthetic mock sequence",
                timestamp_label=f"t={timestamps[index]:.3f}s frame={index:05d}",
                alignment_label="Calibration-prefix alignment (synthetic)",
                status_lines=[
                    f"confidence={3.0 - 1.2 * index / max(frame_count - 1, 1):.3f}",
                    f"position error={metrics['position_error_m'][index]:.4f} m",
                ],
            )
            if index == 0:
                cv2.imwrite(str(preview_path), frame)
            writer.write(frame)
    probe = verify_video(
        video_path,
        ffprobe_path,
        expected_frames=frame_count,
        expected_fps=fps,
    )
    result = {
        "mode": "synthetic_mock_no_real_da3",
        "video": str(video_path),
        "preview": str(preview_path),
        "alignment": alignment,
        "metrics": metrics,
        "ffprobe": probe,
        "panel_validation": panel_validation,
    }
    write_json(output_root / "mock_metrics.json", result)
    return result
