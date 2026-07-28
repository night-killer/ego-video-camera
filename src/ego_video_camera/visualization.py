from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .camera_models import CameraModel


GT_COLOR = (0, 220, 0)
DA3_COLOR = (0, 140, 255)


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    x: int
    y: int
    width: int
    height: int


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.65,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def letterbox(image: np.ndarray, width: int, height: int, background=(16, 16, 16)) -> tuple[np.ndarray, LetterboxTransform]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas, LetterboxTransform(scale, x, y, resized_width, resized_height)


def _finite_pose(pose: np.ndarray | None) -> bool:
    return pose is not None and np.asarray(pose).shape == (4, 4) and np.isfinite(pose).all()


def semantic_pose_directions(
    rotation: np.ndarray, frame_kind: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return semantic right, up and gaze directions in the parent frame."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation, got {rotation.shape}")
    if frame_kind == "head":
        # EgoBody head convention: +X right, +Y up, -Z gaze/forward.
        return rotation[:, 0], rotation[:, 1], -rotation[:, 2]
    if frame_kind == "camera":
        # EgoBody PV/HoloLens camera: +X right, +Y up, -Z gaze/forward.
        return rotation[:, 0], rotation[:, 1], -rotation[:, 2]
    if frame_kind == "opencv_camera":
        # OpenCV camera: +X right, +Y down and +Z gaze/forward.
        return rotation[:, 0], -rotation[:, 1], rotation[:, 2]
    raise ValueError(f"Unsupported pose frame kind: {frame_kind}")


def draw_pose_overlay(
    image: np.ndarray,
    pose: np.ndarray | None,
    camera: CameraModel,
    color: tuple[int, int, int],
    axis_length_m: float = 0.20,
    frame_kind: str = "head",
    history: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, bool]:
    output = image.copy()
    if not _finite_pose(pose):
        return output, False
    pose = np.asarray(pose, dtype=np.float64)
    origin = pose[:3, 3]
    rotation = pose[:3, :3]
    right, up, gaze = semantic_pose_directions(rotation, frame_kind)
    points = np.stack(
        [
            origin,
            origin + axis_length_m * right,
            origin + axis_length_m * up,
            origin + axis_length_m * gaze,
        ]
    )
    pixels, depth_valid = camera.project(points)
    if not depth_valid[0] or not np.isfinite(pixels[0]).all():
        return output, False
    center = tuple(np.rint(pixels[0]).astype(int))
    cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
    cv2.circle(output, center, 11, color, 2, cv2.LINE_AA)
    semantic_axes = [
        (1, (0, 0, 255), "R", 2),
        (2, (0, 255, 0), "UP", 2),
        (3, (255, 0, 0), "GAZE", 4),
    ]
    for index, axis_color, label, thickness in semantic_axes:
        if depth_valid[index] and np.isfinite(pixels[index]).all():
            endpoint = tuple(np.rint(pixels[index]).astype(int))
            cv2.arrowedLine(
                output,
                center,
                endpoint,
                axis_color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.24,
            )
            draw_text(
                output,
                label,
                (endpoint[0] + 5, endpoint[1] - 5),
                axis_color,
                0.48,
                1,
            )
    if history:
        history_points = np.asarray([item[:3, 3] for item in history if _finite_pose(item)])
        if len(history_points) >= 2:
            history_pixels, valid = camera.project(history_points)
            for first, second, first_ok, second_ok in zip(
                history_pixels[:-1], history_pixels[1:], valid[:-1], valid[1:]
            ):
                if first_ok and second_ok and np.isfinite(first).all() and np.isfinite(second).all():
                    cv2.line(
                        output,
                        tuple(np.rint(first).astype(int)),
                        tuple(np.rint(second).astype(int)),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
    return output, True


def compose_triptych(
    ego: np.ndarray,
    exo_gt: np.ndarray,
    exo_da3: np.ndarray,
    *,
    gt_title: str,
    da3_title: str,
    sequence_label: str,
    timestamp_label: str,
    alignment_label: str,
    canvas_size: tuple[int, int] = (1920, 1080),
    status_lines: list[str] | None = None,
) -> np.ndarray:
    width, height = canvas_size
    left_width = width // 2
    right_width = width - left_width
    right_height = height // 2
    left, _ = letterbox(ego, left_width, height)
    top, _ = letterbox(exo_gt, right_width, right_height)
    bottom, _ = letterbox(exo_da3, right_width, height - right_height)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :left_width] = left
    canvas[:right_height, left_width:] = top
    canvas[right_height:, left_width:] = bottom
    cv2.line(canvas, (left_width, 0), (left_width, height), (210, 210, 210), 2)
    cv2.line(canvas, (left_width, right_height), (width, right_height), (210, 210, 210), 2)
    draw_text(canvas, "Ego RGB", (28, 42), scale=0.9, thickness=2)
    draw_text(canvas, gt_title, (left_width + 28, 38), GT_COLOR, scale=0.8, thickness=2)
    draw_text(canvas, da3_title, (left_width + 28, right_height + 38), DA3_COLOR, scale=0.8, thickness=2)
    draw_text(canvas, sequence_label, (28, height - 86), scale=0.60)
    draw_text(canvas, timestamp_label, (28, height - 58), scale=0.60)
    draw_text(canvas, alignment_label, (28, height - 30), (0, 220, 255), scale=0.60)
    for index, line in enumerate(status_lines or []):
        draw_text(canvas, line, (left_width + 28, right_height + 70 + index * 28), scale=0.55)
    return canvas
