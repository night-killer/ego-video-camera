"""External-view Gaussian backgrounds and cumulative camera-frustum overlays."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from .camera import FRUSTUM_EDGES, fov_y_to_intrinsics, frustum_vertices, look_at_c2w, project_points
from .gaussian import GaussianScene, render_camera
from .io_utils import atomic_write_json
from .schema import CameraTrajectory
from .video import H264Writer, video_info


@dataclass(frozen=True)
class ObserverCamera:
    name: str
    camera_to_world: np.ndarray
    K: np.ndarray
    width: int
    height: int
    fov_y_degrees: float

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "camera_to_world": self.camera_to_world.tolist(),
            "K": self.K.tolist(),
            "width": self.width,
            "height": self.height,
            "fov_y_degrees": self.fov_y_degrees,
        }


def _horizontal_primary_direction(centers: np.ndarray) -> np.ndarray:
    horizontal = np.asarray(centers, dtype=np.float64)[:, (0, 2)]
    centered = horizontal - horizontal.mean(axis=0)
    if len(horizontal) < 2 or float(np.linalg.norm(centered)) <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0])
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    if float(values.max()) <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0])
    direction = np.asarray([vector[0], 0.0, vector[1]], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    # Fix PCA's sign ambiguity for reproducible observer videos.
    dominant = int(np.argmax(np.abs(direction)))
    if direction[dominant] < 0:
        direction *= -1.0
    return direction


def build_observer_cameras(
    gt: CameraTrajectory,
    predicted: CameraTrajectory,
    scene_bounds_min: np.ndarray,
    scene_bounds_max: np.ndarray,
    *,
    width: int = 896,
    height: int = 504,
    fov_y_degrees: float = 50.0,
    margin: float = 1.2,
) -> tuple[ObserverCamera, ObserverCamera]:
    gt_poses, _ = gt.matrices()
    predicted_poses, _ = predicted.matrices()
    centers = np.concatenate([gt_poses[:, :3, 3], predicted_poses[:, :3, 3]], axis=0)
    lower = np.minimum(np.asarray(scene_bounds_min, dtype=np.float64), centers.min(axis=0))
    upper = np.maximum(np.asarray(scene_bounds_max, dtype=np.float64), centers.max(axis=0))
    target = (lower + upper) * 0.5
    radius = max(float(np.linalg.norm(upper - lower) * 0.5), 1e-3)
    vertical_half = math.radians(fov_y_degrees * 0.5)
    horizontal_half = math.atan(math.tan(vertical_half) * width / height)
    distance = margin * radius / math.sin(min(vertical_half, horizontal_half))
    primary = _horizontal_primary_direction(gt_poses[:, :3, 3])
    up = np.asarray([0.0, 1.0, 0.0])
    lateral = np.cross(up, primary)
    lateral /= np.linalg.norm(lateral)
    K = fov_y_to_intrinsics(width, height, fov_y_degrees)
    observers = []
    for name, forward in (("principal", primary), ("orthogonal", lateral)):
        position = target - forward * distance
        c2w = look_at_c2w(position, target, up)
        observers.append(ObserverCamera(name, c2w, K.copy(), width, height, fov_y_degrees))
    return observers[0], observers[1]


def tone_gaussian_background(rgb: np.ndarray, *, strength: float = 0.42) -> np.ndarray:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("background strength must be between zero and one")
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    luminance = (
        image[..., 0:1] * 0.2126 + image[..., 1:2] * 0.7152 + image[..., 2:3] * 0.0722
    )
    desaturated = image * 0.35 + luminance * 0.65
    toned = desaturated * strength + np.asarray([0.025, 0.03, 0.035]) * (1.0 - strength)
    return np.ascontiguousarray(np.rint(np.clip(toned, 0.0, 1.0) * 255.0).astype(np.uint8))


def render_observer_background(
    scene: GaussianScene,
    observer: ObserverCamera,
    *,
    strength: float = 0.42,
) -> np.ndarray:
    raw = render_camera(
        scene,
        observer.camera_to_world,
        observer.K,
        width=observer.width,
        height=observer.height,
    )
    return tone_gaussian_background(raw, strength=strength)


def default_frustum_depth(
    gt: CameraTrajectory,
    scene_bounds_min: np.ndarray,
    scene_bounds_max: np.ndarray,
) -> float:
    poses, _ = gt.matrices()
    trajectory_diagonal = float(np.linalg.norm(np.ptp(poses[:, :3, 3], axis=0)))
    scene_diagonal = float(
        np.linalg.norm(np.asarray(scene_bounds_max) - np.asarray(scene_bounds_min))
    )
    return max(trajectory_diagonal * 0.04, scene_diagonal * 0.008, 0.05)


def _draw_projected_segment(
    canvas: np.ndarray,
    points: np.ndarray,
    edge: tuple[int, int],
    observer: ObserverCamera,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    selected = points[np.asarray(edge)]
    pixels, depths = project_points(selected, observer.camera_to_world, observer.K)
    if np.any(depths <= 1e-6) or not np.isfinite(pixels).all():
        return
    limit = max(observer.width, observer.height) * 8
    if np.any(np.abs(pixels) > limit):
        return
    start = tuple(np.rint(pixels[0]).astype(int))
    end = tuple(np.rint(pixels[1]).astype(int))
    cv2.line(canvas, start, end, color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_frustum(
    canvas: np.ndarray,
    camera_to_world: np.ndarray,
    camera_K: np.ndarray,
    camera_width: int,
    camera_height: int,
    observer: ObserverCamera,
    *,
    depth: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    vertices = frustum_vertices(
        camera_to_world,
        camera_K,
        camera_width,
        camera_height,
        depth,
    )
    for edge in FRUSTUM_EDGES:
        _draw_projected_segment(canvas, vertices, edge, observer, color, thickness)


def _draw_path(
    canvas: np.ndarray,
    centers: np.ndarray,
    observer: ObserverCamera,
    color: tuple[int, int, int],
) -> None:
    if len(centers) < 2:
        return
    pixels, depths = project_points(centers, observer.camera_to_world, observer.K)
    for index in range(1, len(pixels)):
        if min(depths[index - 1], depths[index]) <= 1e-6:
            continue
        if not np.isfinite(pixels[index - 1 : index + 1]).all():
            continue
        cv2.line(
            canvas,
            tuple(np.rint(pixels[index - 1]).astype(int)),
            tuple(np.rint(pixels[index]).astype(int)),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )


def _draw_legend(canvas: np.ndarray, frame_index: int, timestamp: float) -> None:
    cv2.rectangle(canvas, (12, 12), (285, 80), (10, 10, 12), thickness=-1)
    cv2.line(canvas, (26, 35), (58, 35), (80, 235, 95), 3, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (68, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 245, 235), 1, cv2.LINE_AA)
    cv2.line(canvas, (130, 35), (162, 35), (240, 75, 70), 3, cv2.LINE_AA)
    cv2.putText(canvas, "DA3", (172, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"frame {frame_index:04d}   t={timestamp:.2f}s",
        (26, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 235),
        1,
        cv2.LINE_AA,
    )


def iter_overlay_frames(
    background: np.ndarray,
    observer: ObserverCamera,
    gt: CameraTrajectory,
    predicted: CameraTrajectory,
    *,
    frustum_depth: float,
) -> Iterator[np.ndarray]:
    if len(gt.frames) != len(predicted.frames):
        raise ValueError("GT and predicted trajectories must have matching frame counts")
    gt_poses, gt_Ks = gt.matrices()
    predicted_poses, predicted_Ks = predicted.matrices()
    gt_centers = gt_poses[:, :3, 3]
    predicted_centers = predicted_poses[:, :3, 3]
    history_green = (38, 145, 52)
    history_red = (160, 42, 42)
    current_green = (80, 235, 95)
    current_red = (240, 75, 70)
    for current in range(len(gt.frames)):
        canvas = np.ascontiguousarray(background.copy())
        _draw_path(canvas, gt_centers[: current + 1], observer, current_green)
        _draw_path(canvas, predicted_centers[: current + 1], observer, current_red)
        for index in range(current):
            _draw_frustum(
                canvas,
                gt_poses[index],
                gt_Ks[index],
                gt.video.width,
                gt.video.height,
                observer,
                depth=frustum_depth,
                color=history_green,
                thickness=1,
            )
            _draw_frustum(
                canvas,
                predicted_poses[index],
                predicted_Ks[index],
                predicted.video.width,
                predicted.video.height,
                observer,
                depth=frustum_depth,
                color=history_red,
                thickness=1,
            )
        _draw_frustum(
            canvas,
            gt_poses[current],
            gt_Ks[current],
            gt.video.width,
            gt.video.height,
            observer,
            depth=frustum_depth * 1.12,
            color=current_green,
            thickness=3,
        )
        _draw_frustum(
            canvas,
            predicted_poses[current],
            predicted_Ks[current],
            predicted.video.width,
            predicted.video.height,
            observer,
            depth=frustum_depth * 1.12,
            color=current_red,
            thickness=3,
        )
        _draw_legend(canvas, gt.frames[current].frame_index, gt.frames[current].timestamp_seconds)
        yield canvas


def write_trajectory_visualizations(
    scene: GaussianScene,
    gt: CameraTrajectory,
    predicted: CameraTrajectory,
    output_dir: Path,
    *,
    background_strength: float = 0.42,
    frustum_depth: float | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if not 0.0 <= background_strength <= 1.0:
        raise ValueError("background_strength must be between zero and one")
    if frustum_depth is not None and frustum_depth <= 0:
        raise ValueError("frustum_depth must be positive")
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "visualization_manifest.json"
    expected_paths = [manifest_path]
    for name in ("principal", "orthogonal"):
        expected_paths.extend(
            [target / f"background_{name}.png", target / f"trajectory_{name}.mp4"]
        )
    if not overwrite:
        existing = [path for path in expected_paths if path.exists()]
        if existing:
            raise FileExistsError(f"output already exists (pass --overwrite): {existing[0]}")
    observers = build_observer_cameras(
        gt,
        predicted,
        scene.robust_bounds_min,
        scene.robust_bounds_max,
        width=gt.video.width,
        height=gt.video.height,
    )
    depth = frustum_depth or default_frustum_depth(
        gt, scene.robust_bounds_min, scene.robust_bounds_max
    )
    outputs: list[dict[str, object]] = []
    for observer in observers:
        background_path = target / f"background_{observer.name}.png"
        video_path = target / f"trajectory_{observer.name}.mp4"
        background = render_observer_background(scene, observer, strength=background_strength)
        Image.fromarray(background, mode="RGB").save(background_path)
        with H264Writer(
            video_path,
            width=observer.width,
            height=observer.height,
            fps=gt.video.fps,
        ) as writer:
            for frame in iter_overlay_frames(
                background,
                observer,
                gt,
                predicted,
                frustum_depth=depth,
            ):
                writer.write(frame)
        encoded_info = video_info(video_path, decode_count=True)
        if (
            encoded_info["decoded_frames"] != len(gt.frames)
            or (encoded_info["width"], encoded_info["height"])
            != (observer.width, observer.height)
            or encoded_info["fps"] is None
            or not math.isclose(float(encoded_info["fps"]), gt.video.fps, abs_tol=1e-6)
            or encoded_info["codec"] != "h264"
            or encoded_info["pixel_format"] != "yuv420p"
        ):
            raise RuntimeError(f"unexpected trajectory video encoding: {encoded_info}")
        outputs.append(
            {
                "name": observer.name,
                "background_path": str(background_path),
                "video_path": str(video_path),
                "frame_count": writer.frame_count,
                "video_info": encoded_info,
                "observer": observer.to_json(),
            }
        )
    manifest = {
        "status": "complete",
        "frame_count": len(gt.frames),
        "fps": gt.video.fps,
        "background_strength": background_strength,
        "frustum_depth": depth,
        "scene_bounds_min": scene.robust_bounds_min.tolist(),
        "scene_bounds_max": scene.robust_bounds_max.tolist(),
        "views": outputs,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
