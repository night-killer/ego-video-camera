"""GT/predicted Gaussian re-rendering and synchronized panel composition."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .gaussian import GaussianScene, render_trajectory_video
from .io_utils import PipelineInputError, atomic_write_json
from .schema import CameraFrame, CameraTrajectory
from .video import compose_side_by_side, video_info


def validate_corresponding_trajectories(gt: CameraTrajectory, predicted: CameraTrajectory) -> None:
    if gt.trajectory_type != "dense":
        raise PipelineInputError(f"GT trajectory must be dense, got {gt.trajectory_type!r}")
    if predicted.trajectory_type != "da3_aligned":
        raise PipelineInputError(
            f"predicted trajectory must be Sim(3)-aligned, got {predicted.trajectory_type!r}"
        )
    if gt.coordinate_system != predicted.coordinate_system:
        raise PipelineInputError(
            f"trajectory coordinate systems differ: {gt.coordinate_system!r} != {predicted.coordinate_system!r}"
        )
    if gt.scene.scene_id != predicted.scene.scene_id:
        raise PipelineInputError(
            f"trajectory scenes differ: {gt.scene.scene_id!r} != {predicted.scene.scene_id!r}"
        )
    if len(gt.frames) != len(predicted.frames):
        raise PipelineInputError(
            f"trajectory frame counts differ: {len(gt.frames)} != {len(predicted.frames)}"
        )
    if (gt.video.width, gt.video.height, gt.video.fps) != (
        predicted.video.width,
        predicted.video.height,
        predicted.video.fps,
    ):
        raise PipelineInputError("trajectory video specifications differ")
    for gt_frame, predicted_frame in zip(gt.frames, predicted.frames, strict=True):
        if gt_frame.frame_index != predicted_frame.frame_index:
            raise PipelineInputError("GT and prediction frame indexes do not match exactly")


def predicted_with_gt_intrinsics(
    predicted: CameraTrajectory,
    gt: CameraTrajectory,
) -> CameraTrajectory:
    validate_corresponding_trajectories(gt, predicted)
    return CameraTrajectory(
        trajectory_type="da3_aligned",
        coordinate_system=predicted.coordinate_system,
        scene=predicted.scene,
        video=gt.video,
        frames=[
            CameraFrame(
                frame_index=predicted_frame.frame_index,
                timestamp_seconds=predicted_frame.timestamp_seconds,
                camera_to_world=predicted_frame.camera_to_world,
                K=gt_frame.K,
            )
            for gt_frame, predicted_frame in zip(gt.frames, predicted.frames, strict=True)
        ],
        source={**predicted.source, "intrinsics_override": "gt_per_corresponding_frame"},
    )


def render_comparisons(
    scene: GaussianScene,
    gt: CameraTrajectory,
    predicted: CameraTrajectory,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    validate_corresponding_trajectories(gt, predicted)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "gt_video": target / "gt.mp4",
        "predicted_full_camera_video": target / "predicted_full_camera.mp4",
        "predicted_pose_only_video": target / "predicted_pose_only.mp4",
        "full_camera_comparison": target / "comparison_full_camera.mp4",
        "pose_only_comparison": target / "comparison_pose_only.mp4",
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"comparison output already exists (pass --overwrite): {existing[0]}")

    pose_only = predicted_with_gt_intrinsics(predicted, gt)
    rendered_counts = {
        "gt": render_trajectory_video(scene, gt, paths["gt_video"]),
        "predicted_full_camera": render_trajectory_video(
            scene, predicted, paths["predicted_full_camera_video"]
        ),
        "predicted_pose_only": render_trajectory_video(
            scene, pose_only, paths["predicted_pose_only_video"]
        ),
    }
    full_count = compose_side_by_side(
        paths["gt_video"],
        paths["predicted_full_camera_video"],
        paths["full_camera_comparison"],
        fps=gt.video.fps,
        left_label="GT",
        right_label="DA3 full camera",
    )
    pose_count = compose_side_by_side(
        paths["gt_video"],
        paths["predicted_pose_only_video"],
        paths["pose_only_comparison"],
        fps=gt.video.fps,
        left_label="GT",
        right_label="DA3 pose only (GT intrinsics)",
    )
    counts = {*rendered_counts.values(), full_count, pose_count}
    if counts != {len(gt.frames)}:
        raise RuntimeError(f"unexpected comparison frame counts: {sorted(counts)}")
    video_infos = {name: video_info(path, decode_count=True) for name, path in paths.items()}
    for name, info in video_infos.items():
        expected_width = gt.video.width * (2 if "comparison" in name else 1)
        if (info["width"], info["height"]) != (expected_width, gt.video.height):
            raise RuntimeError(f"unexpected encoded size for {name}: {info}")
        if info["fps"] is None or not math.isclose(
            float(info["fps"]), gt.video.fps, abs_tol=1e-6
        ):
            raise RuntimeError(f"unexpected encoded FPS for {name}: {info['fps']}")
        if info["decoded_frames"] != len(gt.frames):
            raise RuntimeError(f"unexpected encoded frame count for {name}: {info}")
        if info["codec"] != "h264" or info["pixel_format"] != "yuv420p":
            raise RuntimeError(f"unexpected encoded H.264 format for {name}: {info}")
    manifest = {
        "status": "complete",
        "frame_count": len(gt.frames),
        "fps": gt.video.fps,
        "single_view_size": [gt.video.width, gt.video.height],
        "comparison_size": [gt.video.width * 2, gt.video.height],
        "outputs": {name: str(path) for name, path in paths.items()},
        "video_info": video_infos,
    }
    atomic_write_json(target / "comparison_manifest.json", manifest)
    return manifest
