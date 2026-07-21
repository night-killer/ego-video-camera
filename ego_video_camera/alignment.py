"""Similarity alignment and trajectory error reporting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .schema import CameraFrame, CameraTrajectory, PLY_WORLD_FRAME


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    method: str
    position_rank: int

    def matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.scale * self.rotation
        result[:3, 3] = self.translation
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "matrix": self.matrix().tolist(),
            "method": self.method,
            "position_rank": int(self.position_rank),
        }


def estimate_similarity(
    predicted_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    *,
    degeneracy_tolerance: float = 1e-8,
    rank_relative_tolerance: float = 1e-3,
) -> SimilarityTransform:
    predicted = np.asarray(predicted_c2w, dtype=np.float64)
    gt = np.asarray(gt_c2w, dtype=np.float64)
    if predicted.shape != gt.shape or predicted.ndim != 3 or predicted.shape[1:] != (4, 4):
        raise ValueError("predicted and GT poses must have matching shape (N, 4, 4)")
    if len(predicted) < 2:
        raise ValueError("at least two pose correspondences are required for Sim(3) alignment")

    source = predicted[:, :3, 3]
    target = gt[:, :3, 3]
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    def effective_rank(points: np.ndarray) -> int:
        singular_values = np.linalg.svd(points, compute_uv=False)
        if not len(singular_values) or singular_values[0] <= degeneracy_tolerance:
            return 0
        threshold = max(degeneracy_tolerance, singular_values[0] * rank_relative_tolerance)
        return int(np.count_nonzero(singular_values > threshold))

    position_rank = min(effective_rank(source_centered), effective_rank(target_centered))
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))

    if position_rank >= 2 and source_variance > degeneracy_tolerance:
        covariance = target_centered.T @ source_centered / len(source)
        u, singular_values, vt = np.linalg.svd(covariance)
        correction = np.eye(3, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0:
            correction[-1, -1] = -1.0
        rotation = u @ correction @ vt
        scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
        method = "umeyama_all_camera_centers"
    else:
        relative_rotations = gt[:, :3, :3] @ np.transpose(predicted[:, :3, :3], (0, 2, 1))
        rotation = Rotation.from_matrix(relative_rotations).mean().as_matrix()
        rotated_source = (rotation @ source_centered.T).T
        denominator = float(np.sum(rotated_source * rotated_source))
        if denominator <= degeneracy_tolerance:
            scale = 1.0
            method = "orientation_mean_static_scale_one"
        else:
            scale = float(np.sum(rotated_source * target_centered) / denominator)
            method = "orientation_mean_least_squares_scale"
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"estimated Sim(3) scale must be positive and finite, got {scale}")
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return SimilarityTransform(scale, rotation, translation, method, position_rank)


def apply_similarity(poses: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    source = np.asarray(poses, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(source), axis=0)
    result[:, :3, :3] = transform.rotation[None] @ source[:, :3, :3]
    result[:, :3, 3] = (
        transform.scale * (transform.rotation @ source[:, :3, 3].T).T
        + transform.translation[None]
    )
    return result


def alignment_metrics(aligned_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    aligned = np.asarray(aligned_c2w, dtype=np.float64)
    gt = np.asarray(gt_c2w, dtype=np.float64)
    translation_errors = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    relative = np.transpose(gt[:, :3, :3], (0, 2, 1)) @ aligned[:, :3, :3]
    rotation_errors = np.degrees(Rotation.from_matrix(relative).magnitude())
    return {
        "ate_rmse": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_mean": float(np.mean(translation_errors)),
        "translation_median": float(np.median(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
        "rotation_degrees_mean": float(np.mean(rotation_errors)),
        "rotation_degrees_median": float(np.median(rotation_errors)),
        "rotation_degrees_max": float(np.max(rotation_errors)),
    }


def align_predicted_trajectory(
    predicted: CameraTrajectory,
    gt: CameraTrajectory,
) -> tuple[CameraTrajectory, dict[str, Any]]:
    if predicted.trajectory_type not in {"da3_raw", "dense"}:
        raise ValueError(f"expected raw predicted trajectory, got {predicted.trajectory_type!r}")
    if gt.trajectory_type != "dense":
        raise ValueError(f"expected dense GT trajectory, got {gt.trajectory_type!r}")
    if len(predicted.frames) != len(gt.frames):
        raise ValueError(
            f"prediction/GT frame count mismatch: {len(predicted.frames)} != {len(gt.frames)}"
        )
    for predicted_frame, gt_frame in zip(predicted.frames, gt.frames, strict=True):
        if predicted_frame.frame_index != gt_frame.frame_index:
            raise ValueError("prediction and GT frame indexes must match exactly")

    predicted_poses, predicted_intrinsics = predicted.matrices()
    gt_poses, _ = gt.matrices()
    transform = estimate_similarity(predicted_poses, gt_poses)
    aligned_poses = apply_similarity(predicted_poses, transform)
    metrics = alignment_metrics(aligned_poses, gt_poses)
    frames = [
        CameraFrame(
            frame_index=source.frame_index,
            timestamp_seconds=source.timestamp_seconds,
            camera_to_world=aligned_poses[index].tolist(),
            K=predicted_intrinsics[index].tolist(),
        )
        for index, source in enumerate(predicted.frames)
    ]
    report = {"similarity": transform.to_json(), "metrics": metrics, "frame_count": len(frames)}
    aligned = CameraTrajectory(
        trajectory_type="da3_aligned",
        coordinate_system=PLY_WORLD_FRAME,
        scene=gt.scene,
        video=gt.video.model_copy(update={"fov_y_degrees": predicted.video.fov_y_degrees}),
        frames=frames,
        source={
            **predicted.source,
            "alignment": report,
            "gt_coordinate_system": gt.coordinate_system,
        },
    )
    return aligned, report
