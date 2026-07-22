from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import average_poses, invert_pose, pose_rotation_errors_deg


@dataclass(frozen=True)
class HeadCalibrationResult:
    status: str
    T_E_Q_fixed: np.ndarray | None
    pair_count: int
    coverage: float
    translation_p95_m: float | None
    rotation_p95_deg: float | None
    reason: str | None = None


def calibrate_camera_to_head(
    T_K_E: np.ndarray,
    T_K_Q: np.ndarray,
    valid: np.ndarray | None = None,
    minimum_pairs: int = 20,
    minimum_coverage: float = 0.80,
    max_translation_p95_m: float = 0.05,
    max_rotation_p95_deg: float = 10.0,
) -> HeadCalibrationResult:
    T_K_E = np.asarray(T_K_E, dtype=np.float64)
    T_K_Q = np.asarray(T_K_Q, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(T_K_E).all(axis=(1, 2)) & np.isfinite(T_K_Q).all(axis=(1, 2))
    valid = np.asarray(valid, dtype=bool)
    count = int(valid.sum())
    coverage = float(valid.mean()) if len(valid) else 0.0
    if count < minimum_pairs or coverage < minimum_coverage:
        return HeadCalibrationResult(
            "proxy", None, count, coverage, None, None, "Insufficient valid camera/head pairs"
        )
    offsets = np.asarray([invert_pose(e) @ q for e, q in zip(T_K_E[valid], T_K_Q[valid])])
    provisional = average_poses(offsets)
    translation_errors = np.linalg.norm(offsets[:, :3, 3] - provisional[:3, 3], axis=1)
    rotation_errors = pose_rotation_errors_deg(offsets, np.repeat(provisional[None], len(offsets), axis=0))
    translation_median = np.median(translation_errors)
    rotation_median = np.median(rotation_errors)
    translation_mad = np.median(np.abs(translation_errors - translation_median)) + 1e-9
    rotation_mad = np.median(np.abs(rotation_errors - rotation_median)) + 1e-9
    inliers = (translation_errors <= translation_median + 3 * translation_mad) & (
        rotation_errors <= rotation_median + 3 * rotation_mad
    )
    if int(inliers.sum()) < minimum_pairs:
        return HeadCalibrationResult(
            "proxy", None, int(inliers.sum()), coverage, None, None, "Too few robust inliers"
        )
    fixed = average_poses(offsets[inliers])
    # Estimate from robust inliers, but evaluate the acceptance P95 over every
    # valid pair so that rejected samples cannot make a non-rigid offset pass.
    translation_errors = np.linalg.norm(offsets[:, :3, 3] - fixed[:3, 3], axis=1)
    rotation_errors = pose_rotation_errors_deg(offsets, np.repeat(fixed[None], count, axis=0))
    translation_p95 = float(np.percentile(translation_errors, 95))
    rotation_p95 = float(np.percentile(rotation_errors, 95))
    if translation_p95 > max_translation_p95_m or rotation_p95 > max_rotation_p95_deg:
        return HeadCalibrationResult(
            "proxy",
            None,
            count,
            coverage,
            translation_p95,
            rotation_p95,
            "Camera-to-head transform is not sufficiently rigid",
        )
    return HeadCalibrationResult(
        "head_pose", fixed, count, coverage, translation_p95, rotation_p95
    )


def camera_to_head_poses(T_K_E: np.ndarray, T_E_Q_fixed: np.ndarray) -> np.ndarray:
    return np.einsum("nij,jk->nik", np.asarray(T_K_E), np.asarray(T_E_Q_fixed))
