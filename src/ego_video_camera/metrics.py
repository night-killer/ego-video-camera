from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .transforms import invert_pose, pose_rotation_errors_deg, rotation_error_deg


def trajectory_metrics(
    estimate_c2w: np.ndarray,
    reference_c2w: np.ndarray,
    timestamps_sec: np.ndarray,
    confidence: np.ndarray | None = None,
    rpe_delta_sec: float = 1.0,
) -> dict[str, float | int | None | list[float]]:
    estimate = np.asarray(estimate_c2w, dtype=np.float64)
    reference = np.asarray(reference_c2w, dtype=np.float64)
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    valid = np.isfinite(estimate).all(axis=(1, 2)) & np.isfinite(reference).all(axis=(1, 2))
    position_error = np.full(len(estimate), np.nan)
    rotation_error = np.full(len(estimate), np.nan)
    position_error[valid] = np.linalg.norm(
        estimate[valid, :3, 3] - reference[valid, :3, 3], axis=1
    )
    rotation_error[valid] = pose_rotation_errors_deg(estimate[valid], reference[valid])
    values = position_error[valid]
    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    valid_indices = np.flatnonzero(valid)
    for i in valid_indices:
        target_time = timestamps[i] + rpe_delta_sec
        later = valid_indices[valid_indices > i]
        if not len(later):
            continue
        j = int(later[np.argmin(np.abs(timestamps[later] - target_time))])
        if abs(timestamps[j] - target_time) > max(0.5 * rpe_delta_sec, 0.05):
            continue
        ref_relative = invert_pose(reference[i]) @ reference[j]
        est_relative = invert_pose(estimate[i]) @ estimate[j]
        delta = invert_pose(ref_relative) @ est_relative
        rpe_translation.append(float(np.linalg.norm(delta[:3, 3])))
        rpe_rotation.append(rotation_error_deg(np.eye(3), delta[:3, :3]))
    result: dict[str, float | int | None | list[float]] = {
        "frame_count": len(estimate),
        "valid_count": int(valid.sum()),
        "valid_ratio": float(valid.mean()) if len(valid) else 0.0,
        "ate_rmse_m": float(np.sqrt(np.mean(values**2))) if len(values) else None,
        "ate_median_m": float(np.median(values)) if len(values) else None,
        "ate_p95_m": float(np.percentile(values, 95)) if len(values) else None,
        "rotation_mean_deg": float(np.mean(rotation_error[valid])) if valid.any() else None,
        "rotation_median_deg": float(np.median(rotation_error[valid])) if valid.any() else None,
        "rotation_p95_deg": float(np.percentile(rotation_error[valid], 95)) if valid.any() else None,
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rpe_translation)))) if rpe_translation else None,
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_rotation)))) if rpe_rotation else None,
        "final_position_drift_m": float(values[-1]) if len(values) else None,
        "position_error_m": position_error.tolist(),
        "rotation_error_deg": rotation_error.tolist(),
    }
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64)
        mask = valid & np.isfinite(confidence)
        if mask.sum() >= 3 and np.std(confidence[mask]) > 0 and np.std(position_error[mask]) > 0:
            result["confidence_error_pearson"] = float(pearsonr(confidence[mask], position_error[mask]).statistic)
            result["confidence_error_spearman"] = float(spearmanr(confidence[mask], position_error[mask]).statistic)
        else:
            result["confidence_error_pearson"] = None
            result["confidence_error_spearman"] = None
    return result
