from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..serialization import read_json, write_json
from ..trajectory_alignment import trajectory_excitation, umeyama
from ..transforms import Sim3, invert_pose, pose_rotation_errors_deg, rotation_error_deg
from .reference import load_reference
from .registry import load_frames
from .schema import PoseTrajectory, RunSpec, RunStatus
from .telemetry import utc_now
from .trajectory_io import read_trajectory, validate_prediction


RPE_DELTAS_SEC = (0.1, 1.0, 5.0, 10.0)


def _summary(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}_median": None, f"{prefix}_p95": None, f"{prefix}_rmse": None}
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
    }


def _fixed_prefix_sim3(
    estimate: np.ndarray,
    reference: np.ndarray,
    timestamps_sec: np.ndarray,
    valid: np.ndarray,
    duration_sec: float,
) -> tuple[str, Sim3 | None, dict[str, Any]]:
    if not len(timestamps_sec) or timestamps_sec[-1] + 1e-6 < duration_sec:
        return "unavailable_short_clip", None, {"used_count": 0, "duration_sec": duration_sec}
    mask = valid & (timestamps_sec <= duration_sec + 1e-9)
    source = estimate[mask, :3, 3]
    target = reference[mask, :3, 3]
    source_excitation = trajectory_excitation(source)
    target_excitation = trajectory_excitation(target)
    details = {
        "used_count": int(mask.sum()),
        "duration_sec": duration_sec,
        "source_excitation": source_excitation,
        "target_excitation": target_excitation,
    }
    if (
        mask.sum() < 3
        or source_excitation["span_m"] < 0.10
        or target_excitation["span_m"] < 0.10
        or source_excitation["rank_ratio"] < 0.001
        or target_excitation["rank_ratio"] < 0.001
    ):
        return "prefix_degenerate", None, details
    try:
        return "ok", umeyama(source, target, with_scale=True), details
    except ValueError as error:
        details["reason"] = str(error)
        return "prefix_degenerate", None, details


def _initial_se3(
    estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray
) -> tuple[str, np.ndarray | None]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return "no_common_pose", None
    index = int(indices[0])
    return "ok", reference[index] @ invert_pose(estimate[index])


def _apply_se3(poses: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", transform, poses)


def _rpe(
    estimate: np.ndarray,
    reference: np.ndarray,
    timestamps_sec: np.ndarray,
    valid: np.ndarray,
    delta_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    translation, rotation = [], []
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return np.empty(0), np.empty(0)
    for i in indices:
        candidates = indices[indices > i]
        if not len(candidates):
            continue
        target = timestamps_sec[i] + delta_sec
        j = int(candidates[np.argmin(np.abs(timestamps_sec[candidates] - target))])
        tolerance = min(0.25, max(0.055, 0.05 * delta_sec))
        if abs(timestamps_sec[j] - target) > tolerance:
            continue
        ref_relative = invert_pose(reference[i]) @ reference[j]
        est_relative = invert_pose(estimate[i]) @ estimate[j]
        error = invert_pose(ref_relative) @ est_relative
        translation.append(np.linalg.norm(error[:3, 3]))
        rotation.append(rotation_error_deg(np.eye(3), error[:3, :3]))
    return np.asarray(translation), np.asarray(rotation)


def _path_length(points: np.ndarray, valid: np.ndarray) -> float:
    pair_valid = valid[:-1] & valid[1:]
    if not pair_valid.any():
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0)[pair_valid], axis=1).sum())


def _scale_metrics(
    estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray
) -> dict[str, float | None]:
    estimate_length = _path_length(estimate[:, :3, 3], valid)
    reference_length = _path_length(reference[:, :3, 3], valid)
    ratio = estimate_length / reference_length if reference_length > 1e-8 else None
    result: dict[str, float | None] = {
        "estimate_path_length_m": estimate_length,
        "reference_path_length_m": reference_length,
        "path_scale_ratio": ratio,
        "abs_log_scale_error": abs(float(np.log(ratio))) if ratio and ratio > 0 else None,
        "scale_drift_abs_log": None,
    }
    indices = np.flatnonzero(valid)
    if len(indices) >= 6:
        sections = np.array_split(indices, 3)
        ratios = []
        for section in (sections[0], sections[-1]):
            mask = np.zeros(len(valid), dtype=bool)
            mask[section] = True
            est_len = _path_length(estimate[:, :3, 3], mask)
            ref_len = _path_length(reference[:, :3, 3], mask)
            ratios.append(est_len / ref_len if ref_len > 1e-8 else np.nan)
        if np.isfinite(ratios).all() and min(ratios) > 0:
            result["scale_drift_abs_log"] = abs(float(np.log(ratios[1] / ratios[0])))
    return result


def _derivative_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    timestamps_sec: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | None]:
    dt = np.diff(timestamps_sec)
    pair_valid = valid[:-1] & valid[1:] & (dt > 1e-6)
    if not pair_valid.any():
        return {
            "velocity_error_rmse_mps": None,
            "acceleration_error_rmse_mps2": None,
            "jerk_error_rmse_mps3": None,
        }
    est_velocity = np.full((len(dt), 3), np.nan)
    ref_velocity = np.full((len(dt), 3), np.nan)
    est_velocity[pair_valid] = np.diff(estimate[:, :3, 3], axis=0)[pair_valid] / dt[pair_valid, None]
    ref_velocity[pair_valid] = np.diff(reference[:, :3, 3], axis=0)[pair_valid] / dt[pair_valid, None]
    velocity_error = np.linalg.norm(est_velocity[pair_valid] - ref_velocity[pair_valid], axis=1)

    def next_derivative(values: np.ndarray, valid_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        next_dt = (dt[:-1] + dt[1:]) * 0.5
        mask = valid_values[:-1] & valid_values[1:] & (next_dt > 1e-6)
        output = np.full((len(next_dt), 3), np.nan)
        output[mask] = np.diff(values, axis=0)[mask] / next_dt[mask, None]
        return output, mask

    acceleration_est, acceleration_valid = next_derivative(est_velocity, pair_valid)
    acceleration_ref, _ = next_derivative(ref_velocity, pair_valid)
    acceleration_error = np.linalg.norm(
        acceleration_est[acceleration_valid] - acceleration_ref[acceleration_valid], axis=1
    )
    acceleration_dt = (dt[:-1] + dt[1:]) * 0.5
    jerk_valid = acceleration_valid[:-1] & acceleration_valid[1:]
    jerk_dt = (acceleration_dt[:-1] + acceleration_dt[1:]) * 0.5
    jerk_valid &= jerk_dt > 1e-6
    if jerk_valid.any():
        jerk_est = np.diff(acceleration_est, axis=0)[jerk_valid] / jerk_dt[jerk_valid, None]
        jerk_ref = np.diff(acceleration_ref, axis=0)[jerk_valid] / jerk_dt[jerk_valid, None]
        jerk_error = np.linalg.norm(jerk_est - jerk_ref, axis=1)
    else:
        jerk_error = np.empty(0)
    return {
        "velocity_error_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "acceleration_error_rmse_mps2": (
            float(np.sqrt(np.mean(acceleration_error**2))) if len(acceleration_error) else None
        ),
        "jerk_error_rmse_mps3": (
            float(np.sqrt(np.mean(jerk_error**2))) if len(jerk_error) else None
        ),
    }


def _confidence_metrics(confidence: np.ndarray, error: np.ndarray, valid: np.ndarray) -> dict[str, float | None]:
    mask = valid & np.isfinite(confidence) & np.isfinite(error)
    if mask.sum() < 3 or np.std(confidence[mask]) < 1e-12:
        return {"confidence_error_spearman": None, "risk_coverage_ause": None}
    from scipy.stats import spearmanr

    conf, risk = confidence[mask], error[mask]
    order = np.argsort(-conf)
    oracle = np.argsort(risk)
    fractions = np.arange(1, len(risk) + 1) / len(risk)
    curve = np.cumsum(risk[order]) / np.arange(1, len(risk) + 1)
    oracle_curve = np.cumsum(risk[oracle]) / np.arange(1, len(risk) + 1)
    return {
        "confidence_error_spearman": float(spearmanr(conf, risk).statistic),
        "risk_coverage_ause": float(np.trapz(curve - oracle_curve, fractions)),
    }


def _robustness(prediction: PoseTrajectory, reference: PoseTrajectory) -> dict[str, Any]:
    reference_valid = reference.valid
    valid = prediction.valid & reference_valid
    timestamps = prediction.timestamp_ns.astype(np.float64) * 1e-9
    first = np.flatnonzero(prediction.valid)
    transitions_lost = prediction.valid[:-1] & ~prediction.valid[1:]
    transitions_recovered = ~prediction.valid[:-1] & prediction.valid[1:]
    init_sec = float(timestamps[first[0]] - timestamps[0]) if len(first) else None
    time_to_failure = None
    if len(first):
        failures = np.flatnonzero(transitions_lost & (np.arange(len(transitions_lost)) >= first[0]))
        if len(failures):
            time_to_failure = float(timestamps[failures[0] + 1] - timestamps[first[0]])
    ref_delta = np.linalg.norm(np.diff(reference.c2w[:, :3, 3], axis=0), axis=1)
    ref_rot_delta = np.asarray(
        [
            rotation_error_deg(reference.c2w[i, :3, :3], reference.c2w[i + 1, :3, :3])
            if reference_valid[i] and reference_valid[i + 1]
            else np.nan
            for i in range(max(0, len(reference.c2w) - 1))
        ]
    )
    stationary = (
        reference_valid[:-1]
        & reference_valid[1:]
        & (ref_delta < 0.005)
        & (ref_rot_delta < 0.5)
        & prediction.valid[:-1]
        & prediction.valid[1:]
    )
    pred_delta = np.linalg.norm(np.diff(prediction.c2w[:, :3, 3], axis=0), axis=1)
    return {
        "frame_count": int(len(prediction.valid)),
        "reference_valid_count": int(reference_valid.sum()),
        "output_valid_count": int(prediction.valid.sum()),
        "common_valid_count": int(valid.sum()),
        "output_coverage": float(prediction.valid.mean()) if len(prediction.valid) else 0.0,
        "scorable_coverage": float(valid.sum() / reference_valid.sum()) if reference_valid.any() else 0.0,
        "initialization_time_sec": init_sec,
        "lost_count": int(transitions_lost.sum()),
        "recovery_count": int(transitions_recovered.sum()),
        "reset_count": int(np.asarray(prediction.reset).sum()),
        "time_to_failure_sec": time_to_failure,
        "stationary_jitter_translation_median_m": (
            float(np.median(pred_delta[stationary])) if stationary.any() else None
        ),
    }


def _protocol_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    timestamps_sec: np.ndarray,
    prediction_valid: np.ndarray,
    reference_valid: np.ndarray,
) -> dict[str, Any]:
    valid = prediction_valid & reference_valid
    position_error = np.full(len(valid), np.nan)
    rotation_error = np.full(len(valid), np.nan)
    position_error[valid] = np.linalg.norm(
        estimate[valid, :3, 3] - reference[valid, :3, 3], axis=1
    )
    rotation_error[valid] = pose_rotation_errors_deg(estimate[valid], reference[valid])
    result: dict[str, Any] = {}
    result.update(_summary(position_error, "ate_m"))
    result.update(_summary(rotation_error, "rotation_deg"))
    for delta in RPE_DELTAS_SEC:
        translation, rotation = _rpe(estimate, reference, timestamps_sec, valid, delta)
        label = str(delta).replace(".", "p")
        result.update(_summary(translation, f"rpe_translation_m_{label}s"))
        result.update(_summary(rotation, f"rpe_rotation_deg_{label}s"))
    result["final_position_drift_m"] = (
        float(position_error[np.flatnonzero(valid)[-1]]) if valid.any() else None
    )
    result["final_rotation_drift_deg"] = (
        float(rotation_error[np.flatnonzero(valid)[-1]]) if valid.any() else None
    )
    result["accurate_coverage_5cm_5deg"] = float(
        np.sum(valid & (position_error <= 0.05) & (rotation_error <= 5.0))
        / max(1, reference_valid.sum())
    )
    result["accurate_coverage_10cm_10deg"] = float(
        np.sum(valid & (position_error <= 0.10) & (rotation_error <= 10.0))
        / max(1, reference_valid.sum())
    )
    result.update(_derivative_metrics(estimate, reference, timestamps_sec, valid))
    result["position_error_m"] = position_error.tolist()
    result["rotation_error_deg"] = rotation_error.tolist()
    return result


def evaluate_trajectory(
    prediction: PoseTrajectory,
    reference: PoseTrajectory,
    *,
    metric_scale: bool,
) -> dict[str, Any]:
    if not np.array_equal(prediction.frame_id, reference.frame_id):
        raise ValueError("Prediction and reference frame ids differ")
    if not np.array_equal(prediction.timestamp_ns, reference.timestamp_ns):
        raise ValueError("Prediction and reference timestamps differ")
    timestamps = prediction.timestamp_ns.astype(np.float64) * 1e-9
    common = prediction.valid & reference.valid
    result: dict[str, Any] = {
        "schema_version": 1,
        "metric_scale_claimed": metric_scale,
        "robustness": _robustness(prediction, reference),
        "scale": _scale_metrics(prediction.c2w, reference.c2w, common),
        "protocols": {},
    }
    raw = _protocol_metrics(
        prediction.c2w, reference.c2w, timestamps, prediction.valid, reference.valid
    )
    result["protocols"]["raw_metric"] = {"status": "ok", "metrics": raw}

    initial_status, initial_transform = _initial_se3(prediction.c2w, reference.c2w, common)
    if initial_transform is None:
        result["protocols"]["initial_se3"] = {"status": initial_status, "metrics": None}
    else:
        aligned = _apply_se3(prediction.c2w, initial_transform)
        result["protocols"]["initial_se3"] = {
            "status": "ok",
            "transform": initial_transform,
            "metrics": _protocol_metrics(
                aligned, reference.c2w, timestamps, prediction.valid, reference.valid
            ),
        }

    for seconds in (5.0, 10.0):
        status, transform, details = _fixed_prefix_sim3(
            prediction.c2w, reference.c2w, timestamps, common, seconds
        )
        key = f"prefix_sim3_{int(seconds)}s"
        entry: dict[str, Any] = {"status": status, "alignment": details, "metrics": None}
        if transform is not None:
            aligned = transform.apply_c2w_poses(prediction.c2w)
            entry["transform"] = asdict(transform)
            entry["metrics"] = _protocol_metrics(
                aligned, reference.c2w, timestamps, prediction.valid, reference.valid
            )
        result["protocols"][key] = entry

    oracle_entry: dict[str, Any] = {"status": "insufficient", "metrics": None}
    if common.sum() >= 3:
        try:
            transform = umeyama(
                prediction.c2w[common, :3, 3], reference.c2w[common, :3, 3], with_scale=True
            )
            aligned = transform.apply_c2w_poses(prediction.c2w)
            oracle_entry = {
                "status": "ok",
                "transform": asdict(transform),
                "metrics": _protocol_metrics(
                    aligned, reference.c2w, timestamps, prediction.valid, reference.valid
                ),
            }
        except ValueError as error:
            oracle_entry = {"status": "degenerate", "reason": str(error), "metrics": None}
    result["protocols"]["oracle_sim3"] = oracle_entry

    primary = "initial_se3" if metric_scale else "prefix_sim3_5s"
    result["primary_protocol"] = primary
    primary_metrics = result["protocols"][primary].get("metrics")
    if primary_metrics is not None:
        errors = np.asarray(primary_metrics["position_error_m"], dtype=np.float64)
        result["confidence"] = _confidence_metrics(
            prediction.confidence, errors, common
        )
    else:
        result["confidence"] = {
            "confidence_error_spearman": None,
            "risk_coverage_ause": None,
        }
    return result


def evaluate_to_file(
    prediction: PoseTrajectory,
    reference: PoseTrajectory,
    output_path: str | Path,
    *,
    metric_scale: bool,
) -> dict[str, Any]:
    result = evaluate_trajectory(prediction, reference, metric_scale=metric_scale)
    write_json(output_path, result)
    return result


def _read_dict(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def evaluate_runs(
    config: dict[str, Any],
    runs: Iterable[RunSpec],
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate completed predictions without exposing references to workers."""

    run_list = list(runs)
    output_root = Path(config["benchmark"]["output_root"])
    cache_dir = output_root / "cache" / "frames"
    ffmpeg = str(config["benchmark"].get("ffmpeg", "ffmpeg"))
    frame_cache: dict[str, Any] = {}
    reference_cache: dict[str, PoseTrajectory] = {}
    counts: Counter[str] = Counter()
    records = []

    for run in run_list:
        state_path = run.output_dir / "run.json"
        state = _read_dict(state_path)
        evaluation_path = run.output_dir / "evaluation.json"
        inference_status = (state or {}).get(
            "inference_status", (state or {}).get("status")
        )
        if state is None or inference_status != RunStatus.SUCCESS.value:
            counts["skipped_not_successful"] += 1
            records.append(
                {"run_id": run.run_id, "status": "skipped_not_successful"}
            )
            continue
        evaluation_state = state.get("evaluation") or {}
        evaluation_status = evaluation_state.get("status")
        if evaluation_status == "failed" and not (resume or force):
            counts["skipped_existing_failed"] += 1
            records.append(
                {"run_id": run.run_id, "status": "skipped_existing_failed"}
            )
            continue
        if (
            evaluation_path.is_file()
            and evaluation_status == "success"
            and not force
        ):
            counts["skipped_existing"] += 1
            records.append({"run_id": run.run_id, "status": "skipped_existing"})
            continue
        state["status"] = RunStatus.SUCCESS.value
        state.pop("inference_status", None)
        if evaluation_path.is_file() or evaluation_path.is_symlink():
            evaluation_path.unlink()
        try:
            frames = frame_cache.get(run.sequence.key)
            if frames is None:
                frames = load_frames(
                    run.sequence,
                    cache_dir=cache_dir,
                    ffmpeg=ffmpeg,
                    materialize=True,
                )
                frame_cache[run.sequence.key] = frames
            reference = reference_cache.get(run.sequence.key)
            if reference is None:
                reference = load_reference(run.sequence, frames)
                reference_cache[run.sequence.key] = reference
            manifest = _read_dict(run.output_dir / "worker_manifest.json")
            if manifest is None:
                raise FileNotFoundError("worker_manifest.json is missing or invalid")
            prediction = read_trajectory(run.output_dir / "prediction.npz")
            validate_prediction(prediction, manifest)
            result = evaluate_trajectory(
                prediction, reference, metric_scale=run.method.metric_scale
            )
            result.update(
                {
                    "run_id": run.run_id,
                    "method_id": run.method.method_id,
                    "dataset_id": run.sequence.dataset_id,
                    "sequence_id": run.sequence.sequence_id,
                    "reference_grade": run.sequence.reference_grade,
                    "seed": run.seed,
                }
            )
            write_json(evaluation_path, result)
            state["evaluation"] = {
                "status": "success",
                "ended_at": utc_now(),
                "path": str(evaluation_path),
                "primary_protocol": result["primary_protocol"],
            }
            status = "success"
        except (
            KeyError,
            OSError,
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            state["inference_status"] = state.get("status")
            state["status"] = RunStatus.EVALUATION_FAILED.value
            state["evaluation"] = {
                "status": "failed",
                "ended_at": utc_now(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
            status = RunStatus.EVALUATION_FAILED.value
        write_json(state_path, state)
        counts[status] += 1
        records.append({"run_id": run.run_id, "status": status})
    return {
        "schema_version": 1,
        "run_count": len(run_list),
        "status_counts": dict(sorted(counts.items())),
        "runs": records,
    }
