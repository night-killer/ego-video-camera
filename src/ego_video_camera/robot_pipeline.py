from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .da3_adapter import (
    EXPECTED_DA3_COMMIT,
    OFFICIAL_STITCHED_POSE_CONVENTION,
    _effective_streaming_overlap,
    run_da3_streaming,
)
from .metrics import trajectory_metrics
from .robot_io import ACTIVE_EGO_MODEL_LABEL, RobotClip
from .serialization import read_json, write_json
from .trajectory_alignment import (
    AlignmentResult,
    estimate_full_alignment,
    estimate_prefix_alignment,
)
from .transforms import invert_pose
from .video_io import FFmpegWriter, verify_video
from .visualization import (
    DA3_COLOR,
    GT_COLOR,
    compose_triptych,
    draw_pose_overlay,
    draw_text,
)


def _reference_to_exo(
    clip: RobotClip, reference_from_ego: np.ndarray
) -> np.ndarray:
    poses = np.full_like(np.asarray(reference_from_ego, dtype=np.float64), np.nan)
    for index, (frame, pose) in enumerate(zip(clip.frames, reference_from_ego)):
        if (
            frame.synchronized
            and frame.reference_from_exo is not None
            and np.isfinite(pose).all()
        ):
            poses[index] = invert_pose(frame.reference_from_exo) @ pose
    return poses


def _projection_stats(clip: RobotClip, exo_from_ego: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(exo_from_ego).all(axis=(1, 2))
    pixels = np.full((len(exo_from_ego), 2), np.nan, dtype=np.float64)
    depth = np.zeros(len(exo_from_ego), dtype=bool)
    if finite.any():
        projected, projected_depth = clip.exo_camera.project(
            exo_from_ego[finite, :3, 3]
        )
        pixels[finite] = projected
        depth[finite] = projected_depth
    inside = depth & clip.exo_camera.inside(
        pixels,
        int(clip.exo_camera.width or 0),
        int(clip.exo_camera.height or 0),
    )
    return {
        "frame_count": len(exo_from_ego),
        "finite_count": int(finite.sum()),
        "depth_valid_count": int(depth.sum()),
        "inside_count": int(inside.sum()),
        "inside_ratio": float(inside.mean()) if len(inside) else 0.0,
        "pixels": pixels,
        "depth_valid": depth,
        "inside": inside,
    }


def _mapping_rows(clip: RobotClip) -> list[dict[str, Any]]:
    return [
        {
            "dataset": clip.dataset,
            "sequence_id": clip.sequence_id,
            "sample_index": sample_index,
            "output_index": frame.output_index,
            "timeline_sec": frame.timeline_sec,
            "ego_timestamp_ms": frame.ego_timestamp_ms,
            "ego_image": str(frame.ego_image),
            "exo_timestamp_ms": frame.exo_timestamp_ms,
            "sync_delta_ms": frame.sync_delta_ms,
            "exo_source_frame_index": frame.exo_source_frame_index,
            "exo_image": str(frame.exo_image) if frame.exo_image else None,
            "synchronized": frame.synchronized,
            "reference_pose_valid": bool(
                np.isfinite(frame.reference_from_ego).all()
                and frame.reference_from_exo is not None
            ),
            "da3_pose_valid": False,
            "da3_pose_interpolated": False,
            "da3_confidence": None,
        }
        for sample_index, frame in enumerate(clip.frames)
    ]


def _read_images(frame, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    ego = cv2.imread(str(frame.ego_image))
    if ego is None:
        raise RuntimeError(f"Unable to decode Ego RGB: {frame.ego_image}")
    exo = cv2.imread(str(frame.exo_image)) if frame.exo_image else None
    if exo is None:
        exo = np.full((height, width, 3), 16, dtype=np.uint8)
        draw_text(
            exo,
            "Synchronized exo frame unavailable",
            (40, 75),
            (0, 0, 255),
            0.8,
            2,
        )
    return ego, exo


def _render_reference_only(
    clip: RobotClip,
    output_path: Path,
    preview_path: Path,
    ffmpeg_path: str,
) -> dict[str, Any]:
    exo_from_reference = _reference_to_exo(clip, clip.reference_from_ego)
    history: list[np.ndarray] = []
    width = int(clip.exo_camera.width or 1280)
    height = int(clip.exo_camera.height or 720)
    with FFmpegWriter(output_path, 1920, 1080, clip.fps, ffmpeg_path) as writer:
        for index, frame_record in enumerate(clip.frames):
            ego, exo = _read_images(frame_record, width, height)
            pose = exo_from_reference[index]
            if np.isfinite(pose).all():
                history.append(pose)
            else:
                history.clear()
            reference_panel, projected = draw_pose_overlay(
                exo,
                pose,
                clip.exo_camera,
                GT_COLOR,
                frame_kind="opencv_camera",
                history=history[-max(1, int(round(5 * clip.fps))) :],
            )
            empty_panel = exo.copy()
            status = []
            if not frame_record.synchronized:
                status.append("Exo synchronization unavailable")
            if not projected:
                draw_text(
                    reference_panel,
                    "Kinematic reference projection unavailable",
                    (40, 75),
                    (0, 0, 255),
                    0.8,
                    2,
                )
            composed = compose_triptych(
                ego,
                reference_panel,
                empty_panel,
                gt_title="Exo + Kinematic Reference",
                da3_title=ACTIVE_EGO_MODEL_LABEL,
                sequence_label=f"{clip.dataset} / {clip.sequence_id}",
                timestamp_label=(
                    f"frame={frame_record.output_index:06d} "
                    f"timestamp={frame_record.ego_timestamp_ms} ms"
                ),
                alignment_label="Reference validation",
                status_lines=status,
            )
            if index == 0:
                cv2.imwrite(str(preview_path), composed)
            writer.write(composed)
    return _projection_stats(clip, exo_from_reference)


def validate_robot_reference(
    clip: RobotClip,
    output_dir: str | Path,
    *,
    ffmpeg_path: str,
    ffprobe_path: str,
    minimum_synchronized_ratio: float = 0.95,
    minimum_projection_inside_ratio: float = 0.70,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    video = output / "reference_only_overlay.mp4"
    preview = output / "reference_only_overlay_preview.jpg"
    projection = _render_reference_only(
        clip, video, preview, ffmpeg_path
    )
    synchronized_ratio = float(clip.synchronized.mean())
    report = {
        "schema_version": 1,
        "dataset": clip.dataset,
        "sequence_id": clip.sequence_id,
        "reference_label": "Kinematic Reference",
        "reference_type": clip.reference_type,
        "camera_basis": {
            "+X": "right",
            "+Y": "down",
            "-Y": "up",
            "+Z": "gaze",
        },
        "frame_count": len(clip.frames),
        "fps": clip.fps,
        "synchronized_ratio": synchronized_ratio,
        "projection_inside_ratio": projection["inside_ratio"],
        "minimum_synchronized_ratio": minimum_synchronized_ratio,
        "minimum_projection_inside_ratio": minimum_projection_inside_ratio,
        "exo_manifest_quality": clip.exo_manifest.get("quality", {}),
        "video": str(video),
        "preview": str(preview),
        "ffprobe": verify_video(
            video,
            ffprobe_path,
            expected_frames=len(clip.frames),
            expected_fps=clip.fps,
        ),
    }
    report["passes"] = bool(
        synchronized_ratio >= minimum_synchronized_ratio
        and projection["inside_ratio"] >= minimum_projection_inside_ratio
    )
    write_json(output / "reference_validation.json", report)
    write_json(output / "frame_mapping.json", _mapping_rows(clip))
    if not report["passes"]:
        raise RuntimeError(
            f"Robot reference gate failed for {clip.dataset}/{clip.sequence_id}: "
            f"sync={synchronized_ratio:.3f}, inside={projection['inside_ratio']:.3f}"
        )
    return report


def _robot_da3_resume_matches(
    da3_dir: Path, clip: RobotClip, config: dict[str, Any]
) -> bool:
    npz_path = da3_dir / "da3_poses_raw.npz"
    json_path = da3_dir / "da3_poses_raw.json"
    resolved_path = da3_dir / "da3_resolved_config.json"
    if not (npz_path.is_file() and json_path.is_file() and resolved_path.is_file()):
        return False
    da3 = config["da3"]
    checkpoint = Path(da3["checkpoint_path"]).resolve()
    weight = checkpoint / "model.safetensors"
    try:
        resolved = read_json(resolved_path)
        with np.load(npz_path) as payload:
            ids_match = np.array_equal(payload["frame_ids"], clip.frame_ids)
            timestamps_match = np.array_equal(
                payload["timestamps"], clip.timestamps_ms
            )
        expected = {
            "source_commit": EXPECTED_DA3_COMMIT,
            "checkpoint_status": "user_validated_local",
            "checkpoint_path": str(checkpoint),
            "checkpoint_weight_size": weight.stat().st_size,
            "checkpoint_weight_mtime_ns": weight.stat().st_mtime_ns,
            "use_ray_pose": True,
            "process_res": int(da3["input_resolution"]),
            "chunk_size": int(da3["window_size"]),
            "requested_overlap": int(da3["window_overlap"]),
            "effective_overlap": _effective_streaming_overlap(
                len(clip.frames),
                int(da3["window_size"]),
                int(da3["window_overlap"]),
            ),
            "loop_closure": False,
            "confidence_threshold": float(da3["confidence_threshold"]),
            "output_pose_basis": "opencv",
            "input_count": len(clip.frames),
        }
    except (OSError, KeyError, ValueError):
        return False
    return ids_match and timestamps_match and all(
        resolved.get(key) == value for key, value in expected.items()
    )


def run_robot_da3(
    *,
    repo_root: str | Path,
    clip: RobotClip,
    output_dir: str | Path,
    config: dict[str, Any],
    resume: bool = False,
) -> Path:
    """Run DA3 from Ego RGB only, preserving the official OpenCV C2W basis."""

    da3_dir = Path(output_dir) / "da3"
    if resume and _robot_da3_resume_matches(da3_dir, clip, config):
        return da3_dir
    if da3_dir.exists():
        shutil.rmtree(da3_dir)
    da3 = config["da3"]
    image_paths = clip.ego_images
    frame_ids = clip.frame_ids.tolist()
    timestamps = clip.timestamps_ms.tolist()
    run_da3_streaming(
        repo_root=repo_root,
        source_root=da3["source_root"],
        checkpoint_path=da3["checkpoint_path"],
        image_paths=image_paths,
        frame_ids=frame_ids,
        timestamps=timestamps,
        output_dir=da3_dir,
        process_res=int(da3["input_resolution"]),
        chunk_size=int(da3["window_size"]),
        overlap=int(da3["window_overlap"]),
        confidence_threshold=float(da3["confidence_threshold"]),
        output_pose_basis="opencv",
    )
    return da3_dir


def load_robot_da3_poses(da3_dir: str | Path) -> dict[str, np.ndarray]:
    directory = Path(da3_dir)
    with np.load(directory / "da3_poses_raw.npz") as payload:
        result = {
            "c2w": np.asarray(payload["c2w"], dtype=np.float64),
            "confidence": np.asarray(payload["confidence"], dtype=np.float64),
            "frame_ids": np.asarray(payload["frame_ids"], dtype=np.int64),
            "timestamps": np.asarray(payload["timestamps"], dtype=np.int64),
        }
    write_json(
        directory / "pose_basis_interpretation.json",
        {
            "source": "da3_poses_raw.npz[c2w]",
            "pose_convention": OFFICIAL_STITCHED_POSE_CONVENTION,
            "camera_basis_change_applied": False,
            "camera_basis": {
                "+X": "right",
                "+Y": "down",
                "-Y": "up",
                "+Z": "gaze",
            },
            "input_count": len(result["c2w"]),
        },
    )
    return result


def _align_robot_da3(
    clip: RobotClip,
    da3_c2w: np.ndarray,
    valid: np.ndarray,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    source = da3_c2w[valid]
    target = clip.reference_from_ego[valid]
    timestamps = clip.timestamps_sec
    valid_timestamps = timestamps[valid]
    alignment_config = config["alignment"]
    minimum_span = float(
        alignment_config.get("minimum_translation_span_m", 0.10)
    )
    minimum_rank = float(alignment_config.get("minimum_rank_ratio", 0.001))
    sim3_full = estimate_full_alignment(
        source, target, True, minimum_span, minimum_rank
    )
    se3_full = estimate_full_alignment(
        source, target, False, minimum_span, minimum_rank
    )
    sim3_prefix = estimate_prefix_alignment(
        source,
        target,
        valid_timestamps,
        initial_sec=float(config["clip"].get("calibration_prefix_sec", 3.0)),
        fallback_sec=5.0,
        maximum_prefix_ratio=float(
            config["clip"].get("maximum_prefix_ratio", 0.30)
        ),
        minimum_span_m=minimum_span,
        minimum_rank_ratio=minimum_rank,
        timeline_start_sec=0.0,
        timeline_end_sec=clip.duration_sec,
    )
    results: dict[str, dict[str, Any]] = {}
    for name, alignment in (
        ("sim3_prefix", sim3_prefix),
        ("sim3_full", sim3_full),
        ("se3_full", se3_full),
    ):
        aligned = np.full_like(da3_c2w, np.nan)
        if alignment.transform is not None:
            aligned[valid] = alignment.transform.apply_c2w_poses(source)
        results[name] = {"alignment": alignment, "reference_from_ego": aligned}
    return results


def _plot_trajectory(
    path: Path, reference: np.ndarray, estimate: np.ndarray, title: str
) -> None:
    valid = np.isfinite(estimate).all(axis=(1, 2))
    figure = plt.figure(figsize=(7, 6))
    plt.plot(
        reference[:, 0, 3],
        reference[:, 2, 3],
        "g-",
        label="Kinematic Reference",
    )
    if valid.any():
        plt.plot(
            estimate[valid, 0, 3],
            estimate[valid, 2, 3],
            color="orange",
            label=ACTIVE_EGO_MODEL_LABEL,
        )
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("Reference X [m]")
    plt.ylabel("Reference Z [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _alignment_metrics(
    clip: RobotClip,
    estimate: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    alignment: AlignmentResult,
    evaluate: bool,
) -> dict[str, Any]:
    metrics = (
        trajectory_metrics(
            estimate,
            clip.reference_from_ego,
            clip.timestamps_sec,
            confidence,
        )
        if evaluate
        else {}
    )
    exo_estimate = _reference_to_exo(clip, estimate)
    projected = _projection_stats(clip, exo_estimate)
    finite_before_confidence = np.isfinite(estimate).all(axis=(1, 2))
    metrics.update(
        {
            "alignment_status": alignment.status,
            "alignment_reason": alignment.reason,
            "alignment_scale": (
                alignment.transform.scale if alignment.transform is not None else None
            ),
            "alignment_used_count": alignment.used_count,
            "alignment_used_duration_sec": alignment.used_duration_sec,
            "da3_valid_count": int(valid.sum()),
            "da3_valid_ratio": float(valid.mean()),
            "finite_aligned_pose_ratio": float(finite_before_confidence.mean()),
            "low_confidence_ratio": float((~valid).mean()),
            "interpolated_ratio": 0.0,
            "reference_substitution_count": 0,
            "synchronized_exo_ratio": float(clip.synchronized.mean()),
            "projected_inside_ratio": projected["inside_ratio"],
        }
    )
    return metrics


def _render_comparison(
    *,
    clip: RobotClip,
    estimate_reference: np.ndarray,
    confidence: np.ndarray,
    pose_valid: np.ndarray,
    metrics: dict[str, Any],
    alignment: AlignmentResult,
    alignment_label: str,
    output_path: Path,
    preview_path: Path,
    ffmpeg_path: str,
) -> None:
    reference_exo = _reference_to_exo(clip, clip.reference_from_ego)
    estimate_exo = _reference_to_exo(clip, estimate_reference)
    reference_history: list[np.ndarray] = []
    estimate_history: list[np.ndarray] = []
    history_count = max(1, int(round(5 * clip.fps)))
    width = int(clip.exo_camera.width or 1280)
    height = int(clip.exo_camera.height or 720)
    with FFmpegWriter(output_path, 1920, 1080, clip.fps, ffmpeg_path) as writer:
        for index, frame_record in enumerate(clip.frames):
            ego, exo = _read_images(frame_record, width, height)
            reference_pose = reference_exo[index]
            estimate_pose = estimate_exo[index]
            if np.isfinite(reference_pose).all():
                reference_history.append(reference_pose)
            else:
                reference_history.clear()
            if np.isfinite(estimate_pose).all():
                estimate_history.append(estimate_pose)
            else:
                estimate_history.clear()
            reference_panel, reference_projected = draw_pose_overlay(
                exo,
                reference_pose,
                clip.exo_camera,
                GT_COLOR,
                frame_kind="opencv_camera",
                history=reference_history[-history_count:],
            )
            prediction_panel, prediction_projected = draw_pose_overlay(
                exo,
                estimate_pose,
                clip.exo_camera,
                DA3_COLOR,
                frame_kind="opencv_camera",
                history=estimate_history[-history_count:],
            )
            status = [
                (
                    f"confidence={confidence[index]:.3f}"
                    if np.isfinite(confidence[index])
                    else "confidence=unavailable"
                )
            ]
            if not pose_valid[index]:
                status.append("LOW CONFIDENCE / INVALID - model pose hidden")
            if alignment.status != "ok":
                status.append(f"Alignment unavailable: {alignment.status}")
            if not frame_record.synchronized:
                status.append("Synchronized exo frame unavailable")
            if not prediction_projected:
                status.append(f"{ACTIVE_EGO_MODEL_LABEL} unavailable")
            if not reference_projected:
                draw_text(
                    reference_panel,
                    "Kinematic reference unavailable",
                    (40, 75),
                    (0, 0, 255),
                    0.8,
                    2,
                )
            position_errors = metrics.get("position_error_m")
            if position_errors:
                error = position_errors[index]
                status.append(
                    f"position error={error:.3f} m"
                    if error is not None and np.isfinite(error)
                    else "position error=N/A"
                )
            rotation_errors = metrics.get("rotation_error_deg")
            if rotation_errors:
                error = rotation_errors[index]
                status.append(
                    f"rotation error={error:.2f} deg"
                    if error is not None and np.isfinite(error)
                    else "rotation error=N/A"
                )
            composed = compose_triptych(
                ego,
                reference_panel,
                prediction_panel,
                gt_title="Exo + Kinematic Reference",
                da3_title=ACTIVE_EGO_MODEL_LABEL,
                sequence_label=f"{clip.dataset} / {clip.sequence_id}",
                timestamp_label=(
                    f"frame={frame_record.output_index:06d} "
                    f"timestamp={frame_record.ego_timestamp_ms} ms"
                ),
                alignment_label=f"{alignment_label} [{alignment.status}]",
                status_lines=status,
            )
            if index == 0:
                cv2.imwrite(str(preview_path), composed)
            writer.write(composed)


def postprocess_robot_da3(
    *,
    clip: RobotClip,
    output_dir: str | Path,
    config: dict[str, Any],
    render_comparison: bool,
    evaluate: bool,
) -> dict[str, Any]:
    output = Path(output_dir)
    da3_data = load_robot_da3_poses(output / "da3")
    da3_c2w = da3_data["c2w"]
    confidence = da3_data["confidence"]
    if len(da3_c2w) != len(clip.frames):
        raise RuntimeError("DA3 pose count does not match robot clip inputs")
    if not np.array_equal(da3_data["frame_ids"], clip.frame_ids):
        raise RuntimeError("DA3 frame IDs do not match robot clip inputs")
    if not np.array_equal(da3_data["timestamps"], clip.timestamps_ms):
        raise RuntimeError("DA3 timestamps do not match robot clip inputs")
    threshold = float(config["da3"]["confidence_threshold"])
    valid = (
        np.isfinite(da3_c2w).all(axis=(1, 2))
        & np.isfinite(confidence)
        & (confidence >= threshold)
    )
    aligned = _align_robot_da3(clip, da3_c2w, valid, config)
    all_metrics: dict[str, dict[str, Any]] = {}
    for name in ("sim3_prefix", "sim3_full", "se3_full"):
        item = aligned[name]
        alignment = item["alignment"]
        estimate = item["reference_from_ego"]
        metrics = _alignment_metrics(
            clip, estimate, confidence, valid, alignment, evaluate
        )
        all_metrics[name] = metrics
        write_json(output / f"alignment_{name}.json", alignment)
        np.savez_compressed(
            output / f"da3_poses_aligned_{name}.npz",
            reference_from_ego=estimate,
            valid=valid,
            frame_ids=clip.frame_ids,
            timestamps=clip.timestamps_ms,
        )
        _plot_trajectory(
            output / f"trajectory_{name}.png",
            clip.reference_from_ego,
            estimate,
            name,
        )
    write_json(output / "metrics.json", all_metrics)
    video_manifest = {}
    if render_comparison:
        for name, filename, label in (
            ("sim3_prefix", "comparison_prefix.mp4", "Sim(3) prefix"),
            ("sim3_full", "comparison_oracle.mp4", "Sim(3) full/oracle"),
        ):
            item = aligned[name]
            path = output / filename
            preview = output / filename.replace(".mp4", "_preview.jpg")
            _render_comparison(
                clip=clip,
                estimate_reference=item["reference_from_ego"],
                confidence=confidence,
                pose_valid=valid,
                metrics=all_metrics[name],
                alignment=item["alignment"],
                alignment_label=label,
                output_path=path,
                preview_path=preview,
                ffmpeg_path=config["runtime"]["ffmpeg_path"],
            )
            video_manifest[name] = {
                "video": str(path),
                "preview": str(preview),
                "ffprobe": verify_video(
                    path,
                    config["runtime"]["ffprobe_path"],
                    expected_frames=len(clip.frames),
                    expected_fps=clip.fps,
                ),
            }
        write_json(output / "video_manifest.json", video_manifest)
    mappings = (
        read_json(output / "frame_mapping.json")
        if (output / "frame_mapping.json").is_file()
        else _mapping_rows(clip)
    )
    for index, row in enumerate(mappings):
        row["da3_pose_valid"] = bool(valid[index])
        row["da3_pose_interpolated"] = False
        row["da3_confidence"] = (
            float(confidence[index]) if np.isfinite(confidence[index]) else None
        )
    write_json(output / "frame_mapping.json", mappings)
    return {"metrics": all_metrics, "videos": video_manifest}


def run_robot_clip(
    *,
    repo_root: str | Path,
    clip: RobotClip,
    output_dir: str | Path,
    config: dict[str, Any],
    run_da3: bool,
    render_comparison: bool,
    evaluate: bool,
    resume: bool = False,
    minimum_synchronized_ratio: float = 0.95,
    minimum_projection_inside_ratio: float = 0.70,
) -> dict[str, Any]:
    output = Path(output_dir)
    reference = validate_robot_reference(
        clip,
        output,
        ffmpeg_path=config["runtime"]["ffmpeg_path"],
        ffprobe_path=config["runtime"]["ffprobe_path"],
        minimum_synchronized_ratio=minimum_synchronized_ratio,
        minimum_projection_inside_ratio=minimum_projection_inside_ratio,
    )
    if run_da3:
        run_robot_da3(
            repo_root=repo_root,
            clip=clip,
            output_dir=output,
            config=config,
            resume=resume,
        )
    if not (output / "da3" / "da3_poses_raw.npz").is_file():
        return {"reference": reference, "da3": "not_run_on_cpu"}
    processed = postprocess_robot_da3(
        clip=clip,
        output_dir=output,
        config=config,
        render_comparison=render_comparison,
        evaluate=evaluate,
    )
    return {"reference": reference, **processed}
