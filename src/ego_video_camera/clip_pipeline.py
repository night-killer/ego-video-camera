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
    DA3_STREAMING_TO_EGOBODY_PV_CAMERA,
    EXPECTED_DA3_COMMIT,
    OFFICIAL_STITCHED_POSE_CONVENTION,
    POSE_CONVENTION,
    _effective_streaming_overlap,
    da3_streaming_c2w_to_egobody_pv,
    run_da3_streaming,
)
from .egobody_io import (
    FrameMapping,
    attach_pv_images,
    discover_pv_sequences,
    load_T_K_W,
    load_head_tracking,
    load_master_camera,
    nearest_head_record,
    parse_pv_file,
    synchronize_exact_frame_ids,
)
from .head_pose_conversion import calibrate_camera_to_head, camera_to_head_poses
from .metrics import trajectory_metrics
from .serialization import read_json, write_json
from .trajectory_alignment import estimate_full_alignment, estimate_prefix_alignment
from .transforms import invert_pose
from .video_io import FFmpegWriter, verify_video
from .visualization import DA3_COLOR, GT_COLOR, compose_triptych, draw_pose_overlay, draw_text, letterbox


def _resolve_sequence_dir(data_root: Path, recording: str, sequence: str) -> Path:
    direct = data_root / "egocentric_color" / recording / sequence
    if direct.is_dir():
        return direct
    candidates = [
        path.parent
        for path in data_root.glob(f"**/egocentric_color/{recording}/{sequence}/*_pv.txt")
    ]
    if not candidates:
        candidates = discover_pv_sequences(data_root, recording)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one HoloLens sequence for {recording}/{sequence}, found {len(candidates)}"
        )
    return candidates[0]


def _find_master_dir(data_root: Path, recording: str) -> Path:
    direct = data_root / "kinect_color" / recording / "master"
    if direct.is_dir():
        return direct
    matches = [path for path in data_root.glob(f"**/kinect_color/{recording}/master") if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve master Kinect frames for {recording}")
    return matches[0]


def _find_gaze_file(data_root: Path, recording: str, sequence: str) -> Path | None:
    direct = data_root / "egocentric_gaze" / recording / sequence
    candidates = list(direct.glob("*_head_hand_eye.csv")) if direct.is_dir() else []
    if not candidates:
        candidates = list(
            data_root.glob(
                f"**/egocentric_gaze/{recording}/{sequence}/**/*_head_hand_eye.csv"
            )
        )
    return sorted(candidates)[0] if candidates else None


def load_clip_records(data_root: str | Path, clip: dict[str, Any]):
    root = Path(data_root)
    recording = clip["recording_name"]
    sequence = clip["hololens_sequence"]
    sequence_dir = _resolve_sequence_dir(root, recording, sequence)
    pv_file = next(sequence_dir.glob("*_pv.txt"))
    pv_calibration, records = parse_pv_file(pv_file)
    records = attach_pv_images(records, sequence_dir / "PV")
    selected_timestamps = {int(value) for value in clip.get("timestamps", [])}
    selected_frames = {int(value) for value in clip.get("frame_ids", [])}
    if selected_timestamps:
        records = [record for record in records if record.timestamp in selected_timestamps]
    elif selected_frames:
        records = [record for record in records if record.frame_id in selected_frames]
    else:
        records = [
            record
            for record in records
            if record.frame_id is not None
            and int(clip["start_frame"]) <= record.frame_id <= int(clip["end_frame"])
        ]
    records = [
        record for record in records if record.frame_id is not None and record.image_path is not None
    ]
    if selected_timestamps and len(records) != len(selected_timestamps):
        raise RuntimeError(
            f"Extracted {len(records)}/{len(selected_timestamps)} selected PV frames for {recording}"
        )
    records.sort(key=lambda item: item.timestamp)
    runtime_duration = clip.get("runtime_duration_sec")
    runtime_fps = clip.get("runtime_sample_fps")
    if runtime_duration is not None or runtime_fps is not None or not selected_timestamps:
        from .egobody_io import sample_records

        records = sample_records(
            records,
            float(runtime_fps or clip.get("sample_fps", 8.0)),
            0.0,
            float(runtime_duration or clip.get("duration_sec", 20.0)),
        )
    if not records:
        raise RuntimeError(f"No extracted PV frames found for {recording} clip")
    mappings = synchronize_exact_frame_ids(records, _find_master_dir(root, recording))
    return pv_calibration, records, mappings


def _gt_state(data_root: Path, clip: dict, records, config: dict) -> dict:
    recording = clip["recording_name"]
    T_K_W, calibration_path = load_T_K_W(data_root, recording)
    T_W_E = np.asarray([record.T_W_E for record in records])
    T_K_E = np.einsum("ij,njk->nik", T_K_W, T_W_E)
    timestamps_sec = (
        np.asarray([record.timestamp for record in records], dtype=np.float64) - records[0].timestamp
    ) / 10_000_000.0
    head_poses = np.full_like(T_K_E, np.nan)
    head_valid = np.zeros(len(records), dtype=bool)
    gaze_file = _find_gaze_file(data_root, recording, clip["hololens_sequence"])
    if gaze_file is not None:
        head_records = load_head_tracking(gaze_file)
        tolerance = float(config["clip"].get("sync_tolerance_ms", 50))
        for index, record in enumerate(records):
            head = nearest_head_record(record.timestamp, head_records, tolerance)
            if head is not None:
                head_poses[index] = T_K_W @ head.T_W_Q
                head_valid[index] = True
    prefix_sec = float(config["clip"].get("calibration_prefix_sec", 3))
    prefix_mask = timestamps_sec <= prefix_sec + 1e-9
    head_config = config["head_pose"]
    calibration = calibrate_camera_to_head(
        T_K_E[prefix_mask],
        head_poses[prefix_mask],
        head_valid[prefix_mask],
        int(head_config.get("minimum_pairs", 20)),
        float(head_config.get("minimum_coverage", 0.8)),
        float(head_config.get("max_translation_p95_m", 0.05)),
        float(head_config.get("max_rotation_p95_deg", 10)),
    )
    use_head = calibration.status == "head_pose"
    display_gt = head_poses if use_head else T_K_E
    return {
        "T_K_W": T_K_W,
        "T_W_E": T_W_E,
        "T_K_E": T_K_E,
        "T_K_Q": head_poses,
        "head_valid": head_valid,
        "head_calibration": calibration,
        "display_gt": display_gt,
        "frame_kind": "head" if use_head else "camera",
        "head_mode": "head_pose" if use_head else "camera_center_proxy",
        "timestamps_sec": timestamps_sec,
        "calibration_path": calibration_path,
        "gaze_file": gaze_file,
    }


def _pose_records(records, poses, convention: str) -> list[dict]:
    return [
        {
            "frame_id": record.frame_id,
            "timestamp": record.timestamp,
            "pose": pose,
            "convention": convention,
            "valid": bool(np.isfinite(pose).all()),
        }
        for record, pose in zip(records, poses)
    ]


def _render_gt_only(
    output_path: Path,
    preview_path: Path,
    mappings: list[FrameMapping],
    display_poses: np.ndarray,
    camera,
    fps: float,
    ffmpeg_path: str,
    frame_kind: str,
    head_mode: str,
    sampling_ratio: float,
) -> dict:
    history: list[np.ndarray] = []
    with FFmpegWriter(output_path, 1920, 1080, fps, ffmpeg_path) as writer:
        for index, mapping in enumerate(mappings):
            image = cv2.imread(str(mapping.exo_image)) if mapping.exo_image else None
            if image is None:
                image = np.zeros((540, 960, 3), dtype=np.uint8)
                draw_text(image, "Missing synchronized exo frame", (40, 270), (0, 0, 255), 0.8, 2)
            pose = display_poses[index]
            if np.isfinite(pose).all():
                history.append(pose)
            overlay, projected = draw_pose_overlay(
                image,
                pose,
                camera,
                GT_COLOR,
                frame_kind=frame_kind,
                history=history[-40:],
            )
            frame, _ = letterbox(overlay, 1920, 1080)
            title = "GT Head Pose" if head_mode == "head_pose" else "GT Head Proxy = ego camera center"
            draw_text(frame, title, (24, 34), GT_COLOR, 0.72, 2)
            if sampling_ratio < 1.0 - 1e-9:
                draw_text(
                    frame,
                    f"Source sampling gaps: {(1.0 - sampling_ratio):.1%} (no interpolation)",
                    (24, 68),
                    (0, 165, 255),
                    0.6,
                    2,
                )
            if not projected:
                draw_text(frame, "GT projection unavailable", (24, 102), (0, 0, 255), 0.6, 2)
            if index == 0:
                cv2.imwrite(str(preview_path), frame)
            writer.write(frame)
    return {"video": str(output_path), "preview": str(preview_path)}


def prepare_gt_clip(
    data_root: str | Path,
    clip: dict,
    output_dir: str | Path,
    config: dict,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, records, mappings = load_clip_records(data_root, clip)
    gt = _gt_state(Path(data_root), clip, records, config)
    camera, camera_path = load_master_camera(data_root)
    width, height = 1920, 1080
    if mappings and mappings[0].exo_image:
        image = cv2.imread(str(mappings[0].exo_image))
        if image is not None:
            height, width = image.shape[:2]
    pixels, z_valid = camera.project(gt["display_gt"][:, :3, 3])
    inside = z_valid & camera.inside(pixels, width, height)
    requested_fps = float(
        clip.get("runtime_sample_fps", clip.get("sample_fps", config["da3"]["sample_fps"]))
    )
    window_duration = float(
        clip.get("runtime_duration_sec", clip.get("duration_sec", len(records) / requested_fps))
    )
    fps = len(records) / window_duration
    sampling_ratio = float(
        clip.get(
            "ego_sampling_ratio",
            len(records) / max(1, int(round(requested_fps * window_duration))),
        )
    )
    gt_output = _render_gt_only(
        output / "gt_only_overlay.mp4",
        output / "gt_only_preview.jpg",
        mappings,
        gt["display_gt"],
        camera,
        fps,
        config["runtime"]["ffmpeg_path"],
        gt["frame_kind"],
        gt["head_mode"],
        sampling_ratio,
    )
    write_json(output / "gt_ego_poses.json", _pose_records(records, gt["T_K_E"], "T_K_E: ego PV camera to master Kinect RGB"))
    write_json(output / "gt_head_poses.json", _pose_records(records, gt["T_K_Q"], "T_K_Q: tracked head to master Kinect RGB"))
    mapping_json = [
        {
            "output_frame": mapping.output_frame,
            "ego_frame_id": mapping.ego_frame_id,
            "exo_frame_id": mapping.exo_frame_id,
            "ego_timestamp": mapping.ego_timestamp,
            "exo_timestamp": mapping.exo_timestamp,
            "time_difference": mapping.time_difference,
            "sync_basis": mapping.sync_basis,
            "gt_pose_valid": bool(np.isfinite(gt["display_gt"][index]).all()),
            "da3_pose_valid": False,
            "da3_pose_interpolated": False,
            "da3_confidence": None,
        }
        for index, mapping in enumerate(mappings)
    ]
    write_json(output / "frame_mapping.json", mapping_json)
    report = {
        "recording_name": clip["recording_name"],
        "hololens_sequence": clip["hololens_sequence"],
        "frame_count": len(records),
        "expected_frame_count": int(round(requested_fps * window_duration)),
        "requested_sample_fps": requested_fps,
        "render_fps": fps,
        "window_duration_sec": window_duration,
        "ego_sampling_ratio": sampling_ratio,
        "synchronized_count": sum(mapping.exo_image is not None for mapping in mappings),
        "synchronized_ratio": float(np.mean([mapping.exo_image is not None for mapping in mappings])),
        "projection_z_valid_ratio": float(z_valid.mean()),
        "projection_inside_ratio": float(inside.mean()),
        "projection_depth_m": {
            "minimum": float(np.nanmin(gt["display_gt"][:, 2, 3])),
            "median": float(np.nanmedian(gt["display_gt"][:, 2, 3])),
            "maximum": float(np.nanmax(gt["display_gt"][:, 2, 3])),
        },
        "projected_pixel_bounds": {
            "x_min": float(np.nanmin(pixels[:, 0])) if np.isfinite(pixels[:, 0]).any() else None,
            "x_max": float(np.nanmax(pixels[:, 0])) if np.isfinite(pixels[:, 0]).any() else None,
            "y_min": float(np.nanmin(pixels[:, 1])) if np.isfinite(pixels[:, 1]).any() else None,
            "y_max": float(np.nanmax(pixels[:, 1])) if np.isfinite(pixels[:, 1]).any() else None,
        },
        "head_mode": gt["head_mode"],
        "head_calibration": gt["head_calibration"],
        "calibration_path": gt["calibration_path"],
        "camera_path": camera_path,
        "gaze_file": gt["gaze_file"],
        **gt_output,
    }
    report["ffprobe"] = verify_video(
        gt_output["video"],
        config["runtime"]["ffprobe_path"],
        expected_frames=len(records),
        expected_fps=fps,
    )
    write_json(output / "gt_validation.json", report)
    return {"records": records, "mappings": mappings, "gt": gt, "camera": camera, "report": report}


def _align_da3(da3_c2w: np.ndarray, valid: np.ndarray, gt: dict, config: dict) -> dict:
    timestamps = gt["timestamps_sec"]
    source = da3_c2w[valid]
    target = gt["T_W_E"][valid]
    valid_timestamps = timestamps[valid]
    alignment_config = config["alignment"]
    minimum_span = float(alignment_config.get("minimum_translation_span_m", 0.1))
    minimum_rank = float(alignment_config.get("minimum_rank_ratio", 0.001))
    oracle = estimate_full_alignment(
        source, target, True, minimum_span, minimum_rank
    )
    se3 = estimate_full_alignment(
        source, target, False, minimum_span, minimum_rank
    )
    prefix = estimate_prefix_alignment(
        source,
        target,
        valid_timestamps,
        float(config["clip"].get("calibration_prefix_sec", 3)),
        5.0,
        float(config["clip"].get("maximum_prefix_ratio", 0.3)),
        minimum_span,
        minimum_rank,
        timeline_start_sec=float(timestamps[0]),
        timeline_end_sec=float(timestamps[-1]),
    )
    results = {}
    for name, alignment in (("sim3_full", oracle), ("sim3_prefix", prefix), ("se3_full", se3)):
        aligned_W = np.full_like(da3_c2w, np.nan)
        aligned_K = np.full_like(da3_c2w, np.nan)
        if alignment.transform is not None:
            aligned_W[valid] = alignment.transform.apply_c2w_poses(source)
            aligned_K[valid] = np.einsum("ij,njk->nik", gt["T_K_W"], aligned_W[valid])
        results[name] = {"alignment": alignment, "T_W_E": aligned_W, "T_K_E": aligned_K}
    return results


def _plot_trajectory(path: Path, gt: np.ndarray, estimate: np.ndarray, title: str) -> None:
    valid = np.isfinite(estimate).all(axis=(1, 2))
    plt.figure(figsize=(7, 6))
    plt.plot(gt[:, 0, 3], gt[:, 2, 3], "g-", label="GT")
    if valid.any():
        plt.plot(estimate[valid, 0, 3], estimate[valid, 2, 3], color="orange", label="DA3")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("Kinect X [m]")
    plt.ylabel("Kinect Z [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _head_pose_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    camera,
) -> dict:
    valid = np.isfinite(estimate).all(axis=(1, 2)) & np.isfinite(reference).all(axis=(1, 2))
    position_error = np.full(len(estimate), np.nan)
    orientation_error = np.full(len(estimate), np.nan)
    pixel_error = np.full(len(estimate), np.nan)
    if valid.any():
        position_error[valid] = np.linalg.norm(
            estimate[valid, :3, 3] - reference[valid, :3, 3], axis=1
        )
        from .transforms import pose_rotation_errors_deg

        orientation_error[valid] = pose_rotation_errors_deg(estimate[valid], reference[valid])
        estimate_pixels, estimate_depth = camera.project(estimate[valid, :3, 3])
        reference_pixels, reference_depth = camera.project(reference[valid, :3, 3])
        pixel_valid = estimate_depth & reference_depth
        valid_indices = np.flatnonzero(valid)
        pixel_error[valid_indices[pixel_valid]] = np.linalg.norm(
            estimate_pixels[pixel_valid] - reference_pixels[pixel_valid], axis=1
        )
    position_values = position_error[np.isfinite(position_error)]
    pixel_values = pixel_error[np.isfinite(pixel_error)]
    orientation_values = orientation_error[np.isfinite(orientation_error)]
    return {
        "head_valid_count": int(valid.sum()),
        "head_valid_ratio": float(valid.mean()) if len(valid) else 0.0,
        "head_position_rmse_m": (
            float(np.sqrt(np.mean(position_values**2))) if len(position_values) else None
        ),
        "head_position_median_m": float(np.median(position_values)) if len(position_values) else None,
        "head_position_p95_m": float(np.percentile(position_values, 95)) if len(position_values) else None,
        "head_pixel_rmse_px": (
            float(np.sqrt(np.mean(pixel_values**2))) if len(pixel_values) else None
        ),
        "head_pixel_median_px": float(np.median(pixel_values)) if len(pixel_values) else None,
        "head_pixel_p95_px": float(np.percentile(pixel_values, 95)) if len(pixel_values) else None,
        "head_orientation_mean_deg": (
            float(np.mean(orientation_values)) if len(orientation_values) else None
        ),
        "head_orientation_median_deg": (
            float(np.median(orientation_values)) if len(orientation_values) else None
        ),
        "head_orientation_p95_deg": (
            float(np.percentile(orientation_values, 95)) if len(orientation_values) else None
        ),
        "head_position_error_m": position_error.tolist(),
        "head_pixel_error_px": pixel_error.tolist(),
        "head_orientation_error_deg": orientation_error.tolist(),
    }


def _render_comparison(
    output_path: Path,
    preview_path: Path,
    mappings,
    gt_display,
    estimate_display,
    confidence,
    metrics,
    camera,
    clip,
    fps,
    ffmpeg_path,
    frame_kind,
    head_mode,
    alignment_label,
    pose_valid,
    alignment_status,
) -> None:
    gt_history: list[np.ndarray] = []
    estimate_history: list[np.ndarray] = []
    with FFmpegWriter(output_path, 1920, 1080, fps, ffmpeg_path) as writer:
        for index, mapping in enumerate(mappings):
            ego = cv2.imread(str(mapping.ego_image))
            exo = cv2.imread(str(mapping.exo_image)) if mapping.exo_image else None
            if ego is None:
                ego = np.zeros((480, 640, 3), dtype=np.uint8)
            if exo is None:
                exo = np.zeros((1080, 1920, 3), dtype=np.uint8)
                draw_text(exo, "Missing synchronized exo frame", (60, 100), (0, 0, 255), 1.1, 2)
            gt_pose = gt_display[index]
            estimate_pose = estimate_display[index]
            if np.isfinite(gt_pose).all():
                gt_history.append(gt_pose)
            if np.isfinite(estimate_pose).all():
                estimate_history.append(estimate_pose)
            gt_overlay, gt_projected = draw_pose_overlay(
                exo, gt_pose, camera, GT_COLOR, frame_kind=frame_kind, history=gt_history[-40:]
            )
            da3_overlay, estimate_projected = draw_pose_overlay(
                exo,
                estimate_pose,
                camera,
                DA3_COLOR,
                frame_kind=frame_kind,
                history=estimate_history[-40:],
            )
            if not estimate_projected:
                draw_text(da3_overlay, "DA3 prediction unavailable", (50, 90), (0, 0, 255), 0.9, 2)
            if not gt_projected:
                draw_text(gt_overlay, "GT projection unavailable", (50, 90), (0, 0, 255), 0.9, 2)
            proxy_suffix = " Head Pose" if head_mode == "head_pose" else " Head Proxy"
            status = []
            conf = confidence[index]
            status.append(f"confidence={conf:.3f}" if np.isfinite(conf) else "confidence=unavailable")
            if not pose_valid[index]:
                status.append("LOW CONFIDENCE / INVALID — pose hidden")
            if mapping.exo_image is None:
                status.append("Missing synchronized exo frame")
            sampling_ratio = float(clip.get("ego_sampling_ratio", 1.0))
            if sampling_ratio < 1.0 - 1e-9:
                status.append(
                    f"Source sampling gaps={(1.0 - sampling_ratio):.1%}; no interpolation"
                )
            if alignment_status != "ok":
                status.append(f"Alignment unavailable: {alignment_status}")
            if metrics.get("position_error_m"):
                value = metrics["position_error_m"][index]
                status.append(f"position error={value:.3f} m" if np.isfinite(value) else "position error=N/A")
            if metrics.get("rotation_error_deg"):
                value = metrics["rotation_error_deg"][index]
                status.append(f"rotation error={value:.2f} deg" if np.isfinite(value) else "rotation error=N/A")
            if head_mode != "head_pose":
                status.append("Head proxy = ego camera center")
            frame = compose_triptych(
                ego,
                gt_overlay,
                da3_overlay,
                gt_title="Exo + GT" + proxy_suffix,
                da3_title="Exo + DA3" + proxy_suffix,
                sequence_label=f"{clip.get('difficulty', '')} {clip['recording_name']}",
                timestamp_label=f"frame={mapping.ego_frame_id:05d} timestamp={mapping.ego_timestamp}",
                alignment_label=alignment_label,
                status_lines=status,
            )
            if index == 0:
                cv2.imwrite(str(preview_path), frame)
            writer.write(frame)


def _da3_resume_matches(da3_dir: Path, records, config: dict) -> bool:
    npz_path = da3_dir / "da3_poses_raw.npz"
    json_path = da3_dir / "da3_poses_raw.json"
    resolved_path = da3_dir / "da3_resolved_config.json"
    if not (npz_path.is_file() and json_path.is_file() and resolved_path.is_file()):
        return False
    da3_config = config["da3"]
    checkpoint = Path(da3_config["checkpoint_path"]).resolve()
    weight = checkpoint / "model.safetensors"
    try:
        resolved = read_json(resolved_path)
        with np.load(npz_path) as data:
            frame_ids_match = np.array_equal(
                data["frame_ids"], np.asarray([record.frame_id for record in records])
            )
            timestamps_match = np.array_equal(
                data["timestamps"], np.asarray([record.timestamp for record in records])
            )
        expected_overlap = _effective_streaming_overlap(
            len(records),
            int(da3_config["window_size"]),
            int(da3_config["window_overlap"]),
        )
        expected = {
            "source_commit": EXPECTED_DA3_COMMIT,
            "checkpoint_status": "user_validated_local",
            "checkpoint_path": str(checkpoint),
            "checkpoint_weight_size": weight.stat().st_size,
            "checkpoint_weight_mtime_ns": weight.stat().st_mtime_ns,
            "use_ray_pose": True,
            "process_res": int(da3_config["input_resolution"]),
            "chunk_size": int(da3_config["window_size"]),
            "requested_overlap": int(da3_config["window_overlap"]),
            "effective_overlap": expected_overlap,
            "loop_closure": False,
            "confidence_threshold": float(da3_config["confidence_threshold"]),
            "input_count": len(records),
        }
    except (OSError, KeyError, ValueError):
        return False
    return frame_ids_match and timestamps_match and all(
        resolved.get(key) == value for key, value in expected.items()
    )


def _load_and_document_da3_poses(da3_dir: Path) -> dict[str, np.ndarray]:
    """Load official poses and expose the EgoBody-PV-basis interpretation.

    Existing GPU results predate the explicit ``c2w_egobody_pv`` key. They
    remain reusable: their official stitched matrices are preserved, and the
    fixed camera-basis change is applied during post-processing without
    inference.
    """

    npz_path = da3_dir / "da3_poses_raw.npz"
    with np.load(npz_path) as data:
        official_c2w = np.asarray(data["c2w"], dtype=np.float64)
        if "c2w_egobody_pv" in data.files:
            egobody_pv_c2w = np.asarray(
                data["c2w_egobody_pv"], dtype=np.float64
            )
            interpretation_source = "explicit_c2w_egobody_pv"
        elif "c2w_opencv" in data.files:
            # A short-lived adapter revision used this misleading key for the
            # same right-multiplied EgoBody-PV matrices. Accept it without
            # forcing an expensive inference rerun.
            egobody_pv_c2w = np.asarray(data["c2w_opencv"], dtype=np.float64)
            interpretation_source = "legacy_mislabeled_c2w_opencv"
        else:
            egobody_pv_c2w = da3_streaming_c2w_to_egobody_pv(official_c2w)
            interpretation_source = (
                "legacy_official_c2w_converted_during_postprocess"
            )
        result = {
            "official_c2w": official_c2w,
            "egobody_pv_c2w": egobody_pv_c2w,
            "confidence": np.asarray(data["confidence"]),
            "frame_ids": np.asarray(data["frame_ids"]),
            "timestamps": np.asarray(data["timestamps"]),
        }

    interpretation = {
        "official_pose_convention": OFFICIAL_STITCHED_POSE_CONVENTION,
        "downstream_pose_convention": POSE_CONVENTION,
        "camera_basis_change": {
            "operation": "right_multiply",
            "matrix": DA3_STREAMING_TO_EGOBODY_PV_CAMERA,
            "preserves_camera_centers": True,
        },
        "source": interpretation_source,
        "input_count": len(official_c2w),
    }
    write_json(da3_dir / "pose_basis_interpretation.json", interpretation)

    metadata_path = da3_dir / "da3_poses_raw.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        metadata.update(
            {
                "pose_convention": POSE_CONVENTION,
                "official_stitched_pose_convention": OFFICIAL_STITCHED_POSE_CONVENTION,
                "camera_basis_change_right_multiply": DA3_STREAMING_TO_EGOBODY_PV_CAMERA,
            }
        )
        records = metadata.get("records", [])
        if len(records) == len(official_c2w):
            for record, official_pose, egobody_pv_pose in zip(
                records, official_c2w, egobody_pv_c2w
            ):
                record.pop("opencv_c2w", None)
                record.pop("opencv_w2c", None)
                record.update(
                    {
                        "raw_extrinsics": invert_pose(official_pose),
                        "raw_extrinsics_convention": (
                            "world-to-camera inverse of official_stitched_c2w"
                        ),
                        "stitched_c2w": official_pose,
                        "stitched_c2w_convention": OFFICIAL_STITCHED_POSE_CONVENTION,
                        "egobody_pv_c2w": egobody_pv_pose,
                        "egobody_pv_w2c": invert_pose(egobody_pv_pose),
                        "pose_convention": POSE_CONVENTION,
                    }
                )
        write_json(metadata_path, metadata)
    return result


def run_clip(
    *,
    repo_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    clip: dict,
    config: dict,
    run_da3: bool,
    render_comparison: bool,
    evaluate: bool,
    resume: bool = False,
) -> dict:
    output = Path(output_dir)
    prepared = prepare_gt_clip(data_root, clip, output, config)
    records, mappings, gt, camera = (
        prepared["records"],
        prepared["mappings"],
        prepared["gt"],
        prepared["camera"],
    )
    da3_dir = output / "da3"
    if run_da3 and not (resume and _da3_resume_matches(da3_dir, records, config)):
        if da3_dir.exists():
            shutil.rmtree(da3_dir)
        da3_config = config["da3"]
        run_da3_streaming(
            repo_root=repo_root,
            source_root=da3_config["source_root"],
            checkpoint_path=da3_config["checkpoint_path"],
            image_paths=[record.image_path for record in records],
            frame_ids=[record.frame_id for record in records],
            timestamps=[record.timestamp for record in records],
            output_dir=da3_dir,
            process_res=int(da3_config["input_resolution"]),
            chunk_size=int(da3_config["window_size"]),
            overlap=int(da3_config["window_overlap"]),
            confidence_threshold=float(da3_config["confidence_threshold"]),
        )
    da3_npz = da3_dir / "da3_poses_raw.npz"
    if not da3_npz.is_file():
        return {"gt": prepared["report"], "da3": "not_run_on_cpu"}
    da3_data = _load_and_document_da3_poses(da3_dir)
    da3_c2w = da3_data["egobody_pv_c2w"]
    confidence = da3_data["confidence"]
    da3_frame_ids = da3_data["frame_ids"]
    da3_timestamps = da3_data["timestamps"]
    if not np.array_equal(da3_frame_ids, np.asarray([record.frame_id for record in records])):
        raise RuntimeError("DA3 frame IDs do not match the selected EgoBody inputs")
    if not np.array_equal(da3_timestamps, np.asarray([record.timestamp for record in records])):
        raise RuntimeError("DA3 timestamps do not match the selected EgoBody inputs")
    valid = np.isfinite(da3_c2w).all(axis=(1, 2)) & np.isfinite(confidence) & (
        confidence >= float(config["da3"]["confidence_threshold"])
    )
    aligned = _align_da3(da3_c2w, valid, gt, config)
    all_metrics = {}
    for name, result in aligned.items():
        aligned_camera = result["T_K_E"]
        base_metrics = (
            trajectory_metrics(aligned_camera, gt["T_K_E"], gt["timestamps_sec"], confidence)
            if evaluate
            else {}
        )
        display_estimate = (
            camera_to_head_poses(aligned_camera, gt["head_calibration"].T_E_Q_fixed)
            if gt["head_mode"] == "head_pose" and result["alignment"].transform is not None
            else aligned_camera
        )
        result["display"] = display_estimate
        display_pixels, display_depth = camera.project(display_estimate[:, :3, 3])
        display_inside = display_depth & camera.inside(
            display_pixels, camera.width or 1920, camera.height or 1080
        )
        base_metrics.update(
            {
                "alignment_status": result["alignment"].status,
                "alignment_reason": result["alignment"].reason,
                "alignment_scale": (
                    result["alignment"].transform.scale
                    if result["alignment"].transform is not None
                    else None
                ),
                "finite_pose_ratio_before_confidence": float(
                    np.isfinite(da3_c2w).all(axis=(1, 2)).mean()
                ),
                "low_confidence_ratio": float(
                    (~np.isfinite(confidence) | (confidence < float(config["da3"]["confidence_threshold"]))).mean()
                ),
                "interpolated_ratio": 0.0,
                "synchronized_exo_ratio": float(
                    np.mean([mapping.exo_image is not None for mapping in mappings])
                ),
                "projected_inside_ratio": float(display_inside.mean()),
            }
        )
        if evaluate and gt["head_mode"] == "head_pose":
            base_metrics.update(_head_pose_metrics(display_estimate, gt["T_K_Q"], camera))
        all_metrics[name] = base_metrics
        write_json(output / f"alignment_{name}.json", result["alignment"])
        write_json(
            output / f"da3_poses_aligned_{name}.json",
            _pose_records(records, aligned_camera, f"T_K_E after {name}"),
        )
        _plot_trajectory(output / f"trajectory_{name}.png", gt["T_K_E"], aligned_camera, name)
    write_json(output / "metrics.json", all_metrics)
    if render_comparison:
        requested_fps = float(
            clip.get(
                "runtime_sample_fps", clip.get("sample_fps", config["da3"]["sample_fps"])
            )
        )
        window_duration = float(
            clip.get("runtime_duration_sec", clip.get("duration_sec", len(records) / requested_fps))
        )
        fps = len(records) / window_duration
        video_manifest = {}
        for name, filename, label in (
            ("sim3_prefix", "comparison_prefix.mp4", "Calibration-prefix alignment"),
            ("sim3_full", "comparison_oracle.mp4", "Oracle full-trajectory alignment"),
        ):
            _render_comparison(
                output / filename,
                output / filename.replace(".mp4", "_preview.jpg"),
                mappings,
                gt["display_gt"],
                aligned[name]["display"],
                confidence,
                all_metrics[name],
                camera,
                clip,
                fps,
                config["runtime"]["ffmpeg_path"],
                gt["frame_kind"],
                gt["head_mode"],
                f"{label} [{aligned[name]['alignment'].status}]",
                valid,
                aligned[name]["alignment"].status,
            )
            video_manifest[name] = {
                "video": str(output / filename),
                "preview": str(output / filename.replace(".mp4", "_preview.jpg")),
                "ffprobe": verify_video(
                    output / filename,
                    config["runtime"]["ffprobe_path"],
                    expected_frames=len(records),
                    expected_fps=fps,
                ),
            }
        write_json(output / "video_manifest.json", video_manifest)
    mapping_data = read_json(output / "frame_mapping.json")
    for index, row in enumerate(mapping_data):
        row["da3_pose_valid"] = bool(valid[index])
        row["da3_pose_interpolated"] = False
        row["da3_confidence"] = float(confidence[index]) if np.isfinite(confidence[index]) else None
    write_json(output / "frame_mapping.json", mapping_data)
    return {"gt": prepared["report"], "metrics": all_metrics}
