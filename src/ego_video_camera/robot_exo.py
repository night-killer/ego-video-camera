from __future__ import annotations

import bisect
import csv
import json
import os
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .eval_dataset_download import (
    DatasetDownloadError,
    DownloadContext,
    _checked_download,
    _normalized_tar_name,
    _remove_owned_eval_cache,
    _rh20t_load_dict,
    _rh20t_pose_matrix,
    _sha256,
    _write_json,
    download_google_drive_ranges,
    download_rh20t_wrist,
)
from .transforms import invert_pose


ROBOT_DEMO_DATASETS = ("droid_wrist", "rh20t_wrist")


def robot_dataset_demo_clips(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Return demo-specific clips without changing the base evaluation subset."""

    clips = dataset.get("demo_exo", {}).get("clips", dataset.get("clips", []))
    if not isinstance(clips, list) or not clips:
        raise DatasetDownloadError("Robot demo clip selection is empty")
    if any(not isinstance(clip, dict) or "sequence" not in clip for clip in clips):
        raise DatasetDownloadError("Robot demo clip selection is malformed")
    return clips


def droid_pose_matrix(values: Sequence[float]) -> np.ndarray:
    """Convert DROID xyz + intrinsic-xyz Euler pose to camera-to-base."""

    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise DatasetDownloadError("DROID camera pose must be finite xyz+Euler-xyz")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def nearest_timestamp_indices(
    source_timestamps: Sequence[int],
    target_timestamps: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_timestamps, dtype=np.int64)
    targets = np.asarray(target_timestamps, dtype=np.int64)
    if source.ndim != 1 or len(source) == 0:
        raise DatasetDownloadError("Exo timestamp stream is empty")
    if len(source) > 1 and np.any(np.diff(source) <= 0):
        raise DatasetDownloadError("Exo timestamps are not strictly increasing")
    indices = np.empty(len(targets), dtype=np.int64)
    for output_index, target in enumerate(targets):
        position = bisect.bisect_left(source, int(target))
        candidates = [min(position, len(source) - 1)]
        if position:
            candidates.append(position - 1)
        indices[output_index] = min(
            candidates, key=lambda index: abs(int(source[index]) - int(target))
        )
    return indices, source[indices] - targets


def projection_quality(
    reference_from_ego: np.ndarray,
    reference_from_exo: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    synchronized: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    ego = np.asarray(reference_from_ego, dtype=np.float64)
    exo = np.asarray(reference_from_exo, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if ego.shape != exo.shape or ego.ndim != 3 or ego.shape[1:] != (4, 4):
        raise ValueError("Ego and exo poses must be matching Nx4x4 arrays")
    if intrinsic.shape != (3, 3):
        raise ValueError("Camera intrinsic matrix must be 3x3")
    sync = (
        np.ones(len(ego), dtype=bool)
        if synchronized is None
        else np.asarray(synchronized, dtype=bool)
    )
    exo_from_reference = np.asarray([invert_pose(pose) for pose in exo])
    centers_reference = ego[:, :3, 3]
    centers_exo = np.einsum(
        "nij,nj->ni",
        exo_from_reference[:, :3, :3],
        centers_reference,
    ) + exo_from_reference[:, :3, 3]
    finite = np.isfinite(centers_exo).all(axis=1)
    z_valid = finite & (centers_exo[:, 2] > 1e-6)
    pixels = np.full((len(ego), 2), np.nan, dtype=np.float64)
    if z_valid.any():
        homogeneous = (intrinsic @ centers_exo[z_valid].T).T
        pixels[z_valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    inside = (
        sync
        & z_valid
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    margins = np.minimum.reduce(
        [
            pixels[:, 0],
            width - 1 - pixels[:, 0],
            pixels[:, 1],
            height - 1 - pixels[:, 1],
        ]
    )
    finite_margins = margins[np.isfinite(margins)]
    return {
        "frame_count": len(ego),
        "synchronized_count": int(sync.sum()),
        "synchronized_ratio": float(sync.mean()) if len(sync) else 0.0,
        "z_valid_ratio": float((sync & z_valid).mean()) if len(sync) else 0.0,
        "inside_count": int(inside.sum()),
        "inside_ratio": float(inside.mean()) if len(inside) else 0.0,
        "median_border_margin_px": (
            float(np.median(finite_margins)) if len(finite_margins) else None
        ),
        "minimum_depth_m": (
            float(np.min(centers_exo[z_valid, 2])) if z_valid.any() else None
        ),
        "maximum_depth_m": (
            float(np.max(centers_exo[z_valid, 2])) if z_valid.any() else None
        ),
    }


def select_droid_exo_candidate(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise DatasetDownloadError("No DROID exterior-camera candidates are available")
    return max(
        candidates,
        key=lambda item: (
            float(item["quality"]["inside_ratio"]),
            float(item["quality"].get("median_border_margin_px") or -1e12),
            str(item["serial"]),
        ),
    )


def exo_quality_passes(
    quality: dict[str, Any],
    minimum_synchronized_ratio: float = 0.95,
    minimum_projection_inside_ratio: float = 0.70,
) -> bool:
    return bool(
        float(quality["synchronized_ratio"]) >= minimum_synchronized_ratio
        and float(quality["inside_ratio"]) >= minimum_projection_inside_ratio
    )


def scale_camera_intrinsics(
    intrinsic: np.ndarray,
    source_width: int,
    source_height: int,
    decoded_width: int,
    decoded_height: int,
) -> tuple[np.ndarray, float, float]:
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Camera intrinsic matrix must be 3x3")
    if min(source_width, source_height, decoded_width, decoded_height) <= 0:
        raise ValueError("Camera source and decoded dimensions must be positive")
    scale_x = decoded_width / source_width
    scale_y = decoded_height / source_height
    scaled = matrix.copy()
    scaled[0] *= scale_x
    scaled[1] *= scale_y
    return scaled, scale_x, scale_y


def rh20t_aligned_extrinsics(
    calibration_dir: str | Path,
    camera_config: dict[str, Any],
    wrist_serial: str,
) -> dict[str, np.ndarray]:
    """Return official T_camera_aligned_base for every RH20T camera."""

    calibration = Path(calibration_dir)
    extrinsics = _rh20t_load_dict(
        calibration / "extrinsics.npy", "camera extrinsics"
    )
    try:
        wrist_extrinsic = np.asarray(extrinsics[wrist_serial], dtype=np.float64).squeeze()
        calibration_tcp = np.asarray(
            np.load(calibration / "tcp.npy", allow_pickle=False), dtype=np.float64
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise DatasetDownloadError("Malformed RH20T exo calibration") from error
    tcp_camera = np.asarray(camera_config["tcp_camera_matrix"], dtype=np.float64)
    align_base = np.asarray(camera_config["align_base_matrix"], dtype=np.float64)
    if wrist_extrinsic.shape != (4, 4):
        raise DatasetDownloadError("RH20T wrist extrinsic must be 4x4")
    base_world = (
        np.linalg.inv(wrist_extrinsic)
        @ tcp_camera
        @ np.linalg.inv(_rh20t_pose_matrix(calibration_tcp))
    )
    result: dict[str, np.ndarray] = {}
    for serial, value in extrinsics.items():
        extrinsic = np.asarray(value, dtype=np.float64).squeeze()
        if extrinsic.shape != (4, 4):
            raise DatasetDownloadError(
                f"RH20T extrinsic for {serial} must be 4x4"
            )
        result[str(serial)] = extrinsic @ base_world @ align_base
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, ValueError) as error:
        raise DatasetDownloadError(f"Unable to read CSV: {path}") from error


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return path


def _validate_images(paths: Sequence[Path]) -> tuple[int, int]:
    if not paths:
        raise DatasetDownloadError("No synchronized exo images were extracted")
    size: tuple[int, int] | None = None
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                current = image.size
                if image.mode not in {"RGB", "RGBA"}:
                    raise DatasetDownloadError(f"Exo image is not RGB: {path}")
        except (OSError, ValueError) as error:
            raise DatasetDownloadError(f"Invalid exo image: {path}") from error
        if size is None:
            size = current
        elif current != size:
            raise DatasetDownloadError("Exo frame dimensions are inconsistent")
    assert size is not None
    return size


def _extract_indexed_frames(
    source: Path,
    destination: Path,
    source_indices: Sequence[int],
    output_indices: Sequence[int],
    ffmpeg: str,
    data_root: Path,
) -> list[Path]:
    if len(source_indices) != len(output_indices):
        raise ValueError("Source and output frame indices must have equal lengths")
    expected = [
        destination / f"{output:06d}_source_{source_index:06d}.png"
        for output, source_index in zip(output_indices, source_indices)
    ]
    if (
        destination.is_dir()
        and len([path for path in destination.iterdir() if path.is_file()])
        == len(expected)
        and all(path.is_file() and path.stat().st_size > 0 for path in expected)
    ):
        _validate_images(expected)
        return expected
    staging = destination.parent / ".frames-staging"
    if staging.exists():
        _remove_owned_eval_cache(staging, data_root)
    staging.mkdir(parents=True)
    unique_indices = list(dict.fromkeys(int(value) for value in source_indices))
    expression = "+".join(f"eq(n\\,{index})" for index in unique_indices)
    import subprocess

    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            f"select={expression}",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            str(staging / "decoded_%06d.png"),
        ],
        check=True,
    )
    decoded = sorted(staging.glob("decoded_*.png"))
    if len(decoded) != len(unique_indices):
        raise DatasetDownloadError(
            f"Exo video decode returned {len(decoded)}/{len(unique_indices)} frames"
        )
    decoded_by_index: dict[int, Path] = {}
    for path, source_index in zip(decoded, unique_indices):
        indexed = staging / f"source_{source_index:06d}.png"
        os.replace(path, indexed)
        decoded_by_index[source_index] = indexed
    staged: list[Path] = []
    for output, source_index in zip(output_indices, source_indices):
        target = staging / f"{output:06d}_source_{source_index:06d}.png"
        shutil.copyfile(decoded_by_index[int(source_index)], target)
        staged.append(target)
    for path in decoded_by_index.values():
        path.unlink()
    _validate_images(staged)
    if destination.exists():
        _remove_owned_eval_cache(destination, data_root)
    os.replace(staging, destination)
    return expected


def _matrix_from_csv_row(row: dict[str, str]) -> np.ndarray:
    try:
        values = [float(row[f"m{i}{j}"]) for i in range(4) for j in range(4)]
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetDownloadError("Malformed 4x4 pose CSV row") from error
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _write_exo_artifacts(
    *,
    clip_dir: Path,
    dataset: str,
    sequence: str,
    reference_type: str,
    ego_timestamps: np.ndarray,
    exo_timestamps: np.ndarray,
    deltas_ms: np.ndarray,
    source_indices: np.ndarray,
    synchronized: np.ndarray,
    reference_from_exo: np.ndarray,
    intrinsic_source: np.ndarray,
    source_width: int,
    source_height: int,
    distortion: Sequence[float],
    camera_metadata: dict[str, Any],
    source_video: Path,
    ffmpeg: str,
    data_root: Path,
    quality: dict[str, Any],
    source_info: dict[str, Any],
) -> dict[str, Any]:
    exo_dir = clip_dir / "exo"
    valid_outputs = np.flatnonzero(synchronized)
    images = _extract_indexed_frames(
        source_video,
        exo_dir / "frames",
        source_indices[valid_outputs].tolist(),
        valid_outputs.tolist(),
        ffmpeg,
        data_root,
    )
    width, height = _validate_images(images)
    intrinsic, scale_x, scale_y = scale_camera_intrinsics(
        intrinsic_source,
        source_width,
        source_height,
        width,
        height,
    )
    camera = {
        **camera_metadata,
        "matrix": intrinsic.tolist(),
        "distortion_coefficients": [float(value) for value in distortion],
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
        "intrinsics_scale_x": scale_x,
        "intrinsics_scale_y": scale_y,
        "projection_model": "pinhole",
        "fisheye": False,
    }
    camera_path = exo_dir / "camera.json"
    _write_json(camera_path, camera)
    filename_by_output = {
        output: path.name for output, path in zip(valid_outputs.tolist(), images)
    }
    frame_rows = []
    pose_rows = []
    matrix_fields = [f"m{i}{j}" for i in range(4) for j in range(4)]
    for index in range(len(ego_timestamps)):
        frame_rows.append(
            {
                "output_index": index,
                "ego_timestamp_ms": int(ego_timestamps[index]),
                "exo_timestamp_ms": (
                    int(exo_timestamps[index]) if synchronized[index] else ""
                ),
                "delta_ms": int(deltas_ms[index]) if synchronized[index] else "",
                "source_frame_index": (
                    int(source_indices[index]) if synchronized[index] else ""
                ),
                "filename": filename_by_output.get(index, ""),
                "synchronized": int(synchronized[index]),
            }
        )
        pose_row: dict[str, Any] = {
            "output_index": index,
            "source_frame_index": (
                int(source_indices[index]) if synchronized[index] else ""
            ),
            "ego_timestamp_ms": int(ego_timestamps[index]),
            "exo_timestamp_ms": (
                int(exo_timestamps[index]) if synchronized[index] else ""
            ),
            "delta_ms": int(deltas_ms[index]) if synchronized[index] else "",
            "valid": int(synchronized[index]),
        }
        for field, value in zip(matrix_fields, reference_from_exo[index].reshape(-1)):
            pose_row[field] = f"{float(value):.12g}" if synchronized[index] else ""
        pose_rows.append(pose_row)
    frames_csv = _atomic_write_csv(
        exo_dir / "frames.csv",
        (
            "output_index",
            "ego_timestamp_ms",
            "exo_timestamp_ms",
            "delta_ms",
            "source_frame_index",
            "filename",
            "synchronized",
        ),
        frame_rows,
    )
    poses_csv = _atomic_write_csv(
        exo_dir / "camera_to_reference.csv",
        (
            "output_index",
            "source_frame_index",
            "ego_timestamp_ms",
            "exo_timestamp_ms",
            "delta_ms",
            "valid",
            *matrix_fields,
        ),
        pose_rows,
    )
    frame_manifest = _atomic_write_csv(
        exo_dir / "frame_manifest.csv",
        ("output_index", "filename", "bytes", "sha256"),
        (
            {
                "output_index": output,
                "filename": filename_by_output[output],
                "bytes": (exo_dir / "frames" / filename_by_output[output]).stat().st_size,
                "sha256": _sha256(exo_dir / "frames" / filename_by_output[output]),
            }
            for output in sorted(filename_by_output)
        ),
    )
    recorded_files = []
    for path in (frames_csv, poses_csv, frame_manifest, camera_path):
        recorded_files.append(
            {
                "path": str(path.relative_to(clip_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "dataset": dataset,
        "sequence_id": sequence,
        "reference_type": reference_type,
        "frame_count": len(ego_timestamps),
        "synchronized_count": int(synchronized.sum()),
        "synchronized_ratio": float(synchronized.mean()),
        "camera": camera,
        "quality": quality,
        "source": source_info,
        "files": recorded_files,
    }
    _write_json(exo_dir / "manifest.json", manifest)
    return manifest


def _verify_recorded_files(clip_dir: Path, manifest: dict[str, Any]) -> str | None:
    if manifest.get("status") != "ready":
        return "exo manifest is not ready"
    for item in manifest.get("files", []):
        try:
            relative = Path(str(item["path"]))
            expected_size = int(item["bytes"])
            expected_hash = str(item["sha256"])
        except (KeyError, TypeError, ValueError):
            return "malformed exo file record"
        if relative.is_absolute() or ".." in relative.parts:
            return "unsafe exo file path"
        path = clip_dir / relative
        if not path.is_file() or path.stat().st_size != expected_size:
            return f"missing or changed exo file: {relative}"
        if _sha256(path) != expected_hash:
            return f"exo file SHA-256 mismatch: {relative}"
    try:
        frame_rows = _read_csv(clip_dir / "exo" / "frame_manifest.csv")
    except DatasetDownloadError as error:
        return str(error)
    for row in frame_rows:
        filename = str(row.get("filename", ""))
        if not filename or Path(filename).name != filename:
            return "unsafe exo frame filename"
        path = clip_dir / "exo" / "frames" / filename
        try:
            expected_size = int(row["bytes"])
            expected_hash = str(row["sha256"])
        except (KeyError, TypeError, ValueError):
            return "malformed exo frame record"
        if not path.is_file() or path.stat().st_size != expected_size:
            return f"missing or changed exo frame: {filename}"
        if _sha256(path) != expected_hash:
            return f"exo frame SHA-256 mismatch: {filename}"
    return None


def exo_clip_status(clip_dir: str | Path) -> tuple[str, dict[str, Any] | None, str | None]:
    clip = Path(clip_dir)
    manifest_path = clip / "exo" / "manifest.json"
    exclusion_path = clip / "exo" / "exclusion.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "invalid", None, "unable to parse exo manifest"
        reason = _verify_recorded_files(clip, manifest)
        return ("ready", manifest, None) if reason is None else ("invalid", manifest, reason)
    if exclusion_path.is_file():
        try:
            exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "invalid", None, "unable to parse exo exclusion"
        if exclusion.get("status") == "excluded":
            return "excluded", exclusion, None
        return "invalid", exclusion, "malformed exo exclusion"
    return "missing", None, "exo artifacts are missing"


def _droid_reference_poses(clip_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_csv(clip_dir / "reference" / "camera_to_robot_base.csv")
    timestamps = []
    poses = []
    for row in rows:
        timestamps.append(int(row["estimated_capture_ms"]))
        poses.append(
            droid_pose_matrix(
                [
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                    float(row["rx_xyz_rad"]),
                    float(row["ry_xyz_rad"]),
                    float(row["rz_xyz_rad"]),
                ]
            )
        )
    return np.asarray(timestamps, dtype=np.int64), np.asarray(poses)


def _prepare_droid_exo(
    plan: dict[str, Any],
    data_root: Path,
    ffmpeg: str,
    keep_source: bool,
    sequence_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["droid_wrist"]
    demo = dataset["demo_exo"]
    # Share the base downloader cache so trajectory/intrinsics are reused when
    # --robot-with-exo is part of the same download invocation.
    cache_root = data_root / "_cache" / "droid_wrist"
    pending = []
    results: list[dict[str, Any]] = []
    selected_clips = [
        clip
        for clip in robot_dataset_demo_clips(dataset)
        if sequence_ids is None or str(clip["sequence"]) in sequence_ids
    ]
    for clip in selected_clips:
        clip_dir = data_root / "droid_wrist" / "clips" / str(clip["sequence"])
        status, payload, _ = exo_clip_status(clip_dir)
        expected_excluded = bool(
            demo["sequences"][str(clip["sequence"])].get("excluded", False)
        )
        if (expected_excluded and status == "excluded") or (
            not expected_excluded and status == "ready"
        ):
            assert payload is not None
            results.append(payload)
        else:
            pending.append(clip)
    if not pending:
        return results
    intrinsics_path = _checked_download(
        str(dataset["intrinsics_url"]),
        cache_root / "intrinsics.json",
        int(dataset["intrinsics_bytes"]),
    )
    if _sha256(intrinsics_path) != str(dataset["intrinsics_sha256"]):
        intrinsics_path.unlink(missing_ok=True)
        raise DatasetDownloadError("DROID intrinsics annotation SHA-256 mismatch")
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    for clip in pending:
        sequence = str(clip["sequence"])
        clip_dir = data_root / "droid_wrist" / "clips" / sequence
        metadata = json.loads(
            (clip_dir / "reference" / "metadata.json").read_text(encoding="utf-8")
        )
        source_dir = cache_root / sequence
        trajectory_url = str(json.loads((clip_dir / "clip.json").read_text(encoding="utf-8"))["source"]["trajectory_url"])
        trajectory = _checked_download(
            trajectory_url,
            source_dir / "trajectory.h5",
            int(clip["trajectory_bytes"]),
        )
        ego_timestamps, reference_from_ego = _droid_reference_poses(clip_dir)
        reference_centers = reference_from_ego[:, :3, 3]
        trajectory_span_m = float(
            np.max(
                np.linalg.norm(
                    reference_centers[:, None] - reference_centers[None, :], axis=-1
                )
            )
        )
        candidates = []
        with h5py.File(trajectory, "r") as archive:
            for label in ("ext1", "ext2"):
                serial = str(metadata[f"{label}_cam_serial"])
                timestamp_key = f"observation/timestamp/cameras/{serial}_estimated_capture"
                pose_key = f"observation/camera_extrinsics/{serial}_left"
                if timestamp_key not in archive or pose_key not in archive:
                    raise DatasetDownloadError(
                        f"DROID exo datasets are missing for {sequence}/{serial}"
                    )
                source_timestamps = np.asarray(archive[timestamp_key][:], dtype=np.int64)
                indices, deltas = nearest_timestamp_indices(
                    source_timestamps, ego_timestamps
                )
                synchronized = np.abs(deltas) <= int(demo["sync_tolerance_ms"])
                poses = np.asarray(
                    [droid_pose_matrix(value) for value in archive[pose_key][indices]]
                )
                calibration = intrinsics[sequence][serial]
                fx, cx, fy, cy = [float(value) for value in calibration["cameraMatrix"]]
                intrinsic = np.asarray(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
                )
                width = int(calibration.get("width", 1280))
                height = int(calibration.get("height", 720))
                quality = projection_quality(
                    reference_from_ego,
                    poses,
                    intrinsic,
                    width,
                    height,
                    synchronized,
                )
                quality.update(
                    {
                        "ego_translation_span_m": trajectory_span_m,
                        "maximum_absolute_sync_delta_ms": int(
                            np.max(np.abs(deltas))
                        ),
                        "median_absolute_sync_delta_ms": float(
                            np.median(np.abs(deltas))
                        ),
                    }
                )
                candidates.append(
                    {
                        "serial": serial,
                        "source_indices": indices,
                        "exo_timestamps": source_timestamps[indices],
                        "deltas": deltas,
                        "synchronized": synchronized,
                        "poses": poses,
                        "intrinsic": intrinsic,
                        "width": width,
                        "height": height,
                        "distortion": calibration.get("distCoeffs", []),
                        "quality": quality,
                    }
                )
        selected = select_droid_exo_candidate(candidates)
        expected = demo["sequences"][sequence]
        if selected["serial"] != str(expected["serial"]):
            raise DatasetDownloadError(
                f"DROID exo selection drift for {sequence}: "
                f"{selected['serial']} != {expected['serial']}"
            )
        candidate_report = {
            item["serial"]: item["quality"] for item in candidates
        }
        passes = exo_quality_passes(
            selected["quality"],
            float(demo["minimum_synchronized_ratio"]),
            float(demo["minimum_projection_inside_ratio"]),
        )
        if bool(expected.get("excluded", False)):
            if passes:
                raise DatasetDownloadError(
                    f"DROID exclusion gate unexpectedly passed: {sequence}"
                )
            maximum_span = float(expected["maximum_translation_span_m"])
            exclusion_checks = {
                "projection_inside_ratio_zero": float(
                    selected["quality"]["inside_ratio"]
                )
                == 0.0,
                "trajectory_degenerate": trajectory_span_m <= maximum_span,
                "maximum_translation_span_m": maximum_span,
            }
            if not all(
                exclusion_checks[key]
                for key in (
                    "projection_inside_ratio_zero",
                    "trajectory_degenerate",
                )
            ):
                raise DatasetDownloadError(
                    f"DROID exclusion evidence drifted for {sequence}: "
                    f"{exclusion_checks}"
                )
            exclusion = {
                "schema_version": 1,
                "status": "excluded",
                "dataset": "droid_wrist",
                "sequence_id": sequence,
                "selected_exo_serial": selected["serial"],
                "reason": str(expected["reason"]),
                "quality": selected["quality"],
                "candidate_quality": candidate_report,
                "exclusion_checks": exclusion_checks,
            }
            (clip_dir / "exo" / "manifest.json").unlink(missing_ok=True)
            _write_json(clip_dir / "exo" / "exclusion.json", exclusion)
            results.append(exclusion)
        else:
            if not passes:
                raise DatasetDownloadError(
                    f"DROID exo quality gate failed for {sequence}: "
                    f"{selected['quality']}"
                )
            clip_record = json.loads(
                (clip_dir / "clip.json").read_text(encoding="utf-8")
            )
            video_url = (
                str(clip_record["source"]["video_url"]).rsplit("/", 1)[0]
                + f"/{selected['serial']}.mp4"
            )
            video = _checked_download(
                video_url,
                source_dir / f"{selected['serial']}.mp4",
                int(expected["video_bytes"]),
            )
            manifest = _write_exo_artifacts(
                clip_dir=clip_dir,
                dataset="droid_wrist",
                sequence=sequence,
                reference_type=str(dataset["reference_type"]),
                ego_timestamps=ego_timestamps,
                exo_timestamps=np.asarray(selected["exo_timestamps"]),
                deltas_ms=np.asarray(selected["deltas"]),
                source_indices=np.asarray(selected["source_indices"]),
                synchronized=np.asarray(selected["synchronized"]),
                reference_from_exo=np.asarray(selected["poses"]),
                intrinsic_source=np.asarray(selected["intrinsic"]),
                source_width=int(selected["width"]),
                source_height=int(selected["height"]),
                distortion=selected["distortion"],
                camera_metadata={
                    "stream": "exterior_left",
                    "serial": selected["serial"],
                    "pose_direction": "camera_to_robot_base",
                    "distortion_model": "rectified_zero_distortion",
                    "camera_basis": "opencv_right_down_forward",
                },
                source_video=video,
                ffmpeg=ffmpeg,
                data_root=data_root,
                quality={
                    **selected["quality"],
                    "candidate_quality": candidate_report,
                },
                source_info={
                    "video_url": video_url,
                    "video_bytes": video.stat().st_size,
                    "video_sha256": _sha256(video),
                    "trajectory_url": trajectory_url,
                    "trajectory_sha256": _sha256(trajectory),
                    "retained_episode_sources": keep_source,
                },
            )
            (clip_dir / "exo" / "exclusion.json").unlink(missing_ok=True)
            results.append(manifest)
        if not keep_source and source_dir.exists():
            _remove_owned_eval_cache(source_dir, data_root)
    if not keep_source and cache_root.exists():
        _remove_owned_eval_cache(cache_root, data_root)
    order = {str(clip["sequence"]): index for index, clip in enumerate(selected_clips)}
    return sorted(results, key=lambda item: order[str(item["sequence_id"])])


def _rh20t_reference_poses(clip_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_csv(clip_dir / "reference" / "camera_to_aligned_robot_base.csv")
    timestamps = np.asarray([int(row["timestamp_ms"]) for row in rows], dtype=np.int64)
    poses = np.asarray([_matrix_from_csv_row(row) for row in rows])
    return timestamps, poses


def _stage_rh20t_exo_members(
    archive_path: Path,
    stage_root: Path,
    required: set[str],
    data_root: Path,
) -> None:
    missing = {
        name
        for name in required
        if not (stage_root / PurePosixPath(name)).is_file()
        or (stage_root / PurePosixPath(name)).stat().st_size <= 0
    }
    if not missing:
        return
    stage_root.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                name = _normalized_tar_name(member.name)
                if name not in missing:
                    continue
                if not member.isfile() or member.size <= 0:
                    raise DatasetDownloadError(
                        f"RH20T exo member is not a regular file: {name}"
                    )
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise DatasetDownloadError(f"Unsafe RH20T exo member: {name}")
                source = archive.extractfile(member)
                if source is None:
                    raise DatasetDownloadError(f"Unable to extract RH20T exo member: {name}")
                destination = stage_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + ".part")
                with source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if partial.stat().st_size != member.size:
                    partial.unlink(missing_ok=True)
                    raise DatasetDownloadError(f"Short RH20T exo member: {name}")
                os.replace(partial, destination)
                found.add(name)
                if found == missing:
                    break
    except (OSError, tarfile.TarError) as error:
        raise DatasetDownloadError("Unable to stream RH20T exo members") from error
    remaining = {
        name
        for name in required
        if not (stage_root / PurePosixPath(name)).is_file()
        or (stage_root / PurePosixPath(name)).stat().st_size <= 0
    }
    if remaining:
        raise DatasetDownloadError(
            f"RH20T archive lacks exo members: {sorted(remaining)}"
        )


def _ensure_rh20t_demo_base(
    plan: dict[str, Any],
    data_root: Path,
    ffmpeg: str,
    workers: int,
    keep_source: bool,
    rh20t_archive: Path | None,
    selected_clips: list[dict[str, Any]],
) -> None:
    dataset = plan["datasets"]["rh20t_wrist"]
    if "clips" not in dataset.get("demo_exo", {}):
        return
    demo_plan = deepcopy(plan)
    demo_plan["datasets"]["rh20t_wrist"]["clips"] = deepcopy(selected_clips)
    context = DownloadContext(
        plan=demo_plan,
        data_root=data_root,
        target_fps=int(plan["profile"]["target_fps"]),
        workers=max(1, int(workers)),
        keep_source=keep_source,
        ffmpeg=ffmpeg,
        aria_mode="preview",
        accept_aria_licenses=False,
        egobody_netrc_file=None,
        accept_egobody_license=False,
        egobody_with_exo=False,
        adt_cdn_file=None,
        hot3d_cdn_file=None,
        hot3d_downloader=None,
        rh20t_archive=rh20t_archive,
        archive_tool="bsdtar",
        robot_with_exo=True,
    )
    records = download_rh20t_wrist(context)
    completed = {str(record["sequence_id"]) for record in records}
    expected = {str(clip["sequence"]) for clip in selected_clips}
    if completed != expected:
        raise DatasetDownloadError(
            f"RH20T demo base preparation returned {sorted(completed)}, "
            f"expected {sorted(expected)}"
        )


def _prepare_rh20t_exo(
    plan: dict[str, Any],
    data_root: Path,
    ffmpeg: str,
    workers: int,
    keep_source: bool,
    rh20t_archive: Path | None,
    sequence_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["rh20t_wrist"]
    demo = dataset["demo_exo"]
    serial = str(demo["serial"])
    pending = []
    results: list[dict[str, Any]] = []
    selected_clips = [
        clip
        for clip in robot_dataset_demo_clips(dataset)
        if sequence_ids is None or str(clip["sequence"]) in sequence_ids
    ]
    _ensure_rh20t_demo_base(
        plan,
        data_root,
        ffmpeg,
        workers,
        keep_source,
        rh20t_archive,
        selected_clips,
    )
    for clip in selected_clips:
        clip_dir = data_root / "rh20t_wrist" / "clips" / str(clip["sequence"])
        status, payload, _ = exo_clip_status(clip_dir)
        if status == "ready":
            assert payload is not None
            results.append(payload)
        else:
            pending.append(clip)
    if not pending:
        return results
    # Share the 27.4 GB archive and staged members with the base RH20T subset.
    cache_root = data_root / "_cache" / "rh20t"
    caller_owned = rh20t_archive is not None
    if rh20t_archive is not None:
        archive_path = Path(rh20t_archive).resolve()
        if not archive_path.is_file():
            raise DatasetDownloadError(f"RH20T archive does not exist: {archive_path}")
        if archive_path.stat().st_size != int(dataset["archive_bytes"]):
            raise DatasetDownloadError("RH20T caller archive size mismatch")
        if _sha256(archive_path) != str(dataset["archive_sha256"]):
            raise DatasetDownloadError("RH20T caller archive SHA-256 mismatch")
    else:
        archive_path = download_google_drive_ranges(
            str(dataset["google_drive_id"]),
            cache_root / str(dataset["archive_name"]),
            int(dataset["archive_bytes"]),
            workers=workers,
            mirror_url=str(dataset["mirror_url"]),
            expected_sha256=str(dataset["archive_sha256"]),
        )
    stage_root = cache_root / "extracted"
    required = set()
    for clip in pending:
        base = f"{dataset['archive_root']}/{clip['sequence']}/cam_{serial}"
        required.update({f"{base}/color.mp4", f"{base}/timestamps.npy"})
    _stage_rh20t_exo_members(archive_path, stage_root, required, data_root)
    try:
        for clip in pending:
            sequence = str(clip["sequence"])
            clip_dir = data_root / "rh20t_wrist" / "clips" / sequence
            base = stage_root / str(dataset["archive_root"]) / sequence / f"cam_{serial}"
            video = base / "color.mp4"
            timestamps_payload = _rh20t_load_dict(
                base / "timestamps.npy", "exo color timestamps"
            )
            try:
                source_timestamps = np.asarray(
                    timestamps_payload["color"], dtype=np.int64
                )
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetDownloadError(
                    f"Malformed RH20T exo timestamps: {sequence}"
                ) from error
            ego_timestamps, reference_from_ego = _rh20t_reference_poses(clip_dir)
            indices, deltas = nearest_timestamp_indices(
                source_timestamps, ego_timestamps
            )
            synchronized = np.abs(deltas) <= int(demo["sync_tolerance_ms"])
            camera_config = json.loads(
                (clip_dir / "reference" / "camera.json").read_text(encoding="utf-8")
            )
            calibration_dir = clip_dir / "reference" / "calibration"
            aligned = rh20t_aligned_extrinsics(
                calibration_dir,
                camera_config,
                str(dataset["in_hand_serial"]),
            )
            if serial not in aligned:
                raise DatasetDownloadError(
                    f"RH20T calibration omits demo exo camera {serial}"
                )
            reference_from_exo_fixed = invert_pose(aligned[serial])
            reference_from_exo = np.repeat(
                reference_from_exo_fixed[None], len(ego_timestamps), axis=0
            )
            intrinsics = _rh20t_load_dict(
                calibration_dir / "intrinsics.npy", "camera intrinsics"
            )
            intrinsic = np.asarray(intrinsics[serial], dtype=np.float64)[:3, :3]
            source_width, source_height = [
                int(value) for value in demo["source_resolution"]
            ]
            quality = projection_quality(
                reference_from_ego,
                reference_from_exo,
                intrinsic,
                source_width,
                source_height,
                synchronized,
            )
            quality.update(
                {
                    "maximum_absolute_sync_delta_ms": int(np.max(np.abs(deltas))),
                    "median_absolute_sync_delta_ms": float(
                        np.median(np.abs(deltas))
                    ),
                }
            )
            if not exo_quality_passes(
                quality,
                float(demo["minimum_synchronized_ratio"]),
                float(demo["minimum_projection_inside_ratio"]),
            ):
                raise DatasetDownloadError(
                    f"RH20T exo quality gate failed for {sequence}: {quality}"
                )
            manifest = _write_exo_artifacts(
                clip_dir=clip_dir,
                dataset="rh20t_wrist",
                sequence=sequence,
                reference_type=str(dataset["reference_type"]),
                ego_timestamps=ego_timestamps,
                exo_timestamps=source_timestamps[indices],
                deltas_ms=deltas,
                source_indices=indices,
                synchronized=synchronized,
                reference_from_exo=reference_from_exo,
                intrinsic_source=intrinsic,
                source_width=source_width,
                source_height=source_height,
                distortion=[],
                camera_metadata={
                    "stream": f"cam_{serial}/color.mp4",
                    "serial": serial,
                    "pose_direction": "camera_to_aligned_robot_base",
                    "distortion_model": "no_coefficients_in_release",
                    "camera_basis": "opencv_right_down_forward",
                    "calibration_id": camera_config.get("calibration_id"),
                    "api_repo": dataset["api_repo"],
                    "api_commit": dataset["api_commit"],
                },
                source_video=video,
                ffmpeg=ffmpeg,
                data_root=data_root,
                quality=quality,
                source_info={
                    "archive_bytes": int(dataset["archive_bytes"]),
                    "archive_sha256": dataset["archive_sha256"],
                    "archive_member_video": str(
                        PurePosixPath(dataset["archive_root"])
                        / sequence
                        / f"cam_{serial}"
                        / "color.mp4"
                    ),
                    "video_bytes": video.stat().st_size,
                    "video_sha256": _sha256(video),
                    "caller_owned_local_archive": caller_owned,
                    "retained_pipeline_archive": keep_source,
                },
            )
            (clip_dir / "exo" / "exclusion.json").unlink(missing_ok=True)
            results.append(manifest)
    finally:
        if not keep_source and stage_root.exists():
            _remove_owned_eval_cache(stage_root, data_root)
        if not keep_source and not caller_owned and archive_path.exists():
            archive_path.unlink()
        if not keep_source and cache_root.exists():
            try:
                cache_root.rmdir()
            except OSError:
                pass
    order = {str(clip["sequence"]): index for index, clip in enumerate(selected_clips)}
    return sorted(results, key=lambda item: order[str(item["sequence_id"])])


def prepare_robot_exo(
    plan: dict[str, Any],
    data_root: str | Path,
    ffmpeg: str,
    *,
    workers: int = 8,
    keep_source: bool = False,
    rh20t_archive: str | Path | None = None,
    datasets: Iterable[str] = ROBOT_DEMO_DATASETS,
    sequence_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    selected = tuple(datasets)
    unsupported = set(selected) - set(ROBOT_DEMO_DATASETS)
    if unsupported:
        raise ValueError(f"Unsupported robot exo datasets: {sorted(unsupported)}")
    requested_sequences = (
        {str(value) for value in sequence_ids} if sequence_ids is not None else None
    )
    available_sequences = {
        str(clip["sequence"])
        for dataset_name in selected
        for clip in robot_dataset_demo_clips(plan["datasets"][dataset_name])
    }
    if requested_sequences is not None:
        unknown_sequences = requested_sequences - available_sequences
        if unknown_sequences:
            raise ValueError(
                f"Robot exo sequences are not in the selected datasets: "
                f"{sorted(unknown_sequences)}"
            )
    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": plan["profile"]["id"],
        "data_root": str(root),
        "datasets": {},
    }
    if "droid_wrist" in selected:
        report["datasets"]["droid_wrist"] = _prepare_droid_exo(
            plan, root, ffmpeg, keep_source, requested_sequences
        )
    if "rh20t_wrist" in selected:
        report["datasets"]["rh20t_wrist"] = _prepare_rh20t_exo(
            plan,
            root,
            ffmpeg,
            max(1, int(workers)),
            keep_source,
            Path(rh20t_archive).resolve() if rh20t_archive is not None else None,
            requested_sequences,
        )
    manifest_path = root / "robot_exo_manifest.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    merged_datasets = (
        dict(previous.get("datasets", {}))
        if previous.get("profile") == report["profile"]
        and previous.get("data_root") == report["data_root"]
        and isinstance(previous.get("datasets"), dict)
        else {}
    )
    for dataset_name, records in report["datasets"].items():
        prior_by_sequence = {
            str(record["sequence_id"]): record
            for record in merged_datasets.get(dataset_name, [])
            if isinstance(record, dict) and "sequence_id" in record
        }
        prior_by_sequence.update(
            {str(record["sequence_id"]): record for record in records}
        )
        order = {
            str(clip["sequence"]): index
            for index, clip in enumerate(
                robot_dataset_demo_clips(plan["datasets"][dataset_name])
            )
        }
        merged_datasets[dataset_name] = sorted(
            prior_by_sequence.values(),
            key=lambda record: order.get(str(record["sequence_id"]), len(order)),
        )
    report["datasets"] = {
        dataset_name: merged_datasets[dataset_name]
        for dataset_name in ROBOT_DEMO_DATASETS
        if dataset_name in merged_datasets
    }
    ready = []
    excluded = []
    for dataset, records in report["datasets"].items():
        for record in records:
            item = {"dataset": dataset, "sequence_id": record["sequence_id"]}
            if record["status"] == "ready":
                ready.append(item)
            else:
                excluded.append({**item, "reason": record.get("reason")})
    report["ready_clips"] = ready
    report["excluded_clips"] = excluded
    _write_json(manifest_path, report)
    return report


def verify_robot_exo(
    plan: dict[str, Any],
    data_root: str | Path,
    datasets: Iterable[str] = ROBOT_DEMO_DATASETS,
    sequence_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    result: dict[str, Any] = {"datasets": {}, "ok": True}
    requested_sequences = (
        {str(value) for value in sequence_ids} if sequence_ids is not None else None
    )
    for dataset_name in datasets:
        dataset = plan["datasets"][dataset_name]
        states = {}
        for clip in robot_dataset_demo_clips(dataset):
            sequence = str(clip["sequence"])
            if requested_sequences is not None and sequence not in requested_sequences:
                continue
            status, payload, reason = exo_clip_status(
                root / dataset_name / "clips" / sequence
            )
            states[sequence] = {
                "status": status,
                "reason": reason or (payload or {}).get("reason"),
            }
            expected_excluded = bool(
                dataset.get("demo_exo", {})
                .get("sequences", {})
                .get(sequence, {})
                .get("excluded", False)
            )
            expected_status = "excluded" if expected_excluded else "ready"
            states[sequence]["expected_status"] = expected_status
            if status != expected_status:
                result["ok"] = False
        result["datasets"][dataset_name] = states
    return result
