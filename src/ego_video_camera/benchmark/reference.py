from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .schema import FrameRecord, PoseTrajectory, SequenceRecord


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matrix_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    values = [float(row[f"{prefix}{r}{c}"]) for r in range(4) for c in range(4)]
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _pose_from_tum(values: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = values[1:4]
    pose[:3, :3] = Rotation.from_quat(values[4:8]).as_matrix()
    return pose


def _read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = np.fromstring(stripped, sep=" ", dtype=np.float64)
        if len(values) >= 8 and np.isfinite(values[:8]).all():
            rows.append(values[:8])
    if not rows:
        return np.empty(0), np.empty((0, 4, 4))
    values = np.asarray(rows)
    return values[:, 0], np.asarray([_pose_from_tum(row) for row in values])


def _interpolate(
    source_time: np.ndarray,
    source_c2w: np.ndarray,
    target_time: np.ndarray,
    maximum_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_time = np.asarray(source_time, dtype=np.float64)
    source_c2w = np.asarray(source_c2w, dtype=np.float64)
    finite = np.isfinite(source_time) & np.isfinite(source_c2w).all(axis=(1, 2))
    source_time = source_time[finite]
    source_c2w = source_c2w[finite]
    if len(source_time) < 2:
        return np.full((len(target_time), 4, 4), np.nan), np.zeros(len(target_time), bool)
    order = np.argsort(source_time)
    source_time, source_c2w = source_time[order], source_c2w[order]
    unique = np.concatenate(([True], np.diff(source_time) > 1e-12))
    source_time, source_c2w = source_time[unique], source_c2w[unique]
    right = np.searchsorted(source_time, target_time, side="left")
    left = np.clip(right - 1, 0, len(source_time) - 1)
    right = np.clip(right, 0, len(source_time) - 1)
    gaps = source_time[right] - source_time[left]
    inside = (target_time >= source_time[0]) & (target_time <= source_time[-1])
    valid = inside & (gaps <= maximum_gap_sec + 1e-12)
    output = np.full((len(target_time), 4, 4), np.nan, dtype=np.float64)
    if not valid.any():
        return output, valid
    query = target_time[valid]
    translation = np.column_stack(
        [np.interp(query, source_time, source_c2w[:, axis, 3]) for axis in range(3)]
    )
    rotation = Slerp(source_time, Rotation.from_matrix(source_c2w[:, :3, :3]))(query)
    output[valid] = np.eye(4)
    output[valid, :3, :3] = rotation.as_matrix()
    output[valid, :3, 3] = translation
    return output, valid


def _input_source_times(sequence: SequenceRecord) -> np.ndarray:
    if sequence.input_kind == "video":
        return sequence.start_sec + np.arange(sequence.frame_count) / sequence.target_fps
    rows = _csv_rows(sequence.clip_dir / "frames.csv")
    if rows and "source_timestamp" in rows[0]:
        return np.asarray([float(row["source_timestamp"]) for row in rows])
    if rows and "timestamp" in rows[0]:
        values = np.asarray([int(row["timestamp"]) for row in rows], dtype=np.int64)
        return (values - values[0]).astype(np.float64) * 1e-7
    return np.arange(len(rows), dtype=np.float64) / sequence.target_fps


def _indexed_matrix_reference(
    path: Path, frames: list[FrameRecord], prefix: str
) -> tuple[np.ndarray, np.ndarray]:
    rows = _csv_rows(path)
    if rows and "output_index" in rows[0]:
        lookup = {int(row["output_index"]): row for row in rows}
        frame_key = lambda frame: frame.frame_id
    elif rows and "filename" in rows[0]:
        lookup = {Path(row["filename"]).name: row for row in rows}
        frame_key = lambda frame: frame.image_path.name
    elif rows:
        raise ValueError(
            f"Reference {path} must contain either output_index or filename"
        )
    else:
        lookup = {}
        frame_key = lambda frame: frame.frame_id
    poses = np.full((len(frames), 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        row = lookup.get(frame_key(frame))
        if row is None:
            continue
        try:
            pose = _matrix_from_row(row, prefix)
        except (KeyError, ValueError):
            continue
        if np.isfinite(pose).all():
            poses[index] = pose
            valid[index] = True
    return poses, valid


def _droid_reference(
    sequence: SequenceRecord, frames: list[FrameRecord]
) -> tuple[np.ndarray, np.ndarray, Path]:
    path = sequence.clip_dir / "reference" / "camera_to_robot_base.csv"
    rows = _csv_rows(path)
    lookup = {int(row["output_index"]): row for row in rows}
    poses = np.full((len(frames), 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        row = lookup.get(frame.frame_id)
        if row is None:
            continue
        pose = np.eye(4)
        pose[:3, 3] = [float(row[key]) for key in ("tx", "ty", "tz")]
        pose[:3, :3] = Rotation.from_euler(
            "xyz", [float(row[key]) for key in ("rx_xyz_rad", "ry_xyz_rad", "rz_xyz_rad")]
        ).as_matrix()
        poses[index], valid[index] = pose, True
    return poses, valid, path


def _princeton_reference(
    sequence: SequenceRecord, frames: list[FrameRecord]
) -> tuple[np.ndarray, np.ndarray, Path]:
    reference = sequence.clip_dir / "reference"
    trajectory_path = next(reference.glob("*.gt_trajectory.txt"))
    transform_path = next(reference.glob("*.relative_transform.npy"))
    source_frame, gt_c2w = _read_tum(trajectory_path)
    # The official trajectory is 60 Hz and the first field is its frame index.
    source_time = source_frame / 60.0
    user_to_gt = np.asarray(np.load(transform_path), dtype=np.float64)
    # Official evaluator applies this transform on the right to obtain user C2W.
    user_c2w = gt_c2w @ user_to_gt
    target_time = sequence.start_sec + np.arange(len(frames)) / sequence.target_fps
    poses, valid = _interpolate(source_time, user_c2w, target_time, 1.0 / 30.0)
    return poses, valid, trajectory_path


def load_reference(
    sequence: SequenceRecord, frames: list[FrameRecord]
) -> PoseTrajectory:
    reference_dir = sequence.clip_dir / "reference"
    source_path: Path
    if sequence.dataset_id == "princeton365":
        poses, valid, source_path = _princeton_reference(sequence, frames)
    elif sequence.dataset_id == "egobody":
        source_path = reference_dir / "pv_trajectory.csv"
        poses, valid = _indexed_matrix_reference(source_path, frames, "t")
    elif sequence.dataset_id in {"tum_rgbd", "bonn_rgbd_dynamic", "openloris_office"}:
        source_path = reference_dir / "groundtruth.txt"
        source_time, source_c2w = _read_tum(source_path)
        target_time = _input_source_times(sequence)
        poses, valid = _interpolate(source_time, source_c2w, target_time, 0.25)
    elif sequence.dataset_id == "droid_wrist":
        poses, valid, source_path = _droid_reference(sequence, frames)
    else:
        matrix_files = {
            "holoassist": ("camera_to_hololens_world.csv", "m"),
            "rh20t_wrist": ("camera_to_aligned_robot_base.csv", "m"),
            "stera10m": ("camera_optical_to_arkit_world.csv", "m"),
        }
        if sequence.dataset_id not in matrix_files:
            raise ValueError(f"No reference adapter for dataset {sequence.dataset_id}")
        filename, prefix = matrix_files[sequence.dataset_id]
        source_path = reference_dir / filename
        poses, valid = _indexed_matrix_reference(source_path, frames, prefix)
    return PoseTrajectory(
        timestamp_ns=np.asarray([frame.timestamp_ns for frame in frames]),
        frame_id=np.asarray([frame.frame_id for frame in frames]),
        c2w=poses,
        valid=valid,
        confidence=np.where(valid, 1.0, np.nan),
        metadata={
            "dataset_id": sequence.dataset_id,
            "sequence_id": sequence.sequence_id,
            "reference_grade": sequence.reference_grade,
            "reference_type": sequence.reference_type,
            "source_path": str(source_path),
            "pose_convention": "OpenCV camera-to-world, column-vector transforms",
        },
    )
