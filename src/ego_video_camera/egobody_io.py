from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .camera_models import CameraModel
from .transforms import as_homogeneous, validate_pose


PV_IMAGE_RE = re.compile(r"^(?P<timestamp>\d+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$", re.I)
EXO_IMAGE_RE = re.compile(r"^frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$", re.I)
WINDOWS_TICKS_PER_SECOND = 10_000_000.0


@dataclass(frozen=True)
class PVCalibration:
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class PVRecord:
    timestamp: int
    fx: float
    fy: float
    T_W_E: np.ndarray
    frame_id: int | None = None
    image_path: Path | None = None


@dataclass(frozen=True)
class HeadRecord:
    timestamp: int
    T_W_Q: np.ndarray
    valid: bool


@dataclass(frozen=True)
class FrameMapping:
    output_frame: int
    ego_frame_id: int
    exo_frame_id: int | None
    ego_timestamp: int
    exo_timestamp: int | None
    time_difference: float | None
    sync_basis: str
    ego_image: Path
    exo_image: Path | None


def timestamp_delta_seconds(timestamp_a: int | float, timestamp_b: int | float) -> float:
    return float(timestamp_a - timestamp_b) / WINDOWS_TICKS_PER_SECOND


def parse_pv_file(path: str | Path) -> tuple[PVCalibration, list[PVRecord]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"PV file contains no frame records: {path}")
    cx, cy, width, height = ast.literal_eval(lines[0])
    calibration = PVCalibration(float(cx), float(cy), int(width), int(height))
    records: list[PVRecord] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 19:
            raise ValueError(f"Malformed PV row {path}:{line_number}: expected >=19 fields")
        values = np.asarray(fields[3:19], dtype=np.float64).reshape(4, 4)
        try:
            validate_pose(values, atol=2e-3)
        except ValueError as error:
            raise ValueError(f"Invalid pv2world at {path}:{line_number}: {error}") from error
        records.append(
            PVRecord(timestamp=int(fields[0]), fx=float(fields[1]), fy=float(fields[2]), T_W_E=values)
        )
    return calibration, records


def index_pv_images(pv_dir: str | Path) -> dict[int, tuple[int, Path]]:
    result: dict[int, tuple[int, Path]] = {}
    for path in sorted(Path(pv_dir).glob("*")):
        match = PV_IMAGE_RE.match(path.name)
        if match:
            result[int(match.group("timestamp"))] = (int(match.group("frame")), path)
    return result


def attach_pv_images(records: Iterable[PVRecord], pv_dir: str | Path) -> list[PVRecord]:
    image_index = index_pv_images(pv_dir)
    return [
        PVRecord(r.timestamp, r.fx, r.fy, r.T_W_E, *(image_index.get(r.timestamp, (None, None))))
        for r in records
    ]


def index_exo_images(master_dir: str | Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(Path(master_dir).glob("*")):
        match = EXO_IMAGE_RE.match(path.name)
        if match:
            result[int(match.group("frame"))] = path
    return result


def synchronize_exact_frame_ids(records: Iterable[PVRecord], master_dir: str | Path) -> list[FrameMapping]:
    exo = index_exo_images(master_dir)
    mappings: list[FrameMapping] = []
    for record in records:
        if record.frame_id is None or record.image_path is None:
            continue
        exo_path = exo.get(record.frame_id)
        mappings.append(
            FrameMapping(
                output_frame=len(mappings),
                ego_frame_id=record.frame_id,
                exo_frame_id=record.frame_id if exo_path else None,
                ego_timestamp=record.timestamp,
                exo_timestamp=None,
                time_difference=None,
                sync_basis="exact_frame_id",
                ego_image=record.image_path,
                exo_image=exo_path,
            )
        )
    return mappings


def sample_records(records: list[PVRecord], sample_fps: float, start_sec: float, duration_sec: float) -> list[PVRecord]:
    available = [r for r in records if r.frame_id is not None and r.image_path is not None]
    if not available:
        return []
    timestamps = np.asarray([r.timestamp for r in available], dtype=np.float64)
    base = timestamps[0]
    seconds = (timestamps - base) / WINDOWS_TICKS_PER_SECOND
    stop_sec = min(start_sec + duration_sec, seconds[-1] + 1e-9)
    targets = np.arange(start_sec, stop_sec - 1e-9, 1.0 / sample_fps)
    selected: list[PVRecord] = []
    used: set[int] = set()
    for target in targets:
        index = int(np.argmin(np.abs(seconds - target)))
        if index not in used:
            selected.append(available[index])
            used.add(index)
    return selected


def discover_pv_sequences(data_root: str | Path, recording_name: str) -> list[Path]:
    base = Path(data_root) / "egocentric_color" / recording_name
    if not base.exists():
        return []
    return sorted(path.parent for path in base.glob("**/*_pv.txt"))


def resolve_calibration_path(data_root: str | Path, recording_name: str, filename: str) -> Path:
    base = Path(data_root) / "calibrations" / recording_name
    candidates = [base / "cal_trans" / filename, base / filename]
    candidates.extend(sorted(base.glob(f"**/{filename}")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve {filename} for {recording_name} under {base}")


def load_transform_json(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = as_homogeneous(np.asarray(data["trans"], dtype=np.float64))
    validate_pose(matrix, atol=2e-3)
    return matrix


def load_T_K_W(data_root: str | Path, recording_name: str) -> tuple[np.ndarray, Path]:
    path = resolve_calibration_path(data_root, recording_name, "holo_to_kinect12.json")
    return load_transform_json(path), path


def load_master_camera(data_root: str | Path) -> tuple[CameraModel, Path]:
    path = Path(data_root) / "kinect_cam_params" / "kinect_master" / "Color.json"
    if not path.is_file():
        alternatives = sorted((Path(data_root) / "kinect_cam_params").glob("**/Color.json"))
        path = next((item for item in alternatives if "master" in str(item).lower()), path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing master Kinect Color.json under {data_root}")
    return CameraModel.from_egobody_json(path), path


def load_head_tracking(path: str | Path) -> list[HeadRecord]:
    data = np.loadtxt(path, delimiter=",", usecols=range(17), ndmin=2)
    records: list[HeadRecord] = []
    for row in data:
        matrix = row[1:17].reshape(4, 4).astype(np.float64)
        valid = True
        try:
            validate_pose(matrix, atol=5e-3)
        except ValueError:
            valid = False
        records.append(HeadRecord(timestamp=int(row[0]), T_W_Q=matrix, valid=valid))
    records.sort(key=lambda record: record.timestamp)
    return records


def nearest_head_record(
    timestamp: int, records: list[HeadRecord], tolerance_ms: float = 50.0
) -> HeadRecord | None:
    if not records:
        return None
    valid_indices = np.flatnonzero([record.valid for record in records])
    if not len(valid_indices):
        return None
    valid_timestamps = np.asarray(
        [records[index].timestamp for index in valid_indices], dtype=np.int64
    )
    insertion = int(np.searchsorted(valid_timestamps, timestamp))
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(valid_timestamps)
    ]
    best = min(candidates, key=lambda index: abs(int(valid_timestamps[index]) - timestamp))
    delta_ms = (
        abs(timestamp_delta_seconds(int(valid_timestamps[best]), timestamp)) * 1000.0
    )
    return records[int(valid_indices[best])] if delta_ms <= tolerance_ms else None


def read_dataset_metadata(data_root: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    root = Path(data_root)
    info = pd.read_csv(root / "data_info_release.csv")
    splits = pd.read_csv(root / "data_splits.csv")
    split_by_recording: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for recording in splits.get(split, pd.Series(dtype=str)).dropna().astype(str):
            split_by_recording[recording] = split
    return info, split_by_recording


def write_frame_mapping_csv(path: str | Path, mappings: Iterable[FrameMapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FrameMapping.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for mapping in mappings:
            writer.writerow({key: getattr(mapping, key) for key in fields})
