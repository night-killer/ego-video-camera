from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..openloris import DEFAULT_COLOR_CAMERA, read_camera_intrinsics
from ..serialization import write_json
from .schema import FrameRecord, MethodSpec, SequenceRecord


DEFAULT_REFERENCE = {
    "egobody": ("B-device", "hololens_device_tracking"),
    "princeton365": ("A-external", "hidden_marker_rig"),
}


def discover_sequences(config: dict[str, Any]) -> list[SequenceRecord]:
    target_fps = float(config["benchmark"].get("target_fps", 10.0))
    records: list[SequenceRecord] = []
    seen: set[str] = set()
    for source in config["datasets"]["sources"]:
        source_root = Path(source["root"])
        included = set(source.get("include", []))
        for clip_json in sorted(source_root.glob("*/clips/*/clip.json")):
            payload = json.loads(clip_json.read_text(encoding="utf-8"))
            dataset_id = str(payload.get("dataset") or clip_json.parents[2].name)
            if included and dataset_id not in included:
                continue
            sequence_id = str(payload.get("sequence_id") or clip_json.parent.name)
            key = f"{dataset_id}/{sequence_id}"
            if key in seen:
                raise ValueError(f"Duplicate benchmark sequence: {key}")
            seen.add(key)
            input_path = Path(payload["input"])
            if not input_path.is_absolute():
                input_path = (clip_json.parent / input_path).resolve()
            fallback_grade, fallback_type = DEFAULT_REFERENCE.get(
                dataset_id, ("unknown", "unknown")
            )
            duration = float(payload["duration_s"])
            frame_count = int(
                payload.get("frame_count")
                or payload.get("expected_frame_count")
                or round(duration * target_fps)
            )
            records.append(
                SequenceRecord(
                    dataset_id=dataset_id,
                    sequence_id=sequence_id,
                    clip_dir=clip_json.parent.resolve(),
                    clip_json=clip_json.resolve(),
                    input_path=input_path,
                    duration_sec=duration,
                    target_fps=target_fps,
                    reference_grade=str(payload.get("reference_grade", fallback_grade)),
                    reference_type=str(payload.get("reference_type", fallback_type)),
                    stratum=str(payload.get("stratum", "unspecified")),
                    start_sec=float(payload.get("start_s", 0.0)),
                    frame_count=frame_count,
                    input_kind="frames" if input_path.is_dir() else "video",
                )
            )
    order = {name: index for index, name in enumerate(config["datasets"]["order"])}
    records.sort(key=lambda value: (order.get(value.dataset_id, 999), value.sequence_id))
    return records


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _timestamp_ns(rows: list[dict[str, str]], fps: float) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.int64)
    columns = rows[0]
    if "timestamp" in columns:
        values = np.asarray([int(row["timestamp"]) for row in rows], dtype=np.int64)
        # EgoBody timestamps are Windows FILETIME ticks at 100 ns.
        return (values - values[0]) * 100
    if "source_timestamp" in columns:
        values = np.asarray([float(row["source_timestamp"]) for row in rows])
        # DROID/RH20T store Unix milliseconds; RGB-D sources store seconds.
        multiplier = 1e6 if abs(values[0]) > 1e11 else 1e9
        return np.rint((values - values[0]) * multiplier).astype(np.int64)
    if "clip_time_s" in columns:
        values = np.asarray([float(row["clip_time_s"]) for row in rows])
        return np.rint((values - values[0]) * 1e9).astype(np.int64)
    return np.rint(np.arange(len(rows), dtype=np.float64) * 1e9 / fps).astype(np.int64)


def _camera_json(sequence: SequenceRecord) -> tuple[Path | None, dict[str, Any]]:
    candidates = (
        sequence.clip_dir / "reference" / "camera.json",
        sequence.clip_dir / "reference" / "pv_camera.json",
    )
    for path in candidates:
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None, {}


def sequence_intrinsics(sequence: SequenceRecord, frame_count: int) -> np.ndarray | None:
    if sequence.dataset_id == "princeton365":
        paths = sorted((sequence.clip_dir / "reference").glob("*.user_camera_mtx.npy"))
        if not paths:
            return None
        matrix = np.asarray(np.load(paths[0]), dtype=np.float64)
        return np.repeat(matrix[None], frame_count, axis=0)
    _, camera = _camera_json(sequence)
    if all(key in camera for key in ("fx", "fy", "cx", "cy")):
        matrix = np.asarray(
            [
                [camera["fx"], 0.0, camera["cx"]],
                [0.0, camera["fy"], camera["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return np.repeat(matrix[None], frame_count, axis=0)
    if sequence.dataset_id == "openloris_office":
        calibration_path = sequence.clip_dir / "reference" / "sensors.yaml"
        if not calibration_path.is_file():
            return None
        calibration = read_camera_intrinsics(
            calibration_path, str(camera.get("camera_key", DEFAULT_COLOR_CAMERA))
        )
        expected_size = (
            int(calibration["source_calibration_width"]),
            int(calibration["source_calibration_height"]),
        )
        actual_size = (int(camera.get("width", -1)), int(camera.get("height", -1)))
        if actual_size != expected_size:
            raise ValueError(
                f"{sequence.key} camera size {actual_size} does not match "
                f"calibration size {expected_size}"
            )
        matrix = np.asarray(
            [
                [calibration["fx"], 0.0, calibration["cx"]],
                [0.0, calibration["fy"], calibration["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return np.repeat(matrix[None], frame_count, axis=0)
    if sequence.dataset_id == "egobody":
        trajectory = _read_csv(sequence.clip_dir / "reference" / "pv_trajectory.csv")
        cx, cy = float(camera["cx"]), float(camera["cy"])
        matrices = []
        for row in trajectory[:frame_count]:
            matrices.append(
                np.asarray(
                    [[float(row["fx"]), 0, cx], [0, float(row["fy"]), cy], [0, 0, 1]],
                    dtype=np.float64,
                )
            )
        return np.asarray(matrices)
    return None


def _materialize_video_frames(
    sequence: SequenceRecord, cache_dir: Path, ffmpeg: str
) -> list[Path]:
    final_dir = cache_dir / sequence.dataset_id / sequence.sequence_id
    existing = sorted(final_dir.glob("*.jpg")) if final_dir.is_dir() else []
    if len(existing) == sequence.frame_count:
        return existing
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{sequence.sequence_id}.", dir=final_dir.parent)
    )
    try:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(sequence.input_path),
            "-vf",
            f"fps={sequence.target_fps:g}",
            "-frames:v",
            str(sequence.frame_count),
            "-q:v",
            "2",
            str(temporary / "%06d.jpg"),
        ]
        subprocess.run(command, check=True)
        generated = sorted(temporary.glob("*.jpg"))
        if len(generated) != sequence.frame_count:
            raise RuntimeError(
                f"Decoded {len(generated)}/{sequence.frame_count} frames for {sequence.key}"
            )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(temporary, final_dir)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    return sorted(final_dir.glob("*.jpg"))


def load_frames(
    sequence: SequenceRecord,
    *,
    cache_dir: str | Path,
    ffmpeg: str = "ffmpeg",
    materialize: bool = True,
) -> list[FrameRecord]:
    if sequence.input_kind == "frames":
        rows = _read_csv(sequence.clip_dir / "frames.csv")
        paths = [sequence.input_path / row["filename"] for row in rows]
    else:
        if not materialize:
            paths = [
                Path(cache_dir) / sequence.dataset_id / sequence.sequence_id / f"{index + 1:06d}.jpg"
                for index in range(sequence.frame_count)
            ]
        else:
            paths = _materialize_video_frames(sequence, Path(cache_dir), ffmpeg)
        rows = [
            {"output_index": str(index), "filename": path.name}
            for index, path in enumerate(paths)
        ]
    if len(rows) != sequence.frame_count or len(paths) != sequence.frame_count:
        raise ValueError(
            f"{sequence.key} has {len(paths)} frames, expected {sequence.frame_count}"
        )
    timestamps = _timestamp_ns(rows, sequence.target_fps)
    intrinsics = sequence_intrinsics(sequence, len(rows))
    frames = []
    for index, (row, path) in enumerate(zip(rows, paths)):
        if materialize and not path.is_file():
            raise FileNotFoundError(f"Missing benchmark input frame: {path}")
        frames.append(
            FrameRecord(
                frame_id=int(row.get("output_index", index)),
                timestamp_ns=int(timestamps[index]),
                image_path=path.resolve(),
                intrinsic=None if intrinsics is None else intrinsics[index],
            )
        )
    return frames


def write_worker_manifest(
    path: str | Path,
    run_id: str,
    method: MethodSpec,
    sequence: SequenceRecord,
    frames: Iterable[FrameRecord],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    include_intrinsics = method.input_intrinsics == "provided"
    frame_payload = []
    for frame in frames:
        row: dict[str, Any] = {
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "image_path": str(frame.image_path),
        }
        if include_intrinsics:
            if frame.intrinsic is None:
                raise ValueError(f"{sequence.key} has no intrinsics required by {method.method_id}")
            row["intrinsic"] = frame.intrinsic.tolist()
        frame_payload.append(row)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method.method_id,
        "adapter": method.adapter,
        "seed": seed,
        "dataset_id": sequence.dataset_id,
        "sequence_id": sequence.sequence_id,
        "duration_sec": sequence.duration_sec,
        "target_fps": sequence.target_fps,
        "input_intrinsics": method.input_intrinsics,
        "frames": frame_payload,
        "parameters": method.parameters,
    }
    # Deliberately excludes clip_json, reference paths, grade, and reference type.
    write_json(path, payload)
    return payload


def sequence_inventory(records: Iterable[SequenceRecord]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(record),
            "clip_dir": str(record.clip_dir),
            "clip_json": str(record.clip_json),
            "input_path": str(record.input_path),
        }
        for record in records
    ]
