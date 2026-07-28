from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .camera_models import CameraModel
from .robot_exo import (
    ROBOT_DEMO_DATASETS,
    droid_pose_matrix,
    exo_clip_status,
    robot_dataset_demo_clips,
)


ROBOT_DATASET_ALIASES = {
    "all": ROBOT_DEMO_DATASETS,
    "droid": ("droid_wrist",),
    "droid_wrist": ("droid_wrist",),
    "rh20t": ("rh20t_wrist",),
    "rh20t_wrist": ("rh20t_wrist",),
}

ACTIVE_EGO_MODEL_LABEL = "Active Ego Foundation Model"


@dataclass(frozen=True)
class RobotFrame:
    output_index: int
    timeline_sec: float
    ego_timestamp_ms: int
    ego_image: Path
    reference_from_ego: np.ndarray
    exo_timestamp_ms: int | None
    sync_delta_ms: int | None
    exo_source_frame_index: int | None
    exo_image: Path | None
    reference_from_exo: np.ndarray | None
    synchronized: bool


@dataclass(frozen=True)
class RobotClip:
    dataset: str
    sequence_id: str
    clip_dir: Path
    reference_type: str
    source_fps: float
    fps: float
    duration_sec: float
    frames: tuple[RobotFrame, ...]
    exo_camera: CameraModel
    exo_manifest: dict[str, Any]

    @property
    def ego_images(self) -> list[Path]:
        return [frame.ego_image for frame in self.frames]

    @property
    def frame_ids(self) -> np.ndarray:
        return np.asarray([frame.output_index for frame in self.frames], dtype=np.int64)

    @property
    def timestamps_ms(self) -> np.ndarray:
        return np.asarray(
            [frame.ego_timestamp_ms for frame in self.frames], dtype=np.int64
        )

    @property
    def timestamps_sec(self) -> np.ndarray:
        return np.asarray([frame.timeline_sec for frame in self.frames], dtype=np.float64)

    @property
    def reference_from_ego(self) -> np.ndarray:
        return np.asarray([frame.reference_from_ego for frame in self.frames])

    @property
    def synchronized(self) -> np.ndarray:
        return np.asarray([frame.synchronized for frame in self.frames], dtype=bool)


def normalize_robot_datasets(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        keys = [item.strip() for item in value.split(",") if item.strip()]
    else:
        keys = [str(item).strip() for item in value]
    if not keys:
        raise ValueError("At least one robot dataset must be selected")
    selected: list[str] = []
    for key in keys:
        try:
            expanded = ROBOT_DATASET_ALIASES[key]
        except KeyError as error:
            raise ValueError(f"Unsupported robot dataset: {key}") from error
        for name in expanded:
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def robot_demo_selection(
    plan: dict[str, Any],
    datasets: str | Iterable[str] = "all",
    sequence_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the fixed seven clips in DROID then RH20T order."""

    selected = normalize_robot_datasets(datasets)
    clips: list[dict[str, Any]] = []
    for dataset_name in ROBOT_DEMO_DATASETS:
        if dataset_name not in selected:
            continue
        dataset = plan["datasets"][dataset_name]
        for source in robot_dataset_demo_clips(dataset):
            sequence = str(source["sequence"])
            if sequence_id is not None and sequence != sequence_id:
                continue
            exo_config = dataset.get("demo_exo", {})
            sequence_exo = exo_config.get("sequences", {}).get(sequence, {})
            if bool(sequence_exo.get("excluded", False)):
                continue
            clips.append(
                {
                    **source,
                    "dataset": dataset_name,
                    "sequence_id": sequence,
                    "reference_type": str(dataset["reference_type"]),
                }
            )
    if sequence_id is not None and len(clips) != 1:
        raise ValueError(
            f"Robot sequence is not selected, is excluded, or is ambiguous: {sequence_id}"
        )
    return clips


def robot_exo_readiness(
    data_root: str | Path, selections: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Check every selected formal clip before any expensive processing starts."""

    root = Path(data_root).resolve()
    clips = []
    for selection in selections:
        dataset = str(selection["dataset"])
        sequence = str(selection["sequence_id"])
        status, _, reason = exo_clip_status(
            root / dataset / "clips" / sequence
        )
        clips.append(
            {
                "dataset": dataset,
                "sequence_id": sequence,
                "status": status,
                "reason": reason,
                "ready": status == "ready",
            }
        )
    missing = [clip for clip in clips if not clip["ready"]]
    return {
        "schema_version": 1,
        "data_root": str(root),
        "ok": not missing,
        "clips": clips,
        "missing_clips": missing,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unable to read robot clip CSV: {path}") from error


def _matrix_from_fields(row: dict[str, str]) -> np.ndarray:
    try:
        values = [float(row[f"m{i}{j}"]) for i in range(4) for j in range(4)]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Malformed robot 4x4 pose row") from error
    pose = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
        raise RuntimeError("Robot reference pose contains non-finite values")
    return pose


def _load_ego_poses(dataset: str, clip_dir: Path) -> dict[int, np.ndarray]:
    if dataset == "droid_wrist":
        path = clip_dir / "reference" / "camera_to_robot_base.csv"
        result = {}
        for row in _read_csv(path):
            try:
                index = int(row["output_index"])
                values = [
                    float(row[key])
                    for key in (
                        "tx",
                        "ty",
                        "tz",
                        "rx_xyz_rad",
                        "ry_xyz_rad",
                        "rz_xyz_rad",
                    )
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Malformed DROID reference row: {path}") from error
            result[index] = droid_pose_matrix(values)
        return result
    if dataset == "rh20t_wrist":
        path = clip_dir / "reference" / "camera_to_aligned_robot_base.csv"
        return {
            int(row["output_index"]): _matrix_from_fields(row)
            for row in _read_csv(path)
        }
    raise ValueError(f"Unsupported robot dataset: {dataset}")


def _camera_from_manifest(camera: dict[str, Any]) -> CameraModel:
    if "matrix" in camera:
        matrix = np.asarray(camera["matrix"], dtype=np.float64)
    else:
        matrix = np.asarray(
            [
                [float(camera["fx"]), 0.0, float(camera["cx"])],
                [0.0, float(camera["fy"]), float(camera["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RuntimeError("Robot exo camera matrix must be finite 3x3")
    return CameraModel(
        matrix=matrix,
        distortion=np.asarray(
            camera.get("distortion_coefficients", []), dtype=np.float64
        ).reshape(-1),
        width=int(camera["width"]),
        height=int(camera["height"]),
    )


def subsample_indices(
    frame_count: int,
    source_fps: float,
    target_fps: float,
    duration_sec: float | None = None,
) -> np.ndarray:
    if frame_count <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame count and sampling rates must be positive")
    if target_fps > source_fps + 1e-9:
        raise ValueError(
            f"Robot demo cannot upsample {source_fps:g} FPS to {target_fps:g} FPS"
        )
    available_duration = frame_count / source_fps
    duration = available_duration if duration_sec is None else float(duration_sec)
    if duration <= 0:
        raise ValueError("Robot demo duration must be positive")
    duration = min(duration, available_duration)
    output_count = min(frame_count, int(round(duration * target_fps)))
    if output_count <= 0:
        raise ValueError("Robot demo sampling selected no frames")
    indices = np.floor(
        np.arange(output_count, dtype=np.float64) * source_fps / target_fps + 1e-9
    ).astype(np.int64)
    if len(np.unique(indices)) != len(indices) or indices[-1] >= frame_count:
        raise ValueError("Robot demo sampling produced invalid source indices")
    return indices


def load_robot_clip(
    data_root: str | Path,
    dataset: str,
    sequence_id: str,
    *,
    sample_fps: float = 10.0,
    duration_sec: float | None = None,
    source_fps: float = 10.0,
    verify_exo: bool = True,
) -> RobotClip:
    if dataset not in ROBOT_DEMO_DATASETS:
        raise ValueError(f"Unsupported robot dataset: {dataset}")
    clip_dir = Path(data_root).resolve() / dataset / "clips" / sequence_id
    if verify_exo:
        status, manifest, reason = exo_clip_status(clip_dir)
        if status != "ready" or manifest is None:
            raise RuntimeError(
                f"Robot exo clip is not ready: {dataset}/{sequence_id}: "
                f"{reason or status}"
            )
    else:
        try:
            manifest = json.loads(
                (clip_dir / "exo" / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Unable to read robot exo manifest: {clip_dir}") from error

    ego_rows = _read_csv(clip_dir / "frames.csv")
    exo_rows = _read_csv(clip_dir / "exo" / "frames.csv")
    exo_pose_rows = _read_csv(clip_dir / "exo" / "camera_to_reference.csv")
    if not ego_rows or len(exo_rows) != len(ego_rows) or len(exo_pose_rows) != len(ego_rows):
        raise RuntimeError("Robot ego/exo frame mapping counts do not match")
    ego_poses = _load_ego_poses(dataset, clip_dir)
    exo_by_index = {int(row["output_index"]): row for row in exo_rows}
    exo_pose_by_index = {int(row["output_index"]): row for row in exo_pose_rows}
    indices = subsample_indices(
        len(ego_rows), source_fps, float(sample_fps), duration_sec
    )
    frames: list[RobotFrame] = []
    for sampled_index, source_index in enumerate(indices.tolist()):
        ego_row = ego_rows[source_index]
        output_index = int(ego_row["output_index"])
        if output_index not in ego_poses:
            raise RuntimeError(f"Missing ego reference pose for output {output_index}")
        ego_image = clip_dir / "frames" / str(ego_row["filename"])
        if not ego_image.is_file():
            raise FileNotFoundError(ego_image)
        exo_row = exo_by_index.get(output_index)
        exo_pose_row = exo_pose_by_index.get(output_index)
        if exo_row is None or exo_pose_row is None:
            raise RuntimeError(f"Missing exo mapping for output {output_index}")
        synchronized = bool(int(exo_row.get("synchronized", "0")))
        exo_image = (
            clip_dir / "exo" / "frames" / exo_row["filename"]
            if synchronized and exo_row.get("filename")
            else None
        )
        if exo_image is not None and not exo_image.is_file():
            raise FileNotFoundError(exo_image)
        exo_pose = (
            _matrix_from_fields(exo_pose_row)
            if synchronized and bool(int(exo_pose_row.get("valid", "0")))
            else None
        )
        frames.append(
            RobotFrame(
                output_index=output_index,
                timeline_sec=sampled_index / float(sample_fps),
                ego_timestamp_ms=int(round(float(ego_row["source_timestamp"]))),
                ego_image=ego_image,
                reference_from_ego=ego_poses[output_index],
                exo_timestamp_ms=(
                    int(exo_row["exo_timestamp_ms"])
                    if synchronized and exo_row.get("exo_timestamp_ms")
                    else None
                ),
                sync_delta_ms=(
                    int(exo_row["delta_ms"])
                    if synchronized and exo_row.get("delta_ms")
                    else None
                ),
                exo_source_frame_index=(
                    int(exo_row["source_frame_index"])
                    if synchronized and exo_row.get("source_frame_index")
                    else None
                ),
                exo_image=exo_image,
                reference_from_exo=exo_pose,
                synchronized=synchronized and exo_image is not None and exo_pose is not None,
            )
        )
    camera = _camera_from_manifest(manifest["camera"])
    return RobotClip(
        dataset=dataset,
        sequence_id=sequence_id,
        clip_dir=clip_dir,
        reference_type=str(manifest["reference_type"]),
        source_fps=float(source_fps),
        fps=float(sample_fps),
        duration_sec=len(frames) / float(sample_fps),
        frames=tuple(frames),
        exo_camera=camera,
        exo_manifest=manifest,
    )
