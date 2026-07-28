from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..serialization import write_json
from .schema import PoseTrajectory


def write_trajectory(path: str | Path, trajectory: PoseTrajectory) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(
            temporary,
            timestamp_ns=trajectory.timestamp_ns,
            frame_id=trajectory.frame_id,
            c2w=trajectory.c2w,
            valid=trajectory.valid,
            confidence=trajectory.confidence,
            tracking_state=trajectory.tracking_state,
            reset=trajectory.reset,
        )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(path.with_suffix(".json"), trajectory.metadata)


def read_trajectory(path: str | Path) -> PoseTrajectory:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata_path = path.with_suffix(".json")
        metadata: dict[str, Any] = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        return PoseTrajectory(
            timestamp_ns=data["timestamp_ns"],
            frame_id=data["frame_id"],
            c2w=data["c2w"],
            valid=data["valid"],
            confidence=data["confidence"],
            tracking_state=data["tracking_state"],
            reset=data["reset"],
            metadata=metadata,
        )


def validate_prediction(
    trajectory: PoseTrajectory, worker_manifest: dict[str, Any]
) -> None:
    frames = worker_manifest["frames"]
    expected_ids = np.asarray([row["frame_id"] for row in frames], dtype=np.int64)
    expected_times = np.asarray([row["timestamp_ns"] for row in frames], dtype=np.int64)
    if not np.array_equal(trajectory.frame_id, expected_ids):
        raise ValueError("Prediction frame_id values do not exactly match the worker manifest")
    if not np.array_equal(trajectory.timestamp_ns, expected_times):
        raise ValueError("Prediction timestamps do not exactly match the worker manifest")
    valid_poses = trajectory.c2w[trajectory.valid]
    if len(valid_poses):
        bottom = valid_poses[:, 3]
        if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-5):
            raise ValueError("Prediction contains invalid homogeneous pose rows")
        rotations = valid_poses[:, :3, :3]
        gram = np.einsum("nji,njk->nik", rotations, rotations)
        if not np.allclose(gram, np.eye(3), atol=2e-3):
            raise ValueError("Prediction rotations are not orthonormal")
        if np.any(np.linalg.det(rotations) <= 0):
            raise ValueError("Prediction contains an improper rotation")

