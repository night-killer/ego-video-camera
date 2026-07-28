from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..trajectory_alignment import umeyama
from ..transforms import Sim3
from .schema import PoseTrajectory


@dataclass(frozen=True)
class StitchEvent:
    window_index: int
    status: str
    overlap_count: int
    scale: float | None
    first_new_frame_id: int | None


def window_slices(frame_count: int, window_size: int, overlap: int) -> list[slice]:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("Expected window_size > overlap >= 0")
    if frame_count == 0:
        return []
    result = []
    start = 0
    while start < frame_count:
        stop = min(frame_count, start + window_size)
        result.append(slice(start, stop))
        if stop == frame_count:
            break
        start = stop - overlap
    return result


def resample_c2w(
    poses: np.ndarray,
    source_times_sec: np.ndarray,
    target_times_sec: np.ndarray,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    source = np.asarray(source_times_sec, dtype=np.float64)
    target = np.asarray(target_times_sec, dtype=np.float64)
    if poses.shape != (len(source), 4, 4):
        raise ValueError("poses and source_times_sec have incompatible shapes")
    if not len(source):
        return np.empty((0, 4, 4), dtype=np.float64)
    if len(source) == 1:
        return np.repeat(poses, len(target), axis=0)
    if np.any(np.diff(source) <= 0):
        raise ValueError("source_times_sec must be strictly increasing")
    query = np.clip(target, source[0], source[-1])
    output = np.repeat(np.eye(4)[None], len(query), axis=0)
    output[:, :3, 3] = np.column_stack(
        [np.interp(query, source, poses[:, axis, 3]) for axis in range(3)]
    )
    output[:, :3, :3] = Slerp(source, Rotation.from_matrix(poses[:, :3, :3]))(
        query
    ).as_matrix()
    return output


def _fallback_transform(source_pose: np.ndarray, target_pose: np.ndarray, scale: float) -> Sim3:
    rotation = target_pose[:3, :3] @ source_pose[:3, :3].T
    translation = target_pose[:3, 3] - scale * (rotation @ source_pose[:3, 3])
    return Sim3(scale=float(scale), rotation=rotation, translation=translation)


def stitch_pose_windows(
    windows: Iterable[PoseTrajectory],
    *,
    timestamp_ns: np.ndarray,
    frame_id: np.ndarray,
) -> PoseTrajectory:
    """Stitch local trajectories using only their shared predicted poses.

    A disconnected window invalidates itself and all following windows. Existing
    output poses are never overwritten, which keeps overlap ownership stable.
    """

    timestamps = np.asarray(timestamp_ns, dtype=np.int64)
    frame_ids = np.asarray(frame_id, dtype=np.int64)
    if len(timestamps) != len(frame_ids) or len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("Target frame ids and timestamps must be unique and equally sized")
    id_to_index = {int(value): index for index, value in enumerate(frame_ids)}
    output = PoseTrajectory.empty_like_frames(timestamps, frame_ids, stitching="prediction_only")
    output.tracking_state = np.full(len(frame_ids), "invalid", dtype="<U64")
    events: list[StitchEvent] = []
    connected = True
    previous_scale = 1.0

    for window_index, window in enumerate(windows):
        unknown = set(int(value) for value in window.frame_id) - set(id_to_index)
        if unknown:
            raise ValueError(f"Window contains unknown frame ids: {sorted(unknown)[:5]}")
        global_indices = np.asarray([id_to_index[int(value)] for value in window.frame_id])
        if window_index == 0:
            transform = Sim3(1.0, np.eye(3), np.zeros(3))
            overlap_count = 0
            status = "origin"
        elif connected:
            overlap_mask = output.valid[global_indices] & window.valid
            overlap_local = np.flatnonzero(overlap_mask)
            overlap_count = len(overlap_local)
            transform = None
            if overlap_count >= 3:
                try:
                    transform = umeyama(
                        window.c2w[overlap_local, :3, 3],
                        output.c2w[global_indices[overlap_local], :3, 3],
                        with_scale=True,
                    )
                    status = "sim3"
                except ValueError:
                    transform = None
            if transform is None and overlap_count:
                first = int(overlap_local[0])
                transform = _fallback_transform(
                    window.c2w[first], output.c2w[global_indices[first]], previous_scale
                )
                status = "single_pose_fallback"
            if transform is None:
                connected = False
                status = "disconnected"
        else:
            transform = None
            overlap_count = 0
            status = "disconnected_after_gap"

        if transform is None:
            events.append(StitchEvent(window_index, status, overlap_count, None, None))
            continue
        previous_scale = transform.scale
        aligned = transform.apply_c2w_poses(window.c2w)
        new_mask = ~output.valid[global_indices] & window.valid
        new_local = np.flatnonzero(new_mask)
        new_global = global_indices[new_local]
        output.c2w[new_global] = aligned[new_local]
        output.valid[new_global] = True
        output.confidence[new_global] = window.confidence[new_local]
        output.tracking_state[new_global] = window.tracking_state[new_local]
        output.reset[new_global] = window.reset[new_local]
        first_new = int(frame_ids[new_global[0]]) if len(new_global) else None
        events.append(
            StitchEvent(window_index, status, overlap_count, transform.scale, first_new)
        )

    output.metadata["stitch_events"] = [event.__dict__ for event in events]
    output.metadata["pose_convention"] = "OpenCV camera-to-world, column-vector transforms"
    return output


def local_window_trajectory(
    frame_rows: list[dict[str, Any]],
    c2w: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
) -> PoseTrajectory:
    count = len(frame_rows)
    if confidence is None:
        confidence = np.ones(count, dtype=np.float64)
    return PoseTrajectory(
        timestamp_ns=np.asarray([row["timestamp_ns"] for row in frame_rows], dtype=np.int64),
        frame_id=np.asarray([row["frame_id"] for row in frame_rows], dtype=np.int64),
        c2w=np.asarray(c2w, dtype=np.float64),
        valid=np.isfinite(c2w).all(axis=(1, 2)),
        confidence=np.asarray(confidence, dtype=np.float64),
    )
