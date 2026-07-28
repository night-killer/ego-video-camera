from __future__ import annotations

import numpy as np

from .common import WorkerContext


def run(context: WorkerContext):
    context.mark_model_ready()
    count = len(context.frames)
    time_sec = np.asarray([row["timestamp_ns"] for row in context.frames]) * 1e-9
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, 3] = np.column_stack(
        (0.15 * time_sec, 0.03 * np.sin(time_sec), 0.01 * time_sec)
    )
    angles = 0.02 * time_sec
    poses[:, 0, 0] = np.cos(angles)
    poses[:, 0, 2] = np.sin(angles)
    poses[:, 2, 0] = -np.sin(angles)
    poses[:, 2, 2] = np.cos(angles)
    context.mark_first_prediction()
    return context.expected_trajectory(poses, metadata={"adapter": "mock"})
