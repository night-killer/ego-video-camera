"""Time-aware interpolation of manually authored camera keyframes."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

from .schema import CameraFrame, CameraTrajectory


def interpolate_keyframes(keyframes: CameraTrajectory) -> CameraTrajectory:
    """Create a contiguous dense trajectory on the keyframe trajectory's FPS grid."""

    if keyframes.trajectory_type != "keyframes":
        raise ValueError(f"expected keyframes trajectory, got {keyframes.trajectory_type!r}")
    poses, intrinsics = keyframes.matrices()
    source_indexes = np.asarray([frame.frame_index for frame in keyframes.frames], dtype=np.int64)
    source_times = source_indexes.astype(np.float64) / keyframes.video.fps
    target_indexes = np.arange(source_indexes[0], source_indexes[-1] + 1, dtype=np.int64)
    target_times = target_indexes.astype(np.float64) / keyframes.video.fps

    positions = poses[:, :3, 3]
    if len(poses) == 2:
        alpha = ((target_times - source_times[0]) / (source_times[-1] - source_times[0]))[:, None]
        dense_positions = positions[0] * (1.0 - alpha) + positions[-1] * alpha
        rotations = Slerp(source_times, Rotation.from_matrix(poses[:, :3, :3]))(target_times)
    else:
        dense_positions = CubicSpline(source_times, positions, axis=0, bc_type="natural")(target_times)
        rotations = RotationSpline(source_times, Rotation.from_matrix(poses[:, :3, :3]))(target_times)

    # Intrinsics are global for authored GT cameras, but interpolating them keeps
    # the function well-defined for hand-edited inputs and preserves keyframes.
    dense_intrinsics = np.empty((len(target_times), 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            dense_intrinsics[:, row, column] = np.interp(
                target_times,
                source_times,
                intrinsics[:, row, column],
            )

    dense_poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(target_times), axis=0)
    dense_poses[:, :3, :3] = rotations.as_matrix()
    dense_poses[:, :3, 3] = dense_positions

    source_by_index = {frame.frame_index: position for position, frame in enumerate(keyframes.frames)}
    for dense_position, frame_index in enumerate(target_indexes):
        source_position = source_by_index.get(int(frame_index))
        if source_position is None:
            continue
        # Exact replacement avoids tiny spline evaluation drift at authored knots.
        dense_poses[dense_position] = poses[source_position]
        dense_intrinsics[dense_position] = intrinsics[source_position]

    output_frames = [
        CameraFrame(
            frame_index=int(index - target_indexes[0]),
            timestamp_seconds=float((index - target_indexes[0]) / keyframes.video.fps),
            camera_to_world=dense_poses[position].tolist(),
            K=dense_intrinsics[position].tolist(),
        )
        for position, index in enumerate(target_indexes)
    ]
    return CameraTrajectory(
        trajectory_type="dense",
        coordinate_system=keyframes.coordinate_system,
        scene=keyframes.scene,
        video=keyframes.video,
        frames=output_frames,
        source={
            "method": "cubic_position_rotation_spline" if len(poses) > 2 else "linear_position_slerp",
            "keyframe_count": len(keyframes.frames),
            "source_frame_indexes": source_indexes.tolist(),
            **keyframes.source,
        },
    )

