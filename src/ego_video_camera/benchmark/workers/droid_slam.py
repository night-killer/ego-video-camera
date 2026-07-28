from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator

import numpy as np

from .common import WorkerContext, poses_from_t_q


def _stream(context: WorkerContext) -> Iterator[tuple[int, object, object]]:
    import cv2
    import torch

    for index, row in enumerate(context.frames):
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read {row['image_path']}")
        h0, w0 = image.shape[:2]
        scale = np.sqrt((384.0 * 512.0) / (h0 * w0))
        h1, w1 = int(h0 * scale), int(w0 * scale)
        image = cv2.resize(image, (w1, h1))
        h1, w1 = h1 - h1 % 8, w1 - w1 % 8
        image = torch.as_tensor(image[:h1, :w1]).permute(2, 0, 1)[None]
        intrinsic = np.asarray(row["intrinsic"], dtype=np.float32)
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        vector = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        vector[0::2] *= w1 / w0
        vector[1::2] *= h1 / h0
        yield index, image, vector


def run(context: WorkerContext):
    from droid import Droid

    first = next(_stream(context))
    parameters = context.parameters
    args = SimpleNamespace(
        weights=str(context.checkpoint(0)),
        buffer=int(parameters.get("buffer", 512)),
        image_size=[first[1].shape[2], first[1].shape[3]],
        disable_vis=True,
        stereo=False,
        beta=float(parameters.get("beta", 0.3)),
        filter_thresh=float(parameters.get("filter_thresh", 2.4)),
        warmup=int(parameters.get("warmup", 8)),
        keyframe_thresh=float(parameters.get("keyframe_thresh", 4.0)),
        frontend_thresh=float(parameters.get("frontend_thresh", 16.0)),
        frontend_window=int(parameters.get("frontend_window", 25)),
        frontend_radius=int(parameters.get("frontend_radius", 2)),
        frontend_nms=int(parameters.get("frontend_nms", 1)),
        backend_thresh=float(parameters.get("backend_thresh", 22.0)),
        backend_radius=int(parameters.get("backend_radius", 2)),
        backend_nms=int(parameters.get("backend_nms", 3)),
        upsample=False,
    )
    droid = Droid(args)
    context.mark_model_ready()
    for timestamp, image, intrinsic in _stream(context):
        droid.track(timestamp, image, intrinsics=intrinsic)
    vectors = np.asarray(droid.terminate(_stream(context)), dtype=np.float64)
    context.mark_first_prediction()
    return context.expected_trajectory(
        poses_from_t_q(vectors, quaternion_order="xyzw"),
        metadata={"dense_termination": True, "disable_vis": True},
    )
