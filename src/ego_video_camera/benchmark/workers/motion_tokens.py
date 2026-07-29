from __future__ import annotations

from typing import Any

import numpy as np

from ..windowing import local_window_trajectory, resample_c2w
from .common import decode_camera_9d


def _read_rgb(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    from PIL import Image

    try:
        with Image.open(path) as source:
            source.load()
            return np.asarray(source.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise FileNotFoundError(f"Cannot read {path}") from error


def rgb_clip(frame_rows: list[dict[str, Any]], frame_count: int, resolution: int) -> np.ndarray:
    import cv2

    if not frame_rows:
        raise ValueError("Cannot build a token clip from no frames")
    sample_indices = np.rint(np.linspace(0, len(frame_rows) - 1, frame_count)).astype(int)
    frames = []
    for index in sample_indices:
        path = frame_rows[int(index)]["image_path"]
        image = _read_rgb(path)
        height, width = image.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        image = image[top : top + side, left : left + side]
        frames.append(cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA))
    return np.asarray(frames, dtype=np.uint8)[None]


def camera_window(frame_rows: list[dict[str, Any]], camera_9d: np.ndarray):
    c2w_30hz = decode_camera_9d(np.asarray(camera_9d, dtype=np.float64))
    source_time = np.linspace(0.0, 2.0, len(c2w_30hz), endpoint=False)
    timestamps = np.asarray([row["timestamp_ns"] for row in frame_rows], dtype=np.int64)
    target_time = (timestamps - timestamps[0]).astype(np.float64) * 1e-9
    return local_window_trajectory(
        frame_rows,
        resample_c2w(c2w_30hz, source_time, target_time),
    )
