from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_COLOR_CAMERA = "d400_color_optical_frame"


def read_camera_intrinsics(
    path: str | Path, camera_key: str = DEFAULT_COLOR_CAMERA
) -> dict[str, Any]:
    import cv2

    calibration_path = Path(path)
    storage = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        storage.release()
        raise ValueError(f"Cannot open OpenLORIS calibration: {calibration_path}")
    try:
        camera = storage.getNode(camera_key)
        if camera.empty():
            raise ValueError(
                f"OpenLORIS calibration has no camera {camera_key}: {calibration_path}"
            )
        model = camera.getNode("model").string()
        width = int(round(camera.getNode("width").real()))
        height = int(round(camera.getNode("height").real()))
        matrix = camera.getNode("intrinsics").mat()
    except cv2.error as error:
        raise ValueError(f"Invalid OpenLORIS calibration: {calibration_path}") from error
    finally:
        storage.release()

    if model != "pinhole":
        raise ValueError(f"OpenLORIS camera {camera_key} is not pinhole: {model!r}")
    if width <= 0 or height <= 0:
        raise ValueError(
            f"OpenLORIS camera {camera_key} has invalid size {width}x{height}"
        )
    values = np.asarray(matrix, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(
            f"OpenLORIS camera {camera_key} must contain four finite intrinsics"
        )
    fx, cx, fy, cy = (float(value) for value in values)
    if fx <= 0.0 or fy <= 0.0 or not (0.0 <= cx < width) or not (0.0 <= cy < height):
        raise ValueError(f"OpenLORIS camera {camera_key} has invalid intrinsics {values}")
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "source_calibration_width": width,
        "source_calibration_height": height,
        "intrinsics_source": f"{calibration_path.name}:{camera_key}",
    }
