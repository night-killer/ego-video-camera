from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_egobody_json(cls, path: str | Path) -> "CameraModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "camera_mtx" in data:
            matrix = np.asarray(data["camera_mtx"], dtype=np.float64)
        else:
            fx, fy = data["f"]
            cx, cy = data["c"]
            matrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        distortion = np.asarray(data.get("k", []), dtype=np.float64).reshape(-1)
        width = data.get("width") or data.get("w")
        height = data.get("height") or data.get("h")
        return cls(matrix=matrix, distortion=distortion, width=width, height=height)

    def project(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
        pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
        if np.any(valid):
            projected, _ = cv2.projectPoints(
                points[valid], np.zeros(3), np.zeros(3), self.matrix, self.distortion
            )
            pixels[valid] = projected.reshape(-1, 2)
        return pixels, valid

    def inside(self, pixels: np.ndarray, width: int, height: int) -> np.ndarray:
        pixels = np.asarray(pixels)
        return (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
