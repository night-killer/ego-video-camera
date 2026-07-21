"""Standard 3D Gaussian PLY loading and CUDA gsplat rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from plyfile import PlyData

from .io_utils import PipelineInputError, require_input_file
from .schema import CameraTrajectory
from .video import H264Writer


@dataclass(frozen=True)
class GaussianPlyLayout:
    vertex_count: int
    rotation_fields: tuple[str, ...]
    dc_fields: tuple[str, ...]
    rest_fields: tuple[str, ...]
    sh_degree: int


@dataclass(frozen=True)
class GaussianNumpyArrays:
    means: np.ndarray
    quaternions: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    harmonics: np.ndarray
    sh_degree: int


@dataclass
class GaussianScene:
    means: Any
    quaternions: Any
    scales: Any
    opacities: Any
    harmonics: Any
    sh_degree: int
    source_path: Path
    robust_bounds_min: np.ndarray
    robust_bounds_max: np.ndarray

    @property
    def splat_count(self) -> int:
        return int(self.means.shape[0])


def _numbered_fields(names: set[str], prefix: str) -> tuple[str, ...]:
    fields: list[tuple[int, str]] = []
    for name in names:
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            fields.append((int(name[len(prefix) :]), name))
    fields.sort()
    if fields and [index for index, _ in fields] != list(range(len(fields))):
        raise PipelineInputError(f"PLY fields with prefix {prefix!r} are not contiguous")
    return tuple(name for _, name in fields)


def inspect_gaussian_ply(path: Path) -> tuple[np.ndarray, GaussianPlyLayout]:
    source = require_input_file(path, "Gaussian PLY")
    try:
        ply = PlyData.read(str(source), mmap="r")
    except Exception as exc:  # noqa: BLE001 - normalize third-party parser failures.
        raise PipelineInputError(f"failed to read Gaussian PLY {source}: {exc}") from exc
    if "vertex" not in {element.name for element in ply.elements}:
        raise PipelineInputError(f"PLY has no vertex element: {source}")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"}
    missing = sorted(required - names)
    if missing:
        raise PipelineInputError(f"Gaussian PLY is missing fields: {', '.join(missing)}")
    rotation_fields = _numbered_fields(names, "rot_") or _numbered_fields(names, "rotation_")
    if len(rotation_fields) != 4:
        raise PipelineInputError("Gaussian PLY must contain rot_0..rot_3 or rotation_0..rotation_3")
    dc_fields = _numbered_fields(names, "f_dc_")
    if len(dc_fields) != 3:
        raise PipelineInputError("Gaussian PLY must contain f_dc_0..f_dc_2")
    rest_fields = _numbered_fields(names, "f_rest_")
    if len(rest_fields) % 3:
        raise PipelineInputError("Gaussian PLY f_rest field count must be divisible by three")
    coefficients = 1 + len(rest_fields) // 3
    root = int(round(math.sqrt(coefficients)))
    if root * root != coefficients or not 1 <= root <= 4:
        raise PipelineInputError(
            f"unsupported spherical-harmonic coefficient count: {coefficients}"
        )
    layout = GaussianPlyLayout(
        vertex_count=len(vertex),
        rotation_fields=rotation_fields,
        dc_fields=dc_fields,
        rest_fields=rest_fields,
        sh_degree=root - 1,
    )
    if layout.vertex_count <= 0:
        raise PipelineInputError(f"Gaussian PLY contains no vertices: {source}")
    return vertex, layout


def _stack(vertex: np.ndarray, fields: tuple[str, ...], selection: slice | np.ndarray) -> np.ndarray:
    return np.stack(
        [np.asarray(vertex[field][selection], dtype=np.float32) for field in fields], axis=-1
    )


def _chunk_arrays(
    vertex: np.ndarray,
    layout: GaussianPlyLayout,
    selection: slice | np.ndarray,
) -> GaussianNumpyArrays:
    means = _stack(vertex, ("x", "y", "z"), selection)
    log_scales = _stack(vertex, ("scale_0", "scale_1", "scale_2"), selection)
    quaternions = _stack(vertex, layout.rotation_fields, selection)
    opacity_logits = np.asarray(vertex["opacity"][selection], dtype=np.float32)
    dc = _stack(vertex, layout.dc_fields, selection)[:, None, :]
    if layout.rest_fields:
        rest_flat = _stack(vertex, layout.rest_fields, selection)
        rest = rest_flat.reshape(len(means), 3, -1).transpose(0, 2, 1)
        harmonics = np.concatenate([dc, rest], axis=1)
    else:
        harmonics = dc

    raw_arrays = (means, log_scales, quaternions, opacity_logits, harmonics)
    if any(not np.isfinite(array).all() for array in raw_arrays):
        raise PipelineInputError("Gaussian PLY contains non-finite parameter values")
    quaternion_norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(quaternion_norms <= 1e-12):
        raise PipelineInputError("Gaussian PLY contains a zero-length quaternion")
    quaternions = quaternions / quaternion_norms
    scales = np.exp(np.clip(log_scales, -80.0, 80.0)).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-np.clip(opacity_logits, -60.0, 60.0)))).astype(np.float32)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise PipelineInputError("Gaussian PLY contains invalid transformed scales")
    return GaussianNumpyArrays(
        means=np.ascontiguousarray(means),
        quaternions=np.ascontiguousarray(quaternions),
        scales=np.ascontiguousarray(scales),
        opacities=np.ascontiguousarray(opacities),
        harmonics=np.ascontiguousarray(harmonics),
        sh_degree=layout.sh_degree,
    )


def read_gaussian_arrays(path: Path) -> GaussianNumpyArrays:
    """Read a complete PLY into NumPy; intended for small files and tests."""

    vertex, layout = inspect_gaussian_ply(path)
    return _chunk_arrays(vertex, layout, slice(0, layout.vertex_count))


def robust_ply_bounds(
    vertex: np.ndarray,
    layout: GaussianPlyLayout,
    *,
    max_samples: int = 200_000,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = min(layout.vertex_count, max_samples)
    indexes = np.linspace(0, layout.vertex_count - 1, sample_count, dtype=np.int64)
    points = _stack(vertex, ("x", "y", "z"), indexes).astype(np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 8:
        raise PipelineInputError("not enough finite Gaussian positions to estimate scene bounds")
    return (
        np.quantile(points, lower_quantile, axis=0),
        np.quantile(points, upper_quantile, axis=0),
    )


def load_gaussian_scene(
    path: Path,
    *,
    device: str = "cuda:0",
    chunk_size: int = 1_000_000,
) -> GaussianScene:
    import torch

    if not str(device).startswith("cuda"):
        raise RuntimeError("faithful Gaussian rendering requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA; run Gaussian rendering on the GPU machine")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = require_input_file(path, "Gaussian PLY")
    vertex, layout = inspect_gaussian_ply(source)
    bounds_min, bounds_max = robust_ply_bounds(vertex, layout)
    target = torch.device(device)
    means = torch.empty((layout.vertex_count, 3), dtype=torch.float32, device=target)
    quaternions = torch.empty((layout.vertex_count, 4), dtype=torch.float32, device=target)
    scales = torch.empty((layout.vertex_count, 3), dtype=torch.float32, device=target)
    opacities = torch.empty((layout.vertex_count,), dtype=torch.float32, device=target)
    coefficient_count = (layout.sh_degree + 1) ** 2
    harmonics = torch.empty(
        (layout.vertex_count, coefficient_count, 3), dtype=torch.float32, device=target
    )
    for start in range(0, layout.vertex_count, chunk_size):
        end = min(start + chunk_size, layout.vertex_count)
        chunk = _chunk_arrays(vertex, layout, slice(start, end))
        means[start:end].copy_(torch.from_numpy(chunk.means), non_blocking=False)
        quaternions[start:end].copy_(torch.from_numpy(chunk.quaternions), non_blocking=False)
        scales[start:end].copy_(torch.from_numpy(chunk.scales), non_blocking=False)
        opacities[start:end].copy_(torch.from_numpy(chunk.opacities), non_blocking=False)
        harmonics[start:end].copy_(torch.from_numpy(chunk.harmonics), non_blocking=False)
    return GaussianScene(
        means=means,
        quaternions=quaternions,
        scales=scales,
        opacities=opacities,
        harmonics=harmonics,
        sh_degree=layout.sh_degree,
        source_path=source,
        robust_bounds_min=bounds_min,
        robust_bounds_max=bounds_max,
    )


def render_camera(
    scene: GaussianScene,
    camera_to_world: np.ndarray,
    K: np.ndarray,
    *,
    width: int,
    height: int,
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    import torch
    from gsplat.rendering import rasterization

    pose = np.asarray(camera_to_world, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    if pose.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("camera_to_world and K must be 4x4 and 3x3")
    device = scene.means.device
    viewmat = torch.from_numpy(np.linalg.inv(pose).astype(np.float32)).to(device)[None]
    torch_K = torch.from_numpy(intrinsics.astype(np.float32)).to(device)[None]
    background = torch.tensor(background_rgb, dtype=torch.float32, device=device)[None]
    with torch.inference_mode():
        rendered, _, _ = rasterization(
            means=scene.means,
            quats=scene.quaternions,
            scales=scene.scales,
            opacities=scene.opacities,
            colors=scene.harmonics,
            viewmats=viewmat,
            Ks=torch_K,
            width=int(width),
            height=int(height),
            near_plane=0.01,
            far_plane=1.0e10,
            eps2d=0.3,
            sh_degree=scene.sh_degree,
            packed=True,
            backgrounds=background,
            render_mode="RGB",
            rasterize_mode="classic",
            camera_model="pinhole",
        )
    rgb = rendered[0, ..., :3].clamp(0.0, 1.0).detach().float().cpu().numpy()
    return np.ascontiguousarray(np.rint(rgb * 255.0).astype(np.uint8))


def iter_rendered_trajectory(
    scene: GaussianScene,
    trajectory: CameraTrajectory,
) -> Iterator[np.ndarray]:
    for frame in trajectory.frames:
        yield render_camera(
            scene,
            np.asarray(frame.camera_to_world, dtype=np.float64),
            np.asarray(frame.K, dtype=np.float64),
            width=trajectory.video.width,
            height=trajectory.video.height,
        )


def render_trajectory_video(
    scene: GaussianScene,
    trajectory: CameraTrajectory,
    output_path: Path,
    *,
    crf: int = 18,
) -> int:
    with H264Writer(
        output_path,
        width=trajectory.video.width,
        height=trajectory.video.height,
        fps=trajectory.video.fps,
        crf=crf,
    ) as writer:
        for rgb in iter_rendered_trajectory(scene, trajectory):
            writer.write(rgb)
    return writer.frame_count
