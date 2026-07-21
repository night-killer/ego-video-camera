"""Depth Anything 3 camera-only inference adapter."""

from __future__ import annotations

import gc
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

from .alignment import align_predicted_trajectory
from .camera import intrinsics_to_fov_y, scale_intrinsics
from .io_utils import PipelineInputError, atomic_write_json, require_input_file
from .schema import (
    CameraFrame,
    CameraTrajectory,
    SceneSpec,
    VideoSpec,
    load_trajectory,
    save_trajectory,
)
from .video import iter_video_rgb, video_info


DEFAULT_MODEL_DIR = Path("/data/aigc/cyb/zxgu/ckpt/DA3NESTED-GIANT-LARGE")
MAX_SUPPORTED_FRAMES = 180


def _homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    values = np.asarray(extrinsics, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] not in {(3, 4), (4, 4)}:
        raise PipelineInputError(
            f"DA3 extrinsics must have shape (N, 3, 4) or (N, 4, 4), got {values.shape}"
        )
    if values.shape[1:] == (4, 4):
        output = values.copy()
    else:
        output = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
        output[:, :3, :4] = values
    if not np.isfinite(output).all():
        raise PipelineInputError("DA3 extrinsics contain non-finite values")
    if not np.allclose(output[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise PipelineInputError("DA3 extrinsics are not homogeneous affine matrices")
    return output


def _validate_intrinsics(intrinsics: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.shape != (count, 3, 3):
        raise PipelineInputError(f"DA3 intrinsics must have shape {(count, 3, 3)}, got {values.shape}")
    if not np.isfinite(values).all() or np.any(values[:, 0, 0] <= 0) or np.any(values[:, 1, 1] <= 0):
        raise PipelineInputError("DA3 intrinsics contain invalid focal lengths or non-finite values")
    return values


def preflight_da3(
    *,
    video_path: Path,
    gt_trajectory_path: Path,
    da3_root: Path,
    model_dir: Path,
    max_frames: int,
    require_cuda: bool,
) -> dict[str, Any]:
    video = require_input_file(video_path, "GT video")
    gt_path = require_input_file(gt_trajectory_path, "GT trajectory")
    root = Path(da3_root).expanduser().resolve()
    model = Path(model_dir).expanduser().resolve()
    gt = load_trajectory(gt_path)
    info = video_info(video, decode_count=True)
    decoded = int(info["decoded_frames"] or 0)
    errors: list[str] = []
    if max_frames < 2 or max_frames > MAX_SUPPORTED_FRAMES:
        errors.append(
            f"--max-frames must be between 2 and the v1 hard limit "
            f"{MAX_SUPPORTED_FRAMES}, got {max_frames}"
        )
    if gt.trajectory_type != "dense":
        errors.append(f"GT trajectory must be dense, got {gt.trajectory_type!r}")
    if decoded < 2:
        errors.append("DA3 camera prediction requires at least two video frames")
    if decoded != len(gt.frames):
        errors.append(f"decoded video frames ({decoded}) do not match GT trajectory ({len(gt.frames)})")
    effective_limit = min(max(max_frames, 0), MAX_SUPPORTED_FRAMES)
    if decoded > effective_limit:
        errors.append(f"video has {decoded} frames, exceeding the limit {effective_limit}")
    if info["width"] != gt.video.width or info["height"] != gt.video.height:
        errors.append(
            f"video size {info['width']}x{info['height']} does not match GT "
            f"{gt.video.width}x{gt.video.height}"
        )
    encoded_fps = info["fps"]
    if encoded_fps is None or not math.isclose(float(encoded_fps), gt.video.fps, abs_tol=1e-6):
        errors.append(f"video FPS {encoded_fps} does not match GT FPS {gt.video.fps}")
    api_file = root / "src" / "depth_anything_3" / "api.py"
    if not api_file.is_file() or api_file.stat().st_size == 0:
        errors.append(f"DA3 submodule is missing or incomplete: {api_file}")
    for filename in ("config.json", "model.safetensors"):
        checkpoint_file = model / filename
        if not checkpoint_file.is_file() or checkpoint_file.stat().st_size == 0:
            errors.append(f"DA3 checkpoint is missing or empty: {checkpoint_file}")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_version = torch.__version__
    except Exception as exc:  # noqa: BLE001
        cuda_available = False
        torch_version = None
        errors.append(f"PyTorch import failed: {type(exc).__name__}: {exc}")
    if require_cuda and not cuda_available:
        errors.append("CUDA is unavailable")
    return {
        "status": "ready" if not errors else "not_ready",
        "video": info,
        "gt_frame_count": len(gt.frames),
        "max_frames": max_frames,
        "da3_root": str(root),
        "model_dir": str(model),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "errors": errors,
    }


def extract_frames_exact(
    video_path: Path,
    frames_dir: Path,
    *,
    expected_count: int,
    overwrite: bool,
) -> list[Path]:
    target = Path(frames_dir).expanduser().resolve()
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"decoded frame directory already exists (pass --overwrite to decode the current "
                f"video again): {target}"
            )
        shutil.rmtree(target)
    target.mkdir(parents=True)
    paths: list[Path] = []
    for index, rgb in enumerate(iter_video_rgb(video_path)):
        path = target / f"{index:06d}.png"
        Image.fromarray(rgb, mode="RGB").save(path, compress_level=1)
        paths.append(path)
    if len(paths) != expected_count:
        shutil.rmtree(target, ignore_errors=True)
        raise PipelineInputError(
            f"decoded {len(paths)} frames but GT trajectory contains {expected_count} frames"
        )
    return paths


def prediction_to_raw_trajectory(
    *,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    processed_hw: tuple[int, int],
    gt: CameraTrajectory,
    source: dict[str, Any],
) -> CameraTrajectory:
    world_to_camera = _homogeneous_extrinsics(extrinsics)
    raw_K = _validate_intrinsics(intrinsics, len(world_to_camera))
    if len(world_to_camera) != len(gt.frames):
        raise PipelineInputError(
            f"DA3 returned {len(world_to_camera)} cameras for {len(gt.frames)} input frames"
        )
    target_hw = (gt.video.height, gt.video.width)
    target_K = np.stack([scale_intrinsics(K, processed_hw, target_hw) for K in raw_K], axis=0)
    try:
        camera_to_world = np.linalg.inv(world_to_camera)
    except np.linalg.LinAlgError as exc:
        raise PipelineInputError("DA3 returned a singular extrinsic matrix") from exc
    if not np.isfinite(camera_to_world).all():
        raise PipelineInputError("inverted DA3 camera poses contain non-finite values")
    frames = [
        CameraFrame(
            frame_index=gt_frame.frame_index,
            timestamp_seconds=gt_frame.timestamp_seconds,
            camera_to_world=camera_to_world[index].tolist(),
            K=target_K[index].tolist(),
        )
        for index, gt_frame in enumerate(gt.frames)
    ]
    fov_y = intrinsics_to_fov_y(target_K[0], gt.video.height)
    return CameraTrajectory(
        trajectory_type="da3_raw",
        coordinate_system="da3_raw_world",
        scene=gt.scene,
        video=VideoSpec(
            width=gt.video.width,
            height=gt.video.height,
            fps=gt.video.fps,
            fov_y_degrees=fov_y,
        ),
        frames=frames,
        source=source,
    )


def run_da3_camera_prediction(
    *,
    video_path: Path,
    gt_trajectory_path: Path,
    output_dir: Path,
    da3_root: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str = "cuda",
    process_res: int = 504,
    max_frames: int = 180,
    overwrite: bool = False,
) -> dict[str, Any]:
    preflight = preflight_da3(
        video_path=video_path,
        gt_trajectory_path=gt_trajectory_path,
        da3_root=da3_root,
        model_dir=model_dir,
        max_frames=max_frames,
        require_cuda=True,
    )
    if preflight["errors"]:
        raise RuntimeError("DA3 preflight failed: " + "; ".join(preflight["errors"]))
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        target / "camera_prediction.npz",
        target / "raw_trajectory.json",
        target / "aligned_trajectory.json",
        target / "alignment_report.json",
        target / "prediction_manifest.json",
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"DA3 outputs already exist (pass --overwrite): {existing[0]}")

    gt = load_trajectory(gt_trajectory_path)
    frame_paths = extract_frames_exact(
        video_path,
        target / "frames",
        expected_count=len(gt.frames),
        overwrite=overwrite,
    )
    root = Path(da3_root).expanduser().resolve()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    import torch
    from depth_anything_3.api import DepthAnything3

    model = None
    prediction = None
    try:
        model = DepthAnything3.from_pretrained(str(Path(model_dir).expanduser().resolve()))
        model = model.to(device).eval()
        prediction = model.inference(
            image=[str(path) for path in frame_paths],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="middle",
            process_res=process_res,
            process_res_method="upper_bound_resize",
        )
        if prediction.extrinsics is None or prediction.intrinsics is None:
            raise RuntimeError("DA3 prediction did not include camera extrinsics and intrinsics")
        if prediction.processed_images is None:
            raise RuntimeError("DA3 prediction did not include processed image dimensions")
        processed = np.asarray(prediction.processed_images)
        if processed.ndim != 4 or processed.shape[0] != len(frame_paths):
            raise RuntimeError(f"unexpected DA3 processed_images shape: {processed.shape}")
        processed_hw = (int(processed.shape[1]), int(processed.shape[2]))
        raw_extrinsics = np.asarray(prediction.extrinsics)
        raw_intrinsics = np.asarray(prediction.intrinsics)
        raw = prediction_to_raw_trajectory(
            extrinsics=raw_extrinsics,
            intrinsics=raw_intrinsics,
            processed_hw=processed_hw,
            gt=gt,
            source={
                "backend": "depth_anything_3",
                "model_dir": str(Path(model_dir).expanduser().resolve()),
                "video_path": str(Path(video_path).expanduser().resolve()),
                "process_res": process_res,
                "process_res_method": "upper_bound_resize",
                "ref_view_strategy": "middle",
                "use_ray_pose": False,
                "infer_gs": False,
                "processed_hw": list(processed_hw),
            },
        )
        aligned, report = align_predicted_trajectory(raw, gt)
        np.savez_compressed(
            target / "camera_prediction.npz",
            extrinsics=raw_extrinsics,
            intrinsics=raw_intrinsics,
            extrinsics_convention=np.asarray("world_to_camera_opencv_rdf"),
            coordinate_system=np.asarray("da3_raw_world"),
            processed_hw=np.asarray(processed_hw, dtype=np.int64),
            target_hw=np.asarray([gt.video.height, gt.video.width], dtype=np.int64),
            frame_indexes=np.asarray([frame.frame_index for frame in gt.frames], dtype=np.int64),
        )
        save_trajectory(target / "raw_trajectory.json", raw)
        save_trajectory(target / "aligned_trajectory.json", aligned)
        atomic_write_json(target / "alignment_report.json", report)
        manifest = {
            "status": "complete",
            "preflight": preflight,
            "frame_count": len(frame_paths),
            "frames_dir": str(target / "frames"),
            "processed_hw": list(processed_hw),
            "raw_trajectory": str(target / "raw_trajectory.json"),
            "aligned_trajectory": str(target / "aligned_trajectory.json"),
            "alignment_report": str(target / "alignment_report.json"),
            "camera_prediction": str(target / "camera_prediction.npz"),
        }
        atomic_write_json(target / "prediction_manifest.json", manifest)
        return manifest
    finally:
        prediction = None
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
