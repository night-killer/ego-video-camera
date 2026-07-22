from __future__ import annotations

import copy
import gc
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from .download import sha256_file
from .serialization import write_json
from .transforms import invert_pose


POSE_CONVENTION = "OpenCV camera-to-DA3-world (T_D_E), converted from official stitched W2C output"
EXPECTED_DA3_COMMIT = "41736238f5bced4debf3f2a12375d2466874866d"


@lru_cache(maxsize=4)
def _checkpoint_sha256(weight_path: str) -> str:
    return sha256_file(weight_path)


def activate_da3_source(repo_root: str | Path, source_root: str | Path) -> Path:
    source_root = Path(source_root)
    if not source_root.is_absolute():
        source_root = Path(repo_root) / source_root
    source_root = source_root.resolve()
    source_python = source_root / "src"
    if not (source_python / "depth_anything_3").is_dir():
        raise FileNotFoundError(f"DA3 source package is missing: {source_python}")
    for name in list(sys.modules):
        if name == "depth_anything_3" or name.startswith("depth_anything_3."):
            del sys.modules[name]
    normalized = str(source_python)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry).resolve()) != normalized]
    sys.path.insert(0, normalized)
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_DA3_COMMIT:
        raise RuntimeError(f"DA3 submodule commit mismatch: {commit} != {EXPECTED_DA3_COMMIT}")
    return source_root


def load_model_only(repo_root: str | Path, source_root: str | Path, checkpoint_path: str | Path):
    activate_da3_source(repo_root, source_root)
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(str(Path(checkpoint_path).resolve()), local_files_only=True)
    model.eval()
    return model


def _load_streaming_module(source_root: Path):
    streaming_root = source_root / "da3_streaming"
    sys.path.insert(0, str(streaming_root))
    module_path = streaming_root / "da3_streaming.py"
    spec = importlib.util.spec_from_file_location("ego_video_camera._official_da3_streaming", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DA3-Streaming from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _streaming_config(source_root: Path, checkpoint: Path, chunk_size: int, overlap: int) -> dict:
    config_path = source_root / "da3_streaming" / "configs" / "base_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["Weights"]["DA3"] = str(checkpoint / "model.safetensors")
    config["Weights"]["DA3_CONFIG"] = str(checkpoint / "config.json")
    config["Model"].update(
        {
            "chunk_size": int(chunk_size),
            "overlap": int(overlap),
            "loop_enable": False,
            "delete_temp_files": False,
            "save_depth_conf_result": True,
            "save_debug_info": True,
            "ref_view_strategy": "middle",
            "ref_view_strategy_loop": "middle",
        }
    )
    return config


def _effective_streaming_overlap(frame_count: int, chunk_size: int, overlap: int) -> int:
    if frame_count <= 0:
        raise ValueError("DA3 requires at least one input frame")
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"Invalid DA3 streaming window: chunk_size={chunk_size}, overlap={overlap}"
        )
    return 0 if frame_count <= chunk_size else overlap


def _make_ray_streaming_class(module, process_res: int):
    class RayPoseStreaming(module.DA3_Streaming):
        def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
            start_idx, end_idx = range_1
            chunk_image_paths = self.img_list[start_idx:end_idx]
            if range_2 is not None:
                start_2, end_2 = range_2
                chunk_image_paths += self.img_list[start_2:end_2]
            ref_view_strategy = self.config["Model"][
                "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
            ]
            torch.cuda.empty_cache()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                predictions = self.model.inference(
                    chunk_image_paths,
                    ref_view_strategy=ref_view_strategy,
                    use_ray_pose=True,
                    process_res=process_res,
                )
                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf = predictions.conf - 1.0
            torch.cuda.empty_cache()
            if is_loop:
                save_dir = self.result_loop_dir
                filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
            else:
                if chunk_idx is None:
                    raise ValueError("chunk_idx must be supplied")
                save_dir = self.result_unaligned_dir
                filename = f"chunk_{chunk_idx}.npy"
            if not is_loop and range_2 is None:
                chunk_range = self.chunk_indices[chunk_idx]
                self.all_camera_poses.append((chunk_range, predictions.extrinsics))
                self.all_camera_intrinsics.append((chunk_range, predictions.intrinsics))
            np.save(os.path.join(save_dir, filename), predictions)
            return predictions

    RayPoseStreaming.__name__ = "RayPoseStreaming"
    return RayPoseStreaming


def _prepare_inputs(image_paths: Sequence[str | Path], output_dir: Path) -> tuple[Path, list[Path]]:
    input_dir = output_dir / "da3_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"{index:06d}.jpg" for index, _ in enumerate(image_paths)
    }
    for existing in input_dir.iterdir():
        if existing.name not in expected_names and (existing.is_symlink() or existing.is_file()):
            existing.unlink()
    staged: list[Path] = []
    for index, source_value in enumerate(image_paths):
        source = Path(source_value).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing DA3 ego RGB input: {source}")
        destination = input_dir / f"{index:06d}.jpg"
        if destination.is_symlink():
            if destination.resolve() != source:
                destination.unlink()
        elif destination.exists():
            raise RuntimeError(f"DA3 staging path is not a managed symlink: {destination}")
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(source)
        staged.append(destination)
    return input_dir, staged


def _read_stitched_outputs(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    poses = []
    for line in (output_dir / "camera_poses.txt").read_text(encoding="utf-8").splitlines():
        poses.append(np.fromstring(line, sep=" ", dtype=np.float64).reshape(4, 4))
    intrinsics = []
    for line in (output_dir / "intrinsic.txt").read_text(encoding="utf-8").splitlines():
        fx, fy, cx, cy = np.fromstring(line, sep=" ", dtype=np.float64)
        intrinsics.append(np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64))
    return np.asarray(poses), np.asarray(intrinsics)


def _read_confidence_medians(
    streaming_dir: Path,
    chunk_indices: Sequence[tuple[int, int]],
    frame_count: int,
) -> np.ndarray:
    values = np.full(frame_count, np.nan, dtype=np.float64)
    result_dir = streaming_dir / "results_output"
    for index in range(frame_count):
        confidence_file = result_dir / f"frame_{index}.npz"
        if confidence_file.is_file():
            with np.load(confidence_file) as data:
                if "conf" in data:
                    values[index] = float(np.median(data["conf"]))
    if np.isfinite(values).all():
        return values
    # Official streaming does not emit results_output for a one-chunk run.
    # Recover only the normalized confidence medians from its unmodified
    # per-chunk Prediction object; no pose or GT data is involved.
    for chunk_index, (start, end) in enumerate(chunk_indices):
        path = streaming_dir / "_tmp_results_unaligned" / f"chunk_{chunk_index}.npy"
        if not path.is_file():
            continue
        prediction = np.load(path, allow_pickle=True).item()
        confidence = np.asarray(prediction.conf)
        for local_index, global_index in enumerate(range(start, end)):
            if not np.isfinite(values[global_index]):
                values[global_index] = float(np.median(confidence[local_index]))
    return values


def run_da3_streaming(
    *,
    repo_root: str | Path,
    source_root: str | Path,
    checkpoint_path: str | Path,
    image_paths: Sequence[str | Path],
    frame_ids: Sequence[int],
    timestamps: Sequence[int],
    output_dir: str | Path,
    process_res: int = 504,
    chunk_size: int = 60,
    overlap: int = 30,
    confidence_threshold: float = 1.5,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Real DA3 inference requires a CUDA GPU; use mock mode on this CPU host")
    if len(image_paths) != len(frame_ids) or len(image_paths) != len(timestamps):
        raise ValueError("image_paths, frame_ids and timestamps must have equal lengths")
    repo_root = Path(repo_root).resolve()
    source_root_resolved = activate_da3_source(repo_root, source_root)
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_dir, staged = _prepare_inputs(image_paths, output)
    module = _load_streaming_module(source_root_resolved)
    cls = _make_ray_streaming_class(module, process_res)
    effective_overlap = _effective_streaming_overlap(len(image_paths), chunk_size, overlap)
    config = _streaming_config(
        source_root_resolved, checkpoint, chunk_size, effective_overlap
    )
    write_json(
        output / "da3_resolved_config.json",
        {
            "source_commit": EXPECTED_DA3_COMMIT,
            "checkpoint_status": "user_validated_local",
            "checkpoint_path": str(checkpoint),
            "checkpoint_weight_size": (checkpoint / "model.safetensors").stat().st_size,
            "checkpoint_weight_mtime_ns": (checkpoint / "model.safetensors").stat().st_mtime_ns,
            "use_ray_pose": True,
            "process_res": process_res,
            "chunk_size": chunk_size,
            "requested_overlap": overlap,
            "effective_overlap": effective_overlap,
            "loop_closure": False,
            "confidence_threshold": confidence_threshold,
            "input_count": len(staged),
        },
    )
    runner = cls(str(input_dir), str(output / "da3_streaming"), copy.deepcopy(config))
    runner.run()
    c2w, intrinsics = _read_stitched_outputs(output / "da3_streaming")
    if len(c2w) != len(image_paths):
        raise RuntimeError(f"DA3 returned {len(c2w)} poses for {len(image_paths)} images")
    records = []
    confidence_values = _read_confidence_medians(
        output / "da3_streaming", runner.chunk_indices, len(image_paths)
    )
    for index, (pose, intrinsic) in enumerate(zip(c2w, intrinsics)):
        normalized_confidence = (
            float(confidence_values[index]) if np.isfinite(confidence_values[index]) else None
        )
        valid = bool(
            np.isfinite(pose).all()
            and normalized_confidence is not None
            and normalized_confidence >= confidence_threshold
        )
        records.append(
            {
                "frame_id": int(frame_ids[index]),
                "timestamp": int(timestamps[index]),
                "source_image": str(Path(image_paths[index]).resolve()),
                "raw_extrinsics": invert_pose(pose),
                "stitched_c2w": pose,
                "predicted_intrinsics": intrinsic,
                "confidence_raw_median": None if normalized_confidence is None else normalized_confidence + 1.0,
                "confidence_normalized_median": normalized_confidence,
                "low_confidence": normalized_confidence is None or normalized_confidence < confidence_threshold,
                "valid": valid,
                "interpolated": False,
                "pose_convention": POSE_CONVENTION,
            }
        )
    config_json = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    metadata = {
        "model_name": config_json.get("model_name"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_status": "user_validated_local",
        "checkpoint_sha256": _checkpoint_sha256(str(checkpoint / "model.safetensors")),
        "source_root": str(source_root_resolved),
        "source_commit": EXPECTED_DA3_COMMIT,
        "use_ray_pose": True,
        "input_resolution": process_res,
        "window_size": chunk_size,
        "requested_window_overlap": overlap,
        "effective_window_overlap": effective_overlap,
        "loop_closure": False,
        "input_count": len(staged),
        "pose_convention": POSE_CONVENTION,
        "records": records,
    }
    write_json(output / "da3_poses_raw.json", metadata)
    np.savez_compressed(
        output / "da3_poses_raw.npz",
        c2w=c2w,
        w2c=np.asarray([invert_pose(pose) for pose in c2w]),
        intrinsics=intrinsics,
        confidence=confidence_values,
        frame_ids=np.asarray(frame_ids),
        timestamps=np.asarray(timestamps),
    )
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return metadata
