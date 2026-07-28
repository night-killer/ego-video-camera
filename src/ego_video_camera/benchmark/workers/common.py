from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ...serialization import write_json
from ..schema import PoseTrajectory


_DINOV2_HUB_ALIASES = {
    "facebookresearch/dinov2",
    "facebookresearch/dinov2:main",
    "torchhub/facebookresearch_dinov2_main",
}


@contextmanager
def local_dinov2_hub(repository: Path) -> Iterator[None]:
    hubconf = repository / "hubconf.py"
    if not repository.is_dir() or not hubconf.is_file():
        raise FileNotFoundError(f"Missing local DINOv2 torchhub repository: {repository}")

    import torch

    original_load = torch.hub.load

    def redirected_load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).rstrip("/") in _DINOV2_HUB_ALIASES:
            kwargs["source"] = "local"
            kwargs["pretrained"] = False
            repo_or_dir = str(repository)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = redirected_load
    try:
        yield
    finally:
        torch.hub.load = original_load


@dataclass
class WorkerContext:
    manifest_path: Path
    output_dir: Path
    repo: Path
    checkpoints: tuple[Path, ...]
    manifest: dict[str, Any]
    events_path: Path
    _event_start: float = field(default_factory=time.monotonic)

    @property
    def frames(self) -> list[dict[str, Any]]:
        return self.manifest["frames"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self.manifest.get("parameters", {})

    @property
    def seed(self) -> int:
        return int(self.manifest["seed"])

    @property
    def image_paths(self) -> list[Path]:
        return [Path(row["image_path"]) for row in self.frames]

    def checkpoint(self, index: int) -> Path:
        try:
            return self.checkpoints[index]
        except IndexError as error:
            raise ValueError(f"Adapter requested missing checkpoint index {index}") from error

    def emit(self, event: str, **details: Any) -> None:
        payload = {
            "event": event,
            "monotonic_sec": time.monotonic(),
            "elapsed_sec": time.monotonic() - self._event_start,
            **details,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def mark_model_ready(self) -> None:
        self.emit("model_ready")

    def mark_first_prediction(self) -> None:
        self.emit("first_prediction")

    def stage_frames(self, name: str = "input_frames") -> Path:
        target = self.output_dir / "work" / name
        target.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        for index, source in enumerate(self.image_paths):
            suffix = source.suffix.lower() or ".jpg"
            filename = f"{index:06d}{suffix}"
            expected.add(filename)
            destination = target / filename
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source.resolve())
        for path in target.iterdir():
            if path.name not in expected and (path.is_file() or path.is_symlink()):
                path.unlink()
        return target

    def expected_trajectory(
        self,
        c2w: np.ndarray,
        *,
        valid: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        tracking_state: np.ndarray | None = None,
        reset: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PoseTrajectory:
        count = len(self.frames)
        poses = np.asarray(c2w, dtype=np.float64)
        if poses.shape != (count, 4, 4):
            raise ValueError(f"Adapter returned poses {poses.shape}, expected {(count, 4, 4)}")
        if valid is None:
            valid = np.isfinite(poses).all(axis=(1, 2))
        if confidence is None:
            confidence = np.where(valid, 1.0, np.nan)
        payload = {
            "method_id": self.manifest["method_id"],
            "seed": self.seed,
            "pose_convention": "OpenCV camera-to-world, column-vector transforms",
            **(metadata or {}),
        }
        return PoseTrajectory(
            timestamp_ns=np.asarray([row["timestamp_ns"] for row in self.frames]),
            frame_id=np.asarray([row["frame_id"] for row in self.frames]),
            c2w=poses,
            valid=np.asarray(valid, dtype=bool),
            confidence=np.asarray(confidence, dtype=np.float64),
            tracking_state=tracking_state,
            reset=reset,
            metadata=payload,
        )


def configure_process(context: WorkerContext) -> None:
    random.seed(context.seed)
    np.random.seed(context.seed)
    os.environ["PYTHONHASHSEED"] = str(context.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for path in (context.repo, context.repo / "src"):
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)
    try:
        import torch

        torch.manual_seed(context.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(context.seed)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def w2c_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    values = np.asarray(extrinsics, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected Nx3x4 or Nx4x4 extrinsics, got {values.shape}")
    homogeneous = np.repeat(np.eye(4)[None], len(values), axis=0)
    homogeneous[:, :3, :4] = values[:, :3, :4]
    return np.linalg.inv(homogeneous)


def poses_from_t_q(vectors: np.ndarray, *, quaternion_order: str = "xyzw") -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Expected Nx7 pose vectors, got {values.shape}")
    quaternions = values[:, 3:]
    if quaternion_order == "wxyz":
        quaternions = quaternions[:, [1, 2, 3, 0]]
    elif quaternion_order != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {quaternion_order}")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Pose vectors contain invalid quaternions")
    x, y, z, w = (quaternions / norms).T
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :3] = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    output[:, :3, 3] = values[:, :3]
    return output


def decode_camera_9d(values: np.ndarray) -> np.ndarray:
    camera = np.asarray(values, dtype=np.float64)
    if camera.ndim == 3 and camera.shape[1:] == (4, 4):
        return camera.copy()
    if camera.ndim != 2 or camera.shape[1] != 9:
        raise ValueError(f"Expected Tx9 or Tx4x4 camera prediction, got {camera.shape}")
    packed = camera.reshape(-1, 3, 3).transpose(0, 2, 1)
    first, second = packed[:, :, 0], packed[:, :, 1]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    output = np.repeat(np.eye(4)[None], len(camera), axis=0)
    output[:, :3, :3] = np.stack((first, second, third), axis=-1)
    output[:, :3, 3] = packed[:, :, 2]
    return output


def load_context(
    manifest_path: str | Path,
    output_dir: str | Path,
    repo: str | Path,
    checkpoints: list[str | Path],
) -> WorkerContext:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forbidden = {"clip_json", "reference", "groundtruth", "ground_truth", "gt_path"}
    if forbidden & set(manifest):
        raise ValueError("Worker manifest contains forbidden reference fields")
    context = WorkerContext(
        manifest_path=manifest_path,
        output_dir=Path(output_dir).resolve(),
        repo=Path(repo).resolve(),
        checkpoints=tuple(Path(path).resolve() for path in checkpoints),
        manifest=manifest,
        events_path=Path(output_dir).resolve() / "worker_events.jsonl",
    )
    configure_process(context)
    return context


def write_worker_summary(context: WorkerContext, value: dict[str, Any]) -> None:
    write_json(context.output_dir / "worker_summary.json", value)
