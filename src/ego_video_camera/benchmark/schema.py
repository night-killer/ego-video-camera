from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    METHOD_FAILED = "method_failed"
    TIMEOUT = "timeout"
    OOM = "oom"
    INVALID_OUTPUT = "invalid_output"
    INPUT_ERROR = "input_error"
    EVALUATION_FAILED = "evaluation_failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_ns: int
    image_path: Path
    intrinsic: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.intrinsic is not None:
            matrix = np.asarray(self.intrinsic, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("intrinsic must be a finite 3x3 matrix")
            object.__setattr__(self, "intrinsic", matrix)


@dataclass(frozen=True)
class SequenceRecord:
    dataset_id: str
    sequence_id: str
    clip_dir: Path
    clip_json: Path
    input_path: Path
    duration_sec: float
    target_fps: float
    reference_grade: str
    reference_type: str
    stratum: str
    start_sec: float
    frame_count: int
    input_kind: str

    @property
    def key(self) -> str:
        return f"{self.dataset_id}/{self.sequence_id}"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    family: str
    display_name: str
    adapter: str
    conda_env: str
    repo: Path
    checkpoint_paths: tuple[Path, ...]
    seeds: tuple[int, ...]
    input_intrinsics: str
    causal: bool
    metric_scale: bool
    canonical: bool = True
    subset: tuple[str, ...] = ()
    timeout_sec: int = 7200
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: MethodSpec
    sequence: SequenceRecord
    seed: int
    output_dir: Path


@dataclass
class PoseTrajectory:
    timestamp_ns: np.ndarray
    frame_id: np.ndarray
    c2w: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    tracking_state: np.ndarray | None = None
    reset: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp_ns = np.asarray(self.timestamp_ns, dtype=np.int64)
        self.frame_id = np.asarray(self.frame_id, dtype=np.int64)
        self.c2w = np.asarray(self.c2w, dtype=np.float64)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        count = len(self.timestamp_ns)
        expected = {
            "frame_id": len(self.frame_id),
            "c2w": len(self.c2w),
            "valid": len(self.valid),
            "confidence": len(self.confidence),
        }
        if any(value != count for value in expected.values()):
            raise ValueError(f"Trajectory arrays have inconsistent lengths: {expected}")
        if self.c2w.shape != (count, 4, 4):
            raise ValueError(f"Expected c2w shape {(count, 4, 4)}, got {self.c2w.shape}")
        if count and np.any(np.diff(self.timestamp_ns) < 0):
            raise ValueError("Trajectory timestamps must be non-decreasing")
        if count and len(np.unique(self.frame_id)) != count:
            raise ValueError("Trajectory frame_id values must be unique")
        finite = np.isfinite(self.c2w).all(axis=(1, 2))
        self.valid &= finite
        if self.tracking_state is None:
            self.tracking_state = np.where(self.valid, "tracking", "invalid")
        else:
            self.tracking_state = np.asarray(self.tracking_state, dtype=str)
            if len(self.tracking_state) != count:
                raise ValueError("tracking_state length does not match trajectory")
        if self.reset is None:
            self.reset = np.zeros(count, dtype=bool)
        else:
            self.reset = np.asarray(self.reset, dtype=bool)
            if len(self.reset) != count:
                raise ValueError("reset length does not match trajectory")

    @classmethod
    def empty_like_frames(
        cls, timestamp_ns: np.ndarray, frame_id: np.ndarray, **metadata: Any
    ) -> "PoseTrajectory":
        count = len(timestamp_ns)
        return cls(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            c2w=np.full((count, 4, 4), np.nan, dtype=np.float64),
            valid=np.zeros(count, dtype=bool),
            confidence=np.full(count, np.nan, dtype=np.float64),
            metadata=metadata,
        )

