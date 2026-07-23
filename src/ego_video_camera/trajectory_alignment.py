from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import Sim3


@dataclass(frozen=True)
class AlignmentResult:
    status: str
    transform: Sim3 | None
    used_count: int
    used_duration_sec: float | None
    reason: str | None = None


def umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> Sim3:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError("At least three points are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Source trajectory has zero spatial variance")
    scale = float(np.sum(singular_values * signs) / variance) if with_scale else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid estimated scale: {scale}")
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def trajectory_excitation(points: np.ndarray) -> dict[str, float | int]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return {"count": len(points), "span_m": 0.0, "rank_ratio": 0.0}
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    span = float(np.max(np.linalg.norm(points[:, None] - points[None, :], axis=-1)))
    ratio = float(singular[1] / singular[0]) if len(singular) > 1 and singular[0] > 1e-12 else 0.0
    return {"count": len(points), "span_m": span, "rank_ratio": ratio}


def estimate_prefix_alignment(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    timestamps_sec: np.ndarray,
    initial_sec: float = 3.0,
    fallback_sec: float = 5.0,
    maximum_prefix_ratio: float = 0.30,
    minimum_span_m: float = 0.10,
    minimum_rank_ratio: float = 0.001,
    timeline_start_sec: float | None = None,
    timeline_end_sec: float | None = None,
) -> AlignmentResult:
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if len(source_c2w) != len(target_c2w) or len(timestamps) != len(source_c2w):
        raise ValueError("Source, target and timestamps must have equal lengths")
    if len(timestamps) < 3:
        return AlignmentResult(
            status="invalid",
            transform=None,
            used_count=len(timestamps),
            used_duration_sec=None,
            reason="Fewer than three valid poses are available for prefix alignment",
        )
    timeline_start = (
        float(timestamps[0]) if timeline_start_sec is None else float(timeline_start_sec)
    )
    timeline_end = (
        float(timestamps[-1]) if timeline_end_sec is None else float(timeline_end_sec)
    )
    if timeline_end < timeline_start:
        raise ValueError("Prefix alignment timeline ends before it starts")
    duration = timeline_end - timeline_start
    maximum_sec = max(initial_sec, duration * maximum_prefix_ratio)
    candidates = sorted(set([initial_sec, min(fallback_sec, maximum_sec), maximum_sec]))
    for seconds in candidates:
        indices = np.flatnonzero(
            (timestamps >= timeline_start - 1e-9)
            & (timestamps - timeline_start <= seconds + 1e-9)
        )
        if len(indices) < 3:
            continue
        source_points = source_c2w[indices, :3, 3]
        target_points = target_c2w[indices, :3, 3]
        excitation = trajectory_excitation(source_points)
        target_excitation = trajectory_excitation(target_points)
        if (
            excitation["span_m"] >= minimum_span_m
            and target_excitation["span_m"] >= minimum_span_m
            and excitation["rank_ratio"] >= minimum_rank_ratio
            and target_excitation["rank_ratio"] >= minimum_rank_ratio
        ):
            try:
                transform = umeyama(source_points, target_points, with_scale=True)
            except ValueError:
                continue
            return AlignmentResult("ok", transform, len(indices), float(seconds))
    return AlignmentResult(
        status="degenerate",
        transform=None,
        used_count=0,
        used_duration_sec=None,
        reason="Insufficient translation or spatial rank within the allowed prefix",
    )


def estimate_full_alignment(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    with_scale: bool,
    minimum_span_m: float = 0.0,
    minimum_rank_ratio: float = 0.0,
) -> AlignmentResult:
    valid = np.isfinite(source_c2w).all(axis=(1, 2)) & np.isfinite(target_c2w).all(axis=(1, 2))
    if valid.sum() < 3:
        return AlignmentResult("invalid", None, int(valid.sum()), None, "Fewer than three valid poses")
    source_points = source_c2w[valid, :3, 3]
    target_points = target_c2w[valid, :3, 3]
    source_excitation = trajectory_excitation(source_points)
    target_excitation = trajectory_excitation(target_points)
    if (
        source_excitation["span_m"] < minimum_span_m
        or target_excitation["span_m"] < minimum_span_m
        or source_excitation["rank_ratio"] < minimum_rank_ratio
        or target_excitation["rank_ratio"] < minimum_rank_ratio
    ):
        return AlignmentResult(
            "degenerate",
            None,
            int(valid.sum()),
            None,
            (
                "Insufficient full-trajectory excitation: "
                f"source span={source_excitation['span_m']:.6g} m, "
                f"target span={target_excitation['span_m']:.6g} m, "
                f"source rank_ratio={source_excitation['rank_ratio']:.6g}, "
                f"target rank_ratio={target_excitation['rank_ratio']:.6g}"
            ),
        )
    try:
        transform = umeyama(source_points, target_points, with_scale)
    except ValueError as error:
        return AlignmentResult("degenerate", None, int(valid.sum()), None, str(error))
    return AlignmentResult("ok", transform, int(valid.sum()), None)
