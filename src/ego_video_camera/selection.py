from __future__ import annotations

import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .download import extract_members
from .egobody_io import (
    EXO_IMAGE_RE,
    PV_IMAGE_RE,
    PVRecord,
    load_T_K_W,
    load_master_camera,
    parse_pv_file,
    read_dataset_metadata,
    sample_records,
)
from .serialization import write_json
from .transforms import rotation_error_deg

if TYPE_CHECKING:
    from .remote_zip import RemoteZipCache


@dataclass
class ClipCandidate:
    recording_name: str
    hololens_sequence: str
    start_sec: float
    duration_sec: float
    start_frame: int
    end_frame: int
    start_timestamp: int
    end_timestamp: int
    source_fps: float
    sample_fps: float
    frame_count: int
    expected_frame_count: int
    ego_sampling_ratio: float
    total_translation_m: float
    mean_angular_velocity_deg_s: float
    p95_angular_velocity_deg_s: float
    turn_excursion_deg: float
    visible_ratio: float
    synchronized_ratio: float
    missing_ratio: float
    texture_score: float | None = None
    blur_score: float | None = None
    ego_motion_score: float | None = None
    exo_motion_score: float | None = None
    ego_nonrigid_motion_score: float | None = None
    exo_nonrigid_motion_score: float | None = None
    difficulty: str | None = None
    selection_reason: str | None = None
    frame_ids: list[int] = field(default_factory=list)
    timestamps: list[int] = field(default_factory=list)
    ego_members: list[str] = field(default_factory=list, repr=False)
    exo_members: list[str] = field(default_factory=list, repr=False)

    def public_dict(self) -> dict:
        data = asdict(self)
        data.pop("ego_members", None)
        data.pop("exo_members", None)
        return data


def _parts_after(name: str, marker: str) -> tuple[str, ...] | None:
    parts = PurePosixPath(name).parts
    try:
        index = parts.index(marker)
    except ValueError:
        return None
    return parts[index + 1 :]


def index_ego_archive(path: str | Path) -> dict[tuple[str, str, int], tuple[int, str]]:
    index: dict[tuple[str, str, int], tuple[int, str]] = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            tail = _parts_after(item.filename, "egocentric_color")
            if tail is None or len(tail) < 4 or tail[-2] != "PV":
                continue
            match = PV_IMAGE_RE.match(tail[-1])
            if match:
                index[(tail[0], tail[1], int(match.group("timestamp")))] = (
                    int(match.group("frame")),
                    item.filename,
                )
    return index


def index_exo_archive(path: str | Path) -> dict[tuple[str, int], str]:
    index: dict[tuple[str, int], str] = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            tail = _parts_after(item.filename, "kinect_color")
            if tail is None or len(tail) < 3 or tail[1] != "master":
                continue
            match = EXO_IMAGE_RE.match(tail[-1])
            if match:
                index[(tail[0], int(match.group("frame")))] = item.filename
    return index


def _modality_relative(member: str, marker: str) -> Path:
    parts = PurePosixPath(member).parts
    return Path(*parts[parts.index(marker) :])


def _attach_archive_images(
    records: list[PVRecord],
    recording: str,
    sequence: str,
    ego_index: dict[tuple[str, str, int], tuple[int, str]],
    data_root: Path,
) -> list[PVRecord]:
    attached = []
    for record in records:
        indexed = ego_index.get((recording, sequence, record.timestamp))
        if indexed:
            frame_id, member = indexed
            attached.append(
                PVRecord(
                    record.timestamp,
                    record.fx,
                    record.fy,
                    record.T_W_E,
                    frame_id,
                    data_root / _modality_relative(member, "egocentric_color"),
                )
            )
    return attached


def _trajectory_stats(sampled: list[PVRecord], T_K_W: np.ndarray, camera, exo_index, recording) -> dict:
    timestamps = np.asarray([r.timestamp for r in sampled], dtype=np.float64) / 10_000_000.0
    poses = np.asarray([T_K_W @ r.T_W_E for r in sampled])
    positions = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(timestamps)
    angular_steps = np.asarray(
        [rotation_error_deg(a[:3, :3], b[:3, :3]) for a, b in zip(poses[:-1], poses[1:])]
    )
    angular_velocity = angular_steps / np.maximum(dt, 1e-6)
    pixels, z_valid = camera.project(positions)
    width = camera.width or 1920
    height = camera.height or 1080
    visible = z_valid & camera.inside(pixels, width, height)
    synchronized = np.asarray([(recording, r.frame_id) in exo_index for r in sampled], dtype=bool)
    return {
        "poses": poses,
        "total_translation_m": float(steps.sum()),
        "mean_angular_velocity_deg_s": float(angular_velocity.mean()) if len(angular_velocity) else 0.0,
        "p95_angular_velocity_deg_s": float(np.percentile(angular_velocity, 95)) if len(angular_velocity) else 0.0,
        "turn_excursion_deg": max(
            (rotation_error_deg(poses[0, :3, :3], pose[:3, :3]) for pose in poses), default=0.0
        ),
        "visible_ratio": float(visible.mean()),
        "synchronized_ratio": float(synchronized.mean()),
        "missing_ratio": float(1.0 - synchronized.mean()),
    }


def _fixed_window_starts(
    total_duration_sec: float,
    window_duration_sec: float,
    stride_sec: float,
) -> np.ndarray:
    """Return starts only for complete, fixed-duration candidate windows."""

    if total_duration_sec + 1e-6 < window_duration_sec:
        return np.empty(0, dtype=np.float64)
    return np.arange(
        0.0,
        total_duration_sec - window_duration_sec + 1e-6,
        stride_sec,
        dtype=np.float64,
    )


def build_candidates(
    data_root: str | Path,
    duration_sec: float = 20.0,
    sample_fps: float = 8.0,
    stride_sec: float = 10.0,
    archive_paths: dict[str, Path] | None = None,
) -> tuple[list[ClipCandidate], dict, dict]:
    root = Path(data_root)
    info, split_by_recording = read_dataset_metadata(root)
    archive_paths = archive_paths or {}
    ego_archive = archive_paths.get(
        "egocentric_color.zip", root / "_archives" / "egocentric_color.zip"
    )
    exo_archive = archive_paths.get(
        "kinect_color.zip", root / "_archives" / "kinect_color.zip"
    )
    ego_index = index_ego_archive(ego_archive)
    exo_index = index_exo_archive(exo_archive)
    camera, _ = load_master_camera(root)
    row_by_recording = info.set_index("recording_name").to_dict("index")
    candidates: list[ClipCandidate] = []
    for pv_path in sorted(root.glob("**/*_pv.txt")):
        parts = pv_path.parts
        if "egocentric_color" not in parts:
            continue
        marker = parts.index("egocentric_color")
        if len(parts) <= marker + 2:
            continue
        recording, sequence = parts[marker + 1], parts[marker + 2]
        if split_by_recording.get(recording) not in {"train", "val"}:
            continue
        metadata = row_by_recording.get(recording)
        if metadata is None:
            continue
        _, raw_records = parse_pv_file(pv_path)
        records = _attach_archive_images(raw_records, recording, sequence, ego_index, root)
        records = [
            record
            for record in records
            if int(metadata["start_frame"]) <= int(record.frame_id) <= int(metadata["end_frame"])
        ]
        if len(records) < 3:
            continue
        seconds = (records[-1].timestamp - records[0].timestamp) / 10_000_000.0
        if seconds + 1e-6 < duration_sec:
            continue
        try:
            T_K_W, _ = load_T_K_W(root, recording)
        except (FileNotFoundError, ValueError):
            continue
        starts = _fixed_window_starts(seconds, duration_sec, stride_sec)
        for start in starts:
            sampled = sample_records(records, sample_fps, float(start), duration_sec)
            expected_frame_count = int(round(sample_fps * duration_sec))
            if len(sampled) < max(20, int(expected_frame_count * 0.7)):
                continue
            stats = _trajectory_stats(sampled, T_K_W, camera, exo_index, recording)
            ego_sampling_ratio = min(1.0, len(sampled) / expected_frame_count)
            stats["visible_ratio"] *= ego_sampling_ratio
            stats["synchronized_ratio"] *= ego_sampling_ratio
            stats["missing_ratio"] = 1.0 - stats["synchronized_ratio"]
            ego_members = [ego_index[(recording, sequence, r.timestamp)][1] for r in sampled]
            exo_members = [exo_index[(recording, r.frame_id)] for r in sampled if (recording, r.frame_id) in exo_index]
            dt = np.diff([r.timestamp for r in records]) / 10_000_000.0
            candidates.append(
                ClipCandidate(
                    recording_name=recording,
                    hololens_sequence=sequence,
                    start_sec=float(start),
                    duration_sec=float(duration_sec),
                    start_frame=int(sampled[0].frame_id),
                    end_frame=int(sampled[-1].frame_id),
                    start_timestamp=sampled[0].timestamp,
                    end_timestamp=sampled[-1].timestamp,
                    source_fps=float(1.0 / np.median(dt)) if len(dt) else sample_fps,
                    sample_fps=sample_fps,
                    frame_count=len(sampled),
                    expected_frame_count=expected_frame_count,
                    ego_sampling_ratio=ego_sampling_ratio,
                    total_translation_m=stats["total_translation_m"],
                    mean_angular_velocity_deg_s=stats["mean_angular_velocity_deg_s"],
                    p95_angular_velocity_deg_s=stats["p95_angular_velocity_deg_s"],
                    turn_excursion_deg=stats["turn_excursion_deg"],
                    visible_ratio=stats["visible_ratio"],
                    synchronized_ratio=stats["synchronized_ratio"],
                    missing_ratio=stats["missing_ratio"],
                    frame_ids=[int(record.frame_id) for record in sampled],
                    timestamps=[int(record.timestamp) for record in sampled],
                    ego_members=ego_members,
                    exo_members=exo_members,
                )
            )
    return candidates, ego_index, exo_index


def _robust_normalize(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values_array, [5, 95]) if len(values_array) > 1 else (0.0, 1.0)
    return np.clip((values_array - low) / max(high - low, 1e-9), 0, 1)


def _trajectory_rankings(candidates: list[ClipCandidate], pool_size: int) -> dict[str, list[ClipCandidate]]:
    eligible = [c for c in candidates if c.visible_ratio >= 0.80 and c.missing_ratio <= 0.05]
    if len({candidate.recording_name for candidate in eligible}) < 3:
        eligible = [c for c in candidates if c.visible_ratio >= 0.70 and c.missing_ratio <= 0.10]
    if len({candidate.recording_name for candidate in eligible}) < 3:
        eligible = sorted(
            [candidate for candidate in candidates if candidate.visible_ratio > 0.0],
            key=lambda candidate: (-candidate.visible_ratio, candidate.missing_ratio),
        )
    if len({candidate.recording_name for candidate in eligible}) < 3:
        raise RuntimeError("Fewer than three recordings contain a trackable candidate window")
    trans = _robust_normalize([c.total_translation_m for c in eligible])
    angular = _robust_normalize([c.p95_angular_velocity_deg_s for c in eligible])
    turn = _robust_normalize([c.turn_excursion_deg for c in eligible])
    easy_score = trans + angular + np.asarray([1 - c.visible_ratio for c in eligible])
    medium_score = -trans - turn + 0.4 * angular + np.asarray([1 - c.visible_ratio for c in eligible])
    hard_score = -angular - 0.5 * turn + np.asarray([1 - c.synchronized_ratio for c in eligible])
    def diverse_pool(score: np.ndarray) -> list[ClipCandidate]:
        result = []
        recordings = set()
        for index in np.argsort(score):
            candidate = eligible[index]
            if candidate.recording_name in recordings:
                continue
            result.append(candidate)
            recordings.add(candidate.recording_name)
            if len(result) >= pool_size:
                break
        return result

    return {
        "easy": diverse_pool(easy_score),
        "medium": diverse_pool(medium_score),
        "hard": diverse_pool(hard_score),
    }


def _nonrigid_motion(first: np.ndarray, second: np.ndarray) -> float:
    points = cv2.goodFeaturesToTrack(first, maxCorners=300, qualityLevel=0.01, minDistance=5)
    if points is None or len(points) < 8:
        return 0.0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
    if tracked is None or status is None:
        return 0.0
    valid = status.reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    target = tracked.reshape(-1, 2)[valid]
    if len(source) < 8:
        return 0.0
    affine, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if affine is None:
        return 0.0
    predicted = source @ affine[:, :2].T + affine[:, 2]
    residual = np.linalg.norm(target - predicted, axis=1)
    return float(np.mean(np.clip(residual, 0.0, 20.0)))


def _image_metrics(paths: list[Path]) -> tuple[float, float, float, float]:
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    images = [cv2.resize(image, (320, 180)) for image in images if image is not None]
    if not images:
        return 0.0, 0.0, 0.0, 0.0
    blur = float(np.mean([cv2.Laplacian(image, cv2.CV_64F).var() for image in images]))
    texture = float(np.mean([np.mean(np.abs(cv2.Sobel(image, cv2.CV_32F, 1, 1))) for image in images]))
    motion = float(np.mean([np.mean(cv2.absdiff(a, b)) for a, b in zip(images[:-1], images[1:])])) if len(images) > 1 else 0.0
    nonrigid = float(np.mean([_nonrigid_motion(a, b) for a, b in zip(images[:-1], images[1:])])) if len(images) > 1 else 0.0
    return texture, blur, motion, nonrigid


def select_toy_clips(
    data_root: str | Path,
    output_root: str | Path,
    duration_sec: float = 20.0,
    sample_fps: float = 8.0,
    pool_size: int = 12,
    remote_cache: "RemoteZipCache | None" = None,
) -> dict:
    root = Path(data_root)
    output = Path(output_root) / "selection"
    output.mkdir(parents=True, exist_ok=True)
    archive_paths = {}
    sparse_names: set[str] = set()
    for name in ("egocentric_color.zip", "kinect_color.zip"):
        full_path = root / "_archives" / name
        if full_path.is_file():
            archive_paths[name] = full_path
        elif remote_cache is not None:
            archive_paths[name] = remote_cache.ensure_index(name)
            sparse_names.add(name)
        else:
            raise FileNotFoundError(
                f"Missing {full_path}; finish the archive or use --remote-selective"
            )
    ego_archive = archive_paths["egocentric_color.zip"]
    if "egocentric_color.zip" in sparse_names:
        with zipfile.ZipFile(ego_archive) as archive:
            pv_members = [
                item.filename
                for item in archive.infolist()
                if not item.is_dir() and item.filename.endswith("_pv.txt")
            ]
        ego_archive = remote_cache.ensure_members("egocentric_color.zip", pv_members)
        archive_paths["egocentric_color.zip"] = ego_archive
        extract_members(ego_archive, root, pv_members)
    candidates, _, _ = build_candidates(
        root,
        duration_sec,
        sample_fps,
        archive_paths=archive_paths,
    )
    if len(candidates) < 3:
        raise RuntimeError(f"Only {len(candidates)} valid candidate windows were found")
    pools = _trajectory_rankings(candidates, pool_size)
    unique_pool = {id(candidate): candidate for pool in pools.values() for candidate in pool}.values()
    ego_members = sorted({member for candidate in unique_pool for member in candidate.ego_members[:: max(1, int(sample_fps))]})
    exo_members = sorted({member for candidate in unique_pool for member in candidate.exo_members[:: max(1, int(sample_fps))]})
    if "egocentric_color.zip" in sparse_names:
        ego_archive = remote_cache.ensure_members("egocentric_color.zip", ego_members)
    else:
        ego_archive = archive_paths["egocentric_color.zip"]
    if "kinect_color.zip" in sparse_names:
        exo_archive = remote_cache.ensure_members("kinect_color.zip", exo_members)
    else:
        exo_archive = archive_paths["kinect_color.zip"]
    extract_members(ego_archive, root, ego_members)
    extract_members(exo_archive, root, exo_members)
    for candidate in unique_pool:
        ego_paths = [root / _modality_relative(member, "egocentric_color") for member in candidate.ego_members[:: max(1, int(sample_fps))]]
        exo_paths = [root / _modality_relative(member, "kinect_color") for member in candidate.exo_members[:: max(1, int(sample_fps))]]
        (
            candidate.texture_score,
            candidate.blur_score,
            candidate.ego_motion_score,
            candidate.ego_nonrigid_motion_score,
        ) = _image_metrics(ego_paths)
        (
            _,
            _,
            candidate.exo_motion_score,
            candidate.exo_nonrigid_motion_score,
        ) = _image_metrics(exo_paths)
    ordered_by_difficulty: dict[str, list[ClipCandidate]] = {}
    for difficulty in ("easy", "medium", "hard"):
        pool = pools[difficulty]
        if difficulty == "easy":
            ordered = sorted(pool, key=lambda c: (c.total_translation_m + c.p95_angular_velocity_deg_s / 100, -float(c.texture_score or 0)))
        elif difficulty == "medium":
            ordered = sorted(pool, key=lambda c: (-(c.total_translation_m + c.turn_excursion_deg / 90), -float(c.texture_score or 0)))
        else:
            ordered = sorted(
                [candidate for candidate in pool if candidate.visible_ratio > 0.0],
                key=lambda c: (
                    -(c.p95_angular_velocity_deg_s + float(c.exo_motion_score or 0)),
                    -float(c.exo_nonrigid_motion_score or 0),
                    float(c.blur_score or 0),
                ),
            )
        ordered_by_difficulty[difficulty] = ordered
    combinations = [
        (easy_index + medium_index + hard_index, easy, medium, hard)
        for easy_index, easy in enumerate(ordered_by_difficulty["easy"])
        for medium_index, medium in enumerate(ordered_by_difficulty["medium"])
        for hard_index, hard in enumerate(ordered_by_difficulty["hard"])
        if len({easy.recording_name, medium.recording_name, hard.recording_name}) == 3
    ]
    if not combinations:
        raise RuntimeError("Could not select Easy/Medium/Hard clips from three distinct recordings")
    _, easy, medium, hard = min(combinations, key=lambda item: item[0])
    chosen = {"easy": easy, "medium": medium, "hard": hard}
    for difficulty, candidate in chosen.items():
        candidate.difficulty = difficulty.capitalize()
        candidate.selection_reason = {
            "easy": "Low translation/angular velocity with high exo visibility and usable texture",
            "medium": "Clear translation and turn excursion with usable texture",
            "hard": "High P95 angular velocity, image/non-rigid motion, and nonzero exo tracking visibility",
        }[difficulty]
    final_ego = sorted({member for candidate in chosen.values() for member in candidate.ego_members})
    final_exo = sorted({member for candidate in chosen.values() for member in candidate.exo_members})
    if "egocentric_color.zip" in sparse_names:
        ego_archive = remote_cache.ensure_members("egocentric_color.zip", final_ego)
    if "kinect_color.zip" in sparse_names:
        exo_archive = remote_cache.ensure_members("kinect_color.zip", final_exo)
    extract_members(ego_archive, root, final_ego)
    extract_members(exo_archive, root, final_exo)
    gaze_archive = root / "_archives" / "egocentric_gaze.zip"
    gaze_sparse = False
    if not gaze_archive.is_file() and remote_cache is not None:
        gaze_archive = remote_cache.ensure_index("egocentric_gaze.zip")
        gaze_sparse = True
    if gaze_archive.is_file():
        selected_sequences = {
            (item.recording_name, item.hololens_sequence) for item in chosen.values()
        }
        with zipfile.ZipFile(gaze_archive) as archive:
            gaze_members = []
            for item in archive.infolist():
                if item.is_dir() or not item.filename.endswith("_head_hand_eye.csv"):
                    continue
                tail = _parts_after(item.filename, "egocentric_gaze")
                if tail is not None and len(tail) >= 3 and (tail[0], tail[1]) in selected_sequences:
                    gaze_members.append(item.filename)
        if gaze_sparse:
            gaze_archive = remote_cache.ensure_members(
                "egocentric_gaze.zip", gaze_members
            )
        extract_members(gaze_archive, root, gaze_members)
    result = {
        "data_root": str(root),
        "selection_parameters": {"duration_sec": duration_sec, "sample_fps": sample_fps, "candidate_count": len(candidates)},
        "archive_access": {
            name: {
                "path": str(path),
                "mode": (
                    "full_archive"
                    if path == root / "_archives" / name
                    else "official_remote_sparse_cache"
                ),
            }
            for name, path in archive_paths.items()
        },
        "clips": {difficulty: candidate.public_dict() for difficulty, candidate in chosen.items()},
    }
    result["archive_access"]["egocentric_gaze.zip"] = {
        "path": str(gaze_archive),
        "mode": "official_remote_sparse_cache" if gaze_sparse else "full_archive",
    }
    write_json(output / "selected_clips.json", result)
    _write_selection_report(output / "selection_report.md", result)
    _write_contact_sheet(output / "contact_sheet.jpg", root, chosen)
    return result


def _write_selection_report(path: Path, result: dict) -> None:
    lines = ["# EgoBody Toy Clip Selection", "", f"Candidates: {result['selection_parameters']['candidate_count']}", ""]
    for key, clip in result["clips"].items():
        lines.extend(
            [
                f"## {key.capitalize()}",
                "",
                f"- Recording: `{clip['recording_name']}`",
                f"- HoloLens sequence: `{clip['hololens_sequence']}`",
                f"- Frames: {clip['start_frame']}–{clip['end_frame']}",
                f"- Timestamps: {clip['start_timestamp']}–{clip['end_timestamp']}",
                f"- Source/sample FPS: {clip['source_fps']:.3f}/{clip['sample_fps']:.3f}",
                f"- Sampled frames: {clip['frame_count']}",
                f"- Expected frames / Ego sampling coverage: {clip['expected_frame_count']} / {clip['ego_sampling_ratio']:.3%}",
                f"- Translation: {clip['total_translation_m']:.3f} m",
                f"- Mean/P95 angular velocity: {clip['mean_angular_velocity_deg_s']:.3f}/{clip['p95_angular_velocity_deg_s']:.3f} deg/s",
                f"- Turn excursion: {clip['turn_excursion_deg']:.3f} deg",
                f"- P95 angular velocity: {clip['p95_angular_velocity_deg_s']:.3f} deg/s",
                f"- Visibility: {clip['visible_ratio']:.3%}",
                f"- Synchronized/missing: {clip['synchronized_ratio']:.3%}/{clip['missing_ratio']:.3%}",
                f"- Texture/sharpness: {clip['texture_score']:.3f}/{clip['blur_score']:.3f}",
                f"- Ego/exo motion: {clip['ego_motion_score']:.3f}/{clip['exo_motion_score']:.3f}",
                f"- Ego/exo non-rigid cue: {clip['ego_nonrigid_motion_score']:.3f}/{clip['exo_nonrigid_motion_score']:.3f}",
                f"- Reason: {clip['selection_reason']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_contact_sheet(path: Path, root: Path, chosen: dict[str, ClipCandidate]) -> None:
    panels = []
    for difficulty, candidate in chosen.items():
        sample_members = candidate.ego_members[:: max(1, len(candidate.ego_members) // 4)][:4]
        images = []
        for member in sample_members:
            image = cv2.imread(str(root / _modality_relative(member, "egocentric_color")))
            if image is not None:
                images.append(cv2.resize(image, (320, 180)))
        while len(images) < 4:
            images.append(np.zeros((180, 320, 3), dtype=np.uint8))
        row = np.concatenate(images, axis=1)
        cv2.putText(row, difficulty.capitalize(), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        panels.append(row)
    cv2.imwrite(str(path), np.concatenate(panels, axis=0))
