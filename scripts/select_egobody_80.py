#!/usr/bin/env python3
"""Select a deterministic, metadata-only 80-clip EgoBody manifest.

The public Motion-X table contains inclusive frame endpoints.  This script
keeps those endpoints in the provenance fields and emits fixed windows using
``[frame_start_inclusive, frame_end_exclusive)``.  It reads PV pose text when
it is already present under ``--data-root`` to rank desktop clips by a camera
rotation proxy; it never downloads or decodes RGB images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


SOURCE_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/Motion-X/main/"
    "mocap-dataset-process/egobody_description_all.csv"
)
SOURCE_FPS = 30
DESKTOP_SECONDS = 20
WALKING_SECONDS = 6
MAX_MERGE_GAP = 2
HEAD_TURN_MIN_DEG = 20.0
HEAD_P95_VELOCITY_MIN_DEG_S = 10.0
# PV pose coverage is a useful early warning for RGB archives with long
# missing-frame gaps.  Keep this conservative when enough qualified choices
# exist, while allowing explicit fallback on metadata-only checkouts.
MIN_DESKTOP_PV_COVERAGE = 0.85
DEFAULT_DATA_ROOT = Path("/data/aigc/cyb/zxgu/data/EgoBody")
DEFAULT_ACTION_CSV = Path("/tmp/egobody_description_all.csv")
DEFAULT_OUTPUT_ROOT = Path("outputs/egobody_80")

DESKTOP_RE = re.compile(
    r"draw|write|book|cutting|baking|coffee|pen|table|blackboard|class",
    re.IGNORECASE,
)
WALK_RE = re.compile(r"(?:^|_)(?:walk|wander|run)(?:$|_)", re.IGNORECASE)
PV_IMAGE_RE = re.compile(
    r"^(?P<timestamp>\d+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$", re.I
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_action(value: str) -> str:
    value = value.strip().lower().replace("\r", "")
    return re.sub(r"_clip\d+$", "", value)


def _role(body_idx: str) -> str:
    return "camera_wearer" if body_idx.strip().startswith("1") else "interactee"


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for ordinal, row in enumerate(csv.DictReader(handle)):
            raw_id = row.get("") or row.get("row_id") or ordinal
            body_value = row.get("body_idx_0", "")
            try:
                body_idx = int(body_value.strip().split()[0])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid body_idx_0 at CSV row {ordinal + 2}: {body_value!r}") from exc
            source_end = int(row["frame_interval_end"])
            rows.append(
                {
                    "row_id": int(raw_id),
                    "recording_name": row["recording_name"],
                    "start": int(row["frame_interval_start"]),
                    # Motion-X's end is inclusive.  Internally use a half-open
                    # endpoint so a 600-frame window is exactly 600 frames.
                    "end": source_end + 1,
                    "source_end_inclusive": source_end,
                    "body_idx": body_idx,
                    "role": _role(body_value),
                    "action": _normalise_action(row["body_0_des"]),
                    "action_raw": row["body_0_des"].strip().replace("\r", ""),
                }
            )
    return rows


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + MAX_MERGE_GAP:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = first.T @ second
    cosine = np.clip((float(np.trace(delta)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _read_metadata_bounds(data_root: Path) -> dict[str, tuple[int, int]]:
    path = data_root / "data_info_release.csv"
    if not path.is_file():
        return {}
    result: dict[str, tuple[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                result[row["recording_name"]] = (int(row["start_frame"]), int(row["end_frame"]))
            except (KeyError, TypeError, ValueError):
                continue
    return result


def _pv_paths(data_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    base = data_root / "egocentric_color"
    if not base.is_dir():
        return result
    for path in sorted(base.rglob("*_pv.txt")):
        try:
            marker = path.parts.index("egocentric_color")
            recording = path.parts[marker + 1]
        except (ValueError, IndexError):
            continue
        result[recording].append(path)
    return result


def _load_pv(path: Path, data_root: Path, recording: str, metadata_bounds: dict[str, tuple[int, int]]) -> dict:
    """Load pose rotations and infer source frame numbers without RGB access."""

    timestamps: list[int] = []
    rotations: list[np.ndarray] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {"status": "pending_pv_unreadable", "frame": np.empty(0), "time": np.empty(0), "rotation": []}
    for line in lines[1:]:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 19:
            continue
        try:
            matrix = np.asarray(fields[3:19], dtype=np.float64).reshape(4, 4)
            if not np.isfinite(matrix).all():
                continue
            timestamps.append(int(fields[0]))
            rotations.append(matrix[:3, :3])
        except (TypeError, ValueError):
            continue
    if len(timestamps) < 2:
        return {"status": "pending_insufficient_pv_rows", "frame": np.empty(0), "time": np.empty(0), "rotation": []}

    time = np.asarray(timestamps, dtype=np.float64)
    frame = None
    status = "computed_pv_camera_rotation_proxy_row_index"
    # Existing sparse PV images, when present, provide an exact timestamp to
    # source-frame mapping.  Their filenames are metadata only; no pixels are read.
    anchors: list[tuple[int, int]] = []
    for image in path.parent.glob("PV/*"):
        match = PV_IMAGE_RE.match(image.name)
        if match:
            anchors.append((int(match.group("timestamp")), int(match.group("frame"))))
    if len({item[0] for item in anchors}) >= 2:
        anchor_time = np.asarray([item[0] for item in anchors], dtype=np.float64)
        anchor_frame = np.asarray([item[1] for item in anchors], dtype=np.float64)
        slope, intercept = np.polyfit(anchor_time, anchor_frame, 1)
        frame = slope * time + intercept
        status = "computed_pv_camera_rotation_proxy_timestamp_anchors"
    elif recording in metadata_bounds:
        first, last = metadata_bounds[recording]
        frame = np.linspace(float(first), float(last), len(time))
        status = "computed_pv_camera_rotation_proxy_metadata_span"
    else:
        frame = np.arange(len(time), dtype=np.float64)

    return {
        "status": status,
        "frame": frame,
        "time": time,
        "rotation": rotations,
    }


def _window_pv_metrics(pv: dict | None, start: int, end: int) -> dict:
    pending = {
        # Keep the validated generator's status for the unauthenticated case;
        # PV proxy statuses are distinct and explicit when pose text exists.
        "head_motion_metric_status": "pending_authenticated_pv_pose",
        "head_turn_excursion_deg": None,
        "head_mean_angular_velocity_deg_s": None,
        "head_p95_angular_velocity_deg_s": None,
        "head_motion_qualified": False,
        "pv_coverage_ratio": 0.0,
    }
    if not pv or len(pv.get("frame", ())) < 2:
        return pending
    mask = (pv["frame"] >= start) & (pv["frame"] < end)
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        pending["head_motion_metric_status"] = "pending_pv_window_coverage"
        return pending
    rotations = [pv["rotation"][int(index)] for index in indices]
    times = pv["time"][indices] / 10_000_000.0
    steps = np.asarray(
        [_rotation_error_deg(a, b) for a, b in zip(rotations[:-1], rotations[1:])],
        dtype=np.float64,
    )
    dt = np.maximum(np.diff(times), 1e-6)
    velocity = steps / dt
    excursion = max((_rotation_error_deg(rotations[0], rotation) for rotation in rotations), default=0.0)
    turn = float(excursion)
    p95 = float(np.percentile(velocity, 95)) if len(velocity) else 0.0
    expected_frames = max(1, end - start)
    return {
        "head_motion_metric_status": pv["status"],
        "head_turn_excursion_deg": turn,
        "head_mean_angular_velocity_deg_s": float(velocity.mean()) if len(velocity) else 0.0,
        "head_p95_angular_velocity_deg_s": p95,
        # Require both a meaningful accumulated turn and sustained motion.
        "head_motion_qualified": bool(turn >= HEAD_TURN_MIN_DEG and p95 >= HEAD_P95_VELOCITY_MIN_DEG_S),
        "pv_coverage_ratio": float(min(1.0, len(indices) / expected_frames)),
    }


def _head_prior(actions: Iterable[str]) -> float:
    score = 0.0
    for action in actions:
        if re.search(r"draw|write|coffee|cutting|baking", action):
            score = max(score, 2.0)
        elif re.search(r"class|blackboard", action):
            score = max(score, 1.5)
        elif re.search(r"book|table|pen", action):
            score = max(score, 1.0)
    return score


def _make_clip(
    *, category: str, recording: str, start: int, duration_frames: int,
    source_rows: list[dict], subject_role: str, body_indices: list[int],
    head_required: bool, pv_metrics: dict | None,
) -> dict:
    end = start + duration_frames
    actions = sorted({row["action"] for row in source_rows})
    metrics = _window_pv_metrics(pv_metrics, start, end) if head_required else {
        "head_motion_metric_status": "not_required",
        "head_turn_excursion_deg": None,
        "head_mean_angular_velocity_deg_s": None,
        "head_p95_angular_velocity_deg_s": None,
        "head_motion_qualified": False,
        "pv_coverage_ratio": 0.0,
    }
    return {
        "clip_id": "",
        "category": category,
        "recording_name": recording,
        "hololens_sequence": None,
        "resolve_hololens_sequence": "archive_index_by_recording",
        "source_row_ids": sorted({row["row_id"] for row in source_rows}),
        "source_intervals_inclusive": [
            {"row_id": row["row_id"], "frame_start_inclusive": row["start"], "frame_end_inclusive": row["source_end_inclusive"]}
            for row in sorted(source_rows, key=lambda value: value["row_id"])
        ],
        "source_action_labels": actions,
        "source_action_labels_raw": sorted({row["action_raw"] for row in source_rows}),
        "subject_role": subject_role,
        "body_indices": body_indices,
        "source_roles": sorted({row["role"] for row in source_rows}),
        "frame_start_inclusive": start,
        "frame_end_exclusive": end,
        "frame_end_inclusive": end - 1,
        "frame_count": duration_frames,
        "duration_s": duration_frames / SOURCE_FPS,
        "source_fps": SOURCE_FPS,
        "head_motion_required": head_required,
        **metrics,
        "head_motion_semantic_prior": _head_prior(actions),
    }


def _candidate_desktop(rows: list[dict], pv_by_recording: dict[str, dict], duration_frames: int) -> list[dict]:
    selected = [row for row in rows if row["role"] == "camera_wearer" and DESKTOP_RE.search(row["action"])]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["recording_name"]].append(row)
    candidates: list[dict] = []
    for recording, group in sorted(grouped.items()):
        for interval_start, interval_end in _merge_intervals((row["start"], row["end"]) for row in group):
            for start in range(interval_start, interval_end - duration_frames + 1, duration_frames):
                source = [row for row in group if row["start"] < start + duration_frames and row["end"] > start]
                candidates.append(_make_clip(
                    category="desktop_head_motion", recording=recording, start=start,
                    duration_frames=duration_frames, source_rows=source,
                    subject_role="camera_wearer", body_indices=[1], head_required=True,
                    pv_metrics=pv_by_recording.get(recording),
                ))
    return candidates


def _candidate_walking(rows: list[dict], duration_frames: int) -> list[dict]:
    selected = [row for row in rows if WALK_RE.search(row["action"])]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["recording_name"]].append(row)
    candidates: list[dict] = []
    for recording, group in sorted(grouped.items()):
        for interval_start, interval_end in _merge_intervals((row["start"], row["end"]) for row in group):
            for start in range(interval_start, interval_end - duration_frames + 1, duration_frames):
                source = [row for row in group if row["start"] < start + duration_frames and row["end"] > start]
                candidates.append(_make_clip(
                    category="walking_person", recording=recording, start=start,
                    duration_frames=duration_frames, source_rows=source,
                    subject_role="any_person", body_indices=sorted({row["body_idx"] for row in source}),
                    head_required=False, pv_metrics=None,
                ))
    return candidates


def _round_robin_select(
    candidates: list[dict], count: int, *, qualified_only: bool = False,
    min_pv_coverage: float | None = None,
) -> tuple[list[dict], list[dict]]:
    selection_pool = (
        [item for item in candidates if item.get("head_motion_qualified", False)]
        if qualified_only
        else list(candidates)
    )
    if min_pv_coverage is not None:
        covered = [
            item for item in selection_pool
            if float(item.get("pv_coverage_ratio", 0.0)) >= min_pv_coverage
        ]
        # A clean checkout may not have PV metadata.  Only enforce the quality
        # gate when it still leaves the requested number of candidates.
        if len(covered) >= count:
            selection_pool = covered
    # If PV metadata is unavailable (for example on a clean checkout), keep
    # the requested count with explicit ``pending_*`` statuses.  When enough
    # metrics exist, qualified candidates are ordered first and fill the
    # complete primary set.
    if qualified_only and len(selection_pool) < count:
        selection_pool = list(candidates)
    by_recording: dict[str, list[dict]] = defaultdict(list)
    for candidate in selection_pool:
        by_recording[candidate["recording_name"]].append(candidate)
    for values in by_recording.values():
        values.sort(key=lambda item: (
            not item.get("head_motion_qualified", False),
            -(item.get("head_turn_excursion_deg") or -1.0),
            -(item.get("head_p95_angular_velocity_deg_s") or -1.0),
            -item["head_motion_semantic_prior"],
            item["frame_start_inclusive"],
        ))
    selected: list[dict] = []
    recordings = sorted(by_recording)
    cursor = 0
    while len(selected) < count:
        progress = False
        for recording in recordings:
            values = by_recording[recording]
            if cursor < len(values):
                selected.append(values[cursor])
                progress = True
                if len(selected) >= count:
                    break
        if not progress:
            break
        cursor += 1
    if len(selected) < count:
        used = {id(item) for item in selected}
        remainder = sorted((item for item in selection_pool if id(item) not in used), key=lambda item: (
            not item.get("head_motion_qualified", False),
            -(item.get("head_turn_excursion_deg") or -1.0),
            -(item.get("head_p95_angular_velocity_deg_s") or -1.0),
            item["recording_name"], item["frame_start_inclusive"],
        ))
        selected.extend(remainder[: count - len(selected)])
    selected_ids = {id(item) for item in selected}
    return selected[:count], [item for item in candidates if id(item) not in selected_ids]


def _overlap(first: dict, second: dict) -> bool:
    return (
        first["recording_name"] == second["recording_name"]
        and first["frame_start_inclusive"] < second["frame_end_exclusive"]
        and second["frame_start_inclusive"] < first["frame_end_exclusive"]
    )


def _assign_ids(items: list[dict], prefix: str) -> None:
    for index, item in enumerate(items, start=1):
        item["clip_id"] = f"{prefix}_{index:03d}"


CSV_FIELDS = [
    "tier", "clip_id", "category", "recording_name", "hololens_sequence", "subject_role",
    "body_indices", "frame_start_inclusive", "frame_end_inclusive", "frame_end_exclusive",
    "frame_count", "duration_s", "source_row_ids", "source_action_labels", "head_motion_required",
    "head_motion_metric_status", "head_turn_excursion_deg", "head_mean_angular_velocity_deg_s",
    "head_p95_angular_velocity_deg_s", "head_motion_qualified", "pv_coverage_ratio",
    "head_motion_semantic_prior",
]


def _flatten_csv(categories: dict[str, dict]) -> list[dict]:
    rows = []
    for category, payload in categories.items():
        for tier in ("clips", "reserve_clips"):
            for item in payload[tier]:
                rows.append({
                    "tier": tier, "clip_id": item["clip_id"], "category": category,
                    "recording_name": item["recording_name"], "hololens_sequence": "",
                    "subject_role": item["subject_role"], "body_indices": ";".join(map(str, item["body_indices"])),
                    "frame_start_inclusive": item["frame_start_inclusive"], "frame_end_inclusive": item["frame_end_inclusive"],
                    "frame_end_exclusive": item["frame_end_exclusive"], "frame_count": item["frame_count"],
                    "duration_s": item["duration_s"], "source_row_ids": ";".join(map(str, item["source_row_ids"])),
                    "source_action_labels": ";".join(item["source_action_labels"]),
                    "head_motion_required": item["head_motion_required"],
                    "head_motion_metric_status": item["head_motion_metric_status"],
                    "head_turn_excursion_deg": item["head_turn_excursion_deg"],
                    "head_mean_angular_velocity_deg_s": item["head_mean_angular_velocity_deg_s"],
                    "head_p95_angular_velocity_deg_s": item["head_p95_angular_velocity_deg_s"],
                    "head_motion_qualified": item["head_motion_qualified"],
                    "pv_coverage_ratio": item["pv_coverage_ratio"],
                    "head_motion_semantic_prior": item["head_motion_semantic_prior"],
                })
    return rows


def build_manifest(data_root: Path | str, action_csv: Path | str, desktop_count: int, walking_count: int) -> dict:
    data_root = Path(data_root)
    action_csv = Path(action_csv)
    if not action_csv.is_file():
        raise FileNotFoundError(f"Missing Motion-X action CSV: {action_csv}")
    if desktop_count < 0 or walking_count < 0:
        raise ValueError("clip counts must be non-negative")
    rows = _read_rows(action_csv)
    metadata_bounds = _read_metadata_bounds(data_root)
    pv_by_recording: dict[str, dict] = {}
    for recording, paths in _pv_paths(data_root).items():
        loaded = [_load_pv(path, data_root, recording, metadata_bounds) for path in paths]
        loaded = [item for item in loaded if len(item.get("frame", ())) >= 2]
        if loaded:
            # A recording normally has one sequence.  Keep the sequence with
            # the broadest pose-frame support if multiple are present.
            pv_by_recording[recording] = max(loaded, key=lambda item: len(item["frame"]))
    desktop_frames = DESKTOP_SECONDS * SOURCE_FPS
    walking_frames = WALKING_SECONDS * SOURCE_FPS
    desktop_all = _candidate_desktop(rows, pv_by_recording, desktop_frames)
    walking_all = _candidate_walking(rows, walking_frames)
    walking, walking_reserve = _round_robin_select(walking_all, walking_count)
    if len(walking) != walking_count:
        raise ValueError(f"Insufficient walking candidates: requested {walking_count}, found {len(walking)}")
    disjoint_desktop = [item for item in desktop_all if not any(_overlap(item, walk) for walk in walking)]
    desktop, desktop_reserve = _round_robin_select(
        disjoint_desktop,
        desktop_count,
        qualified_only=True,
        min_pv_coverage=MIN_DESKTOP_PV_COVERAGE,
    )
    if len(desktop) != desktop_count:
        raise ValueError(
            "Insufficient disjoint desktop candidates after walking selection: "
            f"requested {desktop_count}, found {len(desktop)}"
        )
    _assign_ids(desktop, "DESK")
    _assign_ids(desktop_reserve, "DESK_RESERVE")
    _assign_ids(walking, "WALK")
    _assign_ids(walking_reserve, "WALK_RESERVE")
    duplicate_rgb_time_keys = [
        {"desktop_clip_id": desk["clip_id"], "walking_clip_id": walk["clip_id"], "recording_name": desk["recording_name"]}
        for desk in desktop for walk in walking if _overlap(desk, walk)
    ]
    categories = {
        "desktop_head_motion": {
            "count": len(desktop), "duration_s": DESKTOP_SECONDS,
            "subject_role": "camera_wearer", "clips": desktop, "reserve_clips": desktop_reserve[:20],
            "action_rule": DESKTOP_RE.pattern,
            "head_motion_rule": {
                "required": True, "must_be_computed_after_download": not bool(pv_by_recording),
                "metric_source": "PV camera rotation proxy; authenticated head pose can replace it",
                "recommended_metrics": [f"turn_excursion_deg >= {HEAD_TURN_MIN_DEG:g}", f"p95_angular_velocity_deg_s >= {HEAD_P95_VELOCITY_MIN_DEG_S:g}"],
                "combination": "both",
            },
        },
        "walking_person": {
            "count": len(walking), "duration_s": WALKING_SECONDS,
            "subject_role": "any_person", "clips": walking, "reserve_clips": walking_reserve,
            "action_rule": WALK_RE.pattern, "head_motion_rule": {"required": False},
        },
    }
    return {
        "schema_version": "egobody_demo_selection_v1",
        "source": {"url": SOURCE_URL, "local_path": action_csv.name, "sha256": _sha256(action_csv), "row_count": len(rows), "note": "Public Motion-X manual EgoBody action intervals; no RGB archive downloaded"},
        "frame_convention": {"source_fps": SOURCE_FPS, "motion_x_intervals": "frame_start_inclusive, frame_end_inclusive", "emitted_windows": "frame_start_inclusive, frame_end_exclusive", "ego_body_archive_resolution": "resolve hololens_sequence by recording_name from egocentric_color ZIP index"},
        "selection": {"desktop": "wearer-only semantic action intervals, 20 s non-overlap; ranked by PV camera rotation proxy when available", "walking": "union of walk/wander/run intervals by recording, 6 s non-overlap; actor roles retained", "desktop_count": desktop_count, "walking_count": walking_count, "count": desktop_count + walking_count},
        "categories": categories,
        "validation": {"unique_primary_clip_ids": len({item["clip_id"] for category in categories.values() for item in category["clips"]}), "primary_count": sum(len(category["clips"]) for category in categories.values()), "duplicate_rgb_time_keys": duplicate_rgb_time_keys, "head_motion_is_not_guaranteed_by_action_text": True, "pv_recordings_with_rotation_metrics": sorted(pv_by_recording), "desktop_head_motion_qualified_count": sum(item["head_motion_qualified"] for item in desktop)},
        "notes": ["Motion-X body_idx_0=1 denotes the camera wearer; body_idx_0=0 denotes the interactee.", "Motion-X frame_interval_end is inclusive; emitted frame_end_exclusive is one greater.", "PV metrics are camera-rotation proxies, not authenticated head pose. A pending status is retained when PV text is absent or does not cover a window.", "No RGB archive is downloaded or decoded by this script."],
    }


def write_outputs(payload: dict, output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "egobody_80_manifest.json"
    csv_path = output_root / "egobody_80_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    flat = _flatten_csv(payload["categories"])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(flat)
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help=f"Existing EgoBody root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--action-csv", type=Path, default=DEFAULT_ACTION_CSV, help="Motion-X egobody_description_all.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--desktop-count", type=int, default=40)
    parser.add_argument("--walking-count", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_manifest(args.data_root, args.action_csv, args.desktop_count, args.walking_count)
    json_path, csv_path = write_outputs(payload, args.output_root)
    print(json_path)
    print(csv_path)
    print(f"primary {payload['validation']['primary_count']}")
    print(f"PV metric recordings {len(payload['validation']['pv_recordings_with_rotation_metrics'])}")
    print(f"desktop head-motion qualified {payload['validation']['desktop_head_motion_qualified_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
