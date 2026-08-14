"""Run the ActiMind Ego Estimation triptych on selective EgoBody demo clips."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from .clip_pipeline import prepare_loaded_gt_clip, run_clip
from .config import load_config, resolve_path
from .da3_adapter import EXPECTED_DA3_COMMIT
from .egobody_demo_download import download_exo_only
from .egobody_io import FrameMapping, PVRecord, load_T_K_W, load_master_camera, parse_pv_file
from .serialization import read_json, write_json


CATEGORY_ALIASES = {
    "desktop": "desktop_head_motion",
    "desktop_head_motion": "desktop_head_motion",
    "walking": "walking_person",
    "walking_person": "walking_person",
    "all": None,
}
DEFAULT_CONFIG = "configs/egobody_demo80_actimind.yaml"
DEFAULT_MANIFEST = "configs/egobody_demo_80/egobody_80_manifest.json"
DEFAULT_METADATA_ROOT = "/data/aigc/cyb/zxgu/data/EgoBody"
DEFAULT_NETRC = "/data/aigc/cyb/zxgu/.secrets/egobody.netrc"


@dataclass(frozen=True)
class LoadedClip:
    clip: dict[str, Any]
    records: list[PVRecord]
    mappings: list[FrameMapping]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_clip_id(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid clip ID: {value!r}")
    return value


def _manifest_clips(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        raise ValueError(f"Manifest categories are missing: {path}")
    clips: list[dict[str, Any]] = []
    for category in ("desktop_head_motion", "walking_person"):
        item = categories.get(category, {})
        for source in item.get("clips", []) if isinstance(item, dict) else []:
            clip = dict(source)
            clip["clip_id"] = _safe_clip_id(str(clip.get("clip_id", "")))
            clip["category"] = category
            clips.append(clip)
    identifiers = [clip["clip_id"] for clip in clips]
    if not clips or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Manifest contains no clips or duplicate clip IDs: {path}")
    return clips


def select_clips(
    clips: Iterable[dict[str, Any]],
    *,
    clip_ids: Iterable[str] = (),
    category: str | None = None,
    run_all: bool = False,
) -> list[dict[str, Any]]:
    available = list(clips)
    requested_ids = list(dict.fromkeys(_safe_clip_id(value) for value in clip_ids))
    if category is not None and category not in CATEGORY_ALIASES:
        raise ValueError(f"Unsupported category: {category}")
    normalized_category = CATEGORY_ALIASES[category] if category else None
    if not run_all and not requested_ids and category is None:
        raise ValueError("Select clips with --clip-id, --category, or --all")
    by_id = {clip["clip_id"]: clip for clip in available}
    missing = [clip_id for clip_id in requested_ids if clip_id not in by_id]
    if missing:
        raise ValueError(f"Unknown clip ID(s): {', '.join(missing)}")
    if requested_ids:
        selected = [by_id[clip_id] for clip_id in requested_ids]
        if normalized_category is not None:
            selected = [clip for clip in selected if clip["category"] == normalized_category]
    else:
        selected = available
        if normalized_category is not None:
            selected = [clip for clip in selected if clip["category"] == normalized_category]
    if not selected:
        raise ValueError("The requested clip selection is empty")
    return selected


def _read_frame_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"output_index", "source_frame_id", "source_timestamp", "filename"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} is missing required columns: {sorted(required)}")
        for source in reader:
            rows.append(
                {
                    "output_index": int(source["output_index"]),
                    "frame_id": int(source["source_frame_id"]),
                    "timestamp": int(source["source_timestamp"]),
                    "filename": str(source["filename"]),
                }
            )
    if not rows:
        raise ValueError(f"No frame rows in {path}")
    expected_indices = list(range(len(rows)))
    if [row["output_index"] for row in rows] != expected_indices:
        raise ValueError(f"output_index must be contiguous and ordered in {path}")
    for field in ("frame_id", "timestamp"):
        values = [row[field] for row in rows]
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError(f"{field} must be strictly increasing in {path}")
    return rows


def _pv_records_by_timestamp(path: Path) -> dict[int, PVRecord]:
    _, records = parse_pv_file(path)
    result: dict[int, PVRecord] = {}
    for record in records:
        if record.timestamp in result:
            raise ValueError(f"Duplicate PV timestamp {record.timestamp} in {path}")
        result[record.timestamp] = record
    return result


def _exo_path(exo_dir: Path, output_index: int, frame_id: int) -> Path | None:
    candidates = sorted(exo_dir.glob(f"{output_index:06d}_{frame_id:06d}.*"))
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous exo frame for output={output_index}, source={frame_id}: {exo_dir}"
        )
    return candidates[0] if candidates else None


def load_demo_clip(data_root: str | Path, source_clip: dict[str, Any]) -> LoadedClip:
    root = Path(data_root).resolve()
    clip_id = _safe_clip_id(str(source_clip["clip_id"]))
    clip_root = root / "clips" / clip_id
    clip_json_path = clip_root / "clip.json"
    if not clip_json_path.is_file():
        raise FileNotFoundError(f"Missing clip metadata: {clip_json_path}")
    downloaded = read_json(clip_json_path)
    for key in ("clip_id", "recording_name", "category"):
        if str(downloaded.get(key)) != str(source_clip.get(key)):
            raise ValueError(
                f"Manifest/clip.json mismatch for {clip_id}: {key} "
                f"{source_clip.get(key)!r} != {downloaded.get(key)!r}"
            )
    rows = _read_frame_rows(clip_root / "frames.csv")
    expected_count = int(downloaded.get("frame_count", len(rows)))
    if len(rows) != expected_count:
        raise ValueError(f"{clip_id}: frames.csv has {len(rows)} rows, expected {expected_count}")
    references = sorted((clip_root / "reference").glob("*_pv.txt"))
    if len(references) != 1:
        raise FileNotFoundError(f"{clip_id}: expected one PV reference, found {len(references)}")
    pv_by_timestamp = _pv_records_by_timestamp(references[0])
    records: list[PVRecord] = []
    mappings: list[FrameMapping] = []
    for row in rows:
        image_path = clip_root / "frames" / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"{clip_id}: missing ego frame {image_path}")
        pv = pv_by_timestamp.get(row["timestamp"])
        if pv is None:
            raise ValueError(
                f"{clip_id}: timestamp {row['timestamp']} from frames.csv is absent from {references[0]}"
            )
        record = PVRecord(
            timestamp=pv.timestamp,
            fx=pv.fx,
            fy=pv.fy,
            T_W_E=pv.T_W_E,
            frame_id=row["frame_id"],
            image_path=image_path,
        )
        records.append(record)
        exo = _exo_path(clip_root / "exo_frames", row["output_index"], row["frame_id"])
        mappings.append(
            FrameMapping(
                output_frame=row["output_index"],
                ego_frame_id=row["frame_id"],
                exo_frame_id=row["frame_id"] if exo else None,
                ego_timestamp=row["timestamp"],
                exo_timestamp=None,
                time_difference=None,
                sync_basis="exact_source_frame_id",
                ego_image=image_path,
                exo_image=exo,
            )
        )
    sample_fps = float(downloaded.get("sample_fps", 8))
    duration_sec = float(source_clip.get("duration_s", len(records) / sample_fps))
    clip = {**source_clip, **downloaded}
    clip.update(
        {
            "clip_id": clip_id,
            "difficulty": "Desktop" if clip["category"] == "desktop_head_motion" else "Walking",
            "frame_ids": [record.frame_id for record in records],
            "timestamps": [record.timestamp for record in records],
            "duration_sec": duration_sec,
            "runtime_duration_sec": duration_sec,
            "runtime_sample_fps": sample_fps,
            "ego_sampling_ratio": 1.0,
        }
    )
    return LoadedClip(clip=clip, records=records, mappings=mappings)


def _validate_image(path: Path, label: str) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot decode {label}: {path}")


def validate_clip(
    data_root: str | Path,
    source_clip: dict[str, Any],
    *,
    require_exo: bool,
    decode_images: bool,
) -> dict[str, Any]:
    loaded = load_demo_clip(data_root, source_clip)
    missing_exo = [mapping for mapping in loaded.mappings if mapping.exo_image is None]
    if require_exo and missing_exo:
        raise FileNotFoundError(
            f"{loaded.clip['clip_id']}: missing {len(missing_exo)}/{len(loaded.mappings)} exo frames"
        )
    if decode_images:
        for mapping in loaded.mappings:
            _validate_image(mapping.ego_image, "ego frame")
            if mapping.exo_image is not None:
                _validate_image(mapping.exo_image, "exo frame")
    return {
        "clip_id": loaded.clip["clip_id"],
        "category": loaded.clip["category"],
        "frame_count": len(loaded.records),
        "exo_frame_count": len(loaded.mappings) - len(missing_exo),
        "missing_exo_count": len(missing_exo),
        "first_frame_id": loaded.records[0].frame_id,
        "last_frame_id": loaded.records[-1].frame_id,
        "first_timestamp": loaded.records[0].timestamp,
        "last_timestamp": loaded.records[-1].timestamp,
    }


def _prepare_exo_command(root: Path, data_root: Path, manifest_path: Path) -> str:
    return shlex.join(
        [
            str(root / "run_egobody_demo80.sh"),
            "prepare-exo",
            "--data-root",
            str(data_root),
            "--manifest",
            str(manifest_path),
            "--netrc-file",
            DEFAULT_NETRC,
        ]
    )


def preflight_run(
    *,
    root: Path,
    data_root: Path,
    manifest_path: Path,
    metadata_root: Path,
    selected: list[dict[str, Any]],
    config: dict[str, Any],
    run_da3: bool,
) -> list[LoadedClip]:
    errors: list[str] = []
    loaded_clips: list[LoadedClip] = []
    missing_exo_detected = False
    try:
        load_master_camera(metadata_root)
    except (OSError, ValueError, KeyError) as error:
        errors.append(str(error))
    for source_clip in selected:
        try:
            loaded = load_demo_clip(data_root, source_clip)
            missing = sum(mapping.exo_image is None for mapping in loaded.mappings)
            if missing:
                missing_exo_detected = True
                errors.append(
                    f"{loaded.clip['clip_id']}: missing {missing}/{len(loaded.mappings)} exo frames"
                )
            load_T_K_W(metadata_root, loaded.clip["recording_name"])
            loaded_clips.append(loaded)
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{source_clip['clip_id']}: {error}")
    if run_da3:
        checkpoint = Path(config["da3"]["checkpoint_path"]).resolve()
        for name in ("config.json", "model.safetensors"):
            path = checkpoint / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing DA3 checkpoint file: {path}")
        source_root = resolve_path(config["da3"]["source_root"], root)
        if not (source_root / "src" / "depth_anything_3").is_dir():
            errors.append(f"Missing DA3 source package: {source_root}")
        else:
            try:
                commit = subprocess.run(
                    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ).stdout.strip()
                if commit != EXPECTED_DA3_COMMIT:
                    errors.append(
                        f"DA3 source commit mismatch: {commit} != {EXPECTED_DA3_COMMIT}"
                    )
            except (OSError, subprocess.CalledProcessError) as error:
                errors.append(f"Cannot inspect DA3 source: {error}")
    for key in ("ffmpeg_path", "ffprobe_path"):
        path = Path(config["runtime"][key])
        if not path.is_file() or not os.access(path, os.X_OK):
            errors.append(f"Missing executable {key}: {path}")
    if run_da3 and not torch.cuda.is_available():
        errors.append("CUDA is unavailable; real ActiMind Ego Estimation inference requires a GPU")
    if errors:
        lines = "\n".join(f"  - {error}" for error in errors)
        message = f"Demo80 preflight failed before GPU inference:\n{lines}"
        if missing_exo_detected:
            message += (
                "\nFor missing exo frames, run:\n  "
                f"{_prepare_exo_command(root, data_root, manifest_path)}"
            )
        raise RuntimeError(message)
    return loaded_clips


def _resolved_config(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config_path = resolve_path(args.config, root)
    overrides: dict[str, Any] = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.output_root:
        overrides["output_root"] = args.output_root
    da3: dict[str, Any] = {}
    for argument, key in (
        (getattr(args, "checkpoint", None), "checkpoint_path"),
        (getattr(args, "source_root", None), "source_root"),
        (getattr(args, "input_resolution", None), "input_resolution"),
        (getattr(args, "window_size", None), "window_size"),
        (getattr(args, "window_overlap", None), "window_overlap"),
    ):
        if argument is not None:
            da3[key] = argument
    if da3:
        overrides["da3"] = da3
    config = load_config(config_path, overrides)
    data_root = resolve_path(config["data_root"], root)
    output_root = resolve_path(config["output_root"], root)
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    return config, data_root, output_root, metadata_root


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--metadata-root", default=DEFAULT_METADATA_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-exo", help="Download only selected Kinect exo frames")
    _add_common_arguments(prepare)
    prepare.add_argument("--netrc-file", default=DEFAULT_NETRC)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--exo-archive")
    prepare.add_argument("--cache-root")
    prepare.add_argument("--allow-missing-exo", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate selected clip-local inputs")
    _add_common_arguments(validate)
    validate.add_argument("--clip-id", action="append", default=[])
    validate.add_argument("--category", choices=sorted(CATEGORY_ALIASES))
    validate.add_argument("--all", action="store_true")
    validate.add_argument("--require-exo", action="store_true")
    validate.add_argument("--decode-images", action="store_true")

    run = subparsers.add_parser("run", help="Run inference, alignment, and triptych rendering")
    _add_common_arguments(run)
    run.add_argument("--clip-id", action="append", default=[])
    run.add_argument("--category", choices=sorted(CATEGORY_ALIASES))
    run.add_argument("--all", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--run-da3", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--checkpoint")
    run.add_argument("--source-root")
    run.add_argument("--input-resolution", type=int)
    run.add_argument("--window-size", type=int)
    run.add_argument("--window-overlap", type=int)
    run.add_argument(
        "--summary-path",
        help="Run summary path; relative values are resolved under output-root",
    )
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _run_summary_path(value: str | None, output_root: Path) -> Path:
    if value is None:
        return output_root / "run_summary.json"
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (output_root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    config, data_root, output_root, metadata_root = _resolved_config(args, root)
    manifest_path = resolve_path(args.manifest, root)
    all_clips = _manifest_clips(manifest_path)
    if args.command == "prepare-exo":
        result = download_exo_only(
            data_root,
            args.netrc_file,
            workers=args.workers,
            exo_archive=args.exo_archive,
            cache_root=args.cache_root,
            require_all=not args.allow_missing_exo,
        )
        _print_json(
            {
                key: result[key]
                for key in (
                    "clip_count",
                    "requested_frame_count",
                    "materialized_frame_count",
                    "missing_frame_count",
                )
            }
        )
        return 0
    selected = select_clips(
        all_clips,
        clip_ids=args.clip_id,
        category=args.category,
        run_all=args.all,
    )
    if args.command == "validate":
        reports = [
            validate_clip(
                data_root,
                clip,
                require_exo=args.require_exo,
                decode_images=args.decode_images,
            )
            for clip in selected
        ]
        _print_json(
            {
                "status": "ok",
                "clip_count": len(reports),
                "frame_count": sum(item["frame_count"] for item in reports),
                "exo_frame_count": sum(item["exo_frame_count"] for item in reports),
                "missing_exo_count": sum(item["missing_exo_count"] for item in reports),
                "clips": reports,
            }
        )
        return 0
    loaded_clips = preflight_run(
        root=root,
        data_root=data_root,
        manifest_path=manifest_path,
        metadata_root=metadata_root,
        selected=selected,
        config=config,
        run_da3=args.run_da3,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = _run_summary_path(args.summary_path, output_root)
    safe_config = {key: value for key, value in config.items() if not key.startswith("_")}
    write_json(output_root / "config_resolved.json", safe_config)
    results: list[dict[str, Any]] = []
    failed = False
    for loaded in loaded_clips:
        clip_id = loaded.clip["clip_id"]
        output_dir = output_root / clip_id
        started = time.time()
        try:
            prepared = prepare_loaded_gt_clip(
                data_root,
                loaded.clip,
                output_dir,
                config,
                loaded.records,
                loaded.mappings,
                allow_head_tracking=False,
                metadata_root=metadata_root,
            )
            result = run_clip(
                repo_root=root,
                data_root=metadata_root,
                output_dir=output_dir,
                clip=loaded.clip,
                config=config,
                run_da3=args.run_da3,
                render_comparison=args.render,
                evaluate=args.evaluate,
                resume=args.resume,
                prepared_clip=prepared,
            )
            results.append(
                {
                    "clip_id": clip_id,
                    "status": "ok",
                    "elapsed_sec": time.time() - started,
                    "output_dir": str(output_dir),
                    "result": result,
                }
            )
        except Exception as error:
            failed = True
            results.append(
                {
                    "clip_id": clip_id,
                    "status": "failed",
                    "elapsed_sec": time.time() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if not args.continue_on_error:
                write_json(summary_path, {"clips": results})
                raise
    summary = {
        "status": "failed" if failed else "ok",
        "clip_count": len(results),
        "succeeded": sum(item["status"] == "ok" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "clips": results,
    }
    write_json(summary_path, summary)
    _print_json({key: summary[key] for key in ("status", "clip_count", "succeeded", "failed")})
    return 1 if failed else 0


__all__ = [
    "LoadedClip",
    "build_parser",
    "load_demo_clip",
    "main",
    "preflight_run",
    "select_clips",
    "validate_clip",
]
