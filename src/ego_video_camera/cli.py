from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .clip_pipeline import prepare_gt_clip, run_clip
from .commands import generate_gpu_commands
from .compose import compose_all_toys
from .config import load_config, resolve_path
from .download import REQUIRED_FILES, download_required, extract_base_data
from .egobody_io import load_master_camera, load_transform_json, read_dataset_metadata
from .inventory import write_inventories
from .mock_pipeline import run_mock_pipeline
from .remote_zip import RemoteZipCache
from .selection import build_candidates, select_toy_clips
from .serialization import read_json, write_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="EgoBody + Depth Anything 3 ego/exo pose demo")
    result.add_argument("--config", default="configs/egobody_toy.yaml")
    result.add_argument("--data-root")
    result.add_argument("--output-root")
    result.add_argument("--checkpoint")
    result.add_argument("--source-root")
    result.add_argument("--netrc-file")
    result.add_argument("--selected-clips")
    result.add_argument("--sequence-id")
    result.add_argument("--duration-sec", type=float)
    result.add_argument("--sample-fps", type=float)
    result.add_argument("--input-resolution", type=int)
    result.add_argument("--window-size", type=int)
    result.add_argument("--window-overlap", type=int)
    result.add_argument("--inspect-environment", action="store_true")
    result.add_argument("--verify-model-load", action="store_true")
    result.add_argument("--inspect-data", action="store_true")
    result.add_argument("--download", action="store_true")
    result.add_argument("--download-name", action="append", choices=REQUIRED_FILES)
    result.add_argument("--download-connections", type=int, default=1)
    result.add_argument("--extract-base", action="store_true")
    result.add_argument("--select-clips", action="store_true")
    result.add_argument("--remote-selective", action="store_true")
    result.add_argument("--validate-gt", action="store_true")
    result.add_argument("--mock", action="store_true")
    result.add_argument("--run-da3", action="store_true")
    result.add_argument("--run-selected-clips", action="store_true")
    result.add_argument("--render-comparison", action="store_true")
    result.add_argument("--evaluate", action="store_true")
    result.add_argument("--compose-all-toys", action="store_true")
    result.add_argument("--generate-gpu-commands", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def _resolved_config(args) -> tuple[dict, Path, Path, Path]:
    root = repo_root()
    config_path = resolve_path(args.config, root)
    overrides = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.output_root:
        overrides["output_root"] = args.output_root
    da3 = {}
    for argument, key in (
        (args.checkpoint, "checkpoint_path"),
        (args.source_root, "source_root"),
        (args.sample_fps, "sample_fps"),
        (args.input_resolution, "input_resolution"),
        (args.window_size, "window_size"),
        (args.window_overlap, "window_overlap"),
    ):
        if argument is not None:
            da3[key] = argument
    if da3:
        overrides["da3"] = da3
    if args.duration_sec is not None:
        overrides["clip"] = {"duration_sec": args.duration_sec}
    config = load_config(config_path, overrides)
    data_root = resolve_path(config["data_root"], root)
    output_root = resolve_path(config["output_root"], root)
    output_root.mkdir(parents=True, exist_ok=True)
    return config, config_path, data_root, output_root


def _file_size_or_none(path: Path) -> int | None:
    try:
        return path.stat().st_size if path.is_file() else None
    except FileNotFoundError:
        return None


def _data_inventory(data_root: Path) -> dict:
    archives = data_root / "_archives"
    archive_data = {}
    for name in REQUIRED_FILES:
        path = archives / name
        segment_root = archives / "_segments" / name
        segment_bytes = 0
        segment_count = 0
        for segment in segment_root.glob("*.part") if segment_root.is_dir() else []:
            try:
                start, inclusive_end = map(int, segment.stem.split("_"))
                stored_size = segment.stat().st_size
            except (ValueError, FileNotFoundError):
                continue
            expected_size = inclusive_end - start + 1
            if stored_size > expected_size and start <= stored_size <= inclusive_end + 1:
                downloaded_size = stored_size - start
            else:
                downloaded_size = min(stored_size, expected_size)
            segment_bytes += max(0, downloaded_size)
            segment_count += 1
        active_transfers = (
            list(segment_root.glob("*.part.transfer")) if segment_root.is_dir() else []
        )
        active_transfer_bytes = 0
        active_transfer_count = 0
        for transfer in active_transfers:
            try:
                active_transfer_bytes += transfer.stat().st_size
                active_transfer_count += 1
            except FileNotFoundError:
                continue
        partial = path.with_suffix(path.suffix + ".part")
        archive_size = _file_size_or_none(path)
        partial_size = _file_size_or_none(partial)
        archive_data[name] = {
            "path": str(path),
            "exists": archive_size is not None,
            "size": archive_size,
            "partial_size": partial_size,
            "parallel_segment_count": segment_count,
            "parallel_segment_downloaded_bytes": segment_bytes,
            "parallel_active_transfer_count": active_transfer_count,
            "parallel_active_transfer_bytes": active_transfer_bytes,
            "known_downloaded_bytes": (
                (partial_size or 0) + segment_bytes + active_transfer_bytes
            ),
        }
    calibration_paths = sorted(data_root.glob("calibrations/**/holo_to_kinect12.json"))
    invalid_calibrations = []
    for calibration_path in calibration_paths:
        try:
            load_transform_json(calibration_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            invalid_calibrations.append(
                {"path": str(calibration_path), "error": type(error).__name__}
            )
    split_counts = None
    recording_count = None
    if (data_root / "data_info_release.csv").is_file() and (
        data_root / "data_splits.csv"
    ).is_file():
        info, split_by_recording = read_dataset_metadata(data_root)
        recording_count = len(info)
        split_counts = {
            split: sum(value == split for value in split_by_recording.values())
            for split in ("train", "val", "test")
        }
    try:
        camera, camera_path = load_master_camera(data_root)
        camera_inventory = {
            "status": "ok",
            "path": str(camera_path),
            "matrix": camera.matrix,
            "distortion": camera.distortion,
        }
    except FileNotFoundError:
        camera_inventory = {"status": "missing"}
    return {
        "data_root": str(data_root),
        "metadata_present": (data_root / "data_info_release.csv").is_file(),
        "recording_count": recording_count,
        "split_counts": split_counts,
        "calibration_count": len(calibration_paths),
        "invalid_calibrations": invalid_calibrations,
        "master_kinect_camera": camera_inventory,
        "pv_text_count": sum(1 for _ in data_root.glob("**/*_pv.txt")),
        "ego_frame_count": sum(1 for _ in data_root.glob("**/PV/*_frame_*.*")),
        "exo_master_frame_count": sum(1 for _ in data_root.glob("**/master/frame_*.*")),
        "gaze_file_count": sum(1 for _ in data_root.glob("**/*_head_hand_eye.csv")),
        "archives": archive_data,
        "disk_free": shutil.disk_usage(data_root.parent if data_root.parent.exists() else "/data").free,
    }


def _load_selected(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Selected clips file not found: {path}")
    return read_json(path)


def _reconcile_real_artifact_status(
    status_path: Path,
    output_root: Path,
    selected: dict | None,
) -> None:
    if selected is None:
        return
    status = read_json(status_path)
    status["real_clip_selection"] = "complete"
    complete_gt: dict[str, bool] = {}
    for difficulty, clip in selected.get("clips", {}).items():
        directory = output_root / clip["recording_name"]
        report_path = directory / "gt_validation.json"
        required = [
            report_path,
            directory / "gt_only_overlay.mp4",
            directory / "frame_mapping.json",
        ]
        valid = all(path.is_file() for path in required)
        if valid:
            try:
                report = read_json(report_path)
                valid = bool(report.get("ffprobe", {}).get("streams"))
            except (OSError, ValueError, json.JSONDecodeError):
                valid = False
        complete_gt[difficulty] = valid
    if complete_gt.get("easy"):
        status["gt_only_validation"] = "complete"
    if complete_gt and all(complete_gt.values()):
        status["gt_only_all_selected"] = "complete"
    write_json(status_path, status)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = repo_root()
    config, config_path, data_root, output_root = _resolved_config(args)
    safe_config = {key: value for key, value in config.items() if not key.startswith("_")}
    write_json(output_root / "config_resolved.json", safe_config)
    status_path = output_root / "execution_status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    defaults = {
        "host_cuda_available": torch.cuda.is_available(),
        "cpu_model_load_only": "pending",
        "real_clip_selection": "pending",
        "gt_only_validation": "pending",
        "gt_only_all_selected": "pending",
        "gpu_smoke": "not_executed",
        "easy_medium_hard_real_da3": "not_executed",
        "real_da3_inference_on_cpu": "intentionally_not_run",
    }
    for key, value in defaults.items():
        status.setdefault(key, value)
    if (output_root / "selection" / "selected_clips.json").is_file():
        status["real_clip_selection"] = "complete"
    write_json(status_path, status)
    if args.inspect_environment or args.verify_model_load:
        _, model_inventory = write_inventories(
            root,
            output_root,
            config["da3"]["checkpoint_path"],
            config["da3"]["source_root"],
            config["runtime"]["ffmpeg_path"],
            config["runtime"]["ffprobe_path"],
            args.verify_model_load,
        )
        status = read_json(status_path)
        status["cpu_model_load_only"] = model_inventory["load_only_check"]["status"]
        write_json(status_path, status)
    if args.download:
        if not args.netrc_file:
            raise ValueError("--netrc-file is required for authenticated EgoBody downloads")
        download_required(
            data_root,
            args.netrc_file,
            names=args.download_name or REQUIRED_FILES,
            connections=args.download_connections,
        )
    if args.extract_base:
        write_json(output_root / "extraction_report.json", extract_base_data(data_root))
    if args.inspect_data:
        write_json(output_root / "data_inventory.json", _data_inventory(data_root))
    if args.mock:
        result = run_mock_pipeline(
            output_root / "mock",
            config["runtime"]["ffmpeg_path"],
            config["runtime"]["ffprobe_path"],
        )
        print(json.dumps({"mock_video": result["video"]}, ensure_ascii=False))
    selected_path = Path(args.selected_clips) if args.selected_clips else output_root / "selection" / "selected_clips.json"
    if args.select_clips:
        if args.remote_selective and not args.netrc_file:
            raise ValueError("--remote-selective requires --netrc-file")
        remote_cache = (
            RemoteZipCache(
                data_root,
                args.netrc_file,
                connections=args.download_connections,
            )
            if args.remote_selective
            else None
        )
        selected = select_toy_clips(
            data_root,
            output_root,
            float(config["clip"]["duration_sec"]),
            float(config["da3"]["sample_fps"]),
            remote_cache=remote_cache,
        )
        status = read_json(status_path)
        status["real_clip_selection"] = "complete"
        write_json(status_path, status)
    else:
        selected = _load_selected(selected_path) if selected_path.is_file() else None
    _reconcile_real_artifact_status(status_path, output_root, selected)
    if args.validate_gt and selected is None:
        candidates, _, _ = build_candidates(
            data_root,
            float(config["clip"]["duration_sec"]),
            float(config["da3"]["sample_fps"]),
        )
        if not candidates:
            raise RuntimeError("No provisional GT candidate could be built")
        candidate = max(candidates, key=lambda item: (item.visible_ratio, item.synchronized_ratio)).public_dict()
        candidate["difficulty"] = "Provisional"
        prepare_gt_clip(data_root, candidate, output_root / "provisional_gt", config)
    elif args.validate_gt and selected is not None:
        easy = dict(selected["clips"]["easy"])
        prepare_gt_clip(data_root, easy, output_root / easy["recording_name"], config)
        status = read_json(status_path)
        status["gt_only_validation"] = "complete"
        write_json(status_path, status)
    should_run = args.run_selected_clips or args.sequence_id is not None
    if (args.run_da3 or args.render_comparison or args.evaluate) and not should_run:
        raise ValueError(
            "DA3/render/evaluate requires --sequence-id or --run-selected-clips"
        )
    if should_run:
        if selected is None:
            raise RuntimeError("Running clips requires selected_clips.json")
        if args.run_selected_clips:
            clips = list(selected["clips"].values())
        else:
            matches = [
                clip
                for clip in selected["clips"].values()
                if clip["recording_name"] == args.sequence_id
            ]
            if len(matches) != 1:
                raise ValueError(f"Selected recording not found or ambiguous: {args.sequence_id}")
            clips = matches
        for clip_source in clips:
            clip = dict(clip_source)
            if args.duration_sec is not None:
                clip["runtime_duration_sec"] = args.duration_sec
            if args.sample_fps is not None:
                clip["runtime_sample_fps"] = args.sample_fps
            run_clip(
                repo_root=root,
                data_root=data_root,
                output_dir=output_root / clip["recording_name"],
                clip=clip,
                config=config,
                run_da3=args.run_da3,
                render_comparison=args.render_comparison,
                evaluate=args.evaluate,
                resume=args.resume,
            )
        status = read_json(status_path)
        status["gt_only_validation"] = "complete"
        if args.run_selected_clips:
            status["gt_only_all_selected"] = "complete"
        write_json(status_path, status)
        if args.run_da3:
            status = read_json(status_path)
            if args.run_selected_clips:
                status["easy_medium_hard_real_da3"] = "executed"
            elif args.duration_sec is not None and args.duration_sec <= 5:
                status["gpu_smoke"] = "executed"
            write_json(status_path, status)
    if args.compose_all_toys:
        if selected is None:
            raise RuntimeError("Composition requires selected clips")
        compose_all_toys(
            output_root,
            selected,
            config["runtime"]["ffmpeg_path"],
            config["runtime"]["ffprobe_path"],
            float(config["da3"]["sample_fps"]),
        )
    if args.generate_gpu_commands or (selected is not None and args.select_clips):
        commands = generate_gpu_commands(
            root,
            config_path,
            data_root,
            output_root,
            config["da3"]["checkpoint_path"],
            selected,
        )
        command_path = output_root / "gpu_commands.sh"
        command_path.write_text(commands, encoding="utf-8")
        command_path.chmod(0o755)
    if not any(
        vars(args)[key]
        for key in vars(args)
        if key.startswith(
            (
                "inspect",
                "verify",
                "download",
                "extract",
                "select",
                "validate",
                "mock",
                "run_",
                "render",
                "evaluate",
                "compose",
                "generate",
            )
        )
    ):
        parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
