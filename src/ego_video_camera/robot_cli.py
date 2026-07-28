from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .config import load_config, resolve_path
from .eval_dataset_download import load_plan
from .robot_commands import generate_robot_gpu_commands
from .robot_compose import compose_robot_demo
from .robot_exo import prepare_robot_exo, verify_robot_exo
from .robot_io import (
    load_robot_clip,
    normalize_robot_datasets,
    robot_demo_selection,
    robot_exo_readiness,
)
from .robot_mock_pipeline import run_robot_mock_pipeline
from .robot_pipeline import run_robot_clip, validate_robot_reference
from .serialization import write_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="DA3 robot-interaction Ego/Exo pose demo"
    )
    result.add_argument(
        "--config", default="configs/robot_interaction_da3_demo.yaml"
    )
    result.add_argument("--dataset-plan")
    result.add_argument("--data-root")
    result.add_argument("--output-root")
    result.add_argument("--checkpoint")
    result.add_argument("--source-root")
    result.add_argument(
        "--dataset",
        default="all",
        help="all, droid, rh20t, droid_wrist, rh20t_wrist, or a comma-separated set",
    )
    result.add_argument("--sequence-id")
    result.add_argument("--duration-sec", type=float)
    result.add_argument("--sample-fps", type=float)
    result.add_argument("--input-resolution", type=int)
    result.add_argument("--window-size", type=int)
    result.add_argument("--window-overlap", type=int)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--rh20t-archive", type=Path)
    result.add_argument("--keep-source", action="store_true")
    result.add_argument("--prepare-exo", action="store_true")
    result.add_argument("--validate-reference", action="store_true")
    result.add_argument("--run-da3", action="store_true")
    result.add_argument("--render-comparison", action="store_true")
    result.add_argument("--evaluate", action="store_true")
    result.add_argument("--compose-all", action="store_true")
    result.add_argument("--generate-gpu-commands", action="store_true")
    result.add_argument("--mock", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def _resolved_config(
    args: argparse.Namespace,
) -> tuple[dict, Path, Path, Path, Path]:
    root = repo_root()
    config_path = resolve_path(args.config, root)
    if config_path is None:
        raise ValueError("Robot demo config path is required")
    overrides: dict = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.output_root:
        overrides["output_root"] = args.output_root
    if args.dataset_plan:
        overrides["dataset_plan"] = args.dataset_plan
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
    config = load_config(config_path, overrides)
    data_root = resolve_path(config["data_root"], root)
    output_root = resolve_path(config["output_root"], root)
    plan_path = resolve_path(config["dataset_plan"], root)
    if data_root is None or output_root is None or plan_path is None:
        raise ValueError("Robot demo data, output and dataset-plan paths are required")
    output_root.mkdir(parents=True, exist_ok=True)
    return config, config_path, plan_path, data_root, output_root


def _format_exo_not_ready(
    report: dict,
    *,
    config_path: Path,
    plan_path: Path,
    data_root: Path,
    output_root: Path,
) -> str:
    missing = report["missing_clips"]
    missing_datasets = {
        str(record["dataset"]) for record in missing
    }
    if missing_datasets == {"droid_wrist"}:
        dataset_argument = "droid"
    elif missing_datasets == {"rh20t_wrist"}:
        dataset_argument = "rh20t"
    else:
        dataset_argument = "all"
    command = [
        str(repo_root() / "run_robot_demo.sh"),
        "--config",
        str(config_path),
        "--dataset-plan",
        str(plan_path),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--dataset",
        dataset_argument,
        "--prepare-exo",
    ]
    lines = [
        "Selected robot exo data is not ready; DA3 was not started.",
        "Missing clips:",
    ]
    for record in missing:
        reason = record.get("reason") or record.get("status") or "unknown"
        lines.append(
            f"  - {record['dataset']}/{record['sequence_id']}: {reason}"
        )
    lines.extend(["", "Prepare the missing exo artifacts first:", f"  {shlex.join(command)}"])
    if "rh20t_wrist" in missing_datasets:
        lines.extend(
            [
                "  This automatically downloads and verifies the 27.4 GB RH20T cfg3 archive.",
                "  To reuse a local archive, append:",
                "    --rh20t-archive /path/to/RH20T_cfg3.tar.gz",
            ]
        )
    lines.extend(
        [
            "",
            "After preparation succeeds, rerun the original command. Generated GPU",
            "actions use --resume, so completed DROID inference will be reused.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    config, config_path, plan_path, data_root, output_root = _resolved_config(args)
    plan = load_plan(plan_path)
    datasets = normalize_robot_datasets(args.dataset)
    clips = robot_demo_selection(plan, datasets, args.sequence_id)
    safe_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    write_json(output_root / "config_resolved.json", safe_config)
    write_json(
        output_root / "selected_robot_clips.json",
        {
            "schema_version": 1,
            "count": len(clips),
            "order": [
                {
                    "dataset": clip["dataset"],
                    "sequence_id": clip["sequence_id"],
                }
                for clip in clips
            ],
        },
    )

    if args.prepare_exo:
        report = prepare_robot_exo(
            plan,
            data_root,
            config["runtime"]["ffmpeg_path"],
            workers=max(1, int(args.workers)),
            keep_source=args.keep_source,
            rh20t_archive=(
                args.rh20t_archive.resolve() if args.rh20t_archive else None
            ),
            datasets=datasets,
            sequence_ids=(args.sequence_id,) if args.sequence_id else None,
        )
        write_json(output_root / "robot_exo_prepare_report.json", report)
        verification = verify_robot_exo(
            plan,
            data_root,
            datasets,
            sequence_ids=(args.sequence_id,) if args.sequence_id else None,
        )
        write_json(output_root / "robot_exo_verify_report.json", verification)
        if not verification["ok"]:
            raise RuntimeError("Prepared robot exo data did not pass verification")

    if args.mock:
        mock = run_robot_mock_pipeline(
            output_root / "mock",
            config["runtime"]["ffmpeg_path"],
            config["runtime"]["ffprobe_path"],
        )
        print(json.dumps({"mock_video": mock["video"]}, ensure_ascii=False))

    should_process = any(
        (
            args.validate_reference,
            args.run_da3,
            args.render_comparison,
            args.evaluate,
        )
    )
    results = {}
    if should_process:
        readiness = robot_exo_readiness(data_root, clips)
        write_json(output_root / "robot_exo_preflight_report.json", readiness)
        if not readiness["ok"]:
            print(
                _format_exo_not_ready(
                    readiness,
                    config_path=config_path,
                    plan_path=plan_path,
                    data_root=data_root,
                    output_root=output_root,
                ),
                file=sys.stderr,
            )
            return 2
        for selection in clips:
            dataset = str(selection["dataset"])
            sequence = str(selection["sequence_id"])
            demo_exo = plan["datasets"][dataset]["demo_exo"]
            clip = load_robot_clip(
                data_root,
                dataset,
                sequence,
                sample_fps=float(config["da3"]["sample_fps"]),
                duration_sec=args.duration_sec,
                source_fps=float(plan["profile"]["target_fps"]),
            )
            destination = output_root / dataset / sequence
            if args.validate_reference and not any(
                (args.run_da3, args.render_comparison, args.evaluate)
            ):
                result = validate_robot_reference(
                    clip,
                    destination,
                    ffmpeg_path=config["runtime"]["ffmpeg_path"],
                    ffprobe_path=config["runtime"]["ffprobe_path"],
                    minimum_synchronized_ratio=float(
                        demo_exo["minimum_synchronized_ratio"]
                    ),
                    minimum_projection_inside_ratio=float(
                        demo_exo["minimum_projection_inside_ratio"]
                    ),
                )
            else:
                result = run_robot_clip(
                    repo_root=repo_root(),
                    clip=clip,
                    output_dir=destination,
                    config=config,
                    run_da3=args.run_da3,
                    render_comparison=args.render_comparison,
                    evaluate=args.evaluate,
                    resume=args.resume,
                    minimum_synchronized_ratio=float(
                        demo_exo["minimum_synchronized_ratio"]
                    ),
                    minimum_projection_inside_ratio=float(
                        demo_exo["minimum_projection_inside_ratio"]
                    ),
                )
            results[f"{dataset}/{sequence}"] = result
        write_json(output_root / "run_manifest.json", results)

    if args.compose_all:
        compose_robot_demo(
            output_root,
            clips,
            config["runtime"]["ffmpeg_path"],
            config["runtime"]["ffprobe_path"],
            float(config["da3"]["sample_fps"]),
        )

    if args.generate_gpu_commands:
        all_clips = robot_demo_selection(plan, "all")
        commands = generate_robot_gpu_commands(
            repo_root(),
            config_path,
            data_root,
            output_root,
            config["da3"]["checkpoint_path"],
            all_clips,
            python_path=str(config["runtime"]["python_path"]),
        )
        command_path = output_root / "gpu_commands.sh"
        command_path.write_text(commands, encoding="utf-8")
        command_path.chmod(0o755)

    if not any(
        (
            args.prepare_exo,
            args.validate_reference,
            args.run_da3,
            args.render_comparison,
            args.evaluate,
            args.compose_all,
            args.generate_gpu_commands,
            args.mock,
        )
    ):
        argument_parser.print_help()
    return 0
