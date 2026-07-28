from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ..serialization import to_jsonable, write_json
from .config import load_benchmark_config, method_specs
from .evaluation import evaluate_runs
from .plan import build_run_matrix, validate_inventory, write_run_plan
from .preflight import preflight_report
from .registry import discover_sequences, sequence_inventory
from .report import generate_report
from .scheduler import execute_runs, execution_has_failures
from .schema import MethodSpec, RunSpec, SequenceRecord


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "ego_pose_benchmark.yaml"


def _patterns(values: Iterable[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(result)


def _selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Method id glob; repeat or use commas",
    )
    parser.add_argument(
        "--datasets",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Dataset id glob; repeat or use commas",
    )
    parser.add_argument(
        "--sequences",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Sequence id or dataset/sequence glob; repeat or use commas",
    )


def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume", action="store_true", help="Retry incomplete runs")
    parser.add_argument("--force", action="store_true", help="Rerun existing outputs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Write commands without starting workers"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified ego RGB camera-pose benchmark"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", help="Override benchmark.output_root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Validate and list datasets")
    _selection_arguments(inventory)

    preflight = subparsers.add_parser("preflight", help="Check existing local resources")
    _selection_arguments(preflight)
    preflight.add_argument(
        "--no-environment-check",
        action="store_true",
        help="Do not query installed Conda environments",
    )

    plan = subparsers.add_parser("plan", help="Build the deterministic run matrix")
    _selection_arguments(plan)

    run = subparsers.add_parser("run", help="Execute workers serially on one GPU")
    _selection_arguments(run)
    _execution_arguments(run)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate completed predictions")
    _selection_arguments(evaluate)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--force", action="store_true")

    report = subparsers.add_parser("report", help="Generate Markdown/CSV/JSON/PNG reports")
    _selection_arguments(report)

    all_parser = subparsers.add_parser("all", help="Plan, run, evaluate, and report")
    _selection_arguments(all_parser)
    _execution_arguments(all_parser)
    all_parser.add_argument(
        "--no-environment-check",
        action="store_true",
        help="Do not query installed Conda environments",
    )
    all_parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip resource checks before execution",
    )
    return parser


def _load(args: argparse.Namespace) -> tuple[
    dict[str, Any], list[MethodSpec], list[SequenceRecord], list[RunSpec]
]:
    config = load_benchmark_config(args.config)
    if args.output_root:
        path = Path(args.output_root)
        if not path.is_absolute():
            path = Path(config["_repo_root"]) / path
        config["benchmark"]["output_root"] = str(path.resolve())
    methods = method_specs(config)
    sequences = discover_sequences(config)
    validate_inventory(config, sequences)
    runs = build_run_matrix(
        methods,
        sequences,
        config["benchmark"]["output_root"],
        method_patterns=_patterns(args.methods),
        dataset_patterns=_patterns(args.datasets),
        sequence_patterns=_patterns(args.sequences),
    )
    return config, methods, sequences, runs


def _selected_entities(
    runs: list[RunSpec],
) -> tuple[list[MethodSpec], list[SequenceRecord]]:
    methods: dict[str, MethodSpec] = {}
    sequences: dict[str, SequenceRecord] = {}
    for run in runs:
        methods.setdefault(run.method.method_id, run.method)
        sequences.setdefault(run.sequence.key, run.sequence)
    return list(methods.values()), list(sequences.values())


def _print(value: Any) -> None:
    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2))


def _write_inventory(
    output_root: Path, sequences: list[SequenceRecord]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "sequence_count": len(sequences),
        "sequences": sequence_inventory(sequences),
    }
    write_json(output_root / "inventory.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _, _, runs = _load(args)
    selected_methods, selected_sequences = _selected_entities(runs)
    output_root = Path(config["benchmark"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_payload = _write_inventory(output_root, selected_sequences)

    if args.command == "inventory":
        summary = {
            "sequence_count": inventory_payload["sequence_count"],
            "dataset_counts": {
                dataset: sum(item.dataset_id == dataset for item in selected_sequences)
                for dataset in config["datasets"]["order"]
                if any(item.dataset_id == dataset for item in selected_sequences)
            },
            "path": str(output_root / "inventory.json"),
        }
        _print(summary)
        return 0

    plan_payload = write_run_plan(output_root / "plan.json", runs)
    if args.command == "plan":
        _print(plan_payload)
        return 0

    if args.command == "preflight":
        result = preflight_report(
            config,
            selected_methods,
            selected_sequences,
            check_environments=not args.no_environment_check,
        )
        write_json(output_root / "preflight.json", result)
        _print({**result, "path": str(output_root / "preflight.json")})
        return 0 if result["status"] == "ok" else 2

    if args.command == "run":
        result = execute_runs(
            config,
            runs,
            resume=args.resume,
            force=args.force,
            dry_run=args.dry_run,
        )
        name = "dry_run_commands.json" if args.dry_run else "execution_summary.json"
        write_json(output_root / name, result)
        _print({**result, "path": str(output_root / name)})
        return 0 if args.dry_run or not execution_has_failures(result) else 1

    if args.command == "evaluate":
        result = evaluate_runs(config, runs, resume=args.resume, force=args.force)
        write_json(output_root / "evaluation_summary.json", result)
        _print({**result, "path": str(output_root / "evaluation_summary.json")})
        return 0 if not result["status_counts"].get("evaluation_failed") else 1

    if args.command == "report":
        result = generate_report(config, runs)
        _print(
            {
                "planned_run_count": result["planned_run_count"],
                "evaluated_run_count": result["evaluated_run_count"],
                "artifacts": result["artifacts"],
            }
        )
        return 0

    if args.command == "all":
        if not args.skip_preflight:
            preflight = preflight_report(
                config,
                selected_methods,
                selected_sequences,
                check_environments=not args.no_environment_check,
            )
            write_json(output_root / "preflight.json", preflight)
            if preflight["status"] != "ok":
                _print({**preflight, "path": str(output_root / "preflight.json")})
                return 2
        execution = execute_runs(
            config,
            runs,
            resume=args.resume,
            force=args.force,
            dry_run=args.dry_run,
        )
        execution_name = (
            "dry_run_commands.json" if args.dry_run else "execution_summary.json"
        )
        write_json(output_root / execution_name, execution)
        if args.dry_run:
            report = generate_report(config, runs)
            _print(
                {
                    "execution": execution,
                    "report_artifacts": report["artifacts"],
                }
            )
            return 0
        evaluation = evaluate_runs(config, runs, resume=True, force=args.force)
        write_json(output_root / "evaluation_summary.json", evaluation)
        report = generate_report(config, runs)
        _print(
            {
                "execution_status_counts": execution.get("status_counts", {}),
                "evaluation_status_counts": evaluation.get("status_counts", {}),
                "report_artifacts": report["artifacts"],
            }
        )
        failed = execution_has_failures(execution) or bool(
            evaluation["status_counts"].get("evaluation_failed")
        )
        return 1 if failed else 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
