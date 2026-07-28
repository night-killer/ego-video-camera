from __future__ import annotations

import argparse
import importlib
import traceback
from pathlib import Path
from typing import Callable

from ..serialization import write_json
from .schema import PoseTrajectory
from .trajectory_io import validate_prediction, write_trajectory
from .workers.common import WorkerContext, load_context


def _load_adapter(value: str) -> Callable[[WorkerContext], PoseTrajectory]:
    module_name, separator, function_name = value.partition(":")
    if not separator:
        raise ValueError(f"Adapter must be module:function, got {value!r}")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Adapter {value!r} is not callable")
    return function


def run_worker(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_context(args.manifest, output_dir, args.repo, args.checkpoint)
    context.emit("worker_started", adapter=args.adapter)
    try:
        trajectory = _load_adapter(args.adapter)(context)
        if not isinstance(trajectory, PoseTrajectory):
            raise TypeError("Method adapter did not return PoseTrajectory")
        validate_prediction(trajectory, context.manifest)
        write_trajectory(output_dir / "prediction.npz", trajectory)
        context.emit("prediction_written", valid_count=int(trajectory.valid.sum()))
        write_json(
            output_dir / "worker_result.json",
            {"status": "success", "valid_count": int(trajectory.valid.sum())},
        )
        return 0
    except BaseException as error:
        context.emit("worker_failed", error_type=type(error).__name__, message=str(error))
        write_json(
            output_dir / "worker_result.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated camera-pose benchmark worker")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    return parser


def main() -> int:
    return run_worker(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
