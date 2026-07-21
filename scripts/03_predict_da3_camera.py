#!/usr/bin/env python3
"""Predict video cameras with Depth Anything 3 and align them to GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_video_camera.da3 import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    MAX_SUPPORTED_FRAMES,
    preflight_da3,
    run_da3_camera_prediction,
)
from ego_video_camera.io_utils import atomic_write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DA3 camera-only inference and Sim(3) alignment.")
    parser.add_argument("--video", type=Path, required=True, help="Rendered GT egocentric MP4.")
    parser.add_argument("--gt-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--da3-root", type=Path, default=PROJECT_ROOT / "third_party" / "depth-anything-3")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true", help="Validate all inputs without loading DA3.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 2 <= args.max_frames <= MAX_SUPPORTED_FRAMES:
        raise SystemExit(
            f"--max-frames must be between 2 and the v1 hard limit {MAX_SUPPORTED_FRAMES}"
        )
    if args.process_res <= 0:
        raise SystemExit("--process-res must be positive")
    if args.dry_run:
        report = preflight_da3(
            video_path=args.video,
            gt_trajectory_path=args.gt_trajectory,
            da3_root=args.da3_root,
            model_dir=args.model_dir,
            max_frames=args.max_frames,
            require_cuda=False,
        )
        target = args.output_dir.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target / "preflight.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if not report["errors"] else 2
    manifest = run_da3_camera_prediction(
        video_path=args.video,
        gt_trajectory_path=args.gt_trajectory,
        output_dir=args.output_dir,
        da3_root=args.da3_root,
        model_dir=args.model_dir,
        device=args.device,
        process_res=args.process_res,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    print(f"Aligned DA3 trajectory: {manifest['aligned_trajectory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
