#!/usr/bin/env python3
"""Re-render GT/DA3 cameras and create full-camera and pose-only comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_video_camera.comparison import render_comparisons  # noqa: E402
from ego_video_camera.gaussian import load_gaussian_scene  # noqa: E402
from ego_video_camera.schema import load_trajectory  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synchronized GT/DA3 Gaussian comparisons.")
    parser.add_argument("--gt-trajectory", type=Path, required=True)
    parser.add_argument("--predicted-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ply", type=Path, help="Override the GT trajectory's PLY path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gt = load_trajectory(args.gt_trajectory)
    predicted = load_trajectory(args.predicted_trajectory)
    ply = (args.ply or Path(gt.scene.ply_path)).expanduser().resolve()
    scene = load_gaussian_scene(ply, device=args.device)
    manifest = render_comparisons(
        scene,
        gt,
        predicted,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Full-camera comparison: {manifest['outputs']['full_camera_comparison']}")
    print(f"Pose-only comparison: {manifest['outputs']['pose_only_comparison']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
