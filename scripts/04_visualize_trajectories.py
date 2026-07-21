#!/usr/bin/env python3
"""Create two orthogonal cumulative trajectory videos over Gaussian backgrounds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_video_camera.comparison import validate_corresponding_trajectories  # noqa: E402
from ego_video_camera.gaussian import load_gaussian_scene  # noqa: E402
from ego_video_camera.schema import load_trajectory  # noqa: E402
from ego_video_camera.trajectory_visualization import write_trajectory_visualizations  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT and DA3 camera trajectories over 3DGS.")
    parser.add_argument("--gt-trajectory", type=Path, required=True)
    parser.add_argument("--predicted-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ply", type=Path, help="Override the GT trajectory's PLY path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--background-strength", type=float, default=0.42)
    parser.add_argument("--frustum-depth", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gt = load_trajectory(args.gt_trajectory)
    predicted = load_trajectory(args.predicted_trajectory)
    validate_corresponding_trajectories(gt, predicted)
    ply = (args.ply or Path(gt.scene.ply_path)).expanduser().resolve()
    scene = load_gaussian_scene(ply, device=args.device)
    manifest = write_trajectory_visualizations(
        scene,
        gt,
        predicted,
        args.output_dir,
        background_strength=args.background_strength,
        frustum_depth=args.frustum_depth,
        overwrite=args.overwrite,
    )
    for view in manifest["views"]:
        print(f"Trajectory video ({view['name']}): {view['video_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

