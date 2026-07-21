#!/usr/bin/env python3
"""Interpolate authored keyframes and optionally render the GT egocentric video."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_video_camera.gaussian import load_gaussian_scene, render_trajectory_video  # noqa: E402
from ego_video_camera.interpolation import interpolate_keyframes  # noqa: E402
from ego_video_camera.io_utils import atomic_write_json  # noqa: E402
from ego_video_camera.schema import load_trajectory, save_trajectory  # noqa: E402
from ego_video_camera.video import video_info  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate camera keyframes and render a GT 3DGS video.")
    parser.add_argument("--keyframes", type=Path, required=True, help="camera_trajectory.v1 keyframes JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ply", type=Path, help="Override the PLY recorded in the keyframe JSON.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trajectory-only", action="store_true", help="Interpolate on CPU without rendering.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = args.output_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    trajectory_path = target / "gt_trajectory.json"
    video_path = target / "gt_video.mp4"
    manifest_path = target / "render_manifest.json"
    expected = [trajectory_path, manifest_path, video_path]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"output already exists (pass --overwrite): {existing[0]}")
    if args.trajectory_only and args.overwrite:
        # Do not leave a stale rendered video beside a newly interpolated trajectory.
        video_path.unlink(missing_ok=True)

    keyframes = load_trajectory(args.keyframes)
    dense = interpolate_keyframes(keyframes)
    ply_path = (args.ply or Path(dense.scene.ply_path)).expanduser().resolve()
    if args.ply is not None:
        dense = dense.model_copy(
            update={"scene": dense.scene.model_copy(update={"ply_path": str(ply_path)})}
        )
    save_trajectory(trajectory_path, dense)
    manifest: dict[str, object] = {
        "status": "trajectory_only" if args.trajectory_only else "rendering",
        "keyframes_path": str(args.keyframes.expanduser().resolve()),
        "gt_trajectory_path": str(trajectory_path),
        "ply_path": str(ply_path),
        "frame_count": len(dense.frames),
        "fps": dense.video.fps,
        "resolution": [dense.video.width, dense.video.height],
        "interpolation": dense.source.get("method"),
    }
    if not args.trajectory_only:
        scene = load_gaussian_scene(ply_path, device=args.device)
        rendered = render_trajectory_video(scene, dense, video_path)
        encoded_info = video_info(video_path, decode_count=True)
        if (
            encoded_info["decoded_frames"] != len(dense.frames)
            or (encoded_info["width"], encoded_info["height"])
            != (dense.video.width, dense.video.height)
            or encoded_info["fps"] is None
            or not math.isclose(float(encoded_info["fps"]), dense.video.fps, abs_tol=1e-6)
            or encoded_info["codec"] != "h264"
            or encoded_info["pixel_format"] != "yuv420p"
        ):
            raise RuntimeError(f"unexpected GT video encoding: {encoded_info}")
        manifest.update(
            {
                "status": "complete",
                "gt_video_path": str(video_path),
                "rendered_frame_count": rendered,
                "splat_count": scene.splat_count,
                "sh_degree": scene.sh_degree,
                "device": args.device,
                "video_info": encoded_info,
            }
        )
    atomic_write_json(manifest_path, manifest)
    print(f"Dense GT trajectory: {trajectory_path}")
    if not args.trajectory_only:
        print(f"GT video: {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
