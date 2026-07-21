#!/usr/bin/env python3
"""Launch the browser camera-keyframe annotator for one SuperSplat scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_video_camera.annotator import (  # noqa: E402
    SPARK_ROOT,
    THREE_ROOT,
    build_annotation_context,
    create_annotation_app,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually author camera keyframes in a local 3DGS webpage.")
    parser.add_argument("--ply", type=Path, required=True, help="Source SuperSplat Gaussian PLY.")
    parser.add_argument("--camera-json", type=Path, required=True, help="Matching SuperSplat camera JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Keyframe trajectory JSON to save.")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: localhost only).")
    parser.add_argument("--port", type=int, default=7860, help="Listen port.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    required_vendor_files = (
        SPARK_ROOT / "dist" / "spark.module.js",
        THREE_ROOT / "build" / "three.module.js",
    )
    missing = [path for path in required_vendor_files if not path.is_file()]
    if missing:
        raise SystemExit(
            "Browser dependencies are missing. Run `git submodule update --init --recursive`. "
            f"First missing file: {missing[0]}"
        )
    context = build_annotation_context(
        ply_path=args.ply,
        camera_json_path=args.camera_json,
        output_path=args.output,
    )
    app = create_annotation_app(context)
    import uvicorn

    print(f"Annotator: http://{args.host}:{args.port}")
    print(f"Display asset: {context.display_asset_path}")
    print(f"Keyframes will be saved to: {context.output_path}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

