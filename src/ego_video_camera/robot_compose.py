from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .serialization import write_json
from .video_io import verify_video


def compose_robot_demo(
    output_root: str | Path,
    clips: list[dict[str, Any]],
    ffmpeg_path: str,
    ffprobe_path: str,
    fps: float = 10.0,
) -> Path:
    output = Path(output_root)
    ordered = sorted(
        clips,
        key=lambda item: (
            0 if item["dataset"] == "droid_wrist" else 1,
            clips.index(item),
        ),
    )
    videos = [
        output
        / str(item["dataset"])
        / str(item["sequence_id"])
        / "comparison_prefix.mp4"
        for item in ordered
    ]
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Robot prefix videos are missing: " + ", ".join(map(str, missing))
        )
    metadata_dir = output / "_compose"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    concat_path = metadata_dir / "prefix_concat.txt"
    concat_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in videos),
        encoding="utf-8",
    )
    destination = output / "comparison_all_prefix.mp4"
    temporary = destination.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            f"fps={fps:.8g}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(destination)
    manifest = {
        "schema_version": 1,
        "alignment": "sim3_prefix",
        "order": [
            {"dataset": item["dataset"], "sequence_id": item["sequence_id"]}
            for item in ordered
        ],
        "sources": [str(path) for path in videos],
        "video": str(destination),
        "ffprobe": verify_video(destination, ffprobe_path, expected_fps=fps),
    }
    write_json(output / "comparison_all_prefix_manifest.json", manifest)
    return destination
