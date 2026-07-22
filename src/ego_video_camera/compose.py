from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .serialization import read_json, write_json
from .video_io import FFmpegWriter, verify_video
from .visualization import draw_text


def _title_card(path: Path, fps: float, ffmpeg_path: str, difficulty: str, clip: dict, metrics: dict, head_mode: str) -> None:
    canvas = np.full((1080, 1920, 3), 18, dtype=np.uint8)
    lines = [
        difficulty.capitalize(),
        clip["recording_name"],
        f"Duration: {clip['duration_sec']:.1f}s",
        "Calibration-prefix alignment",
        f"ATE RMSE: {metrics.get('ate_rmse_m', float('nan')):.3f} m" if metrics.get("ate_rmse_m") is not None else "ATE RMSE: N/A",
        f"RPE rotation: {metrics.get('rpe_rotation_rmse_deg', float('nan')):.3f} deg" if metrics.get("rpe_rotation_rmse_deg") is not None else "RPE rotation: N/A",
        "Head Pose" if head_mode == "head_pose" else "Head proxy = ego camera center",
    ]
    for index, line in enumerate(lines):
        scale = 1.5 if index == 0 else 0.8
        draw_text(canvas, line, (160, 210 + index * 105), (255, 255, 255), scale, 2)
    with FFmpegWriter(path, 1920, 1080, fps, ffmpeg_path) as writer:
        for _ in range(max(1, int(round(2 * fps)))):
            writer.write(canvas)


def compose_all_toys(
    output_root: str | Path,
    selected: dict,
    ffmpeg_path: str,
    ffprobe_path: str,
    fps: float = 8.0,
) -> Path:
    output = Path(output_root)
    segments: list[Path] = []
    cards = output / "_title_cards"
    cards.mkdir(parents=True, exist_ok=True)
    for difficulty in ("easy", "medium", "hard"):
        clip = selected["clips"][difficulty]
        sequence_output = output / clip["recording_name"]
        video = sequence_output / "comparison_prefix.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        metrics_path = sequence_output / "metrics.json"
        metrics = read_json(metrics_path).get("sim3_prefix", {}) if metrics_path.is_file() else {}
        validation_path = sequence_output / "gt_validation.json"
        head_mode = read_json(validation_path).get("head_mode", "camera_center_proxy")
        card = cards / f"{difficulty}.mp4"
        _title_card(card, fps, ffmpeg_path, difficulty, clip, metrics, head_mode)
        segments.extend([card, video])
    concat_file = cards / "concat.txt"
    concat_file.write_text("".join(f"file '{path.resolve()}'\n" for path in segments), encoding="utf-8")
    destination = output / "comparison_all_toys.mp4"
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
            str(concat_file),
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
    write_json(
        output / "comparison_all_toys_manifest.json",
        {
            "video": str(destination),
            "ffprobe": verify_video(
                destination,
                ffprobe_path,
                expected_fps=fps,
            ),
        },
    )
    return destination
