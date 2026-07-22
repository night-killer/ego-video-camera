from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


class FFmpegWriter:
    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        ffmpeg_path: str,
        codec: str = "libx264",
        pixel_format: str = "yuv420p",
        crf: int = 20,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp.mp4")
        self.temporary = temporary
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.8g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            pixel_format,
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.width = width
        self.height = height
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("Writer is closed")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected {(self.height, self.width, 3)}, got {frame.shape}")
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.closed:
            return
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        self.closed = True
        if code != 0:
            self.temporary.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg exited with {code}: {stderr.strip()}")
        self.temporary.replace(self.path)

    def __enter__(self) -> "FFmpegWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.close()
        else:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            self.temporary.unlink(missing_ok=True)
            self.closed = True


def probe_video(path: str | Path, ffprobe_path: str) -> dict:
    completed = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def verify_video(
    path: str | Path,
    ffprobe_path: str,
    *,
    width: int = 1920,
    height: int = 1080,
    expected_frames: int | None = None,
    expected_fps: float | None = None,
) -> dict:
    report = probe_video(path, ffprobe_path)
    streams = report.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
    }
    mismatches = {
        key: (stream.get(key), value)
        for key, value in expected.items()
        if stream.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Video stream validation failed for {path}: {mismatches}")
    if expected_frames is not None:
        observed_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
        if observed_frames is None or int(observed_frames) != int(expected_frames):
            raise RuntimeError(
                f"Video frame-count mismatch for {path}: "
                f"{observed_frames} != {expected_frames}"
            )
    if expected_fps is not None:
        raw_rate = stream.get("avg_frame_rate")
        try:
            observed_fps = float(Fraction(raw_rate))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise RuntimeError(f"Invalid ffprobe frame rate for {path}: {raw_rate}") from error
        if abs(observed_fps - expected_fps) > 1e-5 * max(1.0, expected_fps):
            raise RuntimeError(
                f"Video frame-rate mismatch for {path}: {observed_fps} != {expected_fps}"
            )
        if expected_frames is not None:
            observed_duration = float(
                stream.get("duration") or report.get("format", {}).get("duration")
            )
            expected_duration = expected_frames / expected_fps
            tolerance = max(0.05, 1.0 / expected_fps)
            if abs(observed_duration - expected_duration) > tolerance:
                raise RuntimeError(
                    f"Video duration mismatch for {path}: "
                    f"{observed_duration} != {expected_duration}"
                )
    return report
