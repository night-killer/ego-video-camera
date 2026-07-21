"""Deterministic RGB video decoding, encoding, and comparison composition."""

from __future__ import annotations

from contextlib import AbstractContextManager
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Iterator

import av
import cv2
import numpy as np

from .io_utils import PipelineInputError


class H264Writer(AbstractContextManager["H264Writer"]):
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = 18,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("H.264 width and height must be positive even numbers")
        if fps <= 0:
            raise ValueError("FPS must be positive")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = float(fps)
        self.container = av.open(
            str(self.path), mode="w", format="mp4", options={"movflags": "+faststart"}
        )
        rate = Fraction(str(self.fps)).limit_denominator(100_000)
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "crf": str(int(crf)),
            "preset": "medium",
        }
        self.frame_count = 0
        self._closed = False

    def write(self, rgb: np.ndarray) -> None:
        array = np.asarray(rgb)
        if array.shape != (self.height, self.width, 3):
            raise ValueError(
                f"video frame must have shape {(self.height, self.width, 3)}, got {array.shape}"
            )
        if array.dtype != np.uint8:
            raise ValueError(f"video frame must use uint8 RGB, got {array.dtype}")
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self._closed = True
        if self.frame_count == 0:
            self.path.unlink(missing_ok=True)
            raise ValueError("refusing to keep an empty video")

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        if exc_type is None:
            self.close()
        else:
            self.container.close()
            self._closed = True
            self.path.unlink(missing_ok=True)


def write_h264(path: Path, frames: Iterable[np.ndarray], *, fps: float, crf: int = 18) -> int:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot encode an empty frame sequence") from exc
    height, width = first.shape[:2]
    with H264Writer(path, width=width, height=height, fps=fps, crf=crf) as writer:
        writer.write(first)
        for frame in iterator:
            writer.write(frame)
    return writer.frame_count


def iter_video_rgb(path: Path) -> Iterator[np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PipelineInputError(f"video does not exist: {source}")
    with av.open(str(source), mode="r") as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise PipelineInputError(f"expected exactly one video stream in {source}, got {len(streams)}")
        for frame in container.decode(streams[0]):
            yield frame.to_ndarray(format="rgb24")


def video_info(path: Path, *, decode_count: bool = False) -> dict[str, float | int | str | None]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PipelineInputError(f"video does not exist: {source}")
    with av.open(str(source), mode="r") as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise PipelineInputError(f"expected exactly one video stream in {source}")
        stream = streams[0]
        rate = stream.average_rate or stream.base_rate
        result: dict[str, float | int | str | None] = {
            "path": str(source),
            "width": int(stream.width),
            "height": int(stream.height),
            "fps": float(rate) if rate is not None else None,
            "declared_frames": int(stream.frames) if stream.frames else None,
            "codec": stream.codec_context.name,
            "pixel_format": (
                stream.codec_context.format.name if stream.codec_context.format is not None else None
            ),
        }
    if decode_count:
        result["decoded_frames"] = sum(1 for _ in iter_video_rgb(source))
    return result


def add_panel_label(rgb: np.ndarray, label: str) -> np.ndarray:
    output = np.ascontiguousarray(rgb.copy())
    height, width = output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(width, height) / 700.0)
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    x, y = 18, 18
    cv2.rectangle(
        output,
        (x - 8, y - 8),
        (x + text_width + 8, y + text_height + baseline + 8),
        (8, 8, 8),
        thickness=-1,
    )
    cv2.putText(
        output,
        label,
        (x, y + text_height),
        font,
        scale,
        (245, 245, 245),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return output


def compose_side_by_side(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    *,
    fps: float,
    left_label: str,
    right_label: str,
) -> int:
    left_frames = iter_video_rgb(left_path)
    right_frames = iter_video_rgb(right_path)

    def combined() -> Iterator[np.ndarray]:
        for index, pair in enumerate(zip_longest(left_frames, right_frames)):
            left, right = pair
            if left is None or right is None:
                raise PipelineInputError(
                    f"comparison inputs have different frame counts near frame {index}"
                )
            if left.shape != right.shape:
                raise PipelineInputError(
                    f"comparison frame shape mismatch at frame {index}: {left.shape} != {right.shape}"
                )
            yield np.concatenate(
                [add_panel_label(left, left_label), add_panel_label(right, right_label)], axis=1
            )

    return write_h264(output_path, combined(), fps=fps)
