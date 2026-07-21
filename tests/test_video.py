from __future__ import annotations

from pathlib import Path

import numpy as np

from ego_video_camera.video import compose_side_by_side, iter_video_rgb, video_info, write_h264


def test_h264_round_trip_and_comparison(tmp_path: Path) -> None:
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    frames_left = [np.full((64, 96, 3), (index * 30, 20, 40), np.uint8) for index in range(4)]
    frames_right = [np.full((64, 96, 3), (10, index * 30, 50), np.uint8) for index in range(4)]
    assert write_h264(left, frames_left, fps=15) == 4
    assert write_h264(right, frames_right, fps=15) == 4
    assert len(list(iter_video_rgb(left))) == 4
    info = video_info(left, decode_count=True)
    assert info["width"] == 96 and info["height"] == 64
    assert info["decoded_frames"] == 4
    assert info["codec"] == "h264" and info["pixel_format"] == "yuv420p"
    output = tmp_path / "comparison.mp4"
    assert compose_side_by_side(left, right, output, fps=15, left_label="GT", right_label="DA3") == 4
    compared = video_info(output, decode_count=True)
    assert compared["width"] == 192 and compared["height"] == 64
