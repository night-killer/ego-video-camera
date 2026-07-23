from pathlib import Path

from ego_video_camera.mock_pipeline import run_mock_pipeline


def test_mock_pipeline_generates_playable_triptych(tmp_path: Path):
    result = run_mock_pipeline(
        tmp_path,
        "/data/aigc/cyb/zxgu/env/worldsearcher/bin/ffmpeg",
        "/data/aigc/cyb/zxgu/env/worldsearcher/bin/ffprobe",
        frame_count=4,
        fps=2,
    )
    assert Path(result["video"]).is_file()
    stream = result["ffprobe"]["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["width"] == 1920
    assert stream["height"] == 1080
    assert int(stream.get("nb_read_frames") or stream.get("nb_frames")) == 4
    validation = result["panel_validation"]
    assert validation["right_exo_backgrounds_same_source"]
    assert validation["top_gt_primary_pixel_count"] > 0
    assert validation["top_da3_marker_absent"]
    assert validation["bottom_da3_primary_pixel_count"] > 0
    assert validation["bottom_gt_marker_absent"]
