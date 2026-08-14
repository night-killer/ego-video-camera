from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ego_video_camera.clip_pipeline as clip_pipeline
import ego_video_camera.mock_pipeline as mock_pipeline
from ego_video_camera.visualization import ACTIMIND_EGO_ESTIMATION_LABEL


class _MemoryWriter:
    def __init__(self, *args, **kwargs):
        self.frames = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def write(self, frame):
        self.frames.append(frame)


def test_comparison_uses_actimind_overlay_text(monkeypatch, tmp_path: Path):
    visible_text = []
    panel_titles = []
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(clip_pipeline, "FFmpegWriter", _MemoryWriter)
    monkeypatch.setattr(clip_pipeline.cv2, "imread", lambda path: image.copy())
    monkeypatch.setattr(clip_pipeline.cv2, "imwrite", lambda path, frame: True)
    monkeypatch.setattr(
        clip_pipeline,
        "draw_pose_overlay",
        lambda source, *args, **kwargs: (source.copy(), False),
    )
    monkeypatch.setattr(
        clip_pipeline,
        "draw_text",
        lambda source, text, *args, **kwargs: visible_text.append(text),
    )

    def capture_triptych(*args, **kwargs):
        panel_titles.append(kwargs["da3_title"])
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    monkeypatch.setattr(clip_pipeline, "compose_triptych", capture_triptych)
    mapping = SimpleNamespace(
        ego_image=tmp_path / "ego.jpg",
        exo_image=tmp_path / "exo.jpg",
        ego_frame_id=7,
        ego_timestamp=123,
    )
    clip_pipeline._render_comparison(
        tmp_path / "comparison.mp4",
        tmp_path / "preview.jpg",
        [mapping],
        np.eye(4)[None],
        np.eye(4)[None],
        np.asarray([2.0]),
        {},
        object(),
        {"difficulty": "Desk", "recording_name": "recording"},
        8.0,
        "ffmpeg",
        "camera",
        "camera_center_proxy",
        "alignment",
        np.asarray([False]),
        "ok",
    )

    assert ACTIMIND_EGO_ESTIMATION_LABEL == "ActiMind Ego Estimation"
    assert panel_titles == ["Exo + ActiMind Ego Estimation Head Proxy"]
    assert "ActiMind Ego Estimation prediction unavailable" in visible_text
    assert all("DA3" not in text for text in [*panel_titles, *visible_text])


def test_trajectory_legend_uses_actimind_display_name(monkeypatch, tmp_path: Path):
    labels = []
    original_plot = clip_pipeline.plt.plot

    def capture_plot(*args, **kwargs):
        labels.append(kwargs.get("label"))
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(clip_pipeline.plt, "plot", capture_plot)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 1.0
    clip_pipeline._plot_trajectory(tmp_path / "trajectory.png", poses, poses, "test")

    assert labels == ["GT", "ActiMind Ego Estimation"]


def test_mock_pipeline_uses_actimind_panel_title(monkeypatch, tmp_path: Path):
    panel_titles = []
    original_compose = mock_pipeline.compose_triptych

    def capture_triptych(*args, **kwargs):
        panel_titles.append(kwargs["da3_title"])
        return original_compose(*args, **kwargs)

    monkeypatch.setattr(mock_pipeline, "compose_triptych", capture_triptych)
    monkeypatch.setattr(mock_pipeline, "FFmpegWriter", _MemoryWriter)
    monkeypatch.setattr(mock_pipeline.cv2, "imwrite", lambda path, frame: True)
    monkeypatch.setattr(mock_pipeline, "verify_video", lambda *args, **kwargs: {})
    mock_pipeline.run_mock_pipeline(
        tmp_path,
        "ffmpeg",
        "ffprobe",
        frame_count=4,
        fps=2.0,
    )

    assert panel_titles == ["Exo + ActiMind Ego Estimation Head Pose"] * 4
