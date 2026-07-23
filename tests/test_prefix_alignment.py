import numpy as np

from ego_video_camera.trajectory_alignment import (
    estimate_full_alignment,
    estimate_prefix_alignment,
)


def test_prefix_alignment_reports_degenerate_static_prefix():
    source = np.repeat(np.eye(4)[None], 30, axis=0)
    target = source.copy()
    timestamps = np.linspace(0, 10, 30)
    result = estimate_prefix_alignment(source, target, timestamps)
    assert result.status == "degenerate"
    assert result.transform is None


def test_prefix_alignment_succeeds_with_2d_motion():
    source = np.repeat(np.eye(4)[None], 30, axis=0)
    source[:, 0, 3] = np.linspace(0, 1, 30)
    source[:, 1, 3] = 0.1 * np.sin(np.linspace(0, 4, 30))
    target = source.copy()
    target[:, :3, 3] = 1.5 * source[:, :3, 3] + [1, 2, 3]
    timestamps = np.linspace(0, 10, 30)
    result = estimate_prefix_alignment(source, target, timestamps)
    assert result.status == "ok"
    assert np.isclose(result.transform.scale, 1.5)


def test_prefix_alignment_does_not_slide_past_clip_prefix():
    source = np.repeat(np.eye(4)[None], 8, axis=0)
    source[:, 0, 3] = np.linspace(0, 1, len(source))
    source[:, 1, 3] = 0.1 * np.sin(np.linspace(0, 3, len(source)))
    target = source.copy()
    timestamps = np.linspace(7, 10, len(source))
    result = estimate_prefix_alignment(
        source,
        target,
        timestamps,
        timeline_start_sec=0.0,
        timeline_end_sec=20.0,
    )
    assert result.status == "degenerate"
    assert result.transform is None


def test_full_alignment_rejects_underexcited_target_trajectory():
    source = np.repeat(np.eye(4)[None], 20, axis=0)
    target = source.copy()
    source[:, 0, 3] = np.linspace(0, 0.3, len(source))
    source[:, 1, 3] = 0.05 * np.sin(np.linspace(0, 3, len(source)))
    target[:, 0, 3] = np.linspace(0, 0.03, len(target))
    target[:, 1, 3] = 0.005 * np.sin(np.linspace(0, 3, len(target)))
    result = estimate_full_alignment(
        source,
        target,
        with_scale=True,
        minimum_span_m=0.1,
        minimum_rank_ratio=0.001,
    )
    assert result.status == "degenerate"
    assert result.transform is None
    assert "target span=" in result.reason
