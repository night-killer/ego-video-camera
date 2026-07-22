import pytest

from ego_video_camera.da3_adapter import _effective_streaming_overlap


def test_one_chunk_da3_disables_overlap_to_avoid_official_tail_drop():
    assert _effective_streaming_overlap(25, 60, 30) == 0
    assert _effective_streaming_overlap(160, 60, 30) == 30


def test_invalid_streaming_window_is_rejected():
    with pytest.raises(ValueError):
        _effective_streaming_overlap(100, 30, 30)
