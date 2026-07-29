import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "thirdparty" / "HaWoR" / "lib" / "pipeline" / "metric_scale.py"
)
SPEC = importlib.util.spec_from_file_location("hawor_metric_scale", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
metric_scale = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metric_scale)


def _inputs(*, moving: bool = False):
    trajectory = np.zeros((2, 7), dtype=np.float64)
    if moving:
        trajectory[1, 0] = 1.0
    return {
        "predicted_depths": [np.ones((2, 2)), np.ones((2, 2))],
        "masks": np.zeros((2, 2, 2), dtype=np.uint8),
        "timestamps": np.asarray([0, 1]),
        "trajectory": trajectory,
    }


def test_hawor_fills_only_empty_upsampled_disparities_from_coarse_maps():
    upsampled = np.asarray(
        [
            np.zeros((4, 6), dtype=np.float32),
            np.full((4, 6), 3.0, dtype=np.float32),
            np.zeros((4, 6), dtype=np.float32),
        ]
    )
    coarse = np.asarray(
        [
            [[1.0, 2.0, 0.0], [3.0, 4.0, np.nan]],
            [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    result, fallback_count = metric_scale.fill_missing_upsampled_disparities(
        upsampled,
        coarse,
    )

    assert result is upsampled
    assert fallback_count == 1
    np.testing.assert_array_equal(result[0, :2, :2], np.ones((2, 2)))
    np.testing.assert_array_equal(result[0, 2:, :2], np.full((2, 2), 3.0))
    assert np.all(result[1] == 3.0)
    assert np.all(result[2] == 0.0)


def test_hawor_metric_scale_retries_are_bounded_and_mask_invalid_disparity():
    calls = []

    def estimator(_slam_depth, _predicted_depth, **kwargs):
        calls.append(kwargs)
        return np.nan

    scale, sample_count, source = metric_scale.estimate_metric_scale(
        np.asarray([[[1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        estimator=estimator,
        **_inputs(),
    )

    assert len(calls) == len(metric_scale.DEFAULT_THRESHOLD_SCHEDULE)
    assert calls[0]["msk"][0, 1] == 1
    assert (scale, sample_count, source) == (
        1.0,
        0,
        "static_translation_fallback",
    )


def test_hawor_metric_scale_resizes_input_mask_to_droid_resolution():
    received_masks = []

    def estimator(_slam_depth, _predicted_depth, **kwargs):
        received_masks.append(kwargs["msk"])
        return 2.0

    inputs = _inputs(moving=True)
    inputs["masks"] = np.zeros((2, 6, 8), dtype=np.uint8)
    inputs["masks"][0, :3, :4] = 1
    scale, sample_count, source = metric_scale.estimate_metric_scale(
        np.ones((2, 2, 2)),
        estimator=estimator,
        **inputs,
    )

    assert (scale, sample_count, source) == (2.0, 2, "estimated")
    assert received_masks[0].shape == (2, 2)
    np.testing.assert_array_equal(received_masks[0], [[1, 0], [0, 0]])


def test_hawor_metric_scale_accepts_only_finite_positive_values():
    values = iter([np.nan, -2.0, 2.5])

    scale, sample_count, source = metric_scale.estimate_metric_scale(
        np.ones((2, 2, 2)),
        estimator=lambda *_args, **_kwargs: next(values, 2.5),
        **_inputs(moving=True),
    )

    assert scale == pytest.approx(2.5)
    assert sample_count == 2
    assert source == "estimated"


def test_hawor_metric_scale_rejects_unscaled_moving_trajectory():
    with pytest.raises(
        RuntimeError,
        match=(
            "moving trajectory .*valid_disparity_frames=0/2.*"
            "translation_span=1"
        ),
    ):
        metric_scale.estimate_metric_scale(
            np.zeros((2, 2, 2)),
            estimator=lambda *_args, **_kwargs: np.nan,
            **_inputs(moving=True),
        )
