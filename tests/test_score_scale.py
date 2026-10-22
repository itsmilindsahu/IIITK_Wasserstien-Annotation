import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from standard_tvsum_cv import (  # noqa: E402
    DEFAULT_FEATURES,
    FEATURES_ENV,
    ANNOTATION_ENV,
    INFO_ENV,
    canonical_summary,
    load_tvsum_features,
)


@pytest.fixture(scope="module")
def real_video():
    annotation_path = os.environ.get(ANNOTATION_ENV)
    info_path = os.environ.get(INFO_ENV)
    if not annotation_path or not info_path:
        pytest.skip(
            f"Set {ANNOTATION_ENV} and {INFO_ENV} to run this real-data test."
        )
    paths = [DEFAULT_FEATURES, Path(annotation_path), Path(info_path)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.skip(f"Real TVSum test data is unavailable: {', '.join(missing)}")

    videos = load_tvsum_features(
        DEFAULT_FEATURES,
        Path(annotation_path),
        Path(info_path),
    )
    return videos["video_1"]


def _softmax_prediction(length: int) -> np.ndarray:
    logits = np.random.default_rng(12345).normal(size=length)
    scores = np.exp(logits - logits.max())
    return scores / scores.sum()


def _shot_integer_scores(scores: np.ndarray, video) -> np.ndarray:
    positions = video.picks
    if positions[-1] != video.n_frames:
        positions = np.concatenate(
            (positions, np.asarray([video.n_frames], dtype=np.int64))
        )
    frame_scores = np.zeros(video.n_frames, dtype=np.float64)
    for index in range(positions.size - 1):
        frame_scores[positions[index]:positions[index + 1]] = scores[index]
    return np.asarray(
        [
            int(1000 * frame_scores[int(start):int(end) + 1].mean())
            for start, end in video.change_points
        ],
        dtype=np.int64,
    )


def test_softmax_predictions_collapse_to_small_integer_shot_scores(real_video):
    prediction = _softmax_prediction(real_video.picks.size)
    shot_scores = _shot_integer_scores(prediction, real_video)
    distinct_count = np.unique(shot_scores).size
    zero_fraction = np.mean(shot_scores == 0)

    print(
        f"{real_video.key} softmax_sum={prediction.sum():.9f} "
        f"max_shot_score={shot_scores.max()} "
        f"zero_fraction={zero_fraction:.3f} "
        f"distinct_shot_scores={distinct_count}/{shot_scores.size}"
    )
    assert np.isclose(prediction.sum(), 1.0)
    assert shot_scores.max() <= 10
    assert zero_fraction >= 0.2
    assert distinct_count < shot_scores.size


def test_canonical_summary_is_invariant_to_positive_score_scale(real_video):
    prediction = _softmax_prediction(real_video.picks.size)
    expected = canonical_summary(prediction, real_video)

    for scale in (1, 10, 1000):
        actual = canonical_summary(scale * prediction, real_video)
        assert np.array_equal(actual, expected), (
            f"summary changed when scores were multiplied by {scale}"
        )
