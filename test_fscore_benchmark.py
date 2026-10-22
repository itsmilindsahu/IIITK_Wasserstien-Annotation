import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

import torch

from standard_tvsum_cv import (
    ARMS,
    BiLSTMRegressor,
    TemporalConvRegressor,
    TVSumVideo,
    canonical_summary,
    evaluate_summary,
    load_dsnet_folds,
    mean_and_barycenter,
    summarize_records,
)


def _toy_video():
    n_frames = 100
    change_points = np.asarray(
        [[start, start + 9] for start in range(0, n_frames, 10)],
        dtype=np.int64,
    )
    user_summary = np.zeros((2, n_frames), dtype=np.uint8)
    user_summary[0, 20:30] = 1
    user_summary[1, 30:40] = 1
    return TVSumVideo(
        key="video_1",
        video_id="toy",
        features=np.zeros((n_frames, 1024), dtype=np.float32),
        annotation_scores=np.ones((20, n_frames), dtype=np.float32),
        user_summary=user_summary,
        change_points=change_points,
        frames_per_segment=np.full(10, 10, dtype=np.int64),
        picks=np.arange(n_frames, dtype=np.int64),
        n_frames=n_frames,
        gtscore=np.ones(n_frames, dtype=np.float32),
        gtsummary=np.ones(n_frames, dtype=np.uint8),
    )


def test_tvsum_summary_uses_shots_and_respects_15_percent_budget():
    video = _toy_video()
    scores = np.zeros(video.picks.size, dtype=np.float32)
    scores[20:30] = 1.0

    summary = canonical_summary(scores, video)

    assert summary.sum() == 10
    assert np.all(summary[20:30] == 1)
    assert summary.sum() <= int(video.n_frames * 0.15)


def test_tvsum_knapsack_preserves_relative_segment_score_ranking():
    video = _toy_video()
    scores = np.zeros(video.picks.size, dtype=np.float32)
    scores[:10] = 0.5001
    scores[10:20] = 0.5009

    summary = canonical_summary(scores, video)

    assert not np.any(summary[:10])
    assert np.all(summary[10:20] == 1)


def test_tvsum_f1_averages_over_users():
    video = _toy_video()
    scores = np.zeros(video.picks.size, dtype=np.float32)
    scores[20:30] = 1.0

    summary = canonical_summary(scores, video)
    metrics = evaluate_summary(summary, video.user_summary)

    assert metrics["f1"] == 50.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_dsnet_five_splits_partition_all_videos_within_each_split():
    known_keys = {f"video_{index}" for index in range(1, 51)}
    folds = load_dsnet_folds(
        Path(PROJECT_ROOT) / "splits" / "tvsum.yml", known_keys
    )

    assert len(folds) == 5
    assert all(len(fold["train"]) == 40 and len(fold["test"]) == 10 for fold in folds)
    assert all(
        not set(fold["train"]) & set(fold["test"])
        and set(fold["train"]) | set(fold["test"]) == known_keys
        for fold in folds
    )


def test_mean_and_wasserstein_barycenter_are_probability_distributions():
    distributions = np.asarray(
        [[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]], dtype=np.float32
    )

    mean_target, barycenter_target = mean_and_barycenter(distributions)

    assert np.isclose(mean_target.sum(), 1.0)
    assert np.isclose(barycenter_target.sum(), 1.0)
    assert np.all(barycenter_target >= 0)


def test_temporal_model_masks_padding_and_normalizes_each_video():
    if not hasattr(torch, "randn"):
        import pytest
        pytest.skip("PyTorch is not installed in the environment.")
    torch.manual_seed(4)
    model = TemporalConvRegressor()
    features = torch.randn(2, 6, 1024)
    valid_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, False, False]]
    )

    predictions = model(features, valid_mask)

    assert torch.isfinite(predictions).all()
    torch.testing.assert_close(predictions.sum(dim=-1), torch.ones(2))
    assert torch.all(predictions[1, 4:] == 0)


def test_bilstm_model_masks_padding_and_normalizes_each_video():
    if not hasattr(torch, "randn"):
        import pytest
        pytest.skip("PyTorch is not installed in the environment.")
    torch.manual_seed(4)
    model = BiLSTMRegressor()
    model.eval()
    features = torch.randn(2, 6, 1024)
    valid_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, False, False]]
    )

    predictions = model(features, valid_mask)

    assert torch.isfinite(predictions).all()
    torch.testing.assert_close(predictions.sum(dim=-1), torch.ones(2))
    assert torch.all(predictions[1, 4:] == 0)



def test_summary_aggregates_seed_and_fold_video_observations():
    records = []
    baselines = {
        "random": [],
        "uniform": [],
        "human_leave_one_out": [],
    }
    for fold in range(1, 6):
        for video_index in range(2):
            video_key = f"video_{video_index + 1}"
            for arm_index, arm in enumerate(ARMS):
                for seed in (11, 23):
                    f1 = 40.0 + arm_index + fold + video_index
                    for metric, value in (
                        ("f1", f1),
                        ("spearman_rho", 0.1 * arm_index),
                        ("kendall_tau", 0.05 * arm_index),
                    ):
                        records.append({
                            "fold": fold,
                            "seed": seed,
                            "arm": arm,
                            "video_key": video_key,
                            "metric": metric,
                            "value": value,
                            "alpha": 0.5 if arm == "F_blended_cramer" else None,
                        })
            for seed in (11, 23):
                baselines["random"].append({
                    "fold": fold,
                    "seed": seed,
                    "video_key": video_key,
                    "metric": "f1",
                    "value": 35.0 + fold + video_index + (0.5 if seed == 23 else 0),
                })
            for baseline in ("uniform", "human_leave_one_out"):
                baselines[baseline].append({
                    "fold": fold,
                    "video_key": video_key,
                    "metric": "f1",
                    "value": 38.0 + fold + video_index,
                })

    summary, per_fold_video = summarize_records(records, baselines)

    assert len(per_fold_video["A_mean_mse"]) == 10
    assert summary["models"]["A_mean_mse"]["f1_mean_seed_std"] == 0.0
    assert summary["barycenter_effect_under_cramer"]["n"] == 2
    assert summary["baselines"]["random"]["f1_mean_seed_std"] > 0
