"""
tests/test_synthetics_pipeline.py

Pytest suite validating the end-to-end self-contained synthetics package.
"""
import pytest
import numpy as np
import sys
import os

SYNTHETICS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthetics"))
if SYNTHETICS_ROOT not in sys.path:
    sys.path.insert(0, SYNTHETICS_ROOT)

from data.synthetic_dataset import generate_synthetic_tvsum, get_video_distributions
from core.barycenter import wasserstein_barycenter, arithmetic_mean
from core.losses import w2_1d, cramer_1d, grad_cramer
from stages.stage1_consensus import run_stage1
from stages.stage2_disagreement import run_stage2
from stages.stage3_diagnosis import run_stage3
from stages.stage6_transformer import run_stage6


def test_synthetic_data_generation():
    videos = generate_synthetic_tvsum(n_videos=5, K=20)
    assert len(videos) == 5
    for vid, matrix in videos.items():
        assert matrix.shape == (20, 100)
        assert np.all(matrix >= 1.0) and np.all(matrix <= 5.0)

    dist = get_video_distributions(matrix, method="softmax", temperature=0.5)
    assert dist.shape == (20, 100)
    assert np.allclose(dist.sum(axis=1), 1.0)
    assert np.all(dist >= 0.0)


def test_barycenter_and_mean():
    videos = generate_synthetic_tvsum(n_videos=1, K=10)
    matrix = list(videos.values())[0]
    dist = get_video_distributions(matrix, method="softmax", temperature=0.5)

    bary = wasserstein_barycenter(dist)
    mean = arithmetic_mean(dist)

    assert bary.shape == (100,)
    assert mean.shape == (100,)
    assert np.isclose(bary.sum(), 1.0)
    assert np.isclose(mean.sum(), 1.0)
    assert np.all(bary >= 0.0)
    assert np.all(mean >= 0.0)


def test_stage1_oracle_consensus():
    summary, records = run_stage1(n_videos=5, verbose=False)
    assert summary["num_videos"] == 5
    assert summary["barycenter_wins"] >= 4
    assert summary["gap_absolute"]["mean"] > 0


def test_stage2_disagreement():
    res = run_stage2(n_videos=5, verbose=False)
    sharpening = res["sharpening"]
    # Temperature monotonicity: lower tau (sharper peaks, higher disagreement) -> larger gap
    mean_gap_02 = float(np.mean(sharpening[0.2]))
    mean_gap_10 = float(np.mean(sharpening[1.0]))
    assert mean_gap_02 > mean_gap_10


def test_stage3_gradient_norm_diagnosis():
    res = run_stage3(n_videos=2, verbose=False)
    g_norms = res["gradient_norms"]
    assert g_norms["ratio_cramer_over_mse"] > 100.0


def test_stage6_transformer_selftest():
    res = run_stage6(n_train_steps=30, verbose=False)
    assert res["shape_ok"] is True
    assert res["nonneg_ok"] is True
    assert res["norm_ok"] is True
    assert res["loss_decreasing"] is True
    assert res["all_checks_pass"] is True
