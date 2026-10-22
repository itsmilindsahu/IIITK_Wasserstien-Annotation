import numpy as np

from run_tvsum_consensus_oracle import (
    finer_grid_barycenter,
    holm_adjust,
    smooth_distribution,
    tolerance_f1_at_picks,
)


def test_tolerance_f1_matches_selected_frames_within_pick_radius():
    predicted = np.asarray([0, 1, 0, 0, 0], dtype=np.uint8)
    heldout = np.asarray([0, 0, 1, 0, 0], dtype=np.uint8)
    picks = np.arange(predicted.size)

    assert tolerance_f1_at_picks(predicted, heldout, picks, 0) == 0.0
    assert tolerance_f1_at_picks(predicted, heldout, picks, 1) == 100.0


def test_tolerance_f1_requires_both_precision_and_recall_matches():
    predicted = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
    heldout = np.asarray([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.uint8)
    picks = np.arange(predicted.size)

    # Both predicted picks match, but one of three held-out picks does not.
    assert tolerance_f1_at_picks(predicted, heldout, picks, 1) == 80.0


def test_consensus_variants_preserve_probability_mass():
    distributions = np.asarray(
        [[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]], dtype=np.float64
    )
    fine = finer_grid_barycenter(distributions, factor=4)
    smoothed = smooth_distribution(fine, sigma=2)

    assert fine.shape == (3,)
    assert np.all(fine >= 0)
    assert np.isclose(fine.sum(), 1.0)
    assert smoothed.shape == fine.shape
    assert np.all(smoothed >= 0)
    assert np.isclose(smoothed.sum(), 1.0)


def test_holm_adjustment_is_monotone_and_no_less_than_raw_p():
    raw = {"a": 0.001, "b": 0.01, "c": 0.04}
    adjusted = holm_adjust(raw)

    assert adjusted == {"a": 0.003, "b": 0.02, "c": 0.04}
    assert all(adjusted[key] >= raw[key] for key in raw)
