import numpy as np

from run_tvsum_consensus_oracle_8d import discrete_w2


def test_discrete_w2_is_zero_for_identical_distributions():
    distribution = np.asarray([0.2, 0.3, 0.5])

    assert discrete_w2(distribution, distribution) == 0.0


def test_discrete_w2_matches_single_pick_translation():
    left = np.asarray([1.0, 0.0, 0.0])
    right = np.asarray([0.0, 1.0, 0.0])

    assert discrete_w2(left, right) == 1.0


def test_discrete_w2_normalizes_unequal_total_mass():
    left = np.asarray([2.0, 0.0])
    right = np.asarray([0.0, 7.0])

    assert discrete_w2(left, right) == 1.0
