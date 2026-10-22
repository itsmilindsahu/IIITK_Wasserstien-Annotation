import numpy as np

from run_tvsum_mechanism_8e import (
    make_simulated_annotators,
    shifted_noisy_curve,
)


def test_zero_shift_without_noise_preserves_normalized_curve():
    curve = np.asarray([1.0, 2.0, 1.0])

    output = shifted_noisy_curve(
        curve,
        shift_width=0,
        rng=np.random.default_rng(4),
        noise_relative_to_max=0.0,
    )

    assert np.allclose(output, curve / curve.sum())


def test_simulated_annotators_are_reproducible_and_normalized():
    curve = np.asarray([0.1, 0.3, 0.6])

    first = make_simulated_annotators(
        curve, shift_width=2, rng=np.random.default_rng(19), annotator_count=20
    )
    second = make_simulated_annotators(
        curve, shift_width=2, rng=np.random.default_rng(19), annotator_count=20
    )

    assert np.array_equal(first, second)
    assert first.shape == (20, 3)
    assert np.all(first >= 0)
    assert np.allclose(first.sum(axis=1), 1.0)
