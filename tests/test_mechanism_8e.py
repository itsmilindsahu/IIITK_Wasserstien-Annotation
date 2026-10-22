import numpy as np

from run_tvsum_mechanism_8e import (
    peak_diagnostics,
    weighted_selection_overlap,
)


def test_peak_diagnostics_uses_fixed_relative_prominence_and_annotator_modes():
    mean_curve = np.asarray([0.0, 1.0, 0.0, 0.5, 0.0])
    annotators = np.asarray([
        [0.0, 0.8, 0.0, 0.2, 0.0],
        [0.0, 0.6, 0.0, 0.4, 0.0],
    ])

    peak_count, mode_sd, peak_indices = peak_diagnostics(mean_curve, annotators)

    assert peak_count == 2
    assert peak_indices == [1, 3]
    assert mode_sd == 0.0


def test_weighted_overlap_is_mass_weighted_over_mean_selected_picks():
    mean_curve = np.asarray([0.1, 0.3, 0.6])
    mean_selected = np.asarray([True, True, False])
    barycenter_selected = np.asarray([False, True, True])

    fraction, denominator_mass, shared_count = weighted_selection_overlap(
        mean_curve, mean_selected, barycenter_selected
    )

    assert np.isclose(fraction, 0.75)
    assert np.isclose(denominator_mass, 0.4)
    assert shared_count == 1
