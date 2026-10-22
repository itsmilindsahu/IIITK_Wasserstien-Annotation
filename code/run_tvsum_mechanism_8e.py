"""Real-data mechanism analysis and a separately labelled jitter simulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.signal import find_peaks

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import run_tvsum_consensus_oracle as oracle  # noqa: E402
import standard_tvsum_cv as benchmark  # noqa: E402


BOOTSTRAP_RESAMPLES = 10_000
PEAK_PROMINENCE = 0.10
QUALITATIVE_VIDEO_IDS = ("uGu_10sucQo", "i3wAGJaaktw")
QUALITATIVE_ANNOTATORS = (0, 9, 19)
SHIFT_WIDTHS = (0, 1, 2, 4, 8, 16, 32)
SIMULATED_ANNOTATORS = 20
NOISE_STD_RELATIVE_TO_MAX = 0.15


def stable_seed(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = scipy.stats.spearmanr(left, right).statistic
    return 0.0 if not np.isfinite(result) else float(result)


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, seed: int, samples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation bootstrap needs equal, finite vectors with at least two values.")
    if samples != BOOTSTRAP_RESAMPLES:
        raise ValueError(f"This analysis requires exactly {BOOTSTRAP_RESAMPLES} resamples.")
    observed = _finite_spearman(x, y)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, x.size, size=x.size)
        estimates[index] = _finite_spearman(x[selected], y[selected])
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "n_videos": int(x.size),
        "spearman_rho": observed,
        "bootstrap_ci_95_lower": float(lower),
        "bootstrap_ci_95_upper": float(upper),
        "bootstrap_resamples": int(samples),
    }


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, samples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or samples < 1:
        raise ValueError("Mean bootstrap needs finite nonempty data and positive resamples.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def peak_diagnostics(
    mean_curve: np.ndarray, annotator_distributions: np.ndarray
) -> tuple[int, float, list[int]]:
    mean_curve = np.asarray(mean_curve, dtype=np.float64).reshape(-1)
    annotator_distributions = np.asarray(annotator_distributions, dtype=np.float64)
    if (
        mean_curve.size == 0
        or annotator_distributions.ndim != 2
        or annotator_distributions.shape[1] != mean_curve.size
        or not np.isfinite(mean_curve).all()
        or not np.isfinite(annotator_distributions).all()
    ):
        raise ValueError("Peak diagnostics require aligned finite curves.")
    maximum = float(mean_curve.max())
    if maximum <= 0:
        raise ValueError("Mean consensus curve must have positive mass.")
    peaks, _ = find_peaks(mean_curve / maximum, prominence=PEAK_PROMINENCE)
    annotator_modes = np.argmax(annotator_distributions, axis=1)
    return int(peaks.size), float(np.std(annotator_modes, ddof=0)), peaks.tolist()


def weighted_selection_overlap(
    mean_curve: np.ndarray,
    mean_selected: np.ndarray,
    barycenter_selected: np.ndarray,
) -> tuple[float, float, int]:
    mean_curve = np.asarray(mean_curve, dtype=np.float64).reshape(-1)
    mean_selected = np.asarray(mean_selected, dtype=bool).reshape(-1)
    barycenter_selected = np.asarray(barycenter_selected, dtype=bool).reshape(-1)
    if not (
        mean_curve.size
        and mean_curve.size == mean_selected.size == barycenter_selected.size
        and np.isfinite(mean_curve).all()
        and np.all(mean_curve >= 0)
    ):
        raise ValueError("Overlap inputs must be aligned, finite, nonnegative vectors.")
    selected_mass = float(mean_curve[mean_selected].sum())
    if selected_mass <= 0:
        raise ValueError("The mean target's selected picks must contain positive mass.")
    shared = mean_selected & barycenter_selected
    shared_mass = float(mean_curve[shared].sum())
    return shared_mass / selected_mass, selected_mass, int(shared.sum())


def _validate_inputs(
    result_path: Path,
    features_path: Path,
    annotations_path: Path,
    info_path: Path,
) -> tuple[dict[str, Any], dict[str, benchmark.TVSumVideo], dict[str, dict[str, Any]], dict[str, str]]:
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    hashes = {
        "feature_sha256": benchmark.sha256_file(features_path),
        "annotation_sha256": benchmark.sha256_file(annotations_path),
        "info_sha256": benchmark.sha256_file(info_path),
    }
    if any(saved["protocol"].get(key) != value for key, value in hashes.items()):
        raise ValueError("Local TVSum inputs do not match the Checkpoint 8 saved fingerprints.")
    videos = benchmark.load_tvsum_features(features_path, annotations_path, info_path)
    saved_rows = {row["video_id"]: row for row in saved["per_video"]}
    if len(saved_rows) != 50 or set(saved_rows) != {video.video_id for video in videos.values()}:
        raise ValueError("Saved oracle results must align exactly with all 50 real TVSum videos.")
    if saved["protocol"].get("video_count") != 50:
        raise ValueError("Saved oracle protocol does not record the expected 50 videos.")
    return saved, videos, saved_rows, hashes


def _selected_pick_mask(summary: np.ndarray, picks: np.ndarray) -> np.ndarray:
    return np.asarray(summary).reshape(-1)[picks] > 0


def _compute_mechanism_rows(
    videos: dict[str, benchmark.TVSumVideo],
    saved_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows = []
    targets_by_id: dict[str, dict[str, np.ndarray]] = {}
    for video in videos.values():
        distributions = benchmark.annotation_distributions(video)
        mean_target, barycenter_target = benchmark.mean_and_barycenter(distributions)
        peak_count, peak_spread, peak_indices = peak_diagnostics(mean_target, distributions)
        mean_summary = benchmark.canonical_summary(mean_target, video)
        barycenter_summary = benchmark.canonical_summary(barycenter_target, video)
        mean_selected = _selected_pick_mask(mean_summary, video.picks)
        barycenter_selected = _selected_pick_mask(barycenter_summary, video.picks)
        overlap_fraction, mean_selected_mass, shared_pick_count = weighted_selection_overlap(
            mean_target, mean_selected, barycenter_selected
        )
        saved = saved_rows[video.video_id]
        mean_f1 = float(saved["variants"]["mean"]["f1"])
        barycenter_f1 = float(saved["variants"]["barycenter"]["f1"])
        rows.append({
            "video_id": video.video_id,
            "video_key": video.key,
            "mean_f1": mean_f1,
            "barycenter_f1": barycenter_f1,
            "f1_gap_barycenter_minus_mean": barycenter_f1 - mean_f1,
            "mean_consensus_interior_peak_count": peak_count,
            "mean_pairwise_spearman_rho": float(saved["mean_pairwise_spearman_rho"]),
            "annotator_dominant_peak_location_sd_picks": peak_spread,
            "weighted_overlap_fraction_of_mean_selected_mass": overlap_fraction,
            "mean_target_selected_pick_mass": mean_selected_mass,
            "shared_selected_pick_count": shared_pick_count,
            "mean_selected_pick_count": int(mean_selected.sum()),
            "barycenter_selected_pick_count": int(barycenter_selected.sum()),
            "mean_consensus_peak_indices_zero_based": peak_indices,
        })
        targets_by_id[video.video_id] = {
            "mean": mean_target,
            "barycenter": barycenter_target,
        }
    return rows, targets_by_id


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _qualitative_figure(
    video: benchmark.TVSumVideo,
    targets: dict[str, np.ndarray],
    path: Path,
) -> None:
    distributions = benchmark.annotation_distributions(video)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    index = np.arange(video.picks.size)
    for axis, variant, title in (
        (axes[0], "mean", "Arithmetic mean target"),
        (axes[1], "barycenter", "1D Wasserstein barycenter target"),
    ):
        target = targets[variant]
        summary = benchmark.canonical_summary(target, video)
        selected = _selected_pick_mask(summary, video.picks)
        axis.plot(index, target, color="black", linewidth=1.5, label=title)
        axis.fill_between(
            index, 0, target, where=selected, color="orange", alpha=0.30,
            step="mid", label="target's selected 15%-budget picks",
        )
        for annotator in QUALITATIVE_ANNOTATORS:
            curve = distributions[annotator]
            axis.plot(
                index, curve, linewidth=0.9, alpha=0.65,
                label=f"annotator {annotator + 1}",
            )
        axis.set_ylabel("Normalized pick mass")
        axis.set_title(title)
        axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Pick index")
    fig.suptitle(f"Real TVSum qualitative mechanism check: {video.video_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _video_note(row: dict[str, Any]) -> list[str]:
    return [
        (
            f"{row['video_id']}: mean-target F1 is {row['mean_f1']:.3f} and "
            f"barycenter-target F1 is {row['barycenter_f1']:.3f}; the saved "
            f"barycenter-minus-mean gap is {row['f1_gap_barycenter_minus_mean']:+.3f} points."
        ),
        (
            f"The mean consensus has {row['mean_consensus_interior_peak_count']} "
            f"interior peaks at fixed normalized prominence {PEAK_PROMINENCE:.2f}; "
            f"the population SD of annotator dominant-peak locations is "
            f"{row['annotator_dominant_peak_location_sd_picks']:.3f} picks."
        ),
        (
            f"The 15%-duration canonical summaries share "
            f"{row['shared_selected_pick_count']} sampled picks; those shared picks "
            f"contain {100 * row['weighted_overlap_fraction_of_mean_selected_mass']:.2f}% "
            "of the mean target's selected pick mass."
        ),
    ]


def analyze_mechanism(
    result_path: Path,
    features_path: Path,
    annotations_path: Path,
    info_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    started = time.time()
    saved, videos, saved_rows, hashes = _validate_inputs(
        result_path, features_path, annotations_path, info_path
    )
    rows, targets = _compute_mechanism_rows(videos, saved_rows)
    gap = np.asarray([row["f1_gap_barycenter_minus_mean"] for row in rows])
    descriptors = (
        (
            "mean_consensus_interior_peak_count",
            "count; scipy.signal.find_peaks on mean curve divided by its maximum; "
            f"fixed prominence={PEAK_PROMINENCE}",
        ),
        (
            "mean_pairwise_spearman_rho",
            "unitless; saved mean pairwise Spearman rho among 20 annotators",
        ),
        (
            "annotator_dominant_peak_location_sd_picks",
            "pick indices; population SD of each annotator's argmax location",
        ),
        (
            "weighted_overlap_fraction_of_mean_selected_mass",
            "fraction; mean-target probability mass selected by both canonical summaries "
            "divided by mass selected by the mean summary",
        ),
    )
    correlation_rows = []
    for name, definition in descriptors:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        stats = bootstrap_spearman_ci(
            values, gap, stable_seed(f"tvsum-8e-mechanism::{name}")
        )
        correlation_rows.append({
            "descriptor": name,
            "definition": definition,
            **stats,
        })

    output_dir.mkdir(parents=True, exist_ok=False)
    notes: dict[str, list[str]] = {}
    figures = []
    rows_by_id = {row["video_id"]: row for row in rows}
    for video_id in QUALITATIVE_VIDEO_IDS:
        if video_id not in rows_by_id:
            raise ValueError(f"Required qualitative video {video_id} is absent.")
        video = next(item for item in videos.values() if item.video_id == video_id)
        filename = f"qualitative_{video_id}.png"
        _qualitative_figure(video, targets[video_id], output_dir / filename)
        figures.append(filename)
        notes[video_id] = _video_note(rows_by_id[video_id])

    result = {
        "protocol": {
            "source_oracle_json": str(result_path.relative_to(PROJECT_ROOT)),
            "source_oracle_json_sha256": benchmark.sha256_file(result_path),
            **hashes,
            "video_count": len(rows),
            "real_tvsum_data_only": True,
            "model_training": False,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_ci": "percentile 95% CI over videos",
            "peak_detection": {
                "method": "scipy.signal.find_peaks on mean consensus divided by its maximum",
                "prominence": PEAK_PROMINENCE,
                "endpoints_included": False,
            },
            "peak_dispersion": (
                "population standard deviation across annotators' dominant-peak "
                "argmax locations in pick-index units"
            ),
            "selection_overlap": (
                "mean-target mass at picks selected by both the mean and barycenter "
                "canonical 15%-duration summaries, divided by mean-target mass at "
                "picks selected by the mean summary"
            ),
            "f1_gap": "saved per-video barycenter F1 minus mean F1 from Checkpoint 8",
            "interpretation": "associations only; no causal interpretation",
            "qualitative_annotator_indices_zero_based": list(QUALITATIVE_ANNOTATORS),
        },
        "per_video": rows,
        "spearman_associations": correlation_rows,
        "qualitative_notes": notes,
        "figures": figures,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "per_video_mechanism.csv",
        rows,
        (
            "video_id", "mean_f1", "barycenter_f1",
            "f1_gap_barycenter_minus_mean", "mean_consensus_interior_peak_count",
            "mean_pairwise_spearman_rho", "annotator_dominant_peak_location_sd_picks",
            "weighted_overlap_fraction_of_mean_selected_mass",
            "mean_target_selected_pick_mass", "shared_selected_pick_count",
            "mean_selected_pick_count", "barycenter_selected_pick_count",
        ),
    )
    _write_csv(
        output_dir / "mechanism_correlations.csv",
        correlation_rows,
        (
            "descriptor", "definition", "n_videos", "spearman_rho",
            "bootstrap_ci_95_lower", "bootstrap_ci_95_upper", "bootstrap_resamples",
        ),
    )
    summary_lines = [
        "REAL TVSUM MECHANISM ANALYSIS (CHECKPOINT 1)",
        "Associations only; no causal interpretation. No model training.",
        "Every reported statistic below is serialized in results.json.",
        "",
    ]
    for item in correlation_rows:
        summary_lines.append(
            f"{item['descriptor']}: Spearman rho={item['spearman_rho']:.6f}, "
            f"95% bootstrap CI [{item['bootstrap_ci_95_lower']:.6f}, "
            f"{item['bootstrap_ci_95_upper']:.6f}] (n={item['n_videos']})."
        )
    for video_id in QUALITATIVE_VIDEO_IDS:
        summary_lines.extend(["", *notes[video_id]])
    (output_dir / "SUMMARY.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    return result


def shifted_noisy_curve(
    mean_curve: np.ndarray,
    shift_width: int,
    rng: np.random.Generator,
    noise_relative_to_max: float = NOISE_STD_RELATIVE_TO_MAX,
) -> np.ndarray:
    curve = np.asarray(mean_curve, dtype=np.float64).reshape(-1)
    if (
        curve.size == 0 or not np.isfinite(curve).all()
        or np.any(curve < 0) or curve.sum() <= 0
    ):
        raise ValueError("Simulation source curve must be finite, nonnegative, and nonempty.")
    if shift_width < 0 or noise_relative_to_max < 0 or not np.isfinite(noise_relative_to_max):
        raise ValueError("Shift width and relative noise must be finite and nonnegative.")
    normalized = curve / curve.sum()
    shift = int(rng.integers(-shift_width, shift_width + 1))
    positions = np.arange(normalized.size, dtype=np.float64) - shift
    shifted = np.interp(
        positions,
        np.arange(normalized.size, dtype=np.float64),
        normalized,
        left=0.0,
        right=0.0,
    )
    noise_scale = noise_relative_to_max * float(normalized.max())
    perturbed = np.maximum(0.0, shifted + rng.normal(0.0, noise_scale, normalized.size))
    total = float(perturbed.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Jitter simulation produced a zero or invalid distribution.")
    return (perturbed / total).astype(np.float32)


def make_simulated_annotators(
    mean_curve: np.ndarray,
    shift_width: int,
    rng: np.random.Generator,
    annotator_count: int = SIMULATED_ANNOTATORS,
) -> np.ndarray:
    if annotator_count < 2:
        raise ValueError("At least two simulated annotators are required.")
    distributions = np.stack([
        shifted_noisy_curve(mean_curve, shift_width, rng)
        for _ in range(annotator_count)
    ])
    if (
        not np.isfinite(distributions).all()
        or np.any(distributions < 0)
        or not np.allclose(distributions.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    ):
        raise AssertionError("Simulated annotator distributions must be normalized.")
    return distributions


def _simulate_video_at_width(
    video: benchmark.TVSumVideo,
    mean_curve: np.ndarray,
    shift_width: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(
        stable_seed(f"tvsum-simulation-jitter-v1::{video.video_id}::{shift_width}")
    )
    simulated = make_simulated_annotators(mean_curve, shift_width, rng)
    references = [
        benchmark.canonical_summary(user_curve, video)
        for user_curve in simulated
    ]
    mean_f1 = []
    barycenter_f1 = []
    for heldout_index, heldout_reference in enumerate(references):
        train = np.delete(simulated, heldout_index, axis=0)
        mean_target, barycenter_target = benchmark.mean_and_barycenter(train)
        mean_summary = benchmark.canonical_summary(mean_target, video)
        barycenter_summary = benchmark.canonical_summary(barycenter_target, video)
        heldout = heldout_reference.reshape(1, -1)
        mean_f1.append(benchmark.evaluate_summary(mean_summary, heldout)["f1"])
        barycenter_f1.append(benchmark.evaluate_summary(barycenter_summary, heldout)["f1"])
    mean_score = float(np.mean(mean_f1))
    barycenter_score = float(np.mean(barycenter_f1))
    return {
        "video_id": video.video_id,
        "video_key": video.key,
        "shift_width_picks": int(shift_width),
        "simulated_annotator_count": SIMULATED_ANNOTATORS,
        "mean_target_f1": mean_score,
        "barycenter_target_f1": barycenter_score,
        "f1_gain_barycenter_minus_mean": barycenter_score - mean_score,
    }


def _summarize_simulation(per_video: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    for width in SHIFT_WIDTHS:
        group = [row for row in per_video if row["shift_width_picks"] == width]
        gains = np.asarray(
            [row["f1_gain_barycenter_minus_mean"] for row in group],
            dtype=np.float64,
        )
        lower, upper = bootstrap_mean_ci(
            gains, stable_seed(f"tvsum-simulation-gain-ci::{width}")
        )
        summary_rows.append({
            "shift_width_picks": int(width),
            "n_videos": int(gains.size),
            "mean_target_f1": float(np.mean([row["mean_target_f1"] for row in group])),
            "barycenter_target_f1": float(np.mean([row["barycenter_target_f1"] for row in group])),
            "mean_gain_barycenter_minus_mean": float(gains.mean()),
            "bootstrap_ci_95_lower": lower,
            "bootstrap_ci_95_upper": upper,
            "videos_barycenter_improved": int(np.count_nonzero(gains > 0)),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        })
    first_positive = next(
        (row["shift_width_picks"] for row in summary_rows
         if row["mean_gain_barycenter_minus_mean"] > 0),
        None,
    )
    first_supported = next(
        (row["shift_width_picks"] for row in summary_rows
         if row["bootstrap_ci_95_lower"] > 0),
        None,
    )
    for row in summary_rows:
        row["first_positive_mean_gain_shift_width_picks"] = first_positive
        row["first_ci_supported_gain_shift_width_picks"] = first_supported
    return summary_rows


def run_jitter_simulation(
    result_path: Path,
    features_path: Path,
    annotations_path: Path,
    info_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    started = time.time()
    _, videos, _, hashes = _validate_inputs(
        result_path, features_path, annotations_path, info_path
    )
    per_video = []
    for video in videos.values():
        distributions = benchmark.annotation_distributions(video)
        real_mean, _ = benchmark.mean_and_barycenter(distributions)
        for width in SHIFT_WIDTHS:
            per_video.append(
                _simulate_video_at_width(video, real_mean, width)
            )
    summary_rows = _summarize_simulation(per_video)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol": {
            "label": "SIMULATION; never mix into real-data tables",
            "source_oracle_json": str(result_path.relative_to(PROJECT_ROOT)),
            "source_oracle_json_sha256": benchmark.sha256_file(result_path),
            **hashes,
            "source_curves": "real TVSum per-video mean consensus distributions",
            "real_tvsum_video_count": len(videos),
            "simulated_annotators_per_video_per_width": SIMULATED_ANNOTATORS,
            "shift_widths_picks": list(SHIFT_WIDTHS),
            "shift_sampling": "independent integer uniform in [-width, +width] per simulated annotator",
            "shift_boundary": "zero padding; no circular wraparound",
            "noise": {
                "distribution": "Gaussian with independent per-pick noise",
                "standard_deviation_relative_to_source_curve_max": NOISE_STD_RELATIVE_TO_MAX,
                "post_noise": "clip below at zero and renormalize to unit mass",
            },
            "consensus": (
                "leave-one-simulated-annotator-out arithmetic mean and the "
                "benchmark's 1D quantile-averaged Wasserstein barycenter"
            ),
            "heldout_reference": (
                "canonical_summary of one held-out simulated annotator's score curve "
                "using the real video's frame count, picks, shot boundaries, and 15% budget"
            ),
            "bootstrap": "10,000-resample percentile 95% CI over the 50 video-level gains",
            "model_training": False,
        },
        "per_video": per_video,
        "shift_summary": summary_rows,
        "elapsed_seconds": time.time() - started,
    }
    result_path_out = output_dir / "SIMULATION_results.json"
    result_path_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "SIMULATION_per_video.csv",
        per_video,
        (
            "video_id", "shift_width_picks", "simulated_annotator_count",
            "mean_target_f1", "barycenter_target_f1",
            "f1_gain_barycenter_minus_mean",
        ),
    )
    _write_csv(
        output_dir / "SIMULATION_shift_summary.csv",
        summary_rows,
        (
            "shift_width_picks", "n_videos", "mean_target_f1",
            "barycenter_target_f1", "mean_gain_barycenter_minus_mean",
            "bootstrap_ci_95_lower", "bootstrap_ci_95_upper",
            "videos_barycenter_improved", "bootstrap_resamples",
            "first_positive_mean_gain_shift_width_picks",
            "first_ci_supported_gain_shift_width_picks",
        ),
    )
    widths = np.asarray([row["shift_width_picks"] for row in summary_rows])
    gains = np.asarray([row["mean_gain_barycenter_minus_mean"] for row in summary_rows])
    lower = np.asarray([row["bootstrap_ci_95_lower"] for row in summary_rows])
    upper = np.asarray([row["bootstrap_ci_95_upper"] for row in summary_rows])
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(widths, gains, marker="o", color="tab:blue", label="Barycenter − mean F1")
    axis.fill_between(widths, lower, upper, color="tab:blue", alpha=0.2, label="95% video bootstrap CI")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Uniform shift half-width (picks)")
    axis.set_ylabel("F1 gain (points)")
    axis.set_title("SIMULATION: annotator jitter vs consensus F1")
    axis.legend()
    fig.tight_layout()
    figure_name = "SIMULATION_gain_vs_shift_width.png"
    fig.savefig(output_dir / figure_name, dpi=150)
    plt.close(fig)
    result["figure"] = figure_name
    result["elapsed_seconds"] = time.time() - started
    result_path_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    first_positive = summary_rows[0]["first_positive_mean_gain_shift_width_picks"]
    first_supported = summary_rows[0]["first_ci_supported_gain_shift_width_picks"]
    summary_lines = [
        "SIMULATION ONLY — LABELLED JITTER; NOT REAL TVSUM PERFORMANCE",
        "Each cohort uses 20 simulated annotators generated from real-video mean curves.",
        "All gains and CIs are serialized in SIMULATION_results.json.",
        (
            "First declared shift width with positive mean gain: "
            f"{'none' if first_positive is None else str(first_positive) + ' picks'}."
        ),
        (
            "First declared shift width with a 95% CI wholly above zero: "
            f"{'none' if first_supported is None else str(first_supported) + ' picks'}."
        ),
        "",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"Shift width {row['shift_width_picks']} picks: mean-target F1="
            f"{row['mean_target_f1']:.4f}, barycenter-target F1="
            f"{row['barycenter_target_f1']:.4f}, gain="
            f"{row['mean_gain_barycenter_minus_mean']:+.4f}, 95% CI "
            f"[{row['bootstrap_ci_95_lower']:+.4f}, "
            f"{row['bootstrap_ci_95_upper']:+.4f}], improved on "
            f"{row['videos_barycenter_improved']}/{row['n_videos']} videos."
        )
    (output_dir / "SIMULATION_SUMMARY.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    return result


def _input_defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "consensus_oracle_8b" / "results.json",
    )
    parser.add_argument("--features", type=Path, default=benchmark.DEFAULT_FEATURES)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--info", type=Path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mechanism", "simulation", "all"), default="all")
    _input_defaults(parser)
    args = parser.parse_args()
    annotations = args.annotations or (
        Path(os.environ[benchmark.ANNOTATION_ENV])
        if os.environ.get(benchmark.ANNOTATION_ENV)
        else None
    )
    info = args.info or (
        Path(os.environ[benchmark.INFO_ENV])
        if os.environ.get(benchmark.INFO_ENV)
        else None
    )
    if annotations is None:
        parser.error(f"Provide --annotations or set {benchmark.ANNOTATION_ENV}.")
    if info is None:
        parser.error(f"Provide --info or set {benchmark.INFO_ENV}.")
    mechanism_dir = PROJECT_ROOT / "results" / "mechanism_8e"
    simulation_dir = PROJECT_ROOT / "results" / "simulation_jitter"
    if args.mode in ("mechanism", "all"):
        if mechanism_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {mechanism_dir}")
    if args.mode in ("simulation", "all"):
        if simulation_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {simulation_dir}")
    if args.mode in ("mechanism", "all"):
        output = analyze_mechanism(
            args.results, args.features, annotations, info, mechanism_dir
        )
        print(json.dumps({
            "phase": "mechanism",
            "elapsed_seconds": output["elapsed_seconds"],
            "correlations": output["spearman_associations"],
            "qualitative_notes": output["qualitative_notes"],
        }, sort_keys=True), flush=True)
    if args.mode in ("simulation", "all"):
        output = run_jitter_simulation(
            args.results, args.features, annotations, info, simulation_dir
        )
        print(json.dumps({
            "phase": "simulation",
            "elapsed_seconds": output["elapsed_seconds"],
            "shift_summary": output["shift_summary"],
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
