"""Compute leave-one-annotator-out TVSum consensus-summary oracle scores."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter1d

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import standard_tvsum_cv as benchmark  # noqa: E402


TOLERANCE_PICK_RADII = (2, 5, 10)
GAUSSIAN_SIGMAS = (1, 2, 4)
VARIANTS = (
    "mean",
    "barycenter",
    "barycenter_fine4",
    "barycenter_gaussian_sigma_1",
    "barycenter_gaussian_sigma_2",
    "barycenter_gaussian_sigma_4",
    "mean_gaussian_sigma_1",
    "mean_gaussian_sigma_2",
    "mean_gaussian_sigma_4",
)
COMPARISON_VARIANTS = tuple(variant for variant in VARIANTS if variant != "mean")


def tolerance_f1_at_picks(
    predicted_summary: np.ndarray,
    heldout_summary: np.ndarray,
    picks: np.ndarray,
    tolerance_picks: int,
) -> float:
    if tolerance_picks < 0:
        raise ValueError("tolerance_picks must be nonnegative.")
    predicted_frames = np.asarray(predicted_summary).reshape(-1)
    heldout_frames = np.asarray(heldout_summary).reshape(-1)
    pick_indices = np.asarray(picks, dtype=np.int64).reshape(-1)
    if predicted_frames.size != heldout_frames.size:
        raise ValueError("Predicted and held-out summaries must have equal frame counts.")
    if (
        pick_indices.size == 0
        or np.any(pick_indices < 0)
        or np.any(pick_indices >= predicted_frames.size)
        or np.any(np.diff(pick_indices) <= 0)
    ):
        raise ValueError("picks must be nonempty, increasing, and within summary bounds.")
    predicted = predicted_frames[pick_indices] > 0
    heldout = heldout_frames[pick_indices] > 0
    if predicted.size != heldout.size:
        raise ValueError("Predicted and held-out pick vectors must have equal length.")
    predicted_indices = np.flatnonzero(predicted)
    heldout_indices = np.flatnonzero(heldout)
    if predicted_indices.size == 0 or heldout_indices.size == 0:
        return 0.0
    distances = np.abs(predicted_indices[:, None] - heldout_indices[None, :])
    precision = float(np.any(distances <= tolerance_picks, axis=1).mean())
    recall = float(np.any(distances <= tolerance_picks, axis=0).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall) * 100


def finer_grid_barycenter(distributions: np.ndarray, factor: int = 4) -> np.ndarray:
    distributions = np.asarray(distributions, dtype=np.float64)
    if distributions.ndim != 2 or distributions.shape[0] == 0 or distributions.shape[1] == 0:
        raise ValueError("distributions must have shape (users, non-empty picks).")
    if factor < 1 or not np.isfinite(distributions).all() or np.any(distributions < 0):
        raise ValueError("factor must be positive and distributions finite/nonnegative.")
    totals = distributions.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every annotator distribution must have positive mass.")
    normalized = distributions / totals[:, None]
    n_picks = normalized.shape[1]
    quantiles = (np.arange(n_picks * factor, dtype=np.float64) + 0.5) / (n_picks * factor)
    inverse_cdfs = np.stack(
        [np.searchsorted(np.cumsum(row), quantiles, side="left") for row in normalized]
    )
    locations = inverse_cdfs.mean(axis=0)
    left = np.floor(locations).astype(int)
    right = np.minimum(left + 1, n_picks - 1)
    fraction = locations - left
    aggregated = np.zeros(n_picks, dtype=np.float64)
    np.add.at(aggregated, left, (1 - fraction) / locations.size)
    np.add.at(aggregated, right, fraction / locations.size)
    total = aggregated.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Finer-grid barycenter did not produce positive finite mass.")
    return (aggregated / total).astype(np.float32)


def smooth_distribution(distribution: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(distribution, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("distribution must be non-empty, finite, and nonnegative.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")
    smoothed = gaussian_filter1d(values, sigma=sigma, mode="reflect")
    total = smoothed.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Smoothed distribution did not retain positive finite mass.")
    return (smoothed / total).astype(np.float32)


def consensus_variants(
    training_distributions: np.ndarray,
) -> dict[str, np.ndarray]:
    mean_target, barycenter_target = benchmark.mean_and_barycenter(
        training_distributions
    )
    variants = {
        "mean": mean_target,
        "barycenter": barycenter_target,
        "barycenter_fine4": finer_grid_barycenter(training_distributions, factor=4),
    }
    for sigma in GAUSSIAN_SIGMAS:
        variants[f"barycenter_gaussian_sigma_{sigma}"] = smooth_distribution(
            barycenter_target, sigma
        )
        variants[f"mean_gaussian_sigma_{sigma}"] = smooth_distribution(
            mean_target, sigma
        )
    if set(variants) != set(VARIANTS):
        raise AssertionError("Consensus variant set does not match the declared protocol.")
    return variants


def correlation_metrics(prediction: np.ndarray, heldout_scores: np.ndarray) -> tuple[float, float]:
    if np.ptp(prediction) == 0 or np.ptp(heldout_scores) == 0:
        return 0.0, 0.0
    rho = scipy.stats.spearmanr(prediction, heldout_scores).statistic
    tau = scipy.stats.kendalltau(prediction, heldout_scores).statistic
    return (
        0.0 if not np.isfinite(rho) else float(rho),
        0.0 if not np.isfinite(tau) else float(tau),
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("Holm adjustment requires at least one p-value.")
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1].")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[key])
        running_max = max(running_max, candidate)
        adjusted[key] = running_max
    return adjusted


def disagreement_rho(annotation_scores: np.ndarray) -> float:
    pairwise = []
    for first in range(annotation_scores.shape[0]):
        for second in range(first + 1, annotation_scores.shape[0]):
            left = annotation_scores[first]
            right = annotation_scores[second]
            if np.ptp(left) == 0 or np.ptp(right) == 0:
                pairwise.append(0.0)
                continue
            rho = scipy.stats.spearmanr(left, right).statistic
            pairwise.append(0.0 if not np.isfinite(rho) else float(rho))
    if not pairwise:
        raise ValueError("At least two annotators are required to measure disagreement.")
    return float(np.mean(pairwise))


def compute_video_oracle(video: benchmark.TVSumVideo) -> dict[str, Any]:
    distributions = benchmark.annotation_distributions(video)
    scores: dict[str, dict[str, list[float]]] = {
        variant: {
            "f1": [],
            "spearman_rho": [],
            "kendall_tau": [],
            **{f"tolerance_f1_k_{k}": [] for k in TOLERANCE_PICK_RADII},
        }
        for variant in VARIANTS
    }
    for heldout_index in range(distributions.shape[0]):
        training_distributions = np.delete(distributions, heldout_index, axis=0)
        targets = consensus_variants(training_distributions)
        heldout_summary = video.user_summary[heldout_index:heldout_index + 1]
        heldout_scores = video.annotation_scores[heldout_index]
        for variant, target in targets.items():
            predicted_summary = benchmark.canonical_summary(target, video)
            f1 = benchmark.evaluate_summary(predicted_summary, heldout_summary)["f1"]
            scores[variant]["f1"].append(float(f1))
            rho, tau = correlation_metrics(target, heldout_scores)
            scores[variant]["spearman_rho"].append(rho)
            scores[variant]["kendall_tau"].append(tau)
            for tolerance in TOLERANCE_PICK_RADII:
                scores[variant][f"tolerance_f1_k_{tolerance}"].append(
                    tolerance_f1_at_picks(
                        predicted_summary,
                        heldout_summary[0],
                        video.picks,
                        tolerance,
                    )
                )
    return {
        "video_key": video.key,
        "video_id": video.video_id,
        "annotators": int(distributions.shape[0]),
        "frame_count": video.n_frames,
        "pick_count": int(video.picks.size),
        "mean_pairwise_spearman_rho": disagreement_rho(video.annotation_scores),
        "variants": {
            variant: {
                metric: float(np.mean(values))
                for metric, values in metrics.items()
            }
            for variant, metrics in scores.items()
        },
        "heldout_annotator_metrics": scores,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty video group.")
    return {
        "video_count": len(rows),
        "variants": {
            variant: {
                metric: float(np.mean([
                    row["variants"][variant][metric] for row in rows
                ]))
                for metric in (
                    "f1",
                    "spearman_rho",
                    "kendall_tau",
                    *(f"tolerance_f1_k_{k}" for k in TOLERANCE_PICK_RADII),
                )
            }
            for variant in VARIANTS
        },
        "paired_vs_mean": paired_comparisons(rows),
    }


def paired_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    raw_t: dict[str, float] = {}
    raw_wilcoxon: dict[str, float] = {}
    mean_values = [row["variants"]["mean"]["f1"] for row in rows]
    for variant in COMPARISON_VARIANTS:
        variant_values = [row["variants"][variant]["f1"] for row in rows]
        stats = benchmark.paired_statistics(variant_values, mean_values)
        results[variant] = {
            "n": stats["n"],
            "f1_difference_vs_mean": float(np.mean(np.asarray(variant_values) - mean_values)),
            "paired_t_p_raw": stats["paired_t_p"],
            "wilcoxon_p_raw": stats["wilcoxon_p"],
        }
        raw_t[variant] = stats["paired_t_p"]
        raw_wilcoxon[variant] = stats["wilcoxon_p"]
    adjusted_t = holm_adjust(raw_t)
    adjusted_wilcoxon = holm_adjust(raw_wilcoxon)
    for variant in COMPARISON_VARIANTS:
        results[variant]["paired_t_p_holm"] = adjusted_t[variant]
        results[variant]["wilcoxon_p_holm"] = adjusted_wilcoxon[variant]
    return results


def assign_disagreement_tertiles(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (row["mean_pairwise_spearman_rho"], row["video_id"]),
    )
    chunks = np.array_split(np.asarray(ordered, dtype=object), 3)
    names = ("high_disagreement", "middle_disagreement", "low_disagreement")
    return {
        name: [dict(row) for row in chunk.tolist()]
        for name, chunk in zip(names, chunks)
    }


def run_oracle(
    features_path: Path,
    annotations_path: Path,
    info_path: Path,
    output_dir: Path,
    prior_oracle_path: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not prior_oracle_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint 8 oracle is required for reproduction validation: {prior_oracle_path}"
        )
    started = time.time()
    prior_result = json.loads(prior_oracle_path.read_text(encoding="utf-8"))
    videos = benchmark.load_tvsum_features(
        features_path, annotations_path, info_path
    )
    per_video = [compute_video_oracle(video) for video in videos.values()]
    if len(per_video) != 50 or any(row["annotators"] != 20 for row in per_video):
        raise AssertionError("Oracle run must include 50 videos with 20 annotators each.")
    prior_by_id = {row["video_id"]: row for row in prior_result["per_video"]}
    if set(prior_by_id) != {row["video_id"] for row in per_video}:
        raise ValueError("Checkpoint 8 and 8b must contain the same 50 videos.")
    current_hashes = {
        "feature_sha256": benchmark.sha256_file(features_path),
        "annotation_sha256": benchmark.sha256_file(annotations_path),
        "info_sha256": benchmark.sha256_file(info_path),
    }
    if any(
        prior_result["protocol"][key] != value
        for key, value in current_hashes.items()
    ):
        raise ValueError("Checkpoint 8 inputs differ from the Checkpoint 8b inputs.")
    max_reproduction_differences = {"mean": 0.0, "barycenter": 0.0}
    for row in per_video:
        previous = prior_by_id[row["video_id"]]
        for old_key, variant in (
            ("mean_f1", "mean"),
            ("barycenter_f1", "barycenter"),
        ):
            difference = abs(
                previous[old_key] - row["variants"][variant]["f1"]
            )
            max_reproduction_differences[variant] = max(
                max_reproduction_differences[variant], difference
            )
    if any(value > 1e-9 for value in max_reproduction_differences.values()):
        raise AssertionError(
            "Checkpoint 8 mean/barycenter F1 did not reproduce within 1e-9: "
            f"{max_reproduction_differences}"
        )

    tertiles = assign_disagreement_tertiles(per_video)
    tertile_summary = {
        name: summarize_group(rows) for name, rows in tertiles.items()
    }
    result = {
        "protocol": {
            "dataset": "TVSum50",
            "video_count": len(per_video),
            "annotators_per_video": 20,
            "leave_one_out_training_annotators": 19,
            "summary_budget_fraction": benchmark.BUDGET_FRACTION,
            "methods": {
                "mean": "arithmetic mean of the other 19 normalized raw annotation distributions",
                "barycenter": "existing 1D Wasserstein barycenter implementation",
                "barycenter_fine4": (
                    "1D barycenter estimated on a 4x finer uniform quantile grid; "
                    "linearly interpolated mass is aggregated into original pick bins"
                ),
                **{
                    f"{method}_gaussian_sigma_{sigma}": (
                        f"{method} target Gaussian-smoothed over pick positions "
                        f"(sigma={sigma}, mode=reflect), then renormalized"
                    )
                    for method in ("mean", "barycenter")
                    for sigma in GAUSSIAN_SIGMAS
                },
            },
            "evaluation": (
                "canonical_summary with DSNet change_points and 15% knapsack; "
                "F1 against only the held-out annotator user_summary; average over 20 users per video"
            ),
            "tolerance_f1": (
                "At sampled picks, precision is the fraction of predicted selected "
                "picks within +/-k pick indices of any held-out selected pick; recall "
                "is the converse fraction of held-out selected picks; harmonic mean "
                "reported as percent"
            ),
            "correlations": (
                "Spearman rho and Kendall tau between target distribution at picks "
                "and held-out user's raw scores at the same picks; average over users "
                "per video, then across videos"
            ),
            "multiple_comparisons": (
                "Holm step-down adjustment across eight non-mean variants, separately "
                "for paired t and Wilcoxon tests, within each group (overall and each "
                "disagreement tertile)"
            ),
            "disagreement": (
                "mean pairwise Spearman rho among 20 raw annotator score vectors "
                "aligned to HDF5 picks; lower mean rho means greater disagreement"
            ),
            "tertile_assignment": (
                "sort ascending by mean pairwise rho, then video_id for ties; "
                "split into three near-equal groups using numpy.array_split"
            ),
            "feature_sha256": benchmark.sha256_file(features_path),
            "annotation_sha256": benchmark.sha256_file(annotations_path),
            "info_sha256": benchmark.sha256_file(info_path),
            "model_training": False,
        },
        "overall": summarize_group(per_video),
        "tertiles": tertile_summary,
        "tertile_video_ids": {
            name: [row["video_id"] for row in rows]
            for name, rows in tertiles.items()
        },
        "per_video": per_video,
        "reproduction_check": {
            "source": str(prior_oracle_path.relative_to(PROJECT_ROOT)),
            "max_abs_f1_difference_by_variant": max_reproduction_differences,
            "passed": True,
        },
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "results.json"
    temporary_path = output_dir / "results.tmp"
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(json_path)

    table_path = output_dir / "variant_metrics.csv"
    with table_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "group", "video_count", "variant", "f1", "spearman_rho",
            "kendall_tau",
            *(f"tolerance_f1_k_{k}" for k in TOLERANCE_PICK_RADII),
        ])
        for group_name, summary in [
            ("overall", result["overall"]), *tertile_summary.items()
        ]:
            for variant in VARIANTS:
                metrics = summary["variants"][variant]
                writer.writerow([
                    group_name,
                    summary["video_count"],
                    variant,
                    metrics["f1"],
                    metrics["spearman_rho"],
                    metrics["kendall_tau"],
                    *(metrics[f"tolerance_f1_k_{k}"] for k in TOLERANCE_PICK_RADII),
                ])
    paired_table_path = output_dir / "paired_tests.csv"
    with paired_table_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "group", "video_count", "variant", "f1_difference_vs_mean",
            "paired_t_p_raw", "paired_t_p_holm",
            "wilcoxon_p_raw", "wilcoxon_p_holm",
        ])
        for group_name, summary in [
            ("overall", result["overall"]), *tertile_summary.items()
        ]:
            for variant in COMPARISON_VARIANTS:
                comparison = summary["paired_vs_mean"][variant]
                writer.writerow([
                    group_name,
                    summary["video_count"],
                    variant,
                    comparison["f1_difference_vs_mean"],
                    comparison["paired_t_p_raw"],
                    comparison["paired_t_p_holm"],
                    comparison["wilcoxon_p_raw"],
                    comparison["wilcoxon_p_holm"],
                ])
    print(
        f"ORACLE_8B_COMPLETE elapsed_seconds={result['elapsed_seconds']:.3f} "
        f"json={json_path} metrics_csv={table_path} paired_csv={paired_table_path} "
        f"overall={json.dumps(result['overall'], sort_keys=True)} "
        f"tertiles={json.dumps(result['tertiles'], sort_keys=True)}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=benchmark.DEFAULT_FEATURES)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            Path(os.environ[benchmark.ANNOTATION_ENV])
            if os.environ.get(benchmark.ANNOTATION_ENV)
            else None
        ),
    )
    parser.add_argument(
        "--info",
        type=Path,
        default=(
            Path(os.environ[benchmark.INFO_ENV])
            if os.environ.get(benchmark.INFO_ENV)
            else None
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "consensus_oracle_8b",
    )
    parser.add_argument(
        "--prior-oracle",
        type=Path,
        default=PROJECT_ROOT / "results" / "consensus_oracle" / "results.json",
    )
    args = parser.parse_args()
    if args.annotations is None:
        parser.error(f"Provide --annotations or set {benchmark.ANNOTATION_ENV}.")
    if args.info is None:
        parser.error(f"Provide --info or set {benchmark.INFO_ENV}.")
    run_oracle(
        args.features,
        args.annotations,
        args.info,
        args.output_dir,
        args.prior_oracle,
    )


if __name__ == "__main__":
    main()
