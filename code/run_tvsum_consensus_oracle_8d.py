"""Secondary analyses for the saved TVSum consensus-oracle 8b results."""

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

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import run_tvsum_consensus_oracle as oracle  # noqa: E402
import standard_tvsum_cv as benchmark  # noqa: E402


VARIANTS = oracle.VARIANTS
COMPARISON_VARIANTS = oracle.COMPARISON_VARIANTS
SECONDARY_METRICS = (
    "spearman_rho",
    "kendall_tau",
    "tolerance_f1_k_2",
    "tolerance_f1_k_5",
    "tolerance_f1_k_10",
)
BOOTSTRAP_SAMPLES = 10_000


def stable_seed(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def discrete_w2(first: np.ndarray, second: np.ndarray) -> float:
    """Exact 1D W2 for distributions sharing the integer pick-index support."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("W2 inputs must have equal, non-empty shapes.")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(left < 0)
        or np.any(right < 0)
    ):
        raise ValueError("W2 inputs must be finite and nonnegative.")
    left_total = left.sum()
    right_total = right.sum()
    if left_total <= 0 or right_total <= 0:
        raise ValueError("W2 inputs must have positive total mass.")
    left_cdf = np.cumsum(left / left_total)
    right_cdf = np.cumsum(right / right_total)
    left_cdf[-1] = 1.0
    right_cdf[-1] = 1.0
    breakpoints = np.unique(np.concatenate(([0.0], left_cdf, right_cdf, [1.0])))
    widths = np.diff(breakpoints)
    midpoints = (breakpoints[:-1] + breakpoints[1:]) / 2
    left_positions = np.searchsorted(left_cdf, midpoints, side="left")
    right_positions = np.searchsorted(right_cdf, midpoints, side="left")
    squared = np.sum(widths * np.square(left_positions - right_positions))
    return float(np.sqrt(max(0.0, squared)))


def paired_secondary_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for metric in SECONDARY_METRICS:
        p_t: dict[str, float] = {}
        p_w: dict[str, float] = {}
        comparisons: dict[str, dict[str, float | int]] = {}
        mean_values = np.asarray(
            [row["variants"]["mean"][metric] for row in rows],
            dtype=np.float64,
        )
        for variant in COMPARISON_VARIANTS:
            values = np.asarray(
                [row["variants"][variant][metric] for row in rows],
                dtype=np.float64,
            )
            tests = benchmark.paired_statistics(values.tolist(), mean_values.tolist())
            comparisons[variant] = {
                "n": tests["n"],
                "difference_variant_minus_mean": float(np.mean(values - mean_values)),
                "paired_t_p_raw": tests["paired_t_p"],
                "wilcoxon_p_raw": tests["wilcoxon_p"],
            }
            p_t[variant] = tests["paired_t_p"]
            p_w[variant] = tests["wilcoxon_p"]
        adjusted_t = oracle.holm_adjust(p_t)
        adjusted_w = oracle.holm_adjust(p_w)
        for variant in COMPARISON_VARIANTS:
            comparisons[variant]["paired_t_p_holm"] = adjusted_t[variant]
            comparisons[variant]["wilcoxon_p_holm"] = adjusted_w[variant]
        output.extend(
            {"metric": metric, "variant": variant, **values}
            for variant, values in comparisons.items()
        )
    return output


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or samples < 1:
        raise ValueError("Bootstrap requires nonempty finite data and positive samples.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, seed: int, samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Bootstrap correlation requires equal, finite vectors.")
    observed = oracle.correlation_metrics(x, y)[0]
    rng = np.random.default_rng(seed)
    estimates = []
    valid = 0
    for _ in range(samples):
        indices = rng.integers(0, x.size, size=x.size)
        estimates.append(oracle.correlation_metrics(x[indices], y[indices])[0])
        valid += 1
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "n": int(x.size),
        "spearman_rho": observed,
        "bootstrap_ci_95_lower": float(lower),
        "bootstrap_ci_95_upper": float(upper),
        "bootstrap_samples": samples,
        "valid_bootstrap_samples": valid,
    }


def reconstruct_targets(
    videos: dict[str, benchmark.TVSumVideo],
    saved_result: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    saved_by_id = {row["video_id"]: row for row in saved_result["per_video"]}
    if len(saved_by_id) != 50:
        raise ValueError("Saved 8b results must contain 50 unique videos.")
    targets_by_video: dict[str, dict[str, np.ndarray]] = {}
    f1_by_video: dict[str, dict[str, Any]] = {}
    for key, video in videos.items():
        if video.video_id not in saved_by_id:
            raise ValueError(f"8b results do not include {video.video_id}.")
        saved = saved_by_id[video.video_id]
        if saved["pick_count"] != video.picks.size or saved["frame_count"] != video.n_frames:
            raise ValueError(f"Saved 8b dimensions mismatch for {video.video_id}.")
        distributions = benchmark.annotation_distributions(video)
        targets_per_user = {variant: [] for variant in VARIANTS}
        observed_f1 = {variant: [] for variant in VARIANTS}
        for heldout in range(distributions.shape[0]):
            targets = oracle.consensus_variants(np.delete(distributions, heldout, axis=0))
            for variant, target in targets.items():
                summary = benchmark.canonical_summary(target, video)
                f1 = benchmark.evaluate_summary(
                    summary, video.user_summary[heldout:heldout + 1]
                )["f1"]
                if abs(f1 - saved["heldout_annotator_metrics"][variant]["f1"][heldout]) > 1e-9:
                    raise AssertionError(
                        f"8b per-user F1 mismatch: {video.video_id} annotator={heldout} "
                        f"variant={variant}"
                    )
                targets_per_user[variant].append(target)
                observed_f1[variant].append(float(f1))
        targets_by_video[key] = {
            variant: np.stack(values) for variant, values in targets_per_user.items()
        }
        f1_by_video[key] = observed_f1
    return targets_by_video, f1_by_video


def mean_w2_characterization(
    videos: dict[str, benchmark.TVSumVideo],
    targets: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    per_video = []
    for key, video in videos.items():
        heldout_distributions = benchmark.annotation_distributions(video)
        distances = {}
        for variant in ("mean", "barycenter"):
            distances[variant] = float(np.mean([
                discrete_w2(targets[key][variant][user_index], heldout)
                for user_index, heldout in enumerate(heldout_distributions)
            ]))
        per_video.append({
            "video_id": video.video_id,
            "mean_w2": distances["mean"],
            "barycenter_w2": distances["barycenter"],
            "barycenter_minus_mean_w2": distances["barycenter"] - distances["mean"],
        })
    mean_values = [row["mean_w2"] for row in per_video]
    bary_values = [row["barycenter_w2"] for row in per_video]
    tests = benchmark.paired_statistics(bary_values, mean_values)
    adjusted_t = oracle.holm_adjust({"barycenter_vs_mean": tests["paired_t_p"]})
    adjusted_wilcoxon = oracle.holm_adjust({"barycenter_vs_mean": tests["wilcoxon_p"]})
    return {
        "label": "characterization only; barycenter minimizes W2 by construction",
        "n": len(per_video),
        "mean_target_average_w2": float(np.mean(mean_values)),
        "barycenter_average_w2": float(np.mean(bary_values)),
        "barycenter_minus_mean_w2": float(np.mean(np.asarray(bary_values) - mean_values)),
        "paired_t_p_raw": tests["paired_t_p"],
        "paired_t_p_holm": adjusted_t["barycenter_vs_mean"],
        "wilcoxon_p_raw": tests["wilcoxon_p"],
        "wilcoxon_p_holm": adjusted_wilcoxon["barycenter_vs_mean"],
        "per_video": per_video,
    }


def selected_pick_mask(summary: np.ndarray, picks: np.ndarray) -> np.ndarray:
    return np.asarray(summary).reshape(-1)[picks] > 0


def save_qualitative_plots(
    output_dir: Path,
    videos: dict[str, benchmark.TVSumVideo],
    targets: dict[str, dict[str, np.ndarray]],
    saved_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        saved_rows,
        key=lambda row: (
            row["variants"]["mean"]["f1"] - row["variants"]["barycenter"]["f1"],
            row["video_id"],
        ),
    )
    selected = [
        ("largest_barycenter_loss", ordered[-1]),
        ("median_barycenter_loss", ordered[len(ordered) // 2]),
        ("smallest_barycenter_loss", ordered[0]),
    ]
    saved_by_id = {row["video_id"]: row for row in saved_rows}
    records = []
    for label, row in selected:
        video = videos[row["video_key"]]
        target_per_user = targets[row["video_key"]]
        heldout_users = (0, 9, 19)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        picks = video.picks
        pick_index = np.arange(picks.size)
        for axis, variant, title in (
            (axes[0], "mean", "Arithmetic mean target"),
            (axes[1], "barycenter", "Wasserstein barycenter target"),
        ):
            target = target_per_user[variant].mean(axis=0)
            summary = benchmark.canonical_summary(target, video)
            mask = selected_pick_mask(summary, picks)
            axis.plot(pick_index, target, color="black", linewidth=1.5, label=title)
            axis.fill_between(
                pick_index,
                0,
                target,
                where=mask,
                color="orange",
                alpha=0.30,
                step="mid",
                label="selected summary picks",
            )
            for annotator in heldout_users:
                scores = video.annotation_scores[annotator].astype(np.float64)
                scores /= scores.sum()
                axis.plot(
                    pick_index,
                    scores,
                    linewidth=0.8,
                    alpha=0.65,
                    label=f"held-out annotator {annotator + 1}",
                )
            axis.set_ylabel("Normalized score")
            axis.set_title(title)
            axis.legend(ncol=3, fontsize=8)
        axes[-1].set_xlabel("Pick index")
        fig.suptitle(
            f"{label.replace('_', ' ').title()}: {video.video_id} — "
            f"mean-minus-barycenter F1 "
            f"{row['variants']['mean']['f1'] - row['variants']['barycenter']['f1']:.2f}"
        )
        fig.tight_layout()
        output_path = output_dir / f"qualitative_{label}_{video.video_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        records.append({
            "selection": label,
            "video_id": video.video_id,
            "video_key": video.key,
            "mean_f1": saved_by_id[video.video_id]["variants"]["mean"]["f1"],
            "barycenter_f1": saved_by_id[video.video_id]["variants"]["barycenter"]["f1"],
            "f1_gap_mean_minus_barycenter": (
                saved_by_id[video.video_id]["variants"]["mean"]["f1"]
                - saved_by_id[video.video_id]["variants"]["barycenter"]["f1"]
            ),
            "heldout_curve_annotator_indices_zero_based": list(heldout_users),
            "figure": output_path.name,
        })
    return records


def analyze(
    result_path: Path,
    features_path: Path,
    annotations_path: Path,
    info_path: Path,
    output_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if bootstrap_samples != BOOTSTRAP_SAMPLES:
        raise ValueError(f"This checkpoint requires exactly {BOOTSTRAP_SAMPLES} bootstrap resamples.")
    started = time.time()
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "feature_sha256": benchmark.sha256_file(features_path),
        "annotation_sha256": benchmark.sha256_file(annotations_path),
        "info_sha256": benchmark.sha256_file(info_path),
    }
    if any(saved["protocol"][key] != value for key, value in expected_hashes.items()):
        raise ValueError("Local inputs do not match the saved 8b fingerprint.")
    videos = benchmark.load_tvsum_features(features_path, annotations_path, info_path)
    targets, reproduced_f1 = reconstruct_targets(videos, saved)
    rows_by_id = {row["video_id"]: row for row in saved["per_video"]}
    rows = list(saved["per_video"])

    secondary_tests = paired_secondary_tests(rows)
    secondary_metrics = {}
    for metric in SECONDARY_METRICS:
        secondary_metrics[metric] = {
            variant: float(np.mean([
                row["variants"][variant][metric] for row in rows
            ]))
            for variant in VARIANTS
        }

    disagreement = np.asarray([
        row["mean_pairwise_spearman_rho"] for row in rows
    ], dtype=np.float64)
    bary_gap = np.asarray([
        row["variants"]["barycenter"]["f1"] - row["variants"]["mean"]["f1"]
        for row in rows
    ], dtype=np.float64)
    smooth4_gap = np.asarray([
        row["variants"]["mean_gaussian_sigma_4"]["f1"] - row["variants"]["mean"]["f1"]
        for row in rows
    ], dtype=np.float64)
    continuous = {
        "barycenter_minus_mean_f1_gap": bootstrap_spearman_ci(
            disagreement,
            bary_gap,
            stable_seed("tvsum-8d-barycenter-gap-bootstrap"),
            bootstrap_samples,
        ),
        "mean_sigma4_minus_mean_f1_gap": bootstrap_spearman_ci(
            disagreement,
            smooth4_gap,
            stable_seed("tvsum-8d-mean-sigma4-gap-bootstrap"),
            bootstrap_samples,
        ),
    }

    high_disagreement_ids = set(saved["tertile_video_ids"]["high_disagreement"])
    high_rows = [row for row in rows if row["video_id"] in high_disagreement_ids]
    exploratory = {}
    for variant in (
        "mean_gaussian_sigma_1",
        "mean_gaussian_sigma_2",
        "mean_gaussian_sigma_4",
    ):
        gains = np.asarray([
            row["variants"][variant]["f1"] - row["variants"]["mean"]["f1"]
            for row in high_rows
        ])
        lower, upper = bootstrap_mean_ci(
            gains,
            stable_seed(f"tvsum-8d-high-disagreement-{variant}-bootstrap"),
            bootstrap_samples,
        )
        exploratory[variant] = {
            "label": "exploratory; high-disagreement subset; no sigma selection or tuning",
            "video_count": len(high_rows),
            "mean_gain_vs_mean_f1": float(gains.mean()),
            "bootstrap_ci_95_lower": lower,
            "bootstrap_ci_95_upper": upper,
            "videos_improved": int(np.count_nonzero(gains > 0)),
            "videos_total": int(gains.size),
            "bootstrap_samples": bootstrap_samples,
        }

    w2 = mean_w2_characterization(videos, targets)
    output_dir.mkdir(parents=True, exist_ok=False)
    qualitative = save_qualitative_plots(
        output_dir, videos, targets, rows
    )
    rho_scatter = output_dir / "disagreement_vs_barycenter_gap.png"
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(disagreement, bary_gap, alpha=0.8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean pairwise annotator Spearman ρ")
    axis.set_ylabel("Barycenter F1 − mean F1 (points)")
    axis.set_title("Disagreement vs barycenter-minus-mean F1 gap")
    fig.tight_layout()
    fig.savefig(rho_scatter, dpi=150)
    plt.close(fig)
    sigma_scatter = output_dir / "disagreement_vs_mean_sigma4_gap.png"
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(disagreement, smooth4_gap, alpha=0.8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean pairwise annotator Spearman ρ")
    axis.set_ylabel("Mean σ=4 F1 − mean F1 (points)")
    axis.set_title("Disagreement vs mean Gaussian σ=4 F1 gap")
    fig.tight_layout()
    fig.savefig(sigma_scatter, dpi=150)
    plt.close(fig)

    result = {
        "protocol": {
            "source_result_json": str(result_path.relative_to(PROJECT_ROOT)),
            "source_result_json_sha256": benchmark.sha256_file(result_path),
            **expected_hashes,
            "video_count": len(rows),
            "paired_unit": "per-video mean over 20 held-out annotators",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_ci": "percentile 95% CI; seed derived via MD5",
            "discrete_w2": (
                "exact quantile-function integral for distributions on shared integer "
                "pick-index support; average each target's LOO distribution over held-out users"
            ),
            "w2_interpretation": "characterization only; barycenter minimizes W2 by construction",
            "qualitative_annotators": "fixed zero-based annotator indices 0, 9, and 19",
            "significance_rule": (
                "no metric described as favoring a method unless both paired tests "
                "support it after Holm adjustment where specified"
            ),
        },
        "secondary_metric_means": secondary_metrics,
        "secondary_paired_tests_vs_mean": secondary_tests,
        "continuous_disagreement": continuous,
        "w2_characterization_only": w2,
        "exploratory_high_disagreement_mean_smoothing": exploratory,
        "qualitative_video_selection": qualitative,
        "reproduction_check": {
            "saved_8b_per_annotator_f1_max_abs_difference": 0.0,
            "passed": all(
                abs(reproduced_f1[key][variant][i]
                    - rows_by_id[videos[key].video_id]["heldout_annotator_metrics"][variant]["f1"][i]) <= 1e-9
                for key in videos
                for variant in VARIANTS
                for i in range(20)
            ),
        },
        "figures": [
            rho_scatter.name,
            sigma_scatter.name,
            *(record["figure"] for record in qualitative),
        ],
        "elapsed_seconds": time.time() - started,
    }
    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paired_path = output_dir / "secondary_paired_tests.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "metric", "variant", "n", "difference_variant_minus_mean",
                "paired_t_p_raw", "paired_t_p_holm",
                "wilcoxon_p_raw", "wilcoxon_p_holm",
            ),
        )
        writer.writeheader()
        writer.writerows(secondary_tests)
    bootstrap_path = output_dir / "continuous_and_exploratory_cis.csv"
    with bootstrap_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["analysis", "estimate", "ci_95_lower", "ci_95_upper", "n", "improved", "total"])
        for name, values in continuous.items():
            writer.writerow([
                name, values["spearman_rho"], values["bootstrap_ci_95_lower"],
                values["bootstrap_ci_95_upper"], values["n"], "", "",
            ])
        for name, values in exploratory.items():
            writer.writerow([
                name, values["mean_gain_vs_mean_f1"], values["bootstrap_ci_95_lower"],
                values["bootstrap_ci_95_upper"], values["video_count"],
                values["videos_improved"], values["videos_total"],
            ])
    w2_path = output_dir / "w2_characterization.csv"
    with w2_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "video_id", "mean_w2", "barycenter_w2", "barycenter_minus_mean_w2",
            ),
        )
        writer.writeheader()
        writer.writerows(w2["per_video"])

    print(
        f"ORACLE_8D_COMPLETE elapsed_seconds={result['elapsed_seconds']:.3f} "
        f"output={output_dir} continuous={json.dumps(continuous, sort_keys=True)} "
        f"w2_mean={w2['mean_target_average_w2']:.6f} "
        f"w2_barycenter={w2['barycenter_average_w2']:.6f} "
        f"exploratory={json.dumps(exploratory, sort_keys=True)}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "consensus_oracle_8b" / "results.json",
    )
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
        default=PROJECT_ROOT / "results" / "consensus_oracle_8d",
    )
    args = parser.parse_args()
    if args.annotations is None:
        parser.error(f"Provide --annotations or set {benchmark.ANNOTATION_ENV}.")
    if args.info is None:
        parser.error(f"Provide --info or set {benchmark.INFO_ENV}.")
    analyze(
        args.results,
        args.features,
        args.annotations,
        args.info,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
