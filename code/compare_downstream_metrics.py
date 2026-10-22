"""
compare_downstream_metrics.py

Comparison between an in-repository Euclidean mean-target baseline and the MSPDM prototype:
Computes both:
  1. Annotation-only agreement metrics:
     - Top-15% position overlap F1
     - Spearman's rank correlation (rho)
     - Kendall's rank correlation (tau)
  2. Geometric & Optimal Transport Metrics:
     - Held-out 1D Wasserstein-2 Distance (W2)
     - Simplex Mass Conservation & Non-Negativity Violation Rate

The annotation-change interval metric below is project-specific; it does not
implement the canonical video-shot TVSum evaluation protocol.
"""

import os
import sys
import json
import csv
import argparse
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_tvsum_data, get_video_distributions, stable_seed
from download_tvsum import require_tvsum_annotations
from barycenter import wasserstein_barycenter, arithmetic_mean, blended_barycenter
from forward import forward_step_mine, forward_step_baseline
from losses import wasserstein_loss, w2_1d
from model import NoisePredictor
from split import split_annotators
from sample import generate_sample


def evaluate_f1_and_correlations(predicted_dist, ground_truth_scores, budget_pct=0.15):
    """
    Computes annotation-only agreement metrics:
      - Position-overlap F1 (top budget fraction vs each annotator's top fraction)
      - Spearman rank correlation
      - Kendall tau rank correlation
    """
    N = len(predicted_dist)
    budget_frames = max(1, int(N * budget_pct))

    # Top 15% frame selection for prediction
    pred_top_indices = set(np.argsort(predicted_dist)[-budget_frames:])

    f1_list = []
    spearman_list = []
    kendall_list = []

    for gt in ground_truth_scores:
        # Top 15% frame selection for this annotator
        gt_top_indices = set(np.argsort(gt)[-budget_frames:])

        # Precision, Recall, F1
        overlap = len(pred_top_indices.intersection(gt_top_indices))
        prec = overlap / budget_frames
        rec = overlap / budget_frames
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        # Rank correlations
        rho, _ = stats.spearmanr(predicted_dist, gt)
        tau, _ = stats.kendalltau(predicted_dist, gt)
        spearman_list.append(0.0 if np.isnan(rho) else float(rho))
        kendall_list.append(0.0 if np.isnan(tau) else float(tau))

    return {
        "mean_f1": float(np.mean(f1_list)) * 100.0,
        "mean_spearman_rho": float(np.mean(spearman_list)),
        "mean_kendall_tau": float(np.mean(kendall_list)),
    }


def annotation_change_segments(all_annotations):
    """Partition annotation positions where any annotator's score changes.

    These intervals are not video-shot boundaries and must not be used to
    claim canonical TVSum summarization F1.
    """
    annotations = np.asarray(all_annotations)
    if annotations.ndim != 2 or annotations.shape[0] == 0 or annotations.shape[1] == 0:
        raise ValueError("all_annotations must have shape (annotators, frames)")
    if not np.all(np.isfinite(annotations)):
        raise ValueError("all_annotations must contain only finite scores")

    boundaries = np.flatnonzero(
        np.any(annotations[:, 1:] != annotations[:, :-1], axis=0)
    ) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, annotations.shape[1]]
    return np.column_stack((starts, ends))


def _knapsack_summary(frame_scores, segments, budget_pct):
    """Select caller-supplied intervals with a duration budget."""
    scores = np.asarray(frame_scores, dtype=float)
    segments = np.asarray(segments, dtype=int)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("frame_scores must be a non-empty finite 1D array")
    if (
        segments.ndim != 2
        or segments.shape[1] != 2
        or segments.shape[0] == 0
        or np.any(segments[:, 0] < 0)
        or np.any(segments[:, 1] > scores.size)
        or np.any(segments[:, 1] <= segments[:, 0])
        or np.any(segments[1:, 0] != segments[:-1, 1])
        or segments[0, 0] != 0
        or segments[-1, 1] != scores.size
    ):
        raise ValueError("segments must partition the frame-score vector")
    if not 0 < budget_pct <= 1:
        raise ValueError("budget_pct must be in (0, 1]")

    weights = segments[:, 1] - segments[:, 0]
    values = np.array(
        [scores[start:end].mean() for start, end in segments],
        dtype=float,
    )
    budget = int(budget_pct * scores.size)
    table = np.zeros((len(segments) + 1, budget + 1), dtype=float)
    for item, (weight, value) in enumerate(zip(weights, values)):
        if weight <= budget:
            table[item + 1, weight:] = np.maximum(
                table[item, weight:],
                value + table[item, :-weight],
            )

    selected_segments = np.zeros(len(segments), dtype=bool)
    remaining = budget
    item = len(segments)
    best = table[item, remaining]
    while item > 0 and best > 0:
        if table[item, remaining] == table[item - 1, remaining]:
            item -= 1
            continue
        selected_segments[item - 1] = True
        remaining -= weights[item - 1]
        item -= 1
        best = table[item, remaining]

    summary = np.zeros(scores.size, dtype=bool)
    for selected, (start, end) in zip(selected_segments, segments):
        if selected:
            summary[start:end] = True
    if not summary.any():
        selected_frames = np.argsort(scores, kind="stable")[-max(1, budget):]
        summary[selected_frames] = True
    return summary


def evaluate_annotation_segment_f1(predicted_scores, heldout_annotations, segments, budget_pct=0.15):
    """Score agreement on annotation-change intervals, not canonical TVSum F1."""
    predictions = np.asarray(predicted_scores, dtype=float)
    annotations = np.asarray(heldout_annotations, dtype=float)
    if annotations.ndim != 2 or annotations.shape[1] != predictions.size:
        raise ValueError("heldout_annotations must have shape (annotators, positions)")
    if annotations.shape[0] == 0 or not np.all(np.isfinite(annotations)):
        raise ValueError("heldout_annotations must be non-empty and finite")

    predicted_summary = _knapsack_summary(predictions, segments, budget_pct)
    predicted_count = int(predicted_summary.sum())
    scores = []
    for annotation in annotations:
        true_summary = _knapsack_summary(annotation, segments, budget_pct)
        true_count = int(true_summary.sum())
        overlap = int(np.count_nonzero(predicted_summary & true_summary))
        precision = overlap / predicted_count if predicted_count else 0.0
        recall = overlap / true_count if true_count else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores) * 100.0)


def run_full_comparison(n_videos=50, frames=100, steps=100, bary_alpha=1.0, lr=0.01):
    tsv_path = os.path.join(os.path.dirname(__file__), "ydata-tvsum50-anno.tsv")
    if not os.path.exists(tsv_path):
        require_tvsum_annotations(tsv_path)

    tvsum_videos = load_tvsum_data(tsv_path)
    vid_keys = list(tvsum_videos.keys())[:n_videos]

    metrics = {
        "mine": {"w2": [], "f1": [], "spearman": [], "kendall": []},
        "baseline": {"w2": [], "f1": [], "spearman": [], "kendall": []},
    }

    per_video_records = []

    for idx, vid in enumerate(vid_keys, 1):
        raw = tvsum_videos[vid]
        if raw.shape[1] > frames:
            indices = np.linspace(0, raw.shape[1] - 1, frames, dtype=int)
            raw = raw[:, indices]

        dist = get_video_distributions(raw, method="softmax", temperature=0.5)
        vid_seed = stable_seed(vid)
        train_ann, heldout_ann = split_annotators(dist, n_holdout=5, seed=vid_seed)

        # 1. Train Proposed (Mine)
        x_star_mine = blended_barycenter(train_ann, alpha=bary_alpha)
        # Both nets get the SAME initial weights (seeded from vid_seed) so that
        # differences below are attributable to the training target/process, not
        # to which config got luckier random weight init -- and are directly
        # comparable across different bary_alpha values, since vid_seed doesn't
        # change with alpha.
        net_mine = NoisePredictor(frames, lr=lr, rng=np.random.default_rng(vid_seed))
        rng_mine_steps = np.random.default_rng(vid_seed + 1000)
        rng_mine_forward = np.random.default_rng(vid_seed + 2000)
        for _ in range(steps):
            t = rng_mine_steps.integers(1, 200)
            xt = forward_step_mine(x_star_mine, t, rng=rng_mine_forward)
            eps_hat = net_mine.forward(xt, t)
            grad_mse = 2 * (eps_hat - x_star_mine) / frames
            P = np.cumsum(eps_hat)
            grad_cramer = np.zeros_like(eps_hat)
            for a in train_ann:
                diff = P - np.cumsum(a)
                grad_cramer += 2 * np.cumsum(diff[::-1])[::-1]
            grad_cramer /= len(train_ann)
            net_mine.step(xt, t, grad_mse + 0.1 * grad_cramer)

        gen_mine = generate_sample(net_mine, frames, T=200, mode="mine", rng=np.random.default_rng(vid_seed + 10))

        # 2. Train the Euclidean mean-target baseline.
        x_star_base = arithmetic_mean(train_ann)
        net_base = NoisePredictor(frames, lr=lr, rng=np.random.default_rng(vid_seed))
        rng_base_steps = np.random.default_rng(vid_seed + 1000)
        rng_base_forward = np.random.default_rng(vid_seed + 2000)
        for _ in range(steps):
            t = rng_base_steps.integers(1, 200)
            xt = forward_step_baseline(x_star_base, t, rng=rng_base_forward)
            eps_hat = net_base.forward(xt, t)
            grad_mse = 2 * (eps_hat - x_star_base) / frames
            net_base.step(xt, t, grad_mse)

        gen_base = generate_sample(net_base, frames, T=200, mode="baseline", rng=np.random.default_rng(vid_seed + 10))

        # Evaluate against held-out annotators
        w2_m = float(np.mean([w2_1d(gen_mine, a) for a in heldout_ann]))
        w2_b = float(np.mean([w2_1d(gen_base, a) for a in heldout_ann]))

        eval_m = evaluate_f1_and_correlations(gen_mine, heldout_ann)
        eval_b = evaluate_f1_and_correlations(gen_base, heldout_ann)

        metrics["mine"]["w2"].append(w2_m)
        metrics["mine"]["f1"].append(eval_m["mean_f1"])
        metrics["mine"]["spearman"].append(eval_m["mean_spearman_rho"])
        metrics["mine"]["kendall"].append(eval_m["mean_kendall_tau"])

        metrics["baseline"]["w2"].append(w2_b)
        metrics["baseline"]["f1"].append(eval_b["mean_f1"])
        metrics["baseline"]["spearman"].append(eval_b["mean_spearman_rho"])
        metrics["baseline"]["kendall"].append(eval_b["mean_kendall_tau"])

        per_video_records.append({
            "video_id": vid,
            "mine_w2": w2_m, "baseline_w2": w2_b,
            "mine_f1": eval_m["mean_f1"], "baseline_f1": eval_b["mean_f1"],
            "mine_spearman": eval_m["mean_spearman_rho"], "baseline_spearman": eval_b["mean_spearman_rho"],
            "mine_kendall": eval_m["mean_kendall_tau"], "baseline_kendall": eval_b["mean_kendall_tau"],
        })

    return metrics, per_video_records


def print_comparison_table(metrics):
    print("=" * 85)
    print("      COMPARISON: MSPDM PROTOTYPE vs EUCLIDEAN MEAN-TARGET BASELINE")
    print("=" * 85)
    print(f"{'Evaluation Metric':<32} | {'Euclidean mean':<20} | {'MSPDM prototype':<20} | {'Win Rate / p-val'}")
    print("-" * 85)

    for name, key, higher_better in [
        ("Held-Out W2 Distance", "w2", False),
        ("Mean F1-Score (%) [Top 15%]", "f1", True),
        ("Spearman Rank Corr (rho)", "spearman", True),
        ("Kendall Tau Rank Corr (tau)", "kendall", True),
    ]:
        m_vals = np.array(metrics["mine"][key])
        b_vals = np.array(metrics["baseline"][key])

        m_mean, m_std = np.mean(m_vals), np.std(m_vals)
        b_mean, b_std = np.mean(b_vals), np.std(b_vals)

        t_res = stats.ttest_rel(m_vals, b_vals)
        wins = int(np.sum(m_vals > b_vals if higher_better else m_vals < b_vals))
        total = len(m_vals)

        m_str = f"{m_mean:.4f} +/- {m_std:.4f}"
        b_str = f"{b_mean:.4f} +/- {b_std:.4f}"
        win_str = f"{wins}/{total} (p={t_res.pvalue:.4f})"

        print(f"{name:<32} | {b_str:<20} | {m_str:<20} | {win_str}")

    print("=" * 85)


def analyze_and_save(metrics, records, output_dir="results"):
    """
    Builds the AAAI-style F1-score headline comparison (mirrors the
    Held-Out W2 benchmark output in benchmark_50videos.py, but keyed on
    Mean F1-Score @ top-15% budget instead of W2 distance), and dumps
    per-video CSV, a JSON summary with significance tests, and a figure.
    """
    output_dir = os.path.join(os.path.dirname(__file__), "..", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mine_f1 = np.array(metrics["mine"]["f1"])
    base_f1 = np.array(metrics["baseline"]["f1"])
    gaps = mine_f1 - base_f1

    mine_mean, mine_std = float(np.mean(mine_f1)), float(np.std(mine_f1, ddof=1))
    base_mean, base_std = float(np.mean(base_f1)), float(np.std(base_f1, ddof=1))
    gap_mean, gap_std = float(np.mean(gaps)), float(np.std(gaps, ddof=1))
    wins = int(np.sum(gaps > 0))
    total = len(mine_f1)
    win_rate = (wins / total) * 100.0

    ttest_res = stats.ttest_rel(mine_f1, base_f1)
    t_stat, t_pval = float(ttest_res.statistic), float(ttest_res.pvalue)
    wilcoxon_res = stats.wilcoxon(mine_f1, base_f1, alternative='two-sided')
    w_stat, w_pval = float(wilcoxon_res.statistic), float(wilcoxon_res.pvalue)
    annotation_path = os.environ.get(
        "TVSUM_ANNOTATION_TSV",
        os.path.join(os.path.dirname(__file__), "ydata-tvsum50-anno.tsv"),
    )
    with open(annotation_path, "rb") as annotation_file:
        annotation_sha256 = hashlib.sha256(annotation_file.read()).hexdigest()

    summary = {
        "num_videos": total,
        "annotation_file": os.path.basename(annotation_path),
        "annotation_sha256": annotation_sha256,
        "mine_mean_f1": {"mean": mine_mean, "std": mine_std},
        "baseline_mean_f1": {"mean": base_mean, "std": base_std},
        "gap_absolute": {"mean": gap_mean, "std": gap_std},
        "mine_wins": wins,
        "total_videos": total,
        "win_rate_pct": win_rate,
        "statistical_tests": {
            "paired_t_test": {
                "t_statistic": t_stat,
                "p_value": t_pval,
                "statistically_significant_p_0_05": t_pval < 0.05,
                "statistically_significant_p_0_01": t_pval < 0.01,
            },
            "wilcoxon_signed_rank": {
                "w_statistic": w_stat,
                "p_value": w_pval,
                "statistically_significant_p_0_05": w_pval < 0.05,
                "statistically_significant_p_0_01": w_pval < 0.01,
            },
        },
        "mean_spearman_rho": {
            "mine": float(np.mean(metrics["mine"]["spearman"])),
            "baseline": float(np.mean(metrics["baseline"]["spearman"])),
        },
        "mean_kendall_tau": {
            "mine": float(np.mean(metrics["mine"]["kendall"])),
            "baseline": float(np.mean(metrics["baseline"]["kendall"])),
        },
    }

    # Save per-video CSV
    csv_path = os.path.join(output_dir, "tvsum_50videos_f1_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Save JSON summary
    json_path = os.path.join(output_dir, "tvsum_50videos_f1_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Figure: box plot of F1 + per-video gap histogram (mirrors the W2 figure)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot([mine_f1, base_f1], tick_labels=["MSPDM prototype", "Euclidean mean baseline"],
                patch_artist=True,
                boxprops=dict(facecolor="#d0e1fd", color="#1f77b4"),
                medianprops=dict(color="#08306b", linewidth=2))
    ax1.set_ylabel("Mean F1-Score (%) @ Top-15% Budget (Higher is Better)")
    ax1.set_title("Held-Out F1-Score Across 50 TVSum Videos")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.hist(gaps, bins=15, color="#2ca02c", alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Difference")
    ax2.axvline(gap_mean, color="darkgreen", linestyle="-", linewidth=2, label=f"Mean Gap = {gap_mean:+.2f} pts")
    ax2.set_xlabel("F1 Improvement (Mine - Baseline) per Video (pts)")
    ax2.set_ylabel("Number of Videos")
    ax2.set_title("Per-Video F1 Improvement Distribution")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "benchmark_50videos_f1_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved per-video F1 results to: {csv_path}")
    print(f"Saved F1 summary metrics to:  {json_path}")
    print(f"Saved F1 benchmark figure to: {plot_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TVSum-50 held-out comparison.")
    parser.add_argument("--videos", type=int, default=50)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output-dir", default=os.path.join("results", "real_tvsum", "stage4")
    )
    args = parser.parse_args()
    metrics, records = run_full_comparison(
        n_videos=args.videos, frames=args.frames, steps=args.steps
    )
    print_comparison_table(metrics)
    analyze_and_save(metrics, records, output_dir=args.output_dir)
