"""
stages/stage1_consensus.py

Stage 1 — Mathematical Consensus & Optimal Transport
=====================================================

Oracle comparison: Wasserstein-2 barycenter vs arithmetic mean, evaluated
on 50 synthetic videos against held-out annotators.

Replicates the key finding from the paper:
  • Barycenter achieves lower held-out W2 transport cost on all 50 videos.
  • ~16% average W2 error reduction.
  • Statistically significant: p < 1e-20.

Outputs (written to synthetics/results/)
-----------------------------------------
  stage1_oracle_comparison.csv   — per-video W2 distances
  stage1_oracle_summary.json     — aggregate statistics & significance tests
  stage1_oracle_comparison.png   — box-plot + improvement histogram
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from data.synthetic_dataset import generate_synthetic_tvsum, get_video_distributions, stable_seed
from core.barycenter import wasserstein_barycenter, arithmetic_mean
from core.losses import w2_1d
from core.split import split_annotators


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_stage1(n_videos=50, n_holdout=5, verbose=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos = generate_synthetic_tvsum(n_videos=n_videos)
    vid_keys = list(videos.keys())

    records = []
    bary_w2_list = []
    mean_w2_list = []

    for vid in vid_keys:
        raw = videos[vid]  # (20, 100) int array
        dist = get_video_distributions(raw, method="softmax", temperature=0.5)
        vid_seed = stable_seed(vid)
        train_ann, heldout_ann = split_annotators(dist, n_holdout=n_holdout, seed=vid_seed)

        bary = wasserstein_barycenter(train_ann)
        mean = arithmetic_mean(train_ann)

        bary_w2 = float(np.mean([w2_1d(bary, a) for a in heldout_ann]))
        mean_w2 = float(np.mean([w2_1d(mean, a) for a in heldout_ann]))

        gap = mean_w2 - bary_w2
        gap_pct = (gap / mean_w2 * 100.0) if mean_w2 > 0 else 0.0

        records.append({
            "video_id": vid,
            "barycenter_w2": bary_w2,
            "mean_w2": mean_w2,
            "gap_absolute": gap,
            "gap_pct": gap_pct,
            "bary_wins": gap > 0,
        })
        bary_w2_list.append(bary_w2)
        mean_w2_list.append(mean_w2)

    bary_arr = np.array(bary_w2_list)
    mean_arr = np.array(mean_w2_list)
    gaps = mean_arr - bary_arr
    wins = int((gaps > 0).sum())
    total = len(gaps)

    t_stat, t_pval = stats.ttest_rel(bary_arr, mean_arr)
    w_stat, w_pval = stats.wilcoxon(bary_arr, mean_arr, alternative="two-sided")

    summary = {
        "num_videos": total,
        "barycenter_w2": {"mean": float(bary_arr.mean()), "std": float(bary_arr.std(ddof=1))},
        "mean_w2":       {"mean": float(mean_arr.mean()), "std": float(mean_arr.std(ddof=1))},
        "gap_absolute":  {"mean": float(gaps.mean()),     "std": float(gaps.std(ddof=1))},
        "avg_improvement_pct": float((gaps / np.where(mean_arr > 0, mean_arr, 1.0) * 100.0).mean()),
        "barycenter_wins": wins,
        "win_rate_pct": wins / total * 100.0,
        "statistical_tests": {
            "paired_t_test": {"t_statistic": float(t_stat), "p_value": float(t_pval)},
            "wilcoxon": {"w_statistic": float(w_stat), "p_value": float(w_pval)},
        },
    }

    if verbose:
        print("\n" + "=" * 65)
        print("  STAGE 1 — ORACLE W2 BARYCENTER vs ARITHMETIC MEAN (50 VIDEOS)")
        print("=" * 65)
        print(f"  Barycenter W2:   {summary['barycenter_w2']['mean']:.4f} +/- {summary['barycenter_w2']['std']:.4f}")
        print(f"  Arithmetic Mean: {summary['mean_w2']['mean']:.4f} +/- {summary['mean_w2']['std']:.4f}")
        print(f"  Gap ((lower-better)):         {summary['gap_absolute']['mean']:.4f} ({summary['avg_improvement_pct']:+.2f}%)")
        print(f"  Win rate:        {wins}/{total} ({summary['win_rate_pct']:.1f}%)")
        print(f"  Paired t-test:   p = {t_pval:.2e}")
        print(f"  Wilcoxon:        p = {w_pval:.2e}")
        print("=" * 65)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "stage1_oracle_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "stage1_oracle_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot([bary_arr, mean_arr], tick_labels=["W2 Barycenter (MSPDM)", "Arithmetic Mean (Baseline)"],
                patch_artist=True,
                boxprops=dict(facecolor="#d0e1fd", color="#1f77b4"),
                medianprops=dict(color="#08306b", linewidth=2))
    ax1.set_ylabel("Held-Out W2 Distance (Lower is Better)")
    ax1.set_title("Stage 1: Oracle W2 Comparison (50 Synthetic Videos)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.hist(gaps, bins=15, color="#2ca02c", alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero difference")
    ax2.axvline(gaps.mean(), color="darkgreen", linestyle="-", linewidth=2,
                label=f"Mean gap = {gaps.mean():.4f}")
    ax2.set_xlabel("W2 Reduction (Mean − Barycenter) per Video")
    ax2.set_ylabel("Number of Videos")
    ax2.set_title("Per-Video W2 Improvement Distribution")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "stage1_oracle_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\n  Outputs written to {RESULTS_DIR}/")
    print(f"    {os.path.basename(csv_path)}")
    print(f"    {os.path.basename(json_path)}")
    print(f"    {os.path.basename(fig_path)}")

    return summary, records


if __name__ == "__main__":
    run_stage1()
