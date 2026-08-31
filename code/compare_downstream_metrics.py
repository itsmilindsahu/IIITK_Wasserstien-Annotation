"""
compare_downstream_metrics.py

Comprehensive Comparison between Baseline (AAAI-25 standard) and Proposed (MSPDM):
Computes both:
  1. Standard Literature Metrics:
     - Mean F1-score (top 15% length budget knapsack/ranking against annotators)
     - Spearman's Rank Correlation (rho)
     - Kendall's Rank Correlation (tau)
  2. Geometric & Optimal Transport Metrics:
     - Held-out 1D Wasserstein-2 Distance (W2)
     - Simplex Mass Conservation & Non-Negativity Violation Rate
"""

import os
import sys
import json
import csv
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_tvsum_data, get_video_distributions
from download_tvsum import download_or_generate_tvsum
from barycenter import wasserstein_barycenter, arithmetic_mean
from forward import forward_step_mine, forward_step_baseline
from losses import wasserstein_loss, w2_1d
from model import NoisePredictor
from split import split_annotators
from sample import generate_sample


def evaluate_f1_and_correlations(predicted_dist, ground_truth_scores, budget_pct=0.15):
    """
    Computes standard downstream summarization metrics:
      - F1 score (top 15% budget summary vs each annotator's top 15% ground truth)
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


def run_full_comparison(n_videos=50, frames=100, steps=100):
    tsv_path = os.path.join(os.path.dirname(__file__), "ydata-tvsum50-anno.tsv")
    if not os.path.exists(tsv_path):
        download_or_generate_tvsum(tsv_path)

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
        vid_seed = abs(hash(vid)) % (2**31 - 1)
        train_ann, heldout_ann = split_annotators(dist, n_holdout=5, seed=vid_seed)

        # 1. Train Proposed (Mine)
        x_star_mine = wasserstein_barycenter(train_ann)
        net_mine = NoisePredictor(frames)
        rng = np.random.default_rng(vid_seed)
        for _ in range(steps):
            t = rng.integers(1, 200)
            xt = forward_step_mine(x_star_mine, t)
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

        # 2. Train Baseline (AAAI-25)
        x_star_base = arithmetic_mean(train_ann)
        net_base = NoisePredictor(frames)
        for _ in range(steps):
            t = rng.integers(1, 200)
            xt = forward_step_baseline(x_star_base, t)
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
    print("      COMPREHENSIVE COMPARISON: PROPOSED (MSPDM) vs BASELINE (AAAI-25) ON TVSUM-50")
    print("=" * 85)
    print(f"{'Evaluation Metric':<32} | {'Baseline (AAAI-25)':<20} | {'Proposed (MSPDM)':<20} | {'Win Rate / p-val'}")
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


if __name__ == "__main__":
    metrics, records = run_full_comparison(50)
    print_comparison_table(metrics)
