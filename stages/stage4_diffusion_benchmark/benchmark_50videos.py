"""
benchmark_50videos.py

Full TVSum-50 Benchmark with Leak-Free Evaluation:
  1. Iterates over all 50 videos in TVSum.
  2. For each video, splits the 20 annotators into 15 train / 5 held-out annotators.
  3. Trains 'mine' (W2 barycenter target + Dirichlet forward process + Cramér loss & exact gradient)
     and 'baseline' (arithmetic mean target + Gaussian forward process + MSE loss).
  4. Runs full reverse diffusion sampling (generate_sample) for both trained models.
  5. Computes held-out W2 distance of the generated samples against the 5 unseen annotators.
  6. Reports mean +/- std across all 50 videos, win rates, and formal statistical significance
     tests (paired t-test and Wilcoxon signed-rank test).
  7. Dumps detailed per-video CSV, JSON summary, and visualization.
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Ensure imports work regardless of execution directory
sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_tvsum_data, get_video_distributions
from download_tvsum import download_or_generate_tvsum
from barycenter import wasserstein_barycenter, arithmetic_mean
from forward import forward_step_mine, forward_step_baseline
from losses import wasserstein_loss, w2_1d
from model import NoisePredictor
from split import split_annotators
from sample import generate_sample


def train_and_sample(train_annotators, heldout_annotators, mode="mine", steps=100, T=200, seed=7):
    """
    Train a model on train_annotators, then generate a sample from noise
    via reverse diffusion and evaluate its W2 against heldout_annotators.
    """
    K, N = train_annotators.shape
    if mode == "mine":
        x_star = wasserstein_barycenter(train_annotators)
    else:
        x_star = arithmetic_mean(train_annotators)

    rng = np.random.default_rng(seed)
    net = NoisePredictor(N)
    losses = []

    for step in range(steps):
        t = rng.integers(1, T)
        if mode == "mine":
            xt = forward_step_mine(x_star, t, T)
            eps = x_star
            eps_hat = net.forward(xt, t)

            # MSE gradient
            grad_mse = 2 * (eps_hat - eps) / N

            # Cramér distance analytic gradient
            P = np.cumsum(eps_hat)
            grad_cramer = np.zeros_like(eps_hat)
            for a in train_annotators:
                Q = np.cumsum(a)
                diff = P - Q
                grad_cramer += 2 * np.cumsum(diff[::-1])[::-1]
            grad_cramer /= K

            lam = 0.1
            gradOut = grad_mse + lam * grad_cramer
            loss, mse, cramer = wasserstein_loss(eps, eps_hat, eps_hat, train_annotators, lam=lam)
            net.step(xt, t, gradOut)
        else:
            xt = forward_step_baseline(x_star, t, T)
            eps = x_star
            eps_hat = net.forward(xt, t)

            grad_mse = 2 * (eps_hat - eps) / N
            gradOut = grad_mse

            loss = np.mean((eps - eps_hat) ** 2)
            net.step(xt, t, gradOut)

        losses.append(float(loss))

    # Reverse diffusion sample generation from pure noise
    sample_rng = np.random.default_rng(seed + 100)
    generated = generate_sample(net, N, T=T, mode=mode, rng=sample_rng)

    # Generalization W2 to held-out annotators
    heldout_w2 = float(np.mean([w2_1d(generated, a) for a in heldout_annotators]))
    naive_leaky_w2 = float(np.mean([w2_1d(x_star, a) for a in train_annotators]))

    return {
        "final_loss": losses[-1],
        "heldout_w2": heldout_w2,
        "naive_leaky_w2": naive_leaky_w2,
        "losses": losses,
    }


def run_benchmark(n_videos=50, frames=100, method="softmax", temperature=0.5, steps=100):
    tsv_path = os.path.join(os.path.dirname(__file__), "ydata-tvsum50-anno.tsv")
    if not os.path.exists(tsv_path):
        download_or_generate_tvsum(tsv_path)

    tvsum_videos = load_tvsum_data(tsv_path)
    vid_keys = list(tvsum_videos.keys())[:n_videos]
    print(f"=== Running Full TVSum Benchmark on {len(vid_keys)} Videos ===")
    print(f"Distribution Transform: {method} (T={temperature}), Steps: {steps}, Frames: {frames}")
    print("-" * 75)

    results = []
    for idx, vid in enumerate(vid_keys, 1):
        raw_scores = tvsum_videos[vid]
        if raw_scores.shape[1] > frames:
            frame_indices = np.linspace(0, raw_scores.shape[1] - 1, frames, dtype=int)
            raw_scores = raw_scores[:, frame_indices]

        dist = get_video_distributions(raw_scores, method=method, temperature=temperature)
        
        # Per-video deterministic seed for splitting
        vid_seed = abs(hash(vid)) % (2**31 - 1)
        train_ann, heldout_ann = split_annotators(dist, n_holdout=5, seed=vid_seed)

        mine_res = train_and_sample(train_ann, heldout_ann, mode="mine", steps=steps, seed=vid_seed + 1)
        base_res = train_and_sample(train_ann, heldout_ann, mode="baseline", steps=steps, seed=vid_seed + 1)

        gap = base_res["heldout_w2"] - mine_res["heldout_w2"]
        gap_pct = (gap / base_res["heldout_w2"]) * 100.0 if base_res["heldout_w2"] > 0 else 0.0

        results.append({
            "video_id": vid,
            "mine_heldout_w2": mine_res["heldout_w2"],
            "baseline_heldout_w2": base_res["heldout_w2"],
            "gap_mean_minus_bary": gap,
            "gap_pct": gap_pct,
            "mine_final_loss": mine_res["final_loss"],
            "baseline_final_loss": base_res["final_loss"],
            "mine_leaky_w2": mine_res["naive_leaky_w2"],
            "baseline_leaky_w2": base_res["naive_leaky_w2"],
        })

        if idx % 10 == 0 or idx == len(vid_keys):
            print(f"Processed [{idx:02d}/{len(vid_keys)}] videos | "
                  f"Latest ({vid}): Mine W2 = {mine_res['heldout_w2']:.3f}, Base W2 = {base_res['heldout_w2']:.3f} "
                  f"({'MINE WINS' if gap > 0 else 'BASE WINS'})")

    return results


def analyze_and_save(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    mine_w2 = np.array([r["mine_heldout_w2"] for r in results])
    base_w2 = np.array([r["baseline_heldout_w2"] for r in results])
    gaps = np.array([r["gap_mean_minus_bary"] for r in results])
    gap_pcts = np.array([r["gap_pct"] for r in results])

    mine_mean, mine_std = float(np.mean(mine_w2)), float(np.std(mine_w2, ddof=1))
    base_mean, base_std = float(np.mean(base_w2)), float(np.std(base_w2, ddof=1))
    gap_mean, gap_std = float(np.mean(gaps)), float(np.std(gaps, ddof=1))
    avg_gap_pct = float(np.mean(gap_pcts))
    wins = int(np.sum(gaps > 0))
    total = len(results)
    win_rate = (wins / total) * 100.0

    # Significance Tests
    ttest_res = stats.ttest_rel(mine_w2, base_w2)
    t_stat, t_pval = float(ttest_res.statistic), float(ttest_res.pvalue)

    wilcoxon_res = stats.wilcoxon(mine_w2, base_w2, alternative='two-sided')
    w_stat, w_pval = float(wilcoxon_res.statistic), float(wilcoxon_res.pvalue)

    summary = {
        "num_videos": total,
        "mine_heldout_w2": {"mean": mine_mean, "std": mine_std},
        "baseline_heldout_w2": {"mean": base_mean, "std": base_std},
        "gap_absolute": {"mean": gap_mean, "std": gap_std},
        "avg_improvement_pct": avg_gap_pct,
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
            }
        }
    }

    # Print Summary Table
    print("\n" + "=" * 75)
    print("                 TVSUM-50 BENCHMARK SUMMARY STATISTICS")
    print("=" * 75)
    print(f"Total Videos:                  {total}")
    print(f"Mine (Proposed Framework):     {mine_mean:.4f} +/- {mine_std:.4f}")
    print(f"Baseline (AAAI-25 standard):   {base_mean:.4f} +/- {base_std:.4f}")
    print(f"Mean Difference (Base - Mine): {gap_mean:.4f} +/- {gap_std:.4f}  ({avg_gap_pct:+.2f}%)")
    print(f"Win Rate (Mine < Baseline):    {wins}/{total} ({win_rate:.1f}%)")
    print("-" * 75)
    print("STATISTICAL SIGNIFICANCE TESTS (Testing Mine vs Baseline per-video):")
    sig_t = 'p < 0.001 ***' if t_pval < 0.001 else ('p < 0.05 *' if t_pval < 0.05 else 'n.s. (p >= 0.05)')
    sig_w = 'p < 0.001 ***' if w_pval < 0.001 else ('p < 0.05 *' if w_pval < 0.05 else 'n.s. (p >= 0.05)')
    print(f"  Paired Student's t-test:     t = {t_stat:+.4f}, p = {t_pval:.4f}  ({sig_t})")
    print(f"  Wilcoxon signed-rank test:   W = {w_stat:.1f},  p = {w_pval:.4f}  ({sig_w})")
    print("=" * 75)

    # Save CSV
    csv_path = os.path.join(output_dir, "tvsum_50videos_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved per-video results to: {csv_path}")

    # Save JSON Summary
    json_path = os.path.join(output_dir, "tvsum_50videos_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to:  {json_path}")

    # Generate Comparison Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    ax1.boxplot([mine_w2, base_w2], tick_labels=["Proposed (Mine)", "Baseline (AAAI-25)"], patch_artist=True,
                boxprops=dict(facecolor="#d0e1fd", color="#1f77b4"),
                medianprops=dict(color="#08306b", linewidth=2))
    ax1.set_ylabel("Held-Out W2 Distance (Lower is Better)")
    ax1.set_title("Held-Out W2 Across 50 TVSum Videos")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Paired differences histogram
    ax2.hist(gaps, bins=15, color="#2ca02c", alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Difference")
    ax2.axvline(gap_mean, color="darkgreen", linestyle="-", linewidth=2, label=f"Mean Gap = {gap_mean:.2f}")
    ax2.set_xlabel("W2 Reduction (Baseline - Mine) per Video")
    ax2.set_ylabel("Number of Videos")
    ax2.set_title("Per-Video Improvement Distribution")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "benchmark_50videos_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved benchmark figure to: {plot_path}")

    return summary


def main():
    results = run_benchmark(n_videos=50, frames=100, method="softmax", temperature=0.5, steps=100)
    analyze_and_save(results)


if __name__ == "__main__":
    main()
