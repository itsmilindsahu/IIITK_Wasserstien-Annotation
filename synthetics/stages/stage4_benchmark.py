"""
stages/stage4_benchmark.py

Stage 4 — Full End-to-End Diffusion Benchmark (50 Synthetic Videos)
====================================================================

Head-to-head evaluation of MSPDM vs Euclidean DDPM baseline across all
50 synthetic videos using the full reverse diffusion pipeline.

Metrics reported (each mirroring the paper's Table):
  • Downstream F1 @ Top-15% Budget
  • Spearman Rank Correlation rho
  • Kendall Rank Correlation tau
  • Held-Out W2 Transport Cost

Protocol
--------
  Per video: 15-annotator train / 5-annotator held-out split (same seed
  for both MSPDM and baseline so only the training paradigm differs).
  Full reverse diffusion sampling used for evaluation (not x_star).

Outputs (written to synthetics/results/)
-----------------------------------------
  stage4_benchmark_per_video.csv   — per-video all metrics
  stage4_benchmark_summary.json    — aggregate statistics & significance tests
  stage4_benchmark_figure.png      — 4-panel metric comparison
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
from core.losses import w2_1d, grad_cramer
from core.model import NoisePredictor
from core.forward import forward_step_mine, forward_step_baseline
from core.sample import generate_sample
from core.split import split_annotators


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Downstream metric helpers
# ---------------------------------------------------------------------------

def _f1_score(pred, gt, budget_pct=0.15):
    """F1 of top-budget_pct frames selected by pred vs gt."""
    N = len(pred)
    budget = max(1, int(N * budget_pct))
    pred_top = set(np.argsort(pred)[-budget:])
    gt_top   = set(np.argsort(gt)[-budget:])
    overlap  = len(pred_top & gt_top)
    p = overlap / budget
    r = overlap / budget
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def evaluate(predicted_dist, ground_truth_matrix, budget_pct=0.15):
    """
    Compute mean F1, Spearman rho, Kendall tau, and W2 against each held-out
    annotator, then average.
    """
    f1s, rhos, taus, w2s = [], [], [], []
    for gt in ground_truth_matrix:
        f1s.append(_f1_score(predicted_dist, gt, budget_pct))
        rho, _ = stats.spearmanr(predicted_dist, gt)
        tau, _ = stats.kendalltau(predicted_dist, gt)
        rhos.append(0.0 if np.isnan(rho) else float(rho))
        taus.append(0.0 if np.isnan(tau) else float(tau))
        w2s.append(w2_1d(predicted_dist, gt))
    return {
        "f1":      float(np.mean(f1s)) * 100.0,
        "spearman": float(np.mean(rhos)),
        "kendall":  float(np.mean(taus)),
        "w2":       float(np.mean(w2s)),
    }


# ---------------------------------------------------------------------------
# Single-video training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(train_ann, heldout_ann, mode="mine", steps=100, T=200, seed=7):
    K, N = train_ann.shape

    if mode == "mine":
        x_star = wasserstein_barycenter(train_ann)
    else:
        x_star = arithmetic_mean(train_ann)

    net  = NoisePredictor(N, rng=np.random.default_rng(seed))
    rng  = np.random.default_rng(seed + 1)

    for _ in range(steps):
        t = rng.integers(1, T)
        if mode == "mine":
            xt      = forward_step_mine(x_star, t, T, rng=rng)
            eps_hat = net.forward(xt, t)
            g_mse   = 2.0 * (eps_hat - x_star) / N
            g_cram  = grad_cramer(eps_hat, train_ann)
            net.step(xt, t, g_mse + 0.1 * g_cram)
        else:
            xt      = forward_step_baseline(x_star, t, T, rng=rng)
            eps_hat = net.forward(xt, t)
            g_mse   = 2.0 * (eps_hat - x_star) / N
            net.step(xt, t, g_mse)

    gen = generate_sample(net, N, T=T, mode=mode, rng=np.random.default_rng(seed + 100))
    return evaluate(gen, heldout_ann)


# ---------------------------------------------------------------------------
# Full 50-video benchmark
# ---------------------------------------------------------------------------

def run_stage4(n_videos=50, steps=100, verbose=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos  = generate_synthetic_tvsum(n_videos=n_videos)
    vid_keys = list(videos.keys())

    metric_keys = ["f1", "spearman", "kendall", "w2"]
    metrics = {
        "mine":     {k: [] for k in metric_keys},
        "baseline": {k: [] for k in metric_keys},
    }
    records = []

    for idx, vid in enumerate(vid_keys, 1):
        raw      = videos[vid]
        dist     = get_video_distributions(raw, method="softmax", temperature=0.5)
        vid_seed = stable_seed(vid)
        train_ann, heldout_ann = split_annotators(dist, n_holdout=5, seed=vid_seed)

        m_res = train_and_evaluate(train_ann, heldout_ann, mode="mine",     steps=steps, seed=vid_seed)
        b_res = train_and_evaluate(train_ann, heldout_ann, mode="baseline", steps=steps, seed=vid_seed)

        for k in metric_keys:
            metrics["mine"][k].append(m_res[k])
            metrics["baseline"][k].append(b_res[k])

        records.append({
            "video_id": vid,
            **{f"mine_{k}": m_res[k] for k in metric_keys},
            **{f"baseline_{k}": b_res[k] for k in metric_keys},
        })

        if idx % 10 == 0 or idx == len(vid_keys):
            print(f"  [{idx:02d}/{len(vid_keys)}] {vid}  "
                  f"F1: mine={m_res['f1']:.1f}% base={b_res['f1']:.1f}%  "
                  f"W2: mine={m_res['w2']:.3f} base={b_res['w2']:.3f}")

    # ---- Aggregate ----
    summary = {}
    for k, label, higher_better in [
        ("f1",       "Mean F1 (%) @ Top-15%",          True),
        ("spearman", "Spearman rho",                      True),
        ("kendall",  "Kendall tau",                       True),
        ("w2",       "Held-Out W2 ((lower-better))",                False),
    ]:
        m_arr = np.array(metrics["mine"][k])
        b_arr = np.array(metrics["baseline"][k])
        if higher_better:
            wins = int((m_arr > b_arr).sum())
        else:
            wins = int((m_arr < b_arr).sum())
        t_stat, t_pval = stats.ttest_rel(m_arr, b_arr)
        w_stat, w_pval = stats.wilcoxon(m_arr, b_arr, alternative="two-sided")
        summary[k] = {
            "mine_mean": float(m_arr.mean()), "mine_std": float(m_arr.std(ddof=1)),
            "base_mean": float(b_arr.mean()), "base_std": float(b_arr.std(ddof=1)),
            "delta":  float((m_arr - b_arr).mean()),
            "wins": wins, "total": len(m_arr),
            "win_rate_pct": wins / len(m_arr) * 100.0,
            "t_pval": float(t_pval), "w_pval": float(w_pval),
        }

    if verbose:
        print("\n" + "=" * 80)
        print("  STAGE 4 — FULL DIFFUSION BENCHMARK (50 SYNTHETIC VIDEOS)")
        print("=" * 80)
        print(f"  {'Metric':<28} | {'Baseline':>20} | {'MSPDM':>20} | {'Win / p-val'}")
        print(f"  {'-'*28}-+-{'-'*20}-+-{'-'*20}-+-{'-'*18}")
        for k, label in [("f1","F1 (%)"),("spearman","Spearman rho"),("kendall","Kendall tau"),("w2","W2 ((lower-better))")]:
            s = summary[k]
            b_str = f"{s['base_mean']:.4f} +/- {s['base_std']:.4f}"
            m_str = f"{s['mine_mean']:.4f} +/- {s['mine_std']:.4f}"
            w_str = f"{s['wins']}/{s['total']} (p={s['t_pval']:.2e})"
            print(f"  {label:<28} | {b_str:>20} | {m_str:>20} | {w_str}")
        print("=" * 80)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "stage4_benchmark_per_video.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "stage4_benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_cfg = [
        ("f1",       "F1 Score (%) @ Top-15% Budget", True,  axes[0, 0]),
        ("spearman", "Spearman Rank Correlation rho",    True,  axes[0, 1]),
        ("kendall",  "Kendall Rank Correlation tau",     True,  axes[1, 0]),
        ("w2",       "Held-Out W2 Distance ((lower-better))",      False, axes[1, 1]),
    ]

    for k, ylabel, higher_better, ax in panel_cfg:
        m_arr = np.array(metrics["mine"][k])
        b_arr = np.array(metrics["baseline"][k])
        ax.boxplot([b_arr, m_arr],
                   tick_labels=["Baseline\n(Euclidean DDPM)", "MSPDM\n(Simplex W2)"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#d0e1fd", color="#1f77b4"),
                   medianprops=dict(color="#08306b", linewidth=2))
        ax.set_ylabel(ylabel)
        ax.set_title(f"Stage 4: {ylabel}")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Stage 4: MSPDM vs Euclidean DDPM — 50 Synthetic Videos", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "stage4_benchmark_figure.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\n  Outputs written to {RESULTS_DIR}/")
    print(f"    {os.path.basename(csv_path)}")
    print(f"    {os.path.basename(json_path)}")
    print(f"    {os.path.basename(fig_path)}")

    return summary, records


if __name__ == "__main__":
    run_stage4()
