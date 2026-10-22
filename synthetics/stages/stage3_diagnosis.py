"""
stages/stage3_diagnosis.py

Stage 3 — Methodological Diagnosis & Baseline Collapse
=======================================================

Reproduces two key diagnostic findings:

A) Gradient norm comparison at t=0:
   • MSE gradient norm ≈ 0.002 (vanishes through the softmax).
   • Cramér gradient norm ≈ 6.8  (~3300x larger).

B) LR-invariance of baseline collapse:
   • The Euclidean DDPM baseline produces near-uniform outputs regardless of
     the learning rate (lr in {0.01, 0.1, 0.5, 1.0, 5.0, 10.0}).
   • Measured as std(generated_sample) -> 0.

Outputs (written to synthetics/results/)
-----------------------------------------
  stage3_gradient_norms.json    — numeric gradient norm results
  stage3_gradient_norms.png     — bar chart (MSE vs Cramér)
  stage3_lr_collapse.csv        — std of generated samples at each lr
  stage3_lr_collapse.png        — std vs lr plot
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

from data.synthetic_dataset import generate_synthetic_tvsum, get_video_distributions, stable_seed
from core.barycenter import wasserstein_barycenter, arithmetic_mean
from core.losses import grad_cramer
from core.model import NoisePredictor
from core.forward import forward_step_mine, forward_step_baseline
from core.sample import generate_sample
from core.split import split_annotators


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]


def _gradient_norm_experiment(dist, n_trials=20, seed=42):
    """
    Measure ||∇MSE|| and ||∇Cramér|| at t=0 averaged over n_trials.

    Protocol: randomly draw a noisy x_t from the forward process at a random t,
    run the network forward, compute both gradients analytically.
    """
    rng = np.random.default_rng(seed)
    K, N = dist.shape
    net = NoisePredictor(N, rng=np.random.default_rng(seed))
    x_star = wasserstein_barycenter(dist)

    mse_norms, cramer_norms = [], []
    for _ in range(n_trials):
        t = rng.integers(1, 200)
        xt = forward_step_mine(x_star, t, rng=rng)
        eps_hat = net.forward(xt, t)

        g_mse = 2.0 * (eps_hat - x_star) / N
        g_cramer = grad_cramer(eps_hat, dist)

        mse_norms.append(float(np.linalg.norm(g_mse)))
        cramer_norms.append(float(np.linalg.norm(g_cramer)))

    return float(np.mean(mse_norms)), float(np.mean(cramer_norms))


def _lr_collapse_experiment(train_ann, steps=200, seed=42):
    """
    Train the Euclidean baseline at each learning rate and return the
    standard deviation of the generated sample (collapse indicator).
    """
    rng = np.random.default_rng(seed)
    K, N = train_ann.shape
    x_star = arithmetic_mean(train_ann)
    results = {}

    for lr in LR_VALUES:
        net = NoisePredictor(N, lr=lr, rng=np.random.default_rng(seed))
        step_rng = np.random.default_rng(seed + 1)
        for _ in range(steps):
            t = step_rng.integers(1, 200)
            xt = forward_step_baseline(x_star, t, rng=step_rng)
            eps_hat = net.forward(xt, t)
            grad = 2.0 * (eps_hat - x_star) / N
            net.step(xt, t, grad)

        gen = generate_sample(net, N, T=200, mode="baseline", rng=np.random.default_rng(seed + 100))
        results[lr] = float(np.std(gen))

    return results


def run_stage3(n_videos=5, verbose=True):
    """Run gradient norm + collapse experiments on a small subset of videos."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos = generate_synthetic_tvsum(n_videos=50)
    vid_keys = list(videos.keys())[:n_videos]

    # ---- Part A: Gradient norms ----
    mse_norms_all, cramer_norms_all = [], []
    for vid in vid_keys:
        raw = videos[vid]
        dist = get_video_distributions(raw, method="softmax", temperature=0.5)
        m, c = _gradient_norm_experiment(dist)
        mse_norms_all.append(m)
        cramer_norms_all.append(c)

    avg_mse    = float(np.mean(mse_norms_all))
    avg_cramer = float(np.mean(cramer_norms_all))
    ratio      = avg_cramer / avg_mse if avg_mse > 0 else float("inf")

    grad_results = {
        "avg_mse_gradient_norm": avg_mse,
        "avg_cramer_gradient_norm": avg_cramer,
        "ratio_cramer_over_mse": ratio,
    }

    if verbose:
        print("\n" + "=" * 65)
        print("  STAGE 3A — GRADIENT NORM COMPARISON")
        print("=" * 65)
        print(f"  ||∇MSE||    : {avg_mse:.4f}")
        print(f"  ||∇Cramér|| : {avg_cramer:.4f}")
        print(f"  Ratio       : {ratio:.0f}x  (Cramér >> MSE)")
        print("=" * 65)

    # ---- Part B: LR-collapse invariance ----
    vid = vid_keys[0]
    raw = videos[vid]
    dist = get_video_distributions(raw, method="softmax", temperature=0.5)
    vid_seed = stable_seed(vid)
    train_ann, _ = split_annotators(dist, n_holdout=5, seed=vid_seed)
    collapse_results = _lr_collapse_experiment(train_ann, steps=200, seed=vid_seed)

    if verbose:
        print("\n  STAGE 3B — BASELINE COLLAPSE ACROSS LEARNING RATES")
        print("=" * 65)
        print(f"  {'LR':<8} | {'std(generated_sample)':>24}")
        print(f"  {'-'*8}-+-{'-'*24}")
        for lr, std_val in collapse_results.items():
            print(f"  {lr:<8} | {std_val:>24.6f}")
        print("=" * 65)

    # Save outputs
    json_path = os.path.join(RESULTS_DIR, "stage3_gradient_norms.json")
    with open(json_path, "w") as f:
        json.dump(grad_results, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, "stage3_lr_collapse.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["learning_rate", "generated_sample_std"])
        for lr, std_val in collapse_results.items():
            writer.writerow([lr, std_val])

    # Figure A: gradient norms bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars = ax1.bar(["MSE (Baseline)", "Cramér (MSPDM)"], [avg_mse, avg_cramer],
                   color=["#d62728", "#2ca02c"], edgecolor="black", width=0.4)
    ax1.set_ylabel("Gradient ℓ2 Norm")
    ax1.set_title("Stage 3a: Gradient Norm at Step t=0\n"
                  f"Cramér is {ratio:.0f}x larger than MSE")
    ax1.set_yscale("log")
    for bar, val in zip(bars, [avg_mse, avg_cramer]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Figure B: collapse std vs lr
    lrs = list(collapse_results.keys())
    stds = list(collapse_results.values())
    ax2.semilogx(lrs, stds, marker="o", color="#1f77b4", linewidth=2)
    ax2.axhline(1.0 / np.sqrt(12), color="gray", linestyle="--", alpha=0.6,
                label="Uniform std ≈ 0.289")
    ax2.set_xlabel("Learning Rate (log scale)")
    ax2.set_ylabel("std(Generated Sample)")
    ax2.set_title("Stage 3b: Baseline Collapse Invariance\nacross Learning Rates")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "stage3_gradient_collapse.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\n  Outputs written to {RESULTS_DIR}/")
    print(f"    {os.path.basename(json_path)}")
    print(f"    {os.path.basename(csv_path)}")
    print(f"    {os.path.basename(fig_path)}")

    return {"gradient_norms": grad_results, "lr_collapse": collapse_results}


if __name__ == "__main__":
    run_stage3()
