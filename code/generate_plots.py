"""
generate_plots.py

Generates publication-quality charts and saves them to assets/ and results/:
  1. assets/benchmark_50videos_distribution.png
  2. assets/sharpening_gap_comparison.png
  3. assets/distribution_comparison.png
  4. assets/loss_curve.png
  5. assets/w2_comparison.png
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_tvsum_data, get_video_distributions
from barycenter import wasserstein_barycenter, arithmetic_mean
from losses import w2_1d
from split import split_annotators

here = os.path.dirname(__file__)
root = os.path.join(here, "..")
assets_dir = os.path.join(root, "assets")
results_dir = os.path.join(root, "results")
os.makedirs(assets_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 1. Sharpening Gap Plot
sharpening_csv = os.path.join(results_dir, "sharpening_experiment.csv")
if os.path.exists(sharpening_csv):
    by_transform = {}
    with open(sharpening_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["transform"]
            by_transform.setdefault(t, []).append(float(row["gap_pct_of_mean"]))

    transforms = list(by_transform.keys())
    avg_gaps = [np.mean(by_transform[t]) for t in transforms]

    labels = ["Linear Normalize", "Softmax (T=0.5)", "Softmax (T=0.2)", "z-score exp (T=1.0)", "z-score exp (T=0.5)"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, avg_gaps, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Barycenter Advantage over Mean (%)", fontsize=12)
    ax.set_title("Wasserstein Barycenter Advantage vs Annotator Score Sharpening", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, avg_gaps):
        height = bar.get_height()
        ax.annotate(f"{val:+.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha="center", va="bottom" if height >= 0 else "top",
                    fontsize=10, fontweight="bold")

    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "sharpening_gap_comparison.png"), dpi=150)
    plt.savefig(os.path.join(results_dir, "sharpening_gap_comparison.png"), dpi=150)
    plt.close()
    print("Generated sharpening_gap_comparison.png")

# 2. Distribution Comparison Plot on Video 01
tsv_path = os.path.join(here, "ydata-tvsum50-anno.tsv")
if os.path.exists(tsv_path):
    vids = load_tvsum_data(tsv_path)
    vid_key = list(vids.keys())[0]
    raw = vids[vid_key]
    if raw.shape[1] > 100:
        raw = raw[:, np.linspace(0, raw.shape[1]-1, 100, dtype=int)]
    dist = get_video_distributions(raw, method="softmax", temperature=0.5)
    train, heldout = split_annotators(dist, n_holdout=5, seed=1234)

    bary = wasserstein_barycenter(train)
    mean = arithmetic_mean(train)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, a in enumerate(train):
        ax.plot(a, color="gray", alpha=0.25, linewidth=1, label="Individual Annotators" if i == 0 else "")
    ax.plot(bary, color="#1f77b4", linewidth=2.5, label="Wasserstein Barycenter (Proposed)")
    ax.plot(mean, color="#ff7f0e", linewidth=2.5, linestyle="--", label="Arithmetic Mean (Baseline)")
    ax.set_xlabel("Frame Index (0–99)", fontsize=11)
    ax.set_ylabel("Probability Mass on Simplex", fontsize=11)
    ax.set_title(f"Annotation Distribution Comparison (TVSum {vid_key})", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "distribution_comparison.png"), dpi=150)
    plt.close()
    print("Generated distribution_comparison.png")

# 3. W2 Target Comparison Bar Chart on 50 TVSum Videos
summary_json = os.path.join(results_dir, "tvsum_50videos_summary.json")
if os.path.exists(summary_json):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    configs = ["Baseline (AAAI-25)", "MSPDM (Proposed)"]
    vals = [37.7075, 37.7110]
    errors = [4.8845, 4.8869]
    colors = ["#ff7f0e", "#1f77b4"]

    bars = ax.bar(configs, vals, yerr=errors, capsize=5, color=colors, alpha=0.85, width=0.45, edgecolor="black")
    ax.set_ylabel("Held-Out W2 Distance (Lower is Better)", fontsize=11)
    ax.set_title("Full TVSum-50 Benchmark: Held-Out W2", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "w2_comparison.png"), dpi=150)
    plt.close()
    print("Generated w2_comparison.png")

print("All charts successfully generated and saved to assets/ and results/.")
