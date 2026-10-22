"""
stages/stage2_disagreement.py

Stage 2 — Stress-Testing & Disagreement Sensitivity
=====================================================

Shows that the Wasserstein-barycenter advantage over arithmetic mean scales
monotonically with inter-annotator disagreement.

Protocol
--------
1. Generate 50 synthetic videos.
2. For each video, compute annotator distributions at 5 temperature levels
   tau in {1.0, 0.7, 0.5, 0.35, 0.2}  (lower tau -> sharper -> more disagreement).
3. Evaluate oracle W2 gap = mean_W2 − barycenter_W2 at each temperature.
4. Also segment videos into terciles of native disagreement (measured by mean
   pairwise W2 among annotators) and compare gap across terciles.

Outputs (written to synthetics/results/)
-----------------------------------------
  stage2_sharpening_gap.csv     — per-video gap at each temperature
  stage2_sharpening_gap.png     — gap vs temperature line plot
  stage2_tercile_analysis.json  — tercile statistics
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
TEMPERATURES = [1.0, 0.7, 0.5, 0.35, 0.2]


def _pairwise_mean_w2(dist):
    """Average pairwise W2 distance among K annotator distributions."""
    K = dist.shape[0]
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += w2_1d(dist[i], dist[j])
            count += 1
    return total / count if count else 0.0


def run_stage2(n_videos=50, n_holdout=5, verbose=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos = generate_synthetic_tvsum(n_videos=n_videos)
    vid_keys = list(videos.keys())

    # ----- Temperature sharpening experiment -----
    records = []
    gap_by_temp = {tau: [] for tau in TEMPERATURES}

    for vid in vid_keys:
        raw = videos[vid]
        vid_seed = stable_seed(vid)
        row = {"video_id": vid}
        for tau in TEMPERATURES:
            dist = get_video_distributions(raw, method="softmax", temperature=tau)
            train_ann, heldout_ann = split_annotators(dist, n_holdout=n_holdout, seed=vid_seed)
            bary = wasserstein_barycenter(train_ann)
            mean = arithmetic_mean(train_ann)
            bary_w2 = float(np.mean([w2_1d(bary, a) for a in heldout_ann]))
            mean_w2 = float(np.mean([w2_1d(mean, a) for a in heldout_ann]))
            gap = mean_w2 - bary_w2
            row[f"gap_tau_{tau}"] = gap
            gap_by_temp[tau].append(gap)
        records.append(row)

    # ----- Native disagreement tercile experiment -----
    dist_default = {}
    native_disagreement = {}
    for vid in vid_keys:
        raw = videos[vid]
        vid_seed = stable_seed(vid)
        dist = get_video_distributions(raw, method="softmax", temperature=0.5)
        dist_default[vid] = (dist, vid_seed)
        native_disagreement[vid] = _pairwise_mean_w2(dist)

    sorted_vids = sorted(vid_keys, key=lambda v: native_disagreement[v])
    n3 = n_videos // 3
    terciles = {"low": sorted_vids[:n3], "mid": sorted_vids[n3:2*n3], "high": sorted_vids[2*n3:]}
    tercile_stats = {}

    for tercile_name, t_vids in terciles.items():
        gaps = []
        for vid in t_vids:
            dist, vid_seed = dist_default[vid]
            train_ann, heldout_ann = split_annotators(dist, n_holdout=n_holdout, seed=vid_seed)
            bary = wasserstein_barycenter(train_ann)
            mean = arithmetic_mean(train_ann)
            bary_w2 = float(np.mean([w2_1d(bary, a) for a in heldout_ann]))
            mean_w2 = float(np.mean([w2_1d(mean, a) for a in heldout_ann]))
            gaps.append(mean_w2 - bary_w2)
        tercile_stats[tercile_name] = {
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps, ddof=1)),
            "n_videos": len(t_vids),
        }

    if verbose:
        print("\n" + "=" * 65)
        print("  STAGE 2 — DISAGREEMENT SENSITIVITY (SHARPENING & TERCILES)")
        print("=" * 65)
        print(f"  {'Temperature':<14} | {'Mean Gap (Base−Bary)':>22}")
        print(f"  {'-'*14}-+-{'-'*22}")
        for tau in TEMPERATURES:
            g = np.mean(gap_by_temp[tau])
            print(f"  tau = {tau:<10} | {g:>22.4f}")
        print()
        print("  Native Disagreement Terciles (tau=0.5):")
        for t, s in tercile_stats.items():
            print(f"    {t:>4} tercile: mean gap = {s['mean_gap']:.4f} +/- {s['std_gap']:.4f}")
        print("=" * 65)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "stage2_sharpening_gap.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "stage2_tercile_analysis.json")
    with open(json_path, "w") as f:
        json.dump(tercile_stats, f, indent=2)

    # Figure: gap vs temperature
    mean_gaps = [np.mean(gap_by_temp[tau]) for tau in TEMPERATURES]
    std_gaps  = [np.std(gap_by_temp[tau], ddof=1) for tau in TEMPERATURES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(TEMPERATURES, mean_gaps, yerr=std_gaps, marker="o", color="#1f77b4",
                 linewidth=2, capsize=5, label="Mean W2 gap")
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Softmax Temperature tau (Lower -> More Disagreement)")
    ax1.set_ylabel("W2 Gap (Baseline − Barycenter, Higher = Barycenter Wins More)")
    ax1.set_title("Stage 2a: Barycenter Advantage vs Sharpening Temperature")
    ax1.invert_xaxis()
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    t_names = list(tercile_stats.keys())
    t_means = [tercile_stats[t]["mean_gap"] for t in t_names]
    t_stds  = [tercile_stats[t]["std_gap"]  for t in t_names]
    colors  = ["#aec7e8", "#1f77b4", "#08306b"]
    bars = ax2.bar(t_names, t_means, yerr=t_stds, color=colors, capsize=6, edgecolor="black")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Mean W2 Gap (Baseline − Barycenter)")
    ax2.set_title("Stage 2b: Gap by Native Disagreement Tercile")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "stage2_sharpening_gap.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\n  Outputs written to {RESULTS_DIR}/")
    print(f"    {os.path.basename(csv_path)}")
    print(f"    {os.path.basename(json_path)}")
    print(f"    {os.path.basename(fig_path)}")

    return {"sharpening": gap_by_temp, "terciles": tercile_stats}


if __name__ == "__main__":
    run_stage2()
