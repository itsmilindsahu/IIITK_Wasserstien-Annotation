"""
experiment_sharpening.py

Item 3 from the review: the original `get_video_distributions` used a
linear normalize-by-sum, which barely sharpens the raw 1-5 annotator
scores -- so annotators end up looking fairly similar to each other and to
uniform, and it's unclear whether the barycenter fix (vs. arithmetic mean)
would show a bigger effect once distributions actually diverge more.

This script re-runs the barycenter-vs-mean comparison across multiple
videos and multiple sharpening transforms:
    - normalize          (original, T not used)
    - softmax, T=0.5
    - softmax, T=0.2
    - zscore_exp, T=1.0
    - zscore_exp, T=0.5

For each (video, transform), we:
    1. split annotators into 15 train / 5 held-out (same split used
       across transforms for a given video, for comparability)
    2. compute x_star via wasserstein_barycenter(train) and via
       arithmetic_mean(train)
    3. score both against the 5 held-out annotators with the true
       quantile-based w2_1d
    4. record the gap = w2(mean, heldout) - w2(barycenter, heldout)
       (positive => barycenter wins)

We also report a "sharpness" proxy per transform (mean pairwise w2_1d
between train annotators within a video) so we can check whether the gap
actually correlates with how spread out the distributions got.
"""
import os
import numpy as np

from dataset import load_tvsum_data, get_video_distributions
from barycenter import wasserstein_barycenter, arithmetic_mean
from losses import w2_1d
from split import split_annotators

TRANSFORMS = [
    ("normalize", {}),
    ("softmax_T0.5", {"method": "softmax", "temperature": 0.5}),
    ("softmax_T0.2", {"method": "softmax", "temperature": 0.2}),
    ("zscore_exp_T1.0", {"method": "zscore_exp", "temperature": 1.0}),
    ("zscore_exp_T0.5", {"method": "zscore_exp", "temperature": 0.5}),
]


def annotator_spread(dists):
    """Mean pairwise w2_1d among a set of annotator distributions -- a
    proxy for how 'sharp'/divergent the distributions are."""
    K = dists.shape[0]
    vals = []
    for i in range(K):
        for j in range(i + 1, K):
            vals.append(w2_1d(dists[i], dists[j]))
    return float(np.mean(vals)) if vals else 0.0


def main(n_videos=9, frames=100, holdout_seed=1234):
    tsv_path = os.path.join(os.path.dirname(__file__), "ydata-tvsum50-anno.tsv")
    tvsum_videos = load_tvsum_data(tsv_path)
    vid_keys = list(tvsum_videos.keys())[:n_videos]

    rows = []
    for vid in vid_keys:
        raw_scores = tvsum_videos[vid]
        if raw_scores.shape[1] > frames:
            indices = np.linspace(0, raw_scores.shape[1] - 1, frames, dtype=int)
            raw_scores = raw_scores[:, indices]

        for name, kwargs in TRANSFORMS:
            dist = get_video_distributions(raw_scores, **kwargs)
            train, heldout = split_annotators(dist, n_holdout=5, seed=holdout_seed)

            bary = wasserstein_barycenter(train)
            mean = arithmetic_mean(train)

            w2_bary = float(np.mean([w2_1d(bary, a) for a in heldout]))
            w2_mean = float(np.mean([w2_1d(mean, a) for a in heldout]))
            gap = w2_mean - w2_bary
            spread = annotator_spread(train)

            rows.append({
                "video": vid,
                "transform": name,
                "w2_barycenter": w2_bary,
                "w2_mean": w2_mean,
                "gap_mean_minus_bary": gap,
                "gap_pct_of_mean": 100.0 * gap / w2_mean if w2_mean > 0 else float("nan"),
                "annotator_spread": spread,
            })

    return rows


def summarize(rows):
    by_transform = {}
    for r in rows:
        by_transform.setdefault(r["transform"], []).append(r)

    print(f"{'transform':<16} {'avg spread':>12} {'avg w2_bary':>12} {'avg w2_mean':>12} {'avg gap':>10} {'avg gap%':>10} {'#bary wins':>10}")
    for name, _ in TRANSFORMS:
        rs = by_transform[name]
        avg_spread = np.mean([r["annotator_spread"] for r in rs])
        avg_bary = np.mean([r["w2_barycenter"] for r in rs])
        avg_mean = np.mean([r["w2_mean"] for r in rs])
        avg_gap = np.mean([r["gap_mean_minus_bary"] for r in rs])
        avg_gap_pct = np.mean([r["gap_pct_of_mean"] for r in rs])
        wins = sum(1 for r in rs if r["gap_mean_minus_bary"] > 0)
        print(f"{name:<16} {avg_spread:>12.2f} {avg_bary:>12.4f} {avg_mean:>12.4f} {avg_gap:>10.4f} {avg_gap_pct:>9.2f}% {wins:>7}/{len(rs)}")


if __name__ == "__main__":
    rows = main()
    summarize(rows)

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "sharpening_experiment.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")
