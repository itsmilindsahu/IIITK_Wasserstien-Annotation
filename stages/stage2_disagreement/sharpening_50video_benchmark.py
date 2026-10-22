"""
sharpening_50video_benchmark.py

Extension (2): "Extend the sharpening ablation (Table 2) to the full
50-video benchmark."

What this does
---------------
Table 2 in the README (Linear normalize / Softmax tau=0.5 / Softmax tau=0.2
/ z-score exp tau=1.0 / z-score exp tau=0.5) was run on a small sample
(win rates out of 9). This script reruns the exact same five transforms
across ALL 50 TVSum videos, using the same leak-free 15/5 split and W2
held-out protocol as benchmark_50videos.py, and reports win rate out of 50
plus paired significance tests per transform -- i.e. it is Table 2 at the
same n as the main benchmark table.

Integration
------------
This script requires the authorized local TVSum annotation TSV and fails if
it cannot be loaded. Optional analysis helpers use local NumPy implementations
when needed; no synthetic dataset is generated.

Usage
-----
    python sharpening_50video_benchmark.py --seed 0 \
        --out results/sharpening_50video_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Callable

# Ensure code/ is on path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_code_dir = os.path.join(_project_root, "code")
if os.path.isdir(_code_dir) and _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

import numpy as np

try:
    from scipy.stats import wilcoxon, ttest_rel
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


TRANSFORMS = [
    ("Linear normalize", "linear", None),
    ("Softmax tau=0.5", "softmax", 0.5),
    ("Softmax tau=0.2", "softmax", 0.2),
    ("z-score exp tau=1.0", "zscore_exp", 1.0),
    ("z-score exp tau=0.5", "zscore_exp", 0.5),
]


# --------------------------------------------------------------------------- #
# --- INTEGRATION POINT: dataset + transforms --------------------------------
# --------------------------------------------------------------------------- #
def _load_real_tvsum_raw_scores() -> dict:
    """Load authorized raw scores as {video_id: (20, frames)} matrices."""
    import dataset
    from download_tvsum import require_tvsum_annotations

    annotation_path = require_tvsum_annotations(
        os.path.join(_code_dir, "ydata-tvsum50-anno.tsv")
    )
    raw_videos = dataset.load_tvsum_data(annotation_path)
    if len(raw_videos) != 50:
        raise ValueError(f"Expected 50 real TVSum videos; found {len(raw_videos)}.")
    if any(scores.ndim != 2 or scores.shape[0] != 20 for scores in raw_videos.values()):
        raise ValueError("Every real TVSum video must have 20 annotator score vectors.")
    return raw_videos


def _apply_transform(raw_scores: np.ndarray, kind: str, tau) -> np.ndarray:
    """Try your dataset.py's transform functions first; fall back to the
    formulas implied by the README's Table 2 column names."""
    try:
        import dataset
        if hasattr(dataset, "get_video_distributions"):
            method = "normalize" if kind == "linear" else kind
            return dataset.get_video_distributions(raw_scores, method=method, temperature=tau if tau is not None else 1.0)
        if kind == "linear" and hasattr(dataset, "linear_normalize"):
            return dataset.linear_normalize(raw_scores)
        if kind == "softmax" and hasattr(dataset, "softmax_transform"):
            return dataset.softmax_transform(raw_scores, tau)
        if kind == "zscore_exp" and hasattr(dataset, "zscore_exp_transform"):
            return dataset.zscore_exp_transform(raw_scores, tau)
    except ImportError:
        pass

    out = np.zeros_like(raw_scores, dtype=float)
    for k in range(raw_scores.shape[0]):
        row = raw_scores[k]
        if kind == "linear":
            row = row - row.min() + 1e-8
            out[k] = row / row.sum()
        elif kind == "softmax":
            z = (row - row.max()) / tau
            e = np.exp(z)
            out[k] = e / e.sum()
        elif kind == "zscore_exp":
            z = (row - row.mean()) / (row.std() + 1e-8)
            e = np.exp(z / tau)
            out[k] = e / e.sum()
        else:
            raise ValueError(kind)
    return out


# --------------------------------------------------------------------------- #
# --- INTEGRATION POINT: split / barycenter / W2 (same as main benchmark) ---
# --------------------------------------------------------------------------- #
def _get_split_fn() -> Callable:
    try:
        import split
        if hasattr(split, "leak_free_split"):
            return split.leak_free_split
        if hasattr(split, "split_annotators"):
            return lambda mat, n_train=15, n_holdout=5, seed=0: split.split_annotators(mat, n_holdout=n_holdout, seed=seed)
    except ImportError:
        pass

    def _fallback_split(annotator_matrix: np.ndarray, n_train=15, n_holdout=5, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(annotator_matrix.shape[0])
        train_idx, holdout_idx = idx[:n_train], idx[n_train:n_train + n_holdout]
        return annotator_matrix[train_idx], annotator_matrix[holdout_idx]

    return _fallback_split


def _get_barycenter_fns() -> tuple[Callable, Callable]:
    try:
        import barycenter
        wb = getattr(barycenter, "wasserstein_barycenter", None)
        am = getattr(barycenter, "arithmetic_mean", None)
        if wb is not None and am is not None:
            return wb, am
    except ImportError:
        pass

    def _quantile_barycenter(dists: np.ndarray, n_quantiles: int = 512) -> np.ndarray:
        u = (np.arange(n_quantiles) + 0.5) / n_quantiles
        avg_quantiles = np.zeros(n_quantiles)
        for row in dists:
            cdf = np.cumsum(row)
            cdf[-1] = 1.0
            avg_quantiles += np.interp(u, cdf, np.arange(1, len(row) + 1))
        avg_quantiles /= dists.shape[0]
        hist, _ = np.histogram(avg_quantiles, bins=np.arange(0, dists.shape[1] + 1))
        hist = hist.astype(float)
        if hist.sum() == 0:
            hist[:] = 1.0
        return hist / hist.sum()

    def _arithmetic_mean(dists: np.ndarray) -> np.ndarray:
        m = dists.mean(axis=0)
        return m / m.sum()

    return _quantile_barycenter, _arithmetic_mean


def _get_w2_fn() -> Callable:
    try:
        import losses
        if hasattr(losses, "w2_1d"):
            return losses.w2_1d
        if hasattr(losses, "w2_distance"):
            return losses.w2_distance
        if hasattr(losses, "wasserstein2"):
            return losses.wasserstein2
    except ImportError:
        pass

    def _w2(p: np.ndarray, q: np.ndarray, n_quantiles: int = 512) -> float:
        u = (np.arange(n_quantiles) + 0.5) / n_quantiles
        cdf_p = np.cumsum(p); cdf_p[-1] = 1.0
        cdf_q = np.cumsum(q); cdf_q[-1] = 1.0
        fp = np.interp(u, cdf_p, np.arange(1, len(p) + 1))
        fq = np.interp(u, cdf_q, np.arange(1, len(q) + 1))
        return float(np.mean((fp - fq) ** 2))

    return _w2


def annotator_spread(dists: np.ndarray, w2_fn: Callable) -> float:
    k = dists.shape[0]
    vals = [w2_fn(dists[i], dists[j]) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(vals))


def evaluate_one(video_id: str, transformed: np.ndarray, seed: int,
                  split_fn, wb_fn, am_fn, w2_fn) -> dict:
    train, holdout = split_fn(transformed, n_train=15, n_holdout=5, seed=seed)
    bary_target = wb_fn(train)
    mean_target = am_fn(train)
    bary_errs = [w2_fn(bary_target, h) for h in holdout]
    mean_errs = [w2_fn(mean_target, h) for h in holdout]
    w2_bary, w2_mean = float(np.mean(bary_errs)), float(np.mean(mean_errs))
    return {
        "video_id": video_id,
        "spread": annotator_spread(transformed, w2_fn),
        "w2_barycenter": w2_bary,
        "w2_mean": w2_mean,
        "gap_pct": 100.0 * (w2_mean - w2_bary) / w2_mean if w2_mean > 0 else 0.0,
        "bary_wins": w2_bary < w2_mean,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/sharpening_50video_summary.csv")
    args = parser.parse_args()

    raw_videos = _load_real_tvsum_raw_scores()

    split_fn = _get_split_fn()
    wb_fn, am_fn = _get_barycenter_fns()
    w2_fn = _get_w2_fn()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    detail_path = args.out.replace(".csv", "_per_video.csv")

    summary_rows = []
    with open(detail_path, "w", newline="") as f_detail:
        writer = csv.writer(f_detail)
        writer.writerow(["transform", "video_id", "spread", "w2_barycenter",
                          "w2_mean", "gap_pct", "bary_wins"])

        for label, kind, tau in TRANSFORMS:
            per_video = []
            for vid, raw in raw_videos.items():
                transformed = _apply_transform(raw, kind, tau)
                res = evaluate_one(vid, transformed, args.seed, split_fn, wb_fn, am_fn, w2_fn)
                per_video.append(res)
                writer.writerow([label, res["video_id"], res["spread"],
                                  res["w2_barycenter"], res["w2_mean"],
                                  res["gap_pct"], res["bary_wins"]])

            bary = np.array([r["w2_barycenter"] for r in per_video])
            mean = np.array([r["w2_mean"] for r in per_video])
            n = len(per_video)
            wins = sum(r["bary_wins"] for r in per_video)
            row = {
                "transform": label,
                "n_videos": n,
                "mean_spread": float(np.mean([r["spread"] for r in per_video])),
                "w2_barycenter": float(bary.mean()),
                "w2_mean": float(mean.mean()),
                "gap_pct": float(np.mean([r["gap_pct"] for r in per_video])),
                "win_rate": f"{wins}/{n}",
            }
            if _HAVE_SCIPY and n > 1:
                try:
                    _, p_t = ttest_rel(bary, mean)
                    _, p_w = wilcoxon(bary, mean)
                    row["p_ttest"], row["p_wilcoxon"] = float(p_t), float(p_w)
                except ValueError:
                    row["p_ttest"], row["p_wilcoxon"] = None, None
            summary_rows.append(row)

    with open(args.out, "w", newline="") as f:
        fieldnames = ["transform", "n_videos", "mean_spread", "w2_barycenter",
                      "w2_mean", "gap_pct", "win_rate", "p_ttest", "p_wilcoxon"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Per-video results: {detail_path}")
    print(f"Summary (50 real TVSum videos): {args.out}\n")

    header = (f"{'Transform':<20}{'Spread':>10}{'W2 Bary':>10}{'W2 Mean':>10}"
              f"{'Gap%':>8}{'WinRate':>9}")
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(f"{row['transform']:<20}{row['mean_spread']:>10.2f}"
              f"{row['w2_barycenter']:>10.4f}{row['w2_mean']:>10.4f}"
              f"{row['gap_pct']:>8.2f}{row['win_rate']:>9}")
        if row.get("p_wilcoxon") is not None:
            print(f"    (paired t-test p={row['p_ttest']:.4g}, "
                  f"Wilcoxon p={row['p_wilcoxon']:.4g})")


if __name__ == "__main__":
    main()
