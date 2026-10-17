"""
native_disagreement_eval.py

Extension (1): "Evaluate on datasets or video subsets with higher native
annotator disagreement (e.g., ambiguous or multi-genre content) to test the
ablation's prediction directly, rather than inducing disagreement synthetically."

What this does
---------------
The existing sharpening ablation (Table 2 / experiment_sharpening.py) tests
the prediction "barycenter advantage grows with annotator disagreement" by
ARTIFICIALLY sharpening scores with softmax/z-score temperature transforms.
This script tests the same prediction WITHOUT touching the score transform:
it measures each real TVSum video's *native* annotator disagreement (as
given by the raw 20-annotator distributions), buckets the 50 videos into
Low / Medium / High native-disagreement terciles, and checks whether the
Wasserstein-barycenter-vs-arithmetic-mean advantage rises monotonically
across those terciles, using the same leak-free 15/5 split and the same
held-out W2 metric as the main benchmark.

Integration
------------
Drop this file next to your existing `code/` package. It tries to import
your real dataset/split/barycenter/loss functions first, and only falls
back to small self-contained equivalents (so the file is runnable/testable
even if it can't find your modules or the real TVSum download). Check the
`--- INTEGRATION POINT ---` comments below if any import name doesn't match
your actual function signatures — those are the only lines you should need
to touch.

Usage
-----
    python native_disagreement_eval.py --n-videos 50 --seed 0 \
        --out results/native_disagreement_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

# Ensure code/ is on path
_code_dir = os.path.join(os.path.dirname(__file__), "code")
if os.path.isdir(_code_dir) and _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

import numpy as np

try:
    from scipy.stats import wilcoxon, ttest_rel
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# --- INTEGRATION POINT 1: dataset loading -----------------------------------
# --------------------------------------------------------------------------- #
def _load_real_tvsum_videos() -> Optional[dict]:
    """
    Try to load the real TVSum data using your existing dataset.py /
    download_tvsum.py. Expected to return {video_id: annotator_matrix}
    where annotator_matrix has shape (K=20, N_segments), each row already
    normalized to the simplex (sums to 1, nonnegative).

    Adjust the two import lines below to match your actual function names
    if they differ.
    """
    try:
        import dataset  # your code/dataset.py
        if hasattr(dataset, "load_tvsum_data") and hasattr(dataset, "get_video_distributions"):
            tsv_path = os.path.join(_code_dir, "ydata-tvsum50-anno.tsv")
            if not os.path.exists(tsv_path):
                tsv_path = "ydata-tvsum50-anno.tsv"
            raw = dataset.load_tvsum_data(tsv_path)
            return {vid: dataset.get_video_distributions(mat, method="normalize") for vid, mat in raw.items()}
        if hasattr(dataset, "load_all_videos"):
            return dataset.load_all_videos()
        if hasattr(dataset, "load_tvsum_all"):
            return dataset.load_tvsum_all()
    except ImportError:
        pass
    try:
        import download_tvsum
        if hasattr(download_tvsum, "get_all_videos"):
            return download_tvsum.get_all_videos()
    except ImportError:
        pass
    return None


def _synthetic_fallback_videos(n_videos: int, seed: int) -> dict:
    """Synthetic stand-in matching the paper's stated shape (50 videos,
    20 annotators, TVSum-like segment counts), used only if the real
    dataset/download modules aren't importable in this environment."""
    rng = np.random.default_rng(seed)
    videos = {}
    for v in range(n_videos):
        n_segments = rng.integers(80, 140)
        n_annotators = 20
        # vary native disagreement across videos: some videos get annotators
        # drawn from very different underlying "preference" distributions
        # (multi-genre / ambiguous content proxy), others from similar ones.
        n_modes = rng.integers(1, 4)
        mode_centers = rng.choice(n_segments, size=n_modes, replace=False)
        mat = np.zeros((n_annotators, n_segments))
        for k in range(n_annotators):
            center = mode_centers[rng.integers(0, n_modes)]
            width = rng.uniform(5, 25)
            x = np.arange(n_segments)
            raw = np.exp(-0.5 * ((x - center) / width) ** 2)
            raw += rng.uniform(0, 0.05, size=n_segments)
            mat[k] = raw / raw.sum()
        videos[f"synthetic_{v:03d}"] = mat
    return videos


# --------------------------------------------------------------------------- #
# --- INTEGRATION POINT 2: split / barycenter / W2 ---------------------------
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
        """Exact 1D W2 barycenter via quantile-function averaging (Fix 1)."""
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
        """Exact 1D W2^2 distance via quantile functions, per the paper's formula."""
        u = (np.arange(n_quantiles) + 0.5) / n_quantiles
        cdf_p = np.cumsum(p); cdf_p[-1] = 1.0
        cdf_q = np.cumsum(q); cdf_q[-1] = 1.0
        fp = np.interp(u, cdf_p, np.arange(1, len(p) + 1))
        fq = np.interp(u, cdf_q, np.arange(1, len(q) + 1))
        return float(np.mean((fp - fq) ** 2))

    return _w2


# --------------------------------------------------------------------------- #
# Native disagreement metric
# --------------------------------------------------------------------------- #
def native_disagreement(annotator_matrix: np.ndarray, w2_fn: Callable) -> float:
    """Mean pairwise W2 distance among a video's raw (untransformed,
    unsharpened) annotator distributions -- the 'native' spread analogous
    to the 'Annotator Spread' column in Table 2, but computed on real data
    with no synthetic temperature transform applied."""
    k = annotator_matrix.shape[0]
    dists = []
    for i in range(k):
        for j in range(i + 1, k):
            dists.append(w2_fn(annotator_matrix[i], annotator_matrix[j]))
    return float(np.mean(dists))


# --------------------------------------------------------------------------- #
# Per-video evaluation (mirrors the main benchmark's held-out protocol)
# --------------------------------------------------------------------------- #
@dataclass
class VideoResult:
    video_id: str
    native_spread: float
    w2_barycenter: float
    w2_mean: float
    gap_pct: float
    bary_wins: bool


def evaluate_video(video_id: str, annotator_matrix: np.ndarray, seed: int,
                    split_fn: Callable, wb_fn: Callable, am_fn: Callable,
                    w2_fn: Callable) -> VideoResult:
    spread = native_disagreement(annotator_matrix, w2_fn)
    train, holdout = split_fn(annotator_matrix, n_train=15, n_holdout=5, seed=seed)

    bary_target = wb_fn(train)
    mean_target = am_fn(train)

    bary_errs = [w2_fn(bary_target, h) for h in holdout]
    mean_errs = [w2_fn(mean_target, h) for h in holdout]

    w2_bary, w2_mean = float(np.mean(bary_errs)), float(np.mean(mean_errs))
    gap_pct = 100.0 * (w2_mean - w2_bary) / w2_mean if w2_mean > 0 else 0.0

    return VideoResult(
        video_id=video_id, native_spread=spread,
        w2_barycenter=w2_bary, w2_mean=w2_mean,
        gap_pct=gap_pct, bary_wins=w2_bary < w2_mean,
    )


# --------------------------------------------------------------------------- #
# Bucketed summary + significance test per bucket
# --------------------------------------------------------------------------- #
def summarize_by_disagreement_tercile(results: list[VideoResult]) -> list[dict]:
    spreads = np.array([r.native_spread for r in results])
    order = np.argsort(spreads)
    n = len(results)
    third = n // 3
    buckets = {
        "Low native disagreement": [results[i] for i in order[:third]],
        "Medium native disagreement": [results[i] for i in order[third:2 * third]],
        "High native disagreement": [results[i] for i in order[2 * third:]],
    }

    summary = []
    for label, bucket in buckets.items():
        bary = np.array([r.w2_barycenter for r in bucket])
        mean = np.array([r.w2_mean for r in bucket])
        wins = sum(r.bary_wins for r in bucket)
        row = {
            "bucket": label,
            "n_videos": len(bucket),
            "mean_native_spread": float(np.mean([r.native_spread for r in bucket])),
            "w2_barycenter": float(bary.mean()),
            "w2_mean": float(mean.mean()),
            "gap_pct": float(np.mean([r.gap_pct for r in bucket])),
            "win_rate": f"{wins}/{len(bucket)}",
        }
        if _HAVE_SCIPY and len(bucket) > 1:
            try:
                _, p_t = ttest_rel(bary, mean)
                _, p_w = wilcoxon(bary, mean)
                row["p_ttest"] = float(p_t)
                row["p_wilcoxon"] = float(p_w)
            except ValueError:
                row["p_ttest"] = None
                row["p_wilcoxon"] = None
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-videos", type=int, default=50,
                         help="Number of videos to use if falling back to synthetic data.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/native_disagreement_benchmark.csv")
    args = parser.parse_args()

    videos = _load_real_tvsum_videos()
    used_real = videos is not None
    if videos is None:
        print("[native_disagreement_eval] Real dataset modules not found in this "
              "environment; using synthetic fallback so the script is runnable. "
              "On your machine (with dataset.py / download_tvsum.py importable) "
              "this will automatically use the real 50-video TVSum set instead.",
              file=sys.stderr)
        videos = _synthetic_fallback_videos(args.n_videos, args.seed)

    split_fn = _get_split_fn()
    wb_fn, am_fn = _get_barycenter_fns()
    w2_fn = _get_w2_fn()

    results = [
        evaluate_video(vid, mat, args.seed, split_fn, wb_fn, am_fn, w2_fn)
        for vid, mat in videos.items()
    ]

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "native_spread", "w2_barycenter", "w2_mean",
                          "gap_pct", "bary_wins"])
        for r in results:
            writer.writerow([r.video_id, r.native_spread, r.w2_barycenter,
                              r.w2_mean, r.gap_pct, r.bary_wins])
    print(f"Per-video results written to {args.out} "
          f"(using {'real' if used_real else 'synthetic fallback'} data)")

    summary = summarize_by_disagreement_tercile(results)
    print("\nNative-disagreement tercile summary "
          "(tests the sharpening ablation's prediction on real, un-sharpened data):\n")
    header = f"{'Bucket':<26}{'n':>4}{'MeanSpread':>13}{'W2 Bary':>10}{'W2 Mean':>10}{'Gap%':>8}{'WinRate':>9}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['bucket']:<26}{row['n_videos']:>4}{row['mean_native_spread']:>13.4f}"
              f"{row['w2_barycenter']:>10.4f}{row['w2_mean']:>10.4f}{row['gap_pct']:>8.2f}"
              f"{row['win_rate']:>9}")
        if "p_wilcoxon" in row:
            print(f"    (paired t-test p={row['p_ttest']:.4g}, "
                  f"Wilcoxon p={row['p_wilcoxon']:.4g})")

    gaps = [row["gap_pct"] for row in summary]
    monotonic = all(gaps[i] <= gaps[i + 1] for i in range(len(gaps) - 1))
    print(f"\nBarycenter advantage monotonically increases with native "
          f"disagreement across buckets: {monotonic}")
    if not monotonic:
        print("NOTE: this directly reports a negative/mixed result if found -- "
              "do not treat non-monotonicity as a bug to be papered over; it is "
              "itself the finding this script exists to check for.")


if __name__ == "__main__":
    main()
