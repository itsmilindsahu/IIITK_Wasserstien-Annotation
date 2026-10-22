"""
data/synthetic_dataset.py

Generates a synthetic multi-annotator video annotation dataset that faithfully
mimics the statistical properties of TVSum:

  • 50 synthetic "videos", each with N=100 temporal segments.
  • K=20 annotators per video, 1-5 integer importance scores.
  • Multimodal structure: each video has 2–4 "highlight peaks" drawn from
    different Gaussian bumps; annotators disagree about which peak is the most
    salient one.  This creates genuine inter-annotator disagreement that drives
    the Wasserstein-barycenter vs arithmetic-mean gap.

No real video content, no network access, no disk I/O of large files needed.

Usage
-----
    from data.synthetic_dataset import generate_synthetic_tvsum
    videos = generate_synthetic_tvsum()   # dict: video_id -> (20, 100) int array
"""
import numpy as np
import hashlib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_seed(vid: str) -> int:
    """Deterministic, PYTHONHASHSEED-independent per-video integer seed."""
    digest = hashlib.md5(vid.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2 ** 31 - 1)


def _gaussian_bump(N, centre, sigma, amplitude):
    """1-D Gaussian bump on {0, …, N-1}."""
    x = np.arange(N, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def _generate_one_video(rng, N=100, K=20, n_peaks=3):
    """
    Generate one (K, N) synthetic annotation matrix.

    Each annotator gets a personalised preference weight over the n_peaks
    peaks, so different annotators "care about" different parts of the video.
    Scores are rounded to integers in [1, 5] as in TVSum.
    """
    # Global peak positions and widths, shared across annotators for this video
    centres = rng.uniform(10, N - 10, size=n_peaks)
    sigmas = rng.uniform(5, 18, size=n_peaks)

    annotators = []
    for _ in range(K):
        # Each annotator assigns different saliency to each peak
        weights = rng.dirichlet(np.ones(n_peaks) * 0.8)  # sparse preferences
        curve = np.zeros(N)
        for c, s, w in zip(centres, sigmas, weights):
            curve += _gaussian_bump(N, c, s, w)

        # Add mild annotator-specific noise
        curve += rng.exponential(0.05, size=N)

        # Map to integer scores 1–5 (TVSum range)
        lo, hi = curve.min(), curve.max()
        if hi > lo:
            normed = (curve - lo) / (hi - lo)
        else:
            normed = np.ones(N) * 0.5
        scores = np.round(1 + normed * 4).astype(int)
        scores = np.clip(scores, 1, 5)
        annotators.append(scores)

    return np.stack(annotators, axis=0)  # (K, N)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_tvsum(n_videos=50, N=100, K=20, seed=42):
    """
    Generate a synthetic TVSum-style dataset.

    Parameters
    ----------
    n_videos : number of synthetic videos (default 50)
    N        : number of temporal segments per video (default 100)
    K        : annotators per video (default 20)
    seed     : global master seed for reproducibility

    Returns
    -------
    dict mapping video_id (str) to (K, N) int32 numpy array of scores in [1,5]
    """
    master_rng = np.random.default_rng(seed)
    videos = {}
    for i in range(1, n_videos + 1):
        vid = f"video_{i:02d}"
        vid_seed = _stable_seed(vid) ^ int(master_rng.integers(0, 2 ** 30))
        rng = np.random.default_rng(vid_seed)
        n_peaks = rng.integers(2, 5)  # 2 to 4 peaks
        videos[vid] = _generate_one_video(rng, N=N, K=K, n_peaks=n_peaks)
    return videos


def get_video_distributions(video_scores, method="softmax", temperature=0.5):
    """
    Convert 1-5 importance scores -> probability distributions over frames.

    Parameters
    ----------
    video_scores : (K, N) int/float array of raw scores
    method       : "normalize" | "softmax" | "zscore_exp"
    temperature  : sharpening temperature (lower = sharper, for softmax/zscore_exp)

    Returns
    -------
    (K, N) float array, each row sums to 1.
    """
    S = np.asarray(video_scores, dtype=float)

    if method == "normalize":
        row_sums = S.sum(axis=1, keepdims=True)
        return S / np.where(row_sums > 0, row_sums, 1.0)

    elif method == "softmax":
        S = S / temperature
        S -= S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    elif method == "zscore_exp":
        mu = S.mean(axis=1, keepdims=True)
        sigma = S.std(axis=1, keepdims=True) + 1e-8
        Z = (S - mu) / sigma / temperature
        Z -= Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown method: {method!r}")


def stable_seed(vid: str) -> int:
    """Public alias for _stable_seed (used by stage scripts)."""
    return _stable_seed(vid)
