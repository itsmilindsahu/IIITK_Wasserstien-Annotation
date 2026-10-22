"""
core/barycenter.py

Exact 1-D Wasserstein-2 barycenter (Monge / quantile-averaging formula) and
helpers.  No Sinkhorn iteration is needed in 1-D: the barycenter is computed
by averaging the inverse CDFs (quantile functions) of all annotator
distributions, then reading the result back as a histogram.

References
----------
Agueh & Carlier (2011): "Barycenters in the Wasserstein Space."
"""
import numpy as np
from core.sinkhorn import sinkhorn_projection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wasserstein_barycenter(annotator_scores, n_quantiles=10_000, **_kwargs):
    """
    K x N annotator score matrix  ->  1-D exact W2 barycenter on Delta^{N-1}.

    Algorithm
    ---------
    1. Normalise each row to a probability vector.
    2. Build each annotator's CDF F_k.
    3. Average the inverse CDFs (quantile functions) point-wise.
    4. Bin the average quantile back into an N-histogram and re-normalise.

    Parameters
    ----------
    annotator_scores : (K, N) array-like
    n_quantiles      : resolution of the quantile grid (default 10 000)

    Returns
    -------
    bary : (N,) ndarray, sums to 1.
    """
    annotator_scores = np.asarray(annotator_scores, dtype=float)
    K, N = annotator_scores.shape

    u = np.linspace(0.0, 1.0, n_quantiles)
    avg_inv_cdf = np.zeros(n_quantiles)

    for k in range(K):
        a = annotator_scores[k]
        a = a / (a.sum() + 1e-8)
        cdf = np.cumsum(a)
        avg_inv_cdf += np.searchsorted(cdf, u)

    avg_inv_cdf /= K

    # Convert average quantile -> histogram on {0, …, N-1}
    bary = np.zeros(N)
    for val in avg_inv_cdf:
        idx = int(round(val))
        bary[np.clip(idx, 0, N - 1)] += 1

    return bary / (bary.sum() + 1e-8)


def arithmetic_mean(annotator_scores):
    """Row-mean of the annotator matrix (Euclidean baseline target)."""
    return np.asarray(annotator_scores, dtype=float).mean(axis=0)


def blended_barycenter(annotator_scores, alpha=1.0):
    """
    Convex blend between the exact W2 barycenter (alpha=1) and the arithmetic
    mean (alpha=0).  alpha≈0.6 empirically optimises both W2 and F1 simultaneously.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    bary = wasserstein_barycenter(annotator_scores)
    mean = arithmetic_mean(annotator_scores)
    blend = alpha * bary + (1.0 - alpha) * mean
    return blend / (blend.sum() + 1e-8)
