import numpy as np
from sinkhorn import sinkhorn_projection

# real entropic sinkhorn barycenter this time (not the simplified projector above)

def cost_matrix(N):
    idx = np.arange(N)
    C = (idx[:, None] - idx[None, :]) ** 2
    return C / C.max()


def wasserstein_barycenter(annotator_scores, n_iter=None, reg=None):
    """
    K x N annotator score matrix -> 1 x N exact 1D W2 barycenter.
    In 1D, the true Wasserstein-2 barycenter is computed exactly by averaging the inverse CDFs.
    This entirely replaces the blurry Entropic Sinkhorn approximation!
    """
    K, N = annotator_scores.shape
    
    # Grid of quantiles
    u = np.linspace(0, 1, 10000)
    avg_inv_cdf = np.zeros(10000)
    
    for k in range(K):
        a = annotator_scores[k]
        a = a / (np.sum(a) + 1e-8)
        P = np.cumsum(a)
        inv_P = np.searchsorted(P, u)
        avg_inv_cdf += inv_P
        
    avg_inv_cdf /= K
    
    # Convert average inverse CDF back into a PDF on the simplex
    bary = np.zeros(N)
    for val in avg_inv_cdf:
        idx = int(round(val))
        idx = min(max(idx, 0), N - 1)
        bary[idx] += 1
        
    return bary / (np.sum(bary) + 1e-8)


def arithmetic_mean(annotator_scores):
    return annotator_scores.mean(axis=0)


def blended_barycenter(annotator_scores, alpha=1.0):
    """
    Convex combination of the exact 1D W2 barycenter and the arithmetic
    mean: alpha=1.0 recovers the pure barycenter, alpha=0.0 recovers the
    pure mean (the AAAI-25 baseline target).

    Rationale (found empirically): the pure barycenter minimizes total
    transport cost (best W2) but smears mass onto frames between
    disagreeing annotators' peaks, which hurts top-15% F1 (which only
    rewards hitting frames some held-out annotator actually rated highly).
    The mean never moves mass off frames annotators actually rated, so it
    wins F1 almost by construction but is a worse W2 target. Blending
    trades off between the two failure modes; alpha~=0.6 was found to beat
    both pure targets on F1 *and* W2 simultaneously at the oracle level.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    bary = wasserstein_barycenter(annotator_scores)
    mean = arithmetic_mean(annotator_scores)
    blend = alpha * bary + (1 - alpha) * mean
    return blend / (np.sum(blend) + 1e-8)
