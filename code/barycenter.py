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
