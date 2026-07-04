import numpy as np
from sinkhorn import sinkhorn_projection

# real entropic sinkhorn barycenter this time (not the simplified projector above)

def cost_matrix(N):
    idx = np.arange(N)
    C = (idx[:, None] - idx[None, :]) ** 2
    return C / C.max()


def wasserstein_barycenter(annotator_scores, n_iter=40, reg=0.05):
    """K x N annotator score matrix -> 1 x N barycenter on the simplex."""
    K, N = annotator_scores.shape
    C = cost_matrix(N)
    Kmat = np.exp(-C / reg)

    u = np.ones((K, N))
    bary = np.ones(N) / N

    for it in range(n_iter):
        v = annotator_scores / (Kmat.T @ u.T).T.clip(1e-16)
        Ku = (Kmat @ v.T).T
        bary = np.exp(np.mean(np.log(Ku * u + 1e-16), axis=0))
        bary = sinkhorn_projection(bary, n_iter=1)
        u = bary[None, :] / Ku.clip(1e-16)

    return bary


def arithmetic_mean(annotator_scores):
    return annotator_scores.mean(axis=0)
