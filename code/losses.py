import numpy as np


def w2_1d(p, q):
    # True 1D Wasserstein-2 distance using the inverse CDF (quantile) method.
    p = p / (np.sum(p) + 1e-8)
    q = q / (np.sum(q) + 1e-8)
    P = np.cumsum(p)
    Q = np.cumsum(q)
    
    u = np.linspace(0, 1, 1000)
    inv_P = np.searchsorted(P, u)
    inv_Q = np.searchsorted(Q, u)
    return np.mean((inv_P - inv_Q) ** 2)


def wasserstein_loss(eps, eps_hat, x0_hat, annotator_scores, lam=0.1):
    mse = np.mean((eps - eps_hat) ** 2)
    w2_terms = [w2_1d(x0_hat, a) for a in annotator_scores]
    w2 = np.mean(w2_terms)
    total = mse + lam * w2
    return total, mse, w2
