import numpy as np


def w2_1d(p, q):
    # closed form W2 between two 1D distributions via CDFs, cheap and exact
    P = np.cumsum(p)
    Q = np.cumsum(q)
    return np.sum((P - Q) ** 2)


def wasserstein_loss(eps, eps_hat, x0_hat, annotator_scores, lam=0.1):
    mse = np.mean((eps - eps_hat) ** 2)
    w2_terms = [w2_1d(x0_hat, a) for a in annotator_scores]
    w2 = np.mean(w2_terms)
    total = mse + lam * w2
    return total, mse, w2
