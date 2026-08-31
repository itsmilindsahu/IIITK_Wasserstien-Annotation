import numpy as np


def w2_1d(p, q, n_grid=1000):
    """
    True 1D Wasserstein-2 distance (squared), via the inverse-CDF / quantile
    method: W2^2(p, q) = E_u[(F_p^{-1}(u) - F_q^{-1}(u))^2].

    This is the metric we report for EVALUATION (barycenter quality,
    held-out distance, etc). It is *not* used to drive gradients: the
    quantile function F^{-1}(u) = argmin{x : F(x) >= u} is piecewise
    constant in p (it comes from `np.searchsorted`), so d(F_p^{-1})/dp is
    zero almost everywhere and undefined on the jumps. Backpropagating
    "through" this function, as an earlier version of this code did, is
    not computing the gradient of this quantity -- see `cramer_1d` below
    for what is actually differentiable and used during training.
    """
    p = p / (np.sum(p) + 1e-8)
    q = q / (np.sum(q) + 1e-8)
    P = np.cumsum(p)
    Q = np.cumsum(q)

    u = np.linspace(0, 1, n_grid)
    inv_P = np.searchsorted(P, u)
    inv_Q = np.searchsorted(Q, u)
    return float(np.mean((inv_P - inv_Q) ** 2))


def cramer_1d(p, q):
    """
    Squared (discrete, un-normalized) Cramer distance between p and q:

        C(p, q) = sum_x (F_p(x) - F_q(x))^2,   F = cumulative sum (CDF).

    This is what the training loss/gradient in train.py actually
    optimizes. It is related to Wasserstein distance in spirit (both are
    CDF-matching objectives, and for 1D distributions the Cramer distance
    upper-bounds the W1 distance), but it is NOT the same quantity as
    `w2_1d` above -- squaring inside the sum over x rather than in the
    quantile domain changes what gets penalized (e.g. it is far more
    sensitive to *where* along the support two CDFs disagree than true W2
    is). We keep it because, unlike the quantile-based W2, it is exactly
    differentiable in closed form:

        d/dp_i sum_x (F_p(x) - F_q(x))^2 = 2 * sum_{x >= i} (F_p(x) - F_q(x))

    See train.py for the corresponding analytic gradient.
    """
    p = p / (np.sum(p) + 1e-8)
    q = q / (np.sum(q) + 1e-8)
    P = np.cumsum(p)
    Q = np.cumsum(q)
    return float(np.sum((P - Q) ** 2))


def wasserstein_loss(eps, eps_hat, x0_hat, annotator_scores, lam=0.1):
    """
    NOTE ON NAMING: despite the historical name, the regularizer term here
    is the Cramer distance (`cramer_1d`), not true W2 -- it's the term
    that is actually differentiable and whose gradient is used to update
    the network (see grad_cramer in train.py). We report the true
    quantile-based W2 (`w2_1d`) separately, as an evaluation-only metric,
    and never backprop through it. Kept the function name for backwards
    compatibility with existing imports; the returned tuple now also
    surfaces both quantities explicitly.
    """
    mse = np.mean((eps - eps_hat) ** 2)
    cramer_terms = [cramer_1d(x0_hat, a) for a in annotator_scores]
    cramer = float(np.mean(cramer_terms))
    total = mse + lam * cramer
    return total, mse, cramer
