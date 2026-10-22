"""
core/losses.py

Loss functions and evaluation metrics.

w2_1d       — true 1-D Wasserstein-2 distance (evaluation only, not differentiable).
cramer_1d   — Cramér / CDF-squared distance (differentiable training loss).
wasserstein_loss — combined MSE + λ·Cramér objective used during training.
grad_cramer — analytic gradient of cramer_1d w.r.t. the predicted distribution p.
grad_mse    — gradient of MSE objective.
"""
import numpy as np


def w2_1d(p, q, n_grid=1_000):
    """
    Squared 1-D Wasserstein-2 distance via the quantile (inverse-CDF) formula:

        W2^2(p, q) = E_u[ (F_p⁻¹(u) − F_q⁻¹(u))^2 ]

    Used ONLY for evaluation; the gradient of the searchsorted is zero almost
    everywhere and is not used for training.
    """
    p = np.asarray(p, float); p = p / (p.sum() + 1e-8)
    q = np.asarray(q, float); q = q / (q.sum() + 1e-8)
    u = np.linspace(0.0, 1.0, n_grid)
    inv_P = np.searchsorted(np.cumsum(p), u)
    inv_Q = np.searchsorted(np.cumsum(q), u)
    return float(np.mean((inv_P - inv_Q) ** 2))


def cramer_1d(p, q):
    """
    Squared discrete Cramér distance:

        C(p, q) = Σ_x (F_p(x) − F_q(x))^2

    Differentiable in closed form:  d/dp_j = 2 Σ_{x>=j} (F_p(x) − F_q(x))
    """
    p = np.asarray(p, float); p = p / (p.sum() + 1e-8)
    q = np.asarray(q, float); q = q / (q.sum() + 1e-8)
    return float(np.sum((np.cumsum(p) - np.cumsum(q)) ** 2))


def grad_cramer(p_hat, annotators):
    """
    Exact analytic gradient of the mean Cramér distance w.r.t. p_hat.

        ∂/∂p_j  mean_k C(p_hat, a_k)
            = (2/K) Σ_k Σ_{x>=j} (F_{p_hat}(x) − F_{a_k}(x))

    Implemented as a reverse cumsum of the CDF differences.

    Parameters
    ----------
    p_hat     : (N,) predicted distribution (post-softmax)
    annotators: (K, N) training annotator distributions

    Returns
    -------
    grad : (N,) gradient array
    """
    annotators = np.asarray(annotators, float)
    K = len(annotators)
    P = np.cumsum(p_hat)
    g = np.zeros_like(p_hat)
    for a in annotators:
        diff = P - np.cumsum(a)
        g += 2.0 * np.cumsum(diff[::-1])[::-1]
    return g / K


def wasserstein_loss(eps, eps_hat, x0_hat, annotator_scores, lam=0.1):
    """Combined MSE + λ·Cramér objective (returns total, mse, cramer)."""
    mse = float(np.mean((eps - eps_hat) ** 2))
    cramer = float(np.mean([cramer_1d(x0_hat, a) for a in annotator_scores]))
    return mse + lam * cramer, mse, cramer
