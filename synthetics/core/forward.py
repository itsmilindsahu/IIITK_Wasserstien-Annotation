"""
core/forward.py

Forward diffusion processes:

  forward_step_mine     — Simplex-preserving Dirichlet forward process (MSPDM).
  forward_step_baseline — Gaussian forward process clipped to simplex (AAAI-25 baseline).
  alpha_bar             — Cosine noise schedule ᾱ_t.
  dirichlet_noise       — Sample from Dir(alpha·1).
"""
import numpy as np
from core.sinkhorn import sinkhorn_projection


def alpha_bar(t, T=200, s=0.008):
    """Cosine noise schedule:  ᾱ_t = f(t)/f(0),  f(t) = cos^2(π/2 · (t/T + s)/(1 + s))."""
    f = lambda tt: np.cos(((tt / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
    return f(t) / f(0)


def dirichlet_noise(N, alpha=1.0, rng=None):
    """Sample a Dirichlet(alpha·1_N) random vector on the simplex."""
    rng = rng or np.random.default_rng()
    return rng.dirichlet(alpha * np.ones(N))


def forward_step_mine(x_star, t, T=200, rng=None):
    """
    MSPDM simplex-preserving forward step:

        x_t = sqrt(ᾱ_t)·x* + sqrt(1 − ᾱ_t)·δ,   δ ~ Dir(1)
        followed by Proj_Delta (clip + renormalise).
    """
    rng = rng or np.random.default_rng()
    ab = alpha_bar(t, T)
    delta = dirichlet_noise(len(x_star), rng=rng)
    xt = np.sqrt(ab) * x_star + np.sqrt(1.0 - ab) * delta
    return sinkhorn_projection(xt, n_iter=3)


def forward_step_baseline(x_star, t, T=200, rng=None):
    """
    Euclidean (AAAI-25) baseline forward step:

        x_t = sqrt(ᾱ_t)·x* + sqrt(1 − ᾱ_t)·eps,   eps ~ N(0, 0.1^2·I)
        followed by clip(·, 0) + L1-renormalise.
    """
    rng = rng or np.random.default_rng()
    ab = alpha_bar(t, T)
    noise = rng.standard_normal(len(x_star)) * 0.1
    xt = np.sqrt(ab) * x_star + np.sqrt(1.0 - ab) * noise
    xt = np.clip(xt, 0.0, None)
    s = xt.sum()
    return xt / s if s > 0 else np.ones_like(xt) / len(xt)
