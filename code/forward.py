import numpy as np
from sinkhorn import sinkhorn_projection


def alpha_bar(t, T=200, s=0.008):
    # standard cosine schedule (same trick as image diffusion models)
    f = lambda tt: np.cos(((tt / T) + s) / (1 + s) * np.pi / 2) ** 2
    return f(t) / f(0)


def dirichlet_noise(N, alpha=1.0):
    return np.random.dirichlet(alpha * np.ones(N))


def forward_step_mine(x_star, t, T=200):
    ab = alpha_bar(t, T)
    delta = dirichlet_noise(len(x_star))
    xt = np.sqrt(ab) * x_star + np.sqrt(1 - ab) * delta
    xt = sinkhorn_projection(xt, n_iter=3)  # this is the Proj_delta step from Eq 4
    return xt


def forward_step_baseline(x_star, t, T=200):
    # original AAAI-25 style gaussian forward process, kept around for comparison
    ab = alpha_bar(t, T)
    noise = np.random.randn(len(x_star)) * 0.1
    xt = np.sqrt(ab) * x_star + np.sqrt(1 - ab) * noise
    xt = np.clip(xt, 0, None)
    s = xt.sum()
    xt = xt / s if s > 0 else np.ones_like(xt) / len(xt)
    return xt
