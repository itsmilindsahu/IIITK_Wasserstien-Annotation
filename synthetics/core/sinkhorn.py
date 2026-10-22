"""
core/sinkhorn.py
Simplex projection (clip + renormalize), called Proj_Delta in the paper.
"""
import numpy as np


def sinkhorn_projection(x, n_iter=5):
    """Project vector x onto the probability simplex."""
    x = np.array(x, dtype=float)
    for _ in range(n_iter):
        x = np.clip(x, 1e-8, None)
        x = x / x.sum()
    return x
