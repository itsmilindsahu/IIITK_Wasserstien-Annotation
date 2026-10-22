"""
core/split.py

Leak-free annotator split: holds out n_holdout annotators per video so that
the training target (barycenter / mean), the loss, and the gradient never
see the held-out annotators.  Only the final evaluation metric uses them.
"""
import numpy as np


def split_annotators(distributions, n_holdout=5, seed=None):
    """
    Split K annotators into train and held-out sets.

    Parameters
    ----------
    distributions : (K, N) array — one row per annotator.
    n_holdout     : number of annotators to hold out.
    seed          : RNG seed (use a stable per-video seed for reproducibility).

    Returns
    -------
    train, heldout : (K-n, N) and (n, N) arrays.
    """
    distributions = np.asarray(distributions, dtype=float)
    K = distributions.shape[0]
    if n_holdout >= K:
        raise ValueError(f"n_holdout={n_holdout} must be < K={K}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    heldout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    return distributions[train_idx], distributions[heldout_idx]
