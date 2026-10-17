import numpy as np


def split_annotators(distributions, n_holdout=5, seed=None):
    """
    Split the K annotators for one video into a barycenter-construction
    (train) set and a held-out set that is never touched when computing
    x_star, never touched by the training loss/gradient, and never touched
    until final evaluation.

    distributions: (K, N) array, one row per annotator.
    n_holdout: number of annotators to hold out (default 5, i.e. 15/5 for
        TVSum's K=20 annotators-per-video).
    seed: optional int/np.random.Generator seed for reproducibility. Pass a
        per-video seed (e.g. hash of the video id) if you want a different,
        but reproducible, split per video rather than the same 5 indices
        held out every time.

    Returns:
        train, heldout: two (k, N) arrays, k = K - n_holdout and n_holdout.
    """
    K = distributions.shape[0]
    if n_holdout >= K:
        raise ValueError(f"n_holdout={n_holdout} must be < number of annotators K={K}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    heldout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    return distributions[train_idx], distributions[heldout_idx]
