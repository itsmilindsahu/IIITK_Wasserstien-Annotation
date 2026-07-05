import numpy as np
from losses import w2_1d

def cost_matrix(N):
    idx = np.arange(N)
    C = (idx[:, None] - idx[None, :]) ** 2
    return C / C.max()

def wasserstein_barycenter_fix(annotator_scores, n_iter=40, reg=0.05):
    K, N = annotator_scores.shape
    C = cost_matrix(N)
    Kmat = np.exp(-C / reg)

    v = np.ones((K, N))
    
    for it in range(n_iter):
        u = annotator_scores / (Kmat @ v.T).T.clip(1e-16)
        KTu = (Kmat.T @ u.T).T
        a = np.exp(np.mean(np.log(KTu + 1e-16), axis=0))
        v = a[None, :] / KTu.clip(1e-16)

    return a / a.sum()

def wasserstein_barycenter_old(annotator_scores, n_iter=40, reg=0.05):
    K, N = annotator_scores.shape
    C = cost_matrix(N)
    Kmat = np.exp(-C / reg)

    u = np.ones((K, N))
    bary = np.ones(N) / N

    for it in range(n_iter):
        v = annotator_scores / (Kmat.T @ u.T).T.clip(1e-16)
        Ku = (Kmat @ v.T).T
        bary = np.exp(np.mean(np.log(Ku * u + 1e-16), axis=0))
        bary = bary / bary.sum()
        u = bary[None, :] / Ku.clip(1e-16)

from dataset import load_tvsum_data, get_video_distributions
import os

def true_1d_barycenter(annotators):
    # Average the inverse CDFs (quantiles)
    u = np.linspace(0, 1, 1000)
    avg_inv_cdf = np.zeros(1000)
    
    for a in annotators:
        P = np.cumsum(a)
        inv_P = np.searchsorted(P, u)
        avg_inv_cdf += inv_P
    avg_inv_cdf /= len(annotators)
    
    # Convert average inverse CDF back to a PDF (histogram)
    N = annotators.shape[1]
    bary = np.zeros(N)
    
    # Histogram of the inverse CDF values
    for val in avg_inv_cdf:
        idx = int(round(val))
        idx = min(max(idx, 0), N - 1)
        bary[idx] += 1
        
    return bary / bary.sum()

def main():
    tsv_path = os.path.join("code", "ydata-tvsum50-anno.tsv")
    tvsum_videos = load_tvsum_data(tsv_path)
    vid_key = list(tvsum_videos.keys())[0]
    raw_scores = tvsum_videos[vid_key]
    
    if raw_scores.shape[1] > 100:
        indices = np.linspace(0, raw_scores.shape[1] - 1, 100, dtype=int)
        raw_scores = raw_scores[:, indices]
        
    annotators = get_video_distributions(raw_scores)
    
    arith = annotators.mean(axis=0)
    bary_old = wasserstein_barycenter_old(annotators, reg=0.05)
    bary_old2 = wasserstein_barycenter_old(annotators, reg=0.001)
    bary_true = true_1d_barycenter(annotators)
    
    w2_arith = np.mean([w2_1d(arith, a) for a in annotators])
    w2_old = np.mean([w2_1d(bary_old, a) for a in annotators])
    w2_old2 = np.mean([w2_1d(bary_old2, a) for a in annotators])
    w2_true = np.mean([w2_1d(bary_true, a) for a in annotators])
    
    print(f"Arith W2: {w2_arith:.4f}")
    print(f"Bary (reg=0.05) W2: {w2_old:.4f}")
    print(f"Bary (reg=0.001) W2: {w2_old2:.4f}")
    print(f"True 1D Bary W2: {w2_true:.4f}")

if __name__ == "__main__":
    main()
