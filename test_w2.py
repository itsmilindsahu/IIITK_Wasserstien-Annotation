import numpy as np
import os
import matplotlib.pyplot as plt

from barycenter import wasserstein_barycenter, arithmetic_mean
from forward import forward_step_mine, forward_step_baseline
from losses import w2_1d
from model import NoisePredictor
from dataset import load_tvsum_data, get_video_distributions
from download_tvsum import download_or_generate_tvsum

np.random.seed(7)

def run_config(annotator_scores, mode="mine", steps=120, T=200):
    K, N = annotator_scores.shape
    if mode == "mine":
        x_star = wasserstein_barycenter(annotator_scores, n_iter=40)
    else:
        x_star = arithmetic_mean(annotator_scores)

    net = NoisePredictor(N)
    w2_history = []
    
    for step in range(steps):
        t = np.random.randint(1, T)
        if mode == "mine":
            xt = forward_step_mine(x_star, t, T)
        else:
            xt = forward_step_baseline(x_star, t, T)

        eps = x_star

        if mode == "mine":
            eps_hat = net.forward(xt, t)
            
            grad_mse = 2 * (eps_hat - eps) / N
            
            P = np.cumsum(eps_hat)
            grad_w2 = np.zeros_like(eps_hat)
            for a in annotator_scores:
                Q = np.cumsum(a)
                diff = P - Q
                grad_w2 += 2 * np.cumsum(diff[::-1])[::-1]
            grad_w2 /= K
            
            lam = 0.1
            gradOut = grad_mse + lam * grad_w2
            net.step(xt, t, gradOut)
        else:
            eps_hat = net.forward(xt, t)
            grad_mse = 2 * (eps_hat - eps) / N
            net.step(xt, t, grad_mse)

        # Evaluate true W2 against raw annotator scores
        current_w2 = float(np.mean([w2_1d(eps_hat, a) for a in annotator_scores]))
        w2_history.append(current_w2)

    return w2_history

def main():
    tsv_path = os.path.join(os.path.dirname(__file__), "code", "ydata-tvsum50-anno.tsv")
    tvsum_videos = load_tvsum_data(tsv_path)
    vid_key = list(tvsum_videos.keys())[0]
    raw_scores = tvsum_videos[vid_key]
    
    if raw_scores.shape[1] > 100:
        indices = np.linspace(0, raw_scores.shape[1] - 1, 100, dtype=int)
        raw_scores = raw_scores[:, indices]
        
    video = get_video_distributions(raw_scores)

    mine_w2 = run_config(video, mode="mine", steps=150)
    baseline_w2 = run_config(video, mode="baseline", steps=150)

    print(f"Final Mine W2: {mine_w2[-1]:.4f}")
    print(f"Final Baseline W2: {baseline_w2[-1]:.4f}")

if __name__ == "__main__":
    main()
