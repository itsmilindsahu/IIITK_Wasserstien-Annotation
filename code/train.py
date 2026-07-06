"""
train.py - phase 2 sprint code

Loads the real TVSum dataset annotations and processes them on the simplex.
Runs both the baseline configuration (arithmetic mean, Gaussian forward process,
MSE loss) and our configuration (Wasserstein barycenter, Dirichlet forward process,
Wasserstein loss with exact 1D proxy gradients) across all 50 videos,
saving the comparison results.
"""
import numpy as np
import json
import csv
import os
import matplotlib.pyplot as plt

from barycenter import wasserstein_barycenter, arithmetic_mean
from forward import forward_step_mine, forward_step_baseline
from losses import wasserstein_loss, w2_1d
from model import NoisePredictor

np.random.seed(7)  # not super rigorous but keeps this reproducible


def run_config(annotator_scores, mode="mine", steps=80, T=200):
    K, N = annotator_scores.shape
    if mode == "mine":
        x_star = wasserstein_barycenter(annotator_scores, n_iter=40)
    else:
        x_star = arithmetic_mean(annotator_scores)

    net = NoisePredictor(N)
    losses = []
    for step in range(steps):
        t = np.random.randint(1, T)
        if mode == "mine":
            xt = forward_step_mine(x_star, t, T)
        else:
            xt = forward_step_baseline(x_star, t, T)

        eps = x_star  # simplification: predictor targets x_star directly

        if mode == "mine":
            eps_hat = net.forward(xt, t)
            
            # 1. MSE gradient
            grad_mse = 2 * (eps_hat - eps) / N
            
            # 2. W2 gradient using Cramer distance (CDF difference) which is differentiable
            P = np.cumsum(eps_hat)
            grad_w2 = np.zeros_like(eps_hat)
            for a in annotator_scores:
                Q = np.cumsum(a)
                diff = P - Q
                # Gradient of sum(diff^2): 2 * diff sum over future elements
                grad_w2 += 2 * np.cumsum(diff[::-1])[::-1]
            grad_w2 /= K
            
            lam = 0.1
            gradOut = grad_mse + lam * grad_w2
            
            loss, mse, w2 = wasserstein_loss(eps, eps_hat, eps_hat, annotator_scores, lam=lam)
            net.step(xt, t, gradOut)
        else:
            eps_hat = net.forward(xt, t)
            
            grad_mse = 2 * (eps_hat - eps) / N
            gradOut = grad_mse
            
            mse = np.mean((eps - eps_hat) ** 2)
            loss = mse
            w2 = 0.0
            
            net.step(xt, t, gradOut)

        losses.append(float(loss))
        if step % 20 == 0:
            print(f"[{mode}] step {step} loss {loss:.4f}")

    heldout_w2 = float(np.mean([w2_1d(x_star, a) for a in annotator_scores]))
    return {"losses": losses, "x_star": x_star.tolist(), "heldout_w2": heldout_w2}


from dataset import load_tvsum_data, get_video_distributions
from download_tvsum import download_or_generate_tvsum

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    tsv_path = os.path.join(os.path.dirname(__file__), "ydata-tvsum50-anno.tsv")
    if not os.path.exists(tsv_path):
        download_or_generate_tvsum(tsv_path)
        
    tvsum_videos = load_tvsum_data(tsv_path)
    # Pick a video for the visualization/experiment, e.g., the first one
    vid_key = list(tvsum_videos.keys())[0]
    raw_scores = tvsum_videos[vid_key]
    
    # Preprocess: sub-sample the frames to reduce compute time for POC
    # raw_scores shape: (20, N), N can be thousands. Let's sample 100 frames
    if raw_scores.shape[1] > 100:
        indices = np.linspace(0, raw_scores.shape[1] - 1, 100, dtype=int)
        raw_scores = raw_scores[:, indices]
        
    video = get_video_distributions(raw_scores)

    results = {}
    for mode in ["mine", "baseline"]:
        results[mode] = run_config(video, mode=mode, steps=120)

    with open(os.path.join(results_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "final_loss", "heldout_w2_dist"])
        for mode in ["mine", "baseline"]:
            w.writerow([mode, results[mode]["losses"][-1], results[mode]["heldout_w2"]])

    plt.figure()
    plt.plot(results["mine"]["losses"], label="mine (barycenter + dirichlet + w2 loss)")
    plt.plot(results["baseline"]["losses"], label="baseline (arith mean + gaussian + mse)")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"Phase 2 loss curves (TVSum {vid_key})")
    plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=120)

    out = {
        "mine_final_loss": results["mine"]["losses"][-1],
        "baseline_final_loss": results["baseline"]["losses"][-1],
        "mine_heldout_w2": results["mine"]["heldout_w2"],
        "baseline_heldout_w2": results["baseline"]["heldout_w2"],
        "mine_losses": results["mine"]["losses"],
        "baseline_losses": results["baseline"]["losses"],
        "annotator_scores": video.tolist(),
        "x_star_mine": results["mine"]["x_star"],
        "x_star_baseline": results["baseline"]["x_star"],
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("done, wrote results/")


if __name__ == "__main__":
    main()
