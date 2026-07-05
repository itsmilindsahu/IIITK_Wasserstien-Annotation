"""
train.py - phase 1 sprint code (proof of concept)

TVSum isn't downloadable in this environment so this uses synthetic
toy annotator data that mimics the two-camp disagreement setup from
Figure 1 of the proposal (some annotators like the "pets" segment,
some like the "products" segment). good enough to sanity check the
non-regression claim, not a real benchmark run - that's phase 2.
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


def makeToyVideo(N=30, K=5, disagree=True):
    scores = []
    if disagree:
        campA = np.zeros(N)
        campA[5:10] = 1
        campB = np.zeros(N)
        campB[20:25] = 1
        for k in range(K):
            base = campA if k % 2 == 0 else campB
            noisy = base + np.random.rand(N) * 0.3
            noisy = np.clip(noisy, 1e-6, None)
            scores.append(noisy / noisy.sum())
    else:
        base = np.random.rand(N)
        for k in range(K):
            noisy = base + np.random.rand(N) * 0.05
            noisy = np.clip(noisy, 1e-6, None)
            scores.append(noisy / noisy.sum())
    return np.array(scores)


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
            loss, mse, w2 = wasserstein_loss(eps, eps_hat, eps_hat, annotator_scores, lam=0.1)
        else:
            eps_hat = net.forward(xt, t)
            mse = np.mean((eps - eps_hat) ** 2)
            loss = mse
            w2 = 0.0

        net.step(xt, t, eps)
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
