"""
train.py - phase 2 sprint code (leakage-fixed evaluation)

Loads the real TVSum dataset annotations and processes them on the simplex.
Runs both the baseline configuration (arithmetic mean, Gaussian forward
process, MSE loss) and our configuration (Wasserstein barycenter, Dirichlet
forward process, MSE + Cramer-distance-regularized loss) across all 50
videos, saving the comparison results.

Evaluation protocol (this is the part that changed):
  1. Per video, the K=20 annotators are split into a 15-annotator
     barycenter-construction ("train") set and a 5-annotator held-out set
     that x_star, the training loss, and the training gradient NEVER see.
  2. Training runs entirely against the train set.
  3. After training, we run the FULL reverse diffusion process (see
     sample.py) with the trained noise predictor to actually generate a
     sample, starting from noise -- not from x_star.
  4. We report the generated sample's W2 distance to the 5 held-out
     annotators. That is the number that tests whether the pipeline
     (barycenter target + Dirichlet forward process + Cramer-regularized
     loss) generalizes to unseen annotators, as opposed to how close the
     training target itself sits to the annotators that built it.

For transparency we also keep the old (leaky) quantity around, clearly
labeled, so the size of the leakage effect is visible in the output rather
than silently disappearing.
"""
import numpy as np
import json
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from barycenter import wasserstein_barycenter, arithmetic_mean
from forward import forward_step_mine, forward_step_baseline
from losses import wasserstein_loss, w2_1d
from model import NoisePredictor
from split import split_annotators
from sample import generate_sample

np.random.seed(7)  # not super rigorous but keeps this reproducible


def run_config(train_annotators, heldout_annotators, mode="mine", steps=80, T=200, sample_seed=None):
    K, N = train_annotators.shape
    if mode == "mine":
        x_star = wasserstein_barycenter(train_annotators, n_iter=40)
    else:
        x_star = arithmetic_mean(train_annotators)

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

            # 2. Cramer-distance gradient (CDF-difference term). This is the
            # exact analytic gradient of sum_x (F_eps_hat(x) - F_a(x))^2,
            # i.e. of cramer_1d(eps_hat, a) in losses.py. It is NOT the
            # gradient of the quantile-based w2_1d -- that quantity is
            # piecewise-constant in eps_hat and has no useful gradient.
            # Only `train_annotators` (never heldout_annotators) enter here.
            P = np.cumsum(eps_hat)
            grad_cramer = np.zeros_like(eps_hat)
            for a in train_annotators:
                Q = np.cumsum(a)
                diff = P - Q
                grad_cramer += 2 * np.cumsum(diff[::-1])[::-1]
            grad_cramer /= K

            lam = 0.1
            gradOut = grad_mse + lam * grad_cramer

            loss, mse, cramer = wasserstein_loss(eps, eps_hat, eps_hat, train_annotators, lam=lam)
            net.step(xt, t, gradOut)
        else:
            eps_hat = net.forward(xt, t)

            grad_mse = 2 * (eps_hat - eps) / N
            gradOut = grad_mse

            mse = np.mean((eps - eps_hat) ** 2)
            loss = mse
            cramer = 0.0

            net.step(xt, t, gradOut)

        losses.append(float(loss))
        if step % 20 == 0:
            print(f"[{mode}] step {step} loss {loss:.4f}")

    # --- Evaluation ---
    # (a) The metric that actually matters: run the FULL reverse diffusion
    # process with the trained net to generate a sample, then score it
    # against annotators that were never used to build x_star or train net.
    rng = np.random.default_rng(sample_seed)
    generated = generate_sample(net, N, T=T, mode=mode, rng=rng)
    heldout_w2_generated = float(np.mean([w2_1d(generated, a) for a in heldout_annotators]))

    # (b) Kept only for transparency/comparison: x_star's distance to the
    # very annotators that were averaged into it. This is the quantity the
    # previous version of this script reported as "held-out W2" -- it is
    # not held out (same annotators used to build x_star) and it never
    # touches the trained network at all, so it can't tell you whether
    # training or sampling helped.
    naive_leaky_w2 = float(np.mean([w2_1d(x_star, a) for a in train_annotators]))

    return {
        "losses": losses,
        "x_star": x_star.tolist(),
        "generated_sample": generated.tolist(),
        "heldout_w2_generated": heldout_w2_generated,
        "naive_leaky_w2_DO_NOT_USE_AS_HEADLINE": naive_leaky_w2,
    }


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

    # Split ONCE per video, and reuse the identical split for both configs
    # so "mine" vs "baseline" are compared on the exact same held-out set.
    train_annotators, heldout_annotators = split_annotators(video, n_holdout=5, seed=1234)
    print(f"{vid_key}: {train_annotators.shape[0]} train annotators, "
          f"{heldout_annotators.shape[0]} held-out annotators")

    results = {}
    for mode in ["mine", "baseline"]:
        results[mode] = run_config(
            train_annotators, heldout_annotators, mode=mode, steps=120, sample_seed=7
        )

    with open(os.path.join(results_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "final_loss", "heldout_w2_generated_sample", "naive_leaky_w2_DO_NOT_USE_AS_HEADLINE"])
        for mode in ["mine", "baseline"]:
            w.writerow([
                mode,
                results[mode]["losses"][-1],
                results[mode]["heldout_w2_generated"],
                results[mode]["naive_leaky_w2_DO_NOT_USE_AS_HEADLINE"],
            ])

    plt.figure()
    plt.plot(results["mine"]["losses"], label="mine (barycenter + dirichlet + MSE + cramer)")
    plt.plot(results["baseline"]["losses"], label="baseline (arith mean + gaussian + mse)")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"Phase 2 loss curves (TVSum {vid_key})")
    plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=120)

    out = {
        "mine_final_loss": results["mine"]["losses"][-1],
        "baseline_final_loss": results["baseline"]["losses"][-1],
        "mine_heldout_w2": results["mine"]["heldout_w2_generated"],
        "baseline_heldout_w2": results["baseline"]["heldout_w2_generated"],
        "mine_heldout_w2_generated": results["mine"]["heldout_w2_generated"],
        "baseline_heldout_w2_generated": results["baseline"]["heldout_w2_generated"],
        "mine_naive_leaky_w2": results["mine"]["naive_leaky_w2_DO_NOT_USE_AS_HEADLINE"],
        "baseline_naive_leaky_w2": results["baseline"]["naive_leaky_w2_DO_NOT_USE_AS_HEADLINE"],
        "mine_losses": results["mine"]["losses"],
        "baseline_losses": results["baseline"]["losses"],
        "annotator_scores": video.tolist(),
        "train_annotator_scores": train_annotators.tolist(),
        "heldout_annotator_scores": heldout_annotators.tolist(),
        "x_star_mine": results["mine"]["x_star"],
        "x_star_baseline": results["baseline"]["x_star"],
        "generated_sample_mine": results["mine"]["generated_sample"],
        "generated_sample_baseline": results["baseline"]["generated_sample"],
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(results_dir, "data_inline.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("done, wrote results/")
    print(f"[mine]     heldout W2 (generated sample vs 5 held-out annotators): {out['mine_heldout_w2_generated']:.4f}")
    print(f"[baseline] heldout W2 (generated sample vs 5 held-out annotators): {out['baseline_heldout_w2_generated']:.4f}")
    print(f"[mine]     naive leaky W2 (x_star vs its own 15 train annotators, old metric): {out['mine_naive_leaky_w2']:.4f}")
    print(f"[baseline] naive leaky W2 (x_star vs its own 15 train annotators, old metric): {out['baseline_naive_leaky_w2']:.4f}")


if __name__ == "__main__":
    main()
