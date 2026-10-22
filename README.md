# TVSum annotator-consensus experiments

This repository explores consensus distributions over video-importance
annotations. The earlier checked-in annotation file was synthetic and has been
removed; all proxy-derived scores and comparisons are withdrawn. No dataset
fallback or synthetic data generation is used by the real-data benchmark.

## Real-data benchmark

`code/standard_tvsum_cv.py` evaluates a video-conditioned temporal convolution
model on the official TVSum50 annotations and preprocessed 1024-D GoogLeNet
pool5 features. The loader maps official video IDs to HDF5 groups by checking
raw annotator means against the normalized HDF5 `gtscore` at `picks`, and
validates original-frame counts. It uses the five supplied 80/20 splits in
[`splits/tvsum.yml`](splits/tvsum.yml), reserves a fixed slice of each training
split for validation, and tests only on that split's held-out videos. The DSNet
test sets can overlap and need not cover all videos; the result JSON records
unique test coverage instead of assuming a disjoint five-fold partition. All target,
loss, checkpoint, and blend-alpha choices are made without test-fold scores.

The mandatory ablation grid compares the mean annotation distribution and the
one-dimensional Wasserstein barycenter under MSE, cross-entropy, and Cramer
objectives, plus a validation-selected blend. A common temporal-convolution
backbone and training budget are used. Summary F-score expands predictions to
original frames, averages scores within the supplied shot boundaries, selects
shots with 0/1 knapsack under a 15% duration budget, and averages F1 over
TVSum's user summaries. Rank correlations are computed against the raw
per-annotator scores aligned at `picks`.

### Data access and running

TVSum50 is distributed under Yahoo! Webscope terms. Obtain the raw annotation
and metadata TSVs through the authorized source; do not commit or redistribute
them. Obtain the standard preprocessed feature HDF5 described in the runner's
error message and store it locally at
`datasets/eccv16_dataset_tvsum_google_pool5.h5` (or set the environment
variable). Local dataset artifacts are ignored by Git.

Install `requirements.txt` in a Python environment with PyTorch support, then
set the paths to the authorized annotation and info TSVs:

```powershell
$env:TVSUM_ANNOTATION_TSV = "C:\path\to\ydata-tvsum50-anno.tsv"
$env:TVSUM_INFO_TSV = "C:\path\to\ydata-tvsum50-info.tsv"
py -3.12 tvsum_cv_entry.py
```

Optional environment variables `TVSUM_FEATURES_H5` and `TVSUM_RESULTS_JSON`
override the feature and result locations. The run records the input hashes,
all train/validation/test IDs, per-video and per-seed metrics, fold summaries,
paired tests, baselines, and sanity-check outcomes in
`results/standard_tvsum_cv/results.json`. It also writes a CSV table and
updates the generated results section below. The run is computationally
expensive because it trains every ablation and validation alpha for five seeds
on each split (ten epochs maximum per candidate by default; override with
`--epochs` only when that budget is stated alongside the results).

Run the complete discovered test suite with:

```powershell
py -3.12 -m pytest -q
```

## Online blinded study

The prepared-video blind study is available as a static GitHub Pages site, with
anonymous responses recorded in a private Google Sheet. See
[DEPLOYMENT.md](DEPLOYMENT.md) for setup. The static version does not perform
new-video inference.

<!-- REAL TVSUM CV RESULTS START -->

## Results (real TVSum, 5-fold)

**Run status: PASS.** Values below are generated from `results/standard_tvsum_cv_fixed/results.json`; no literature values are mixed in.

| Method | F-score mean ± seed SD (%) | Fold/video SD | Spearman ρ | Kendall τ |
|---|---:|---:|---:|---:|
| A: Mean target + MSE | 60.77 ± 2.51 | 9.41 | 0.148 | 0.113 |
| B: Mean target + cross-entropy | 61.22 ± 2.48 | 9.20 | 0.168 | 0.128 |
| C: Mean target + Cramer | 60.28 ± 2.95 | 9.22 | 0.135 | 0.103 |
| D: Barycenter target + MSE | 60.37 ± 2.69 | 9.66 | 0.116 | 0.088 |
| E: Barycenter target + Cramer | 60.60 ± 2.92 | 9.40 | 0.144 | 0.109 |
| F: Validation-blended target + Cramer | 60.55 ± 2.87 | 9.16 | 0.144 | 0.109 |
| Random summary | 56.31 ± 4.58 | 8.18 | -0.001 | -0.001 |
| Uniform summary | 59.37 ± — | 10.06 | 0.000 | 0.000 |
| Human upper bound (leave-one-out) | 65.83 ± — | 8.92 | — | — |

Paired contrasts average seeds and repeated split appearances per unique test video before testing, avoiding duplicate-video pseudoreplication; positive differences favor the first-listed arm. In particular, E vs C isolates the barycenter-target effect under the Cramer loss.

| Contrast | Mean F1 difference (points) | Paired t p | Wilcoxon p | Interpretation |
|---|---:|---:|---:|---|
| C vs A: Cramer loss at mean target | -0.42 | 0.3514 | 0.3217 | similar within one F1 point; no detected paired difference |
| D vs A: Barycenter target under MSE | -0.53 | 0.1373 | 0.0577 | similar within one F1 point; no detected paired difference |
| E vs C: Barycenter effect under Cramer | +0.28 | 0.2498 | 0.5188 | similar within one F1 point; no detected paired difference |
| E vs D: Cramer loss at barycenter target | +0.39 | 0.3684 | 0.2865 | similar within one F1 point; no detected paired difference |
| F vs E: Validation-selected blend | +0.10 | 0.5353 | 0.2821 | similar within one F1 point; no detected paired difference |

The highest mean-F1 learned arm is B: Mean target + cross-entropy (61.22); it exceeds the random and uniform baselines (56.31 and 59.37) on mean F1.

| Learned arm | Baseline | Mean F1 difference (points) | Paired t p | Wilcoxon p |
|---|---|---:|---:|---:|
| A: Mean target + MSE | Random | +3.87 | 6.175e-06 | 1.603e-05 |
| A: Mean target + MSE | Uniform | +1.28 | 0.2529 | 0.345 |
| A: Mean target + MSE | Human leave-one-out | -5.03 | 4.12e-06 | 2.1e-05 |
| B: Mean target + cross-entropy | Random | +4.61 | 9.67e-08 | 2.004e-06 |
| B: Mean target + cross-entropy | Uniform | +2.02 | 0.08136 | 0.07413 |
| B: Mean target + cross-entropy | Human leave-one-out | -4.30 | 5.13e-05 | 0.0001528 |
| C: Mean target + Cramer | Random | +3.46 | 2.302e-05 | 3.175e-05 |
| C: Mean target + Cramer | Uniform | +0.86 | 0.4737 | 0.7329 |
| C: Mean target + Cramer | Human leave-one-out | -5.45 | 2.438e-06 | 1.288e-06 |
| D: Barycenter target + MSE | Random | +3.34 | 1.896e-05 | 1.264e-05 |
| D: Barycenter target + MSE | Uniform | +0.75 | 0.5205 | 0.775 |
| D: Barycenter target + MSE | Human leave-one-out | -5.57 | 8.016e-07 | 2.98e-07 |
| E: Barycenter target + Cramer | Random | +3.73 | 2.766e-05 | 3.175e-05 |
| E: Barycenter target + Cramer | Uniform | +1.14 | 0.3373 | 0.4429 |
| E: Barycenter target + Cramer | Human leave-one-out | -5.17 | 5.745e-06 | 2.004e-06 |
| F: Validation-blended target + Cramer | Random | +3.83 | 6.614e-06 | 1.424e-05 |
| F: Validation-blended target + Cramer | Uniform | +1.24 | 0.2975 | 0.4211 |
| F: Validation-blended target + Cramer | Human leave-one-out | -5.07 | 1.149e-05 | 7.716e-06 |

Paired-test p-values are unadjusted for multiple comparisons.

All learned methods use the same 1024-D temporal-convolution backbone, optimizer, epoch budget, DSNet folds, and five fixed seeds. Alpha for arm F is selected using only the validation videos carved from that fold's training IDs. F-score follows the preprocessed TVSum protocol: expand scores at `picks`, average within supplied `change_points`, select shots by 0/1 knapsack under 15% duration, and average F1 over the 20 `user_summary` rows. Rank correlations compare sampled model scores with raw per-annotator scores aligned at `picks`.

Sanity checks: random mean F1 over all 50 videos=54.48 (expected 50–55: True); shuffled-feature F1=58.88 versus baseline threshold 64.37 (True).
The random baseline on the supplied DSNet test instances is 56.31; the all-video sanity estimate is separate because the supplied split file does not test every video.

The supplied DSNet split file has 32 unique test videos out of 50; test sets are not required to be mutually disjoint. Metrics average the fold/video test instances. Limitations: only the supplied five TVSum splits are included. Augmented and transfer settings were not run. These results describe this run and make no leading-performance claim. No conditional diffusion model was evaluated, so these results do not validate DDPM against the regressor. The human leave-one-out baseline is an annotator-agreement oracle, not a deployable model. Use the JSON for fold-level scores, paired tests, split IDs, feature/data hashes, selected validation alphas, and per-video/seed metrics.

<!-- REAL TVSUM CV RESULTS END -->

Augmented and transfer protocols have not been run. Results from this
experiment alone make no leading-performance claim; published numbers should
only be compared when their dataset split, features, and evaluation protocol
are demonstrably identical.

## Historical materials

[`PAPER.md`](PAPER.md) and [`STAGES_AND_RESULTS.md`](STAGES_AND_RESULTS.md)
contain historical project context. Earlier proxy-based numerical claims are
withdrawn and are not evidence for the real-data benchmark.
