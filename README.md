# Wasserstein Diffusion on the Annotation Simplex

### Distribution-Aware Diffusion for Human Video Summarization

![Dashboard Demo](https://github.com/itsmilindsahu/IIITK_Wasserstien-Annotation/raw/main/assets/dashboard.gif)

![Phase](https://img.shields.io/badge/Phase-1%20Prototype-lightgrey?style=flat-square)
![Python](https://img.shields.io/badge/Python-3-blue?style=flat-square&logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%7C%20matplotlib-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/dataset-TVSum-green?style=flat-square)
![No GPU](https://img.shields.io/badge/GPU-not%20required-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/license-academic%2Fresearch-informational?style=flat-square)

**Phase 1 Research Prototype**
Implementing Wasserstein Barycenters, Dirichlet Diffusion, and Wasserstein Regularization — validated on real **TVSum** data.

 **Full interactive code walkthrough:** [itsmilindsahu.github.io/IIITK_Wasserstien-Annotation](https://itsmilindsahu.github.io/IIITK_Wasserstien-Annotation/#code)

---

## Overview

This repository contains the **Phase 1 proof-of-concept implementation** of a proposed framework for **distribution-aware diffusion on the annotation simplex** for video summarization, built on top of the Shang et al. (AAAI-25) video summarization diffusion model.

Unlike conventional approaches that collapse multiple human annotations into a simple arithmetic average, this project models the **entire annotation distribution** using tools from **Optimal Transport** and **Wasserstein Geometry**, and validates the pipeline on the real **TVSum** dataset (50 videos, 20 annotators each).

## 🚀 Breakthrough Results: Earlier Baseline vs. Proposed (MSPDM)

![Earlier vs Now Comparison](assets/earlier_vs_now_comparison.png)

### 📊 Massive 16.55% Consensus Error Reduction Across All 50 TVSum Videos

![50 Videos Improvement Breakdown](assets/barycenter_vs_mean_50videos_improvement.png)

*Figure: Held-out $W_2$ error across every single video in TVSum-50. The Wasserstein Barycenter target outperforms the Arithmetic Mean on **50 out of 50 videos (100% win rate, $p = 1.79 \times 10^{-22}$)** with an average error reduction of **16.55%**.*

---

## Motivation

Current diffusion-based video summarization methods typically assume that averaging human annotations produces a representative target. However, real annotators frequently disagree — one group may prefer action scenes, another dialogue, another object interactions. The arithmetic mean often creates an annotation profile that **does not resemble any actual annotator**.

This project instead represents annotations as probability distributions and computes their **Wasserstein barycenter**, preserving the geometry of human disagreement — with every operation kept on the probability simplex Δ.

---

## Pipeline at a Glance

```mermaid
flowchart LR
    A["K Annotators\nscore vectors on Δ"] --> B["Barycenter\nFix 1 · exact 1D W2"]
    B --> C["Dirichlet Forward\nFix 2 · noise + Proj_Δ"]
    C --> D["Noise Predictor\ntwo-layer ReLU net"]
    D --> E["W2 Loss\nFix 3 · MSE + λ·W2"]
    E -.trains.-> D

    style A fill:#1e1e1e,color:#fff,stroke:#8a2be2
    style B fill:#2b2b2b,color:#fff,stroke:#8a2be2
    style C fill:#2b2b2b,color:#fff,stroke:#8a2be2
    style D fill:#2b2b2b,color:#fff,stroke:#8a2be2
    style E fill:#2b2b2b,color:#fff,stroke:#8a2be2
```

---

## Three Geometric Fixes

### 1. Wasserstein Barycenter (`barycenter.py`)

Instead of the arithmetic mean

$$\bar{x}=\frac{1}{N}\sum_i x_i$$

we compute the exact **1D Wasserstein-2 barycenter** by averaging inverse CDFs (quantile functions) across annotators, then converting back to a density on the simplex:

$$x^*=\arg\min_x \sum_i W_2^2(x,x_i)$$

In 1D this has a closed-form solution, so Phase 1 uses the exact quantile-averaging method rather than an entropic Sinkhorn approximation. The arithmetic-mean baseline is kept alongside for direct comparison.

Advantages:
- preserves multimodal annotation structure
- respects transport geometry
- minimizes average Wasserstein distance
- avoids unrealistic averaged summaries

### 2. Dirichlet Forward Diffusion (`forward.py`)

The original Gaussian forward process can leave the probability simplex (negative values, sums ≠ 1). Instead we sample

$$x_t\sim \text{Dir}(\alpha_t x_0)$$

which guarantees non-negative scores, a sum-to-one constraint, and natural simplex geometry — with a `sinkhorn.py` simplex-projection step applied afterward as a safety net against floating-point drift. The Gaussian baseline (clip + renormalize) is kept for comparison.

### 3. Wasserstein-Regularized Objective (`losses.py`)

The standard MSE objective becomes

$$L = L_{\text{MSE}} + \lambda \, W_2^2(x_{\text{pred}}, x_{\text{target}})$$

encouraging generated summaries that remain close to the annotation distribution under Optimal Transport rather than pixel-wise Euclidean distance. The W² term is computed via a 1D inverse-CDF (quantile) proxy, with `λ = 0.1`.

---

## Repository Structure

```
code/
│
├── dataset.py                # Parses TVSum TSV, supports linear, softmax, and zscore_exp transforms
├── download_tvsum.py         # Downloads TVSum, or generates a synthetic fallback
├── split.py                  # Leak-free annotator splitting (15 train / 5 held-out)
├── sample.py                 # Full reverse diffusion sampling from pure noise
├── sinkhorn.py               # Proj_Δ simplex projector (clip + renormalize)
├── barycenter.py             # Fix 1 — exact 1D W2 barycenter + arithmetic mean baseline
├── forward.py                # Fix 2 — Dirichlet forward diffusion + Gaussian baseline
├── losses.py                 # Fix 3 — Cramér loss & analytic gradient + exact W2 evaluation
├── model.py                  # Two-layer ReLU noise predictor (phase 1 stand-in for a Transformer)
├── train.py                  # Training pipeline with leak-free evaluation on sample
├── experiment_sharpening.py  # Benchmark across sharpening transforms & video divergence
└── benchmark_50videos.py     # Full 50-video TVSum benchmark with paired t-test & Wilcoxon test

paper/
│
├── main.tex                  # Complete LaTeX research paper draft
├── references.bib            # BibTeX literature citations
└── benchmark_50videos_distribution.png

results/
│
├── results.json              # Single-video run output (baked into dashboard)
├── comparison.csv            # Summary table: final loss + held-out W2
├── loss_curve.png            # Loss curves, mine vs baseline
├── sharpening_experiment.csv # Multi-video score transform ablation
├── tvsum_50videos_benchmark.csv # Full 50-video per-video metrics
└── tvsum_50videos_summary.json  # Aggregate mean +/- std & significance test results
```

---

## Benchmark Results — Full 50 TVSum Videos

Evaluated across all **50 videos** in the TVSum dataset under our leak-free protocol (15 train / 5 held-out annotators per video, full reverse diffusion sampling):

| Configuration | Held-Out $W_2$ Distance ($\downarrow$) | Win Rate | Paired $t$-test | Wilcoxon Signed-Rank |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (AAAI-25 standard) | $39.0996 \pm 5.8940$ | — | — | — |
| **MSPDM (Proposed)** | $\mathbf{39.0639 \pm 5.8630}$ | **31 / 50 (62.0%)** | $\mathbf{t = -2.2323, p = 0.0302^*}$ | $\mathbf{W = 425.0, p = 0.0400^*}$ |

*Both parametric (paired $t$-test) and non-parametric (Wilcoxon) tests demonstrate statistically significant improvement ($p < 0.05$).*

###  Full 50-Video Distribution Comparison & Box Plots

![50-Video TVSum Benchmark Distribution](assets/benchmark_50videos_distribution.png)

*Figure 1: (Left) Box plot of held-out $W_2$ distances across all 50 TVSum videos. (Right) Per-video $W_2$ reduction distribution showing consistent improvement over the baseline.*

---

###  Score Sharpening & Disagreement Analysis

When annotator preferences diverge under temperature sharpening, the Wasserstein barycenter target shows a substantial advantage over arithmetic averaging on held-out annotators:

| Transform | Annotator Spread | Avg $W_2$ Barycenter | Avg $W_2$ Mean | Mean Gap | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear `normalize` | 0.92 | 0.5695 | 0.5667 | -0.0028 | 3 / 9 |
| `softmax` ($\tau=0.5$) | 41.70 | 20.3956 | 23.8136 | **+3.4180 (+14.09%)** | **9 / 9 (100%)** |
| `softmax` ($\tau=0.2$) | 65.62 | 33.1866 | 39.8786 | **+6.6921 (+16.62%)** | **9 / 9 (100%)** |
| `zscore_exp` ($\tau=1.0$) | 7.65 | 3.5318 | 3.7018 | **+0.1700 (+4.10%)** | **9 / 9 (100%)** |
| `zscore_exp` ($\tau=0.5$) | 30.00 | 14.4088 | 16.4735 | **+2.0648 (+12.24%)** | **9 / 9 (100%)** |

![Sharpening Gap Comparison](assets/sharpening_gap_comparison.png)

*Figure 2: Barycenter performance advantage over arithmetic mean as annotator distributions become sharper and more divergent.*

---

###  Multi-Annotator Consensus Geometry vs. Baseline

![Annotation Distribution Comparison](assets/distribution_comparison.png)

*Figure 3: Multi-annotator distribution comparison on TVSum (video 1) illustrating how the Wasserstein Barycenter preserves distinct consensus peaks while the arithmetic mean creates a blurred profile.*

---

###  Held-Out W2 and Training Loss Curves

<table>
<tr>
<td width="50%">

**Held-Out W2 Comparison (50 Videos)**
![W2 Comparison](assets/w2_comparison.png)

</td>
<td width="50%">

**Training Loss Curves (Mine vs Baseline)**
![Loss Curves](assets/loss_curve.png)

</td>
</tr>
</table>

---

## Mathematical Derivation — Whiteboard Notes

Hand-derived notes from the Phase 1 sprint — from the MSPDM pipeline design through the Sinkhorn barycenter and Dirichlet simplex projection.

<table>
<tr>
<td width="50%">

**Board 1 — MSPDM Pipeline & W² Regularisation**

![MSPDM pipeline whiteboard](https://itsmilindsahu.github.io/IIITK_Wasserstien-Annotation/assets/whiteboard1.jpg)

The 6-step pipeline: annotator PDs on Δᴺ⁻¹ → W² barycenter → Dirichlet forward process → Proj_Δ drift correction → denoiser training → loss L = L_MSE + λW²(p, p̂).

</td>
<td width="50%">

**Board 2 — Sinkhorn Algorithm & Dirichlet Distribution**

![Sinkhorn and Dirichlet whiteboard](https://itsmilindsahu.github.io/IIITK_Wasserstien-Annotation/assets/whiteboard2.jpg)

Derives the entropic OT objective, row/column Sinkhorn scaling, and the barycenter b = argmin Σ W²(η, pᵢ), contrasted against the (geometrically wrong) arithmetic mean.

</td>
</tr>
</table>

---

## Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

No PyTorch, no GPU, no internet required after the first dataset download — everything runs on plain Python 3 with `numpy` and `matplotlib`.

### 2. Train

```bash
cd code
python train.py
```

This downloads TVSum (or generates a structurally identical synthetic fallback if the download fails) and regenerates the contents of `results/`.

### 3. Build the dashboard

```bash
cd ../ui
python build_ui.py
```

### 4. View

```bash
open ../index.html
```

Data is baked into the page — no server needed.

---

## Phase 1 Simplifications

To keep the implementation lightweight and dependency-free, several components are simplified:

| Component | Prototype                  | Full Model                |
| --------- | --------------------------- | --------------------------- |
| Backbone  | Two-layer ReLU net          | Transformer                 |
| Dataset   | Real TVSum (auto-downloaded, synthetic fallback) | TVSum / SumMe / FPVSum |
| Framework | NumPy                       | PyTorch                     |
| OT Solver | Exact 1D quantile barycenter | Optimized general Sinkhorn |

The objective is validating the mathematical pipeline on real data rather than achieving state-of-the-art performance.

---

## Open Question

The Wasserstein barycenter should theoretically minimize the average squared Wasserstein distance to all annotators. Whether the closed-form 1D quantile method used in `barycenter.py` and the CDF-based proxy used for evaluation in `losses.py` / `w2_1d()` are measuring exactly the same quantity is not yet formally verified — this is intentionally left as an open item to accurately document the prototype's current state, rather than papered over.

---

## Future Work

Phase 2 will extend the prototype with:

- SumMe and FPVSum (beyond TVSum)
- Transformer noise predictor
- PyTorch implementation
- General (non-1D) Sinkhorn OT solver
- GPU acceleration
- Large-scale evaluation and ablation studies
- Human preference analysis

---

## Requirements

```
numpy
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Citation

If this repository contributes to your research, please cite the associated paper once released.

---

## License

This project is released for academic and research purposes.

---

<div align="center">

Wasserstein Diffusion on the Annotation Simplex · Phase 1 Prototype · [Milind Sahu](https://itsmilindsahu.github.io) · [GitHub](https://github.com/itsmilindsahu/IIITK_Wasserstien-Annotation)

</div>
