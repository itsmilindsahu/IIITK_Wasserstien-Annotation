# Wasserstein Diffusion on the Annotation Simplex
### Distribution-Aware Diffusion for Human Video Summarization

<p align="center">
  <img src="assets/dashboard.gif" width="900" alt="Dashboard Demo">
</p>

<p align="center">
  <b>Phase 1 Research Prototype</b><br>
  Implementing Wasserstein Barycenters, Dirichlet Diffusion, and Wasserstein Regularization
</p>

---

## Overview

This repository contains the **Phase 1 proof-of-concept implementation** of our proposed framework for **distribution-aware diffusion on the annotation simplex** for video summarization.

Unlike conventional approaches that collapse multiple human annotations into a simple arithmetic average, this project models the **entire annotation distribution** using tools from **Optimal Transport** and **Wasserstein Geometry**.

The implementation reproduces the three proposed algorithmic improvements on a lightweight synthetic benchmark before scaling to TVSum, SumMe and FPVSum in Phase 2.

---

## Motivation

Current diffusion-based video summarization methods typically assume that averaging human annotations produces a representative target.

However, real annotators frequently disagree.

For example:

- one group may prefer action scenes
- another may prefer dialogue
- another may focus on object interactions

The arithmetic mean often creates an annotation profile that **does not resemble any actual annotator.**

This project instead represents annotations as probability distributions and computes their **Wasserstein barycenter**, preserving the geometry of human disagreement.

---

# Proposed Improvements

The prototype implements all three components proposed in the research roadmap.

## 1. Wasserstein Barycenter

Instead of

\[
\bar{x}=\frac{1}{N}\sum_i x_i
\]

we compute the entropic Sinkhorn barycenter

\[
x^*=\arg\min_x \sum_i W_2^2(x,x_i)
\]

Advantages:

- preserves multimodal annotation structure
- respects transport geometry
- minimizes average Wasserstein distance
- avoids unrealistic averaged summaries

---

## 2. Dirichlet Forward Diffusion

The original Gaussian forward process leaves the probability simplex.

Instead we sample

\[
x_t\sim Dir(\alpha_t x_0)
\]

which guarantees

- non-negative scores
- sum-to-one constraint
- natural simplex geometry

---

## 3. Wasserstein-Regularized Objective

The standard MSE objective becomes

\[
L
=
L_{noise}
+
\lambda
W_2^2
(x_{pred},x_{target})
\]

encouraging generated summaries that remain close to the annotation distribution under Optimal Transport rather than Euclidean distance.

---

# Repository Structure

```
code/
│
├── sinkhorn.py        # Simplex projection
├── barycenter.py      # Sinkhorn barycenter + arithmetic mean baseline
├── forward.py         # Dirichlet forward diffusion
├── losses.py          # Wasserstein loss + MSE baseline
├── model.py           # Tiny linear noise predictor
└── train.py           # Training pipeline

results/
│
├── results.json
├── comparison.csv
└── loss_curve.png

ui/
│
├── template.html
├── build_ui.py
└── index.html
```

---

# Dashboard

The repository automatically generates an interactive HTML dashboard comparing both methods.

Features include

- distribution comparison
- arithmetic mean vs Wasserstein barycenter
- individual annotator visualization
- training statistics
- held-out Wasserstein distance
- final evaluation table

Simply open

```
ui/index.html
```

after training.

---

# Example Result

Current synthetic benchmark

| Configuration | Training Loss | Held-Out W₂ Distance |
|---------------|--------------|----------------------|
| Wasserstein Barycenter + Dirichlet + W₂ | **0.3787** | **0.6199** |
| Arithmetic Mean + Gaussian + MSE | 0.0000 | 0.6285 |

Although the improvement is currently modest, the prototype demonstrates an end-to-end implementation of the proposed framework.

---

# Running

## Train

```bash
cd code
python train.py
```

This regenerates

```
results/
```

---

## Build Dashboard

```bash
cd ../ui
python build_ui.py
```

Then simply open

```
ui/index.html
```

No web server required.

---

# Phase 1 Simplifications

To keep the implementation lightweight and dependency-free, several components are simplified.

| Component | Prototype | Full Model |
|------------|-----------|------------|
| Backbone | Linear predictor | Transformer |
| Dataset | Synthetic toy video | TVSum / SumMe / FPVSum |
| Framework | NumPy | PyTorch |
| OT Solver | Minimal Sinkhorn | Optimized implementation |

The objective is validating the mathematical pipeline rather than achieving state-of-the-art performance.

---

# Current Limitation

The current implementation exposes an important open issue.

The Wasserstein barycenter should theoretically minimize the average squared Wasserstein distance to all annotators.

However, using the current evaluation metric (`w2_1d()` based on squared CDF differences), the arithmetic mean occasionally performs slightly better.

This likely indicates a mismatch between

- the Sinkhorn objective optimized during barycenter computation

and

- the proxy Wasserstein distance used for evaluation.

The true 1D Wasserstein distance is computed via squared differences between quantile functions rather than cumulative distributions.

This discrepancy is intentionally left unresolved to accurately document the prototype's current state.

---

# Future Work

Phase 2 will extend the prototype with

- TVSum
- SumMe
- FPVSum
- Transformer noise predictor
- PyTorch implementation
- Exact Sinkhorn OT solver
- GPU acceleration
- Large-scale evaluation
- Ablation studies
- Human preference analysis

---

# Requirements

```
numpy
matplotlib
```

Install

```bash
pip install -r requirements.txt
```

---

# Citation

If this repository contributes to your research, please cite the associated paper once released.

---

# License

This project is released for academic and research purposes.
