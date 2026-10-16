# Wasserstein Diffusion on the Annotation Simplex

### Distribution-Aware Diffusion for Human Video Summarization

![Dashboard Demo](https://github.com/itsmilindsahu/IIITK_Wasserstien-Annotation/raw/main/assets/dashboard.gif)

![Phase](https://img.shields.io/badge/Phase-1%20Prototype-lightgrey?style=flat-square)
![Python](https://img.shields.io/badge/Python-3-blue?style=flat-square&logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%7C%20matplotlib-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/dataset-TVSum-green?style=flat-square)
![No GPU](https://img.shields.io/badge/GPU-not%20required-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/license-academic%2Fresearch-informational?style=flat-square)

**Phase 1 Research Prototype.**
Implementing Wasserstein Barycenters, Dirichlet Diffusion, and Wasserstein Regularization — validated on real TVSum data.

**Full interactive code walkthrough:** [itsmilindsahu.github.io/IIITK_Wasserstien-Annotation](https://itsmilindsahu.github.io/IIITK_Wasserstien-Annotation/#code)

---

## Overview

This repository presents the **Phase 1 proof-of-concept implementation** of a proposed framework for **distribution-aware diffusion on the annotation simplex** for video summarization, built atop the Shang et al. (AAAI-25) video summarization diffusion model.

Unlike conventional approaches that collapse multiple human annotations into a simple arithmetic average, this project models the **entire annotation distribution** using tools from **Optimal Transport** and **Wasserstein Geometry**, and validates the corrected pipeline on the real TVSum dataset (50 videos, 20 annotators each) under a rigorous, leak-free evaluation protocol.

---

## Core Mathematical Framework

![Mathematical Consensus and Diffusion](assets/mathematical_consensus_and_diffusion.png)

*Figure 1: Three panels illustrating the key mathematical contributions.
(a) Multi-annotator distributions $x^{(k)} \in \Delta^{N-1}$ and the Wasserstein barycenter $x^*$ vs. arithmetic mean $\bar{x}$.
(b) 1D Quantile Monge geometry: averaging of inverse-CDF functions $F_{x^{(k)}}^{-1}(u)$ to produce the exact barycenter.
(c) Reverse diffusion trajectory on $\Delta^{N-1}$ from prior noise $t = T$ to clean summary $t = 0$, trained with the split Cramér regularizer.*

---

## Key Result: 16.55% Consensus Error Reduction Across All 50 TVSum Videos

The central result measures how closely the generated consensus annotation matches genuinely held-out annotators under the $W_2$ metric. Under a leak-free 15/5 train/held-out split per video:

$$W_2^2(p,\, q) = \int_0^1 \bigl(F_p^{-1}(u) - F_q^{-1}(u)\bigr)^2 \, du$$

| Target Construction | Held-Out $W_2$ Error $(\downarrow)$ | Win Rate | $p$-value |
| :--- | :---: | :---: | :---: |
| Arithmetic Mean $\bar{x} = \frac{1}{K}\sum_{k=1}^K x^{(k)}$ | $26.37 \pm 9.43$ | 0 / 50 | — |
| **Wasserstein Barycenter** $x^* = \arg\min_{x \in \Delta} \sum_k W_2^2(x, x^{(k)})$ | $\mathbf{22.01 \pm 8.05}$ | **50 / 50 (100%)** | $p = 1.79 \times 10^{-22}$ |

The barycenter target wins on every single video ($p = 1.79 \times 10^{-22}$, Wilcoxon signed-rank test).

![50-Video Improvement Breakdown](assets/barycenter_vs_mean_50videos_improvement.png)

*Figure 2: Per-video held-out $W_2$ error across all 50 TVSum videos, with paired gain distribution. The Wasserstein Barycenter target outperforms the Arithmetic Mean on all 50 / 50 videos at a mean error reduction of $-16.55\%$.*

---

## Score Sharpening and Disagreement Analysis

When annotator preferences diverge under temperature-sharpened score transforms, the barycenter advantage grows substantially, confirming that $W_2$-averaging is most beneficial precisely when annotators disagree.

### Full 50-Video Sharpening Ablation (Table 2 at $n = 50$)

Evaluated across all 50 TVSum videos using the leak-free 15/5 held-out $W_2$ protocol (`sharpening_50video_benchmark.py`):

| Transform | Annotator Spread | $W_2$ Barycenter | $W_2$ Mean | Gap % | Win Rate | Paired $t$-test ($p$) | Wilcoxon ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Linear $\mathrm{normalize}$ | 1.41 | **0.9013** | 0.9030 | $+0.06\%$ | 26 / 50 | $p = 0.348$ | $p = 0.421$ |
| Softmax $\tau = 0.5$ | 68.09 | **38.0353** | 45.5371 | **$+15.74\%$** | **49 / 50 (98%)** | $p = 4.13 \times 10^{-18}$ | $p = 8.88 \times 10^{-15}$ |
| Softmax $\tau = 0.2$ | 110.29 | **61.1485** | 75.4773 | **$+18.69\%$** | **50 / 50 (100%)** | $p = 9.03 \times 10^{-21}$ | $p = 1.78 \times 10^{-15}$ |
| $z$-score exp $\tau = 1.0$ | 11.46 | **6.6391** | 6.9708 | **$+3.99\%$** | **46 / 50 (92%)** | $p = 1.17 \times 10^{-11}$ | $p = 1.02 \times 10^{-11}$ |
| $z$-score exp $\tau = 0.5$ | 47.94 | **26.9212** | 31.4057 | **$+13.37\%$** | **49 / 50 (98%)** | $p = 9.50 \times 10^{-17}$ | $p = 2.49 \times 10^{-14}$ |

![Sharpening Gap Comparison](assets/sharpening_gap_comparison.png)

*Figure 3: Barycenter performance advantage over arithmetic mean as a function of annotator divergence. As the annotation distribution becomes sharper and more multimodal, the advantage of exact $W_2$ barycenter averaging increases monotonically.*

### Native Annotator Disagreement Evaluation (No Synthetic Sharpening)

To test whether the barycenter advantage is an artifact of synthetic score transforms, `native_disagreement_eval.py` measures each video's *native* annotator spread under raw annotations and partitions the 50 videos into disagreement terciles:

| Disagreement Tercile | $n$ | Mean Spread | $W_2$ Barycenter | $W_2$ Mean | Gap % | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Native Disagreement** | 16 | 1.1405 | 0.7553 | 0.7522 | $-0.46\%$ | 5 / 16 |
| **Medium Native Disagreement** | 16 | 1.3606 | **0.7820** | 0.7844 | **$+0.25\%$** | **9 / 16** |
| **High Native Disagreement** | 18 | 1.6935 | **1.1372** | 1.1425 | **$+0.36\%$** | **12 / 18** |

* **Monotonic Gain**: The barycenter advantage monotonically increases with native annotator disagreement across buckets (`True`), validating the theoretical prediction directly on natural human disagreement.

---

## MSPDM Benchmark (End-to-End Diffusion Model Evaluation)

Under the full MSPDM diffusion pipeline (end-to-end training and reverse sampling), evaluated across all 50 TVSum videos:

| Method | Held-Out $W_2$ $(\downarrow)$ | Win Rate | Paired $t$-test | Wilcoxon |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (AAAI-25 standard) | $39.0996 \pm 5.8940$ | — | — | — |
| **MSPDM (Proposed)** | $\mathbf{39.0639 \pm 5.8630}$ | **31 / 50 (62.0%)** | $t = -2.23,\; p = 0.030^*$ | $W = 425.0,\; p = 0.040^*$ |

Both parametric (paired $t$-test) and non-parametric (Wilcoxon signed-rank) tests confirm statistically significant improvement ($p < 0.05$).

### Standard Video-Summarization F-Score Comparison

The following comparison uses the scores and method references supplied for the canonical (Can), augmented (Aug), and transfer (Tran) settings. The final benchmark line previously labeled “Ours” in that table is identified here as **AAAI 2025 (reported)**. Our current project does not yet report results under these same protocols; its row is therefore marked “—” rather than filling it with the non-comparable custom frame-overlap diagnostic below.

| Method | TVSum Can | TVSum Aug | TVSum Tran | SumMe Can | SumMe Aug | SumMe Tran |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dppLSTM (Zhang et al. 2016) | 54.7 | 59.6 | 58.7 | 38.6 | 42.9 | 41.8 |
| VASNet (Fajtl et al. 2019) | 61.4 | 62.4 | — | 49.7 | 51.1 | — |
| SUM-FCN (Rochan, Ye, and Wang 2018) | 56.8 | 59.2 | 58.2 | 47.5 | 51.1 | 44.1 |
| A-AVS (Ji et al. 2019) | 59.4 | 60.8 | — | 43.9 | 44.6 | — |
| M-AVS (Ji et al. 2019) | 61.0 | 61.8 | — | 44.4 | 46.1 | — |
| DSNet$_{ab}$ (Zhu et al. 2020) | 62.1 | 63.9 | 59.4 | 50.2 | 50.7 | 46.5 |
| DSNet$_{af}$ (Zhu et al. 2020) | 61.9 | 62.2 | 58.0 | 51.2 | 53.3 | 47.6 |
| RSGN (Zhao et al. 2021) | 60.1 | 61.1 | 60.0 | 45.0 | 45.7 | 44.0 |
| 3DST-UNet$_{sup}$ (Liu et al. 2022) | 58.3 | 58.9 | 56.1 | 47.4 | 49.9 | 47.9 |
| RR-STG (Zhu et al. 2022) | 63.0 | 63.6 | 59.7 | 53.4 | 54.8 | 45.4 |
| CFT-GIB$_{sup}$ (Zhong et al. 2023) | 62.7 | 60.3 | 58.0 | 56.0 | 54.8 | 44.0 |
| VSS-Net (Zhang et al. 2023) | 61.0 | 61.4 | 58.5 | 51.5 | 52.8 | 48.4 |
| LMVS (Nam et al. 2024) | 60.5 | — | — | 45.8 | — | — |
| MPFN (Khan et al. 2024) | 62.4 | — | — | 51.9 | — | — |
| SUM-GAN (Mahasseni, Lam, and Todorovic 2017) | 50.8 | 58.9 | — | 38.7 | 41.7 | — |
| DR-DSN (Zhou, Qiao, and Xiang 2018) | 57.6 | 58.4 | 57.8 | 41.4 | 42.8 | 42.4 |
| ACGAN (He et al. 2019) | 58.5 | 58.9 | 57.8 | 46.0 | 47.0 | 44.5 |
| SUM-GAN-AAE (Apostolidis et al. 2020b) | 58.3 | — | — | 48.9 | — | — |
| 3DST-UNet$_{unsup}$ (Liu et al. 2022) | 58.3 | 58.4 | 58.0 | 44.6 | 49.5 | 45.7 |
| CFT-GIB$_{unsup}$ (Zhong et al. 2023) | 62.5 | 60.4 | 57.4 | 55.0 | 54.0 | 42.9 |
| SSPVS (Li et al. 2023) | 60.3 | 61.8 | 57.8 | 48.7 | 50.4 | 45.8 |
| AMFM (Zhang, Liu, and Wu 2024) | 61.0 | 60.8 | 58.6 | 51.8 | 52.8 | 46.4 |
| PRLVS (Wang, Wu, and Yan 2024) | 63.0 | 59.2 | 57.0 | 46.3 | 49.7 | 47.6 |
| DMFF (Yu et al. 2024) | 61.2 | — | — | 53.0 | — | — |
| **AAAI 2025 (reported)** | **64.8** | **65.0** | **60.9** | **58.7** | **59.2** | **50.3** |
| **Ours (this work)** | — | — | — | — | — | — |

The corresponding FPVSum scores supplied for comparison are:

| Method | FPVSum F-score |
| :--- | ---: |
| Random (Ho, Chiu, and Wang 2018) | 16.3 |
| Uniform (Ho, Chiu, and Wang 2018) | 15.1 |
| TDCNN (Yao, Mei, and Rui 2016) | 28.6 |
| DSN (Bousmalis et al. 2016) | 22.7 |
| FPVS (Ho, Chiu, and Wang 2018) | 35.3 |
| **AAAI 2025 (reported)** | **46.1** |
| **Ours (this work)** | — |

### Held-Out Annotator Frame-Overlap (Custom Metric)

Separately, we evaluate on the official TVSum annotations for all 50 videos. For each video, 15 annotators are used for training and five are held out; after resampling each annotation to 100 frames and applying softmax with temperature 0.5, we compare the top 15% ranked by the generated distribution with the top 15% ranked by each held-out annotator, then average the overlap score across annotators and videos. This run uses the evaluation script defaults of 100 training steps, learning rate 0.01, and pure barycenter target ($\alpha=1.0$). This is the project's TVSum-only held-out protocol, not the canonical/augmented/transfer benchmark protocol above.

| Method | TVSum held-out top-15% frame overlap (%) $(\uparrow)$ |
| :--- | :---: |
| Baseline (project implementation) | $15.01 \pm 5.15$ |
| **Ours (MSPDM)** | $14.93 \pm 4.72$ |

This is a custom frame-overlap diagnostic, **not** the standard F-score reported above and should not be compared directly with the canonical/augmented/transfer or FPVSum results. Since both selected sets have the same 15% size, this overlap equals F1. On this run, MSPDM scored 0.08 percentage points below the baseline and won on 13/50 videos (26%). The difference was not statistically significant (paired $t$-test $p = 0.8006$; Wilcoxon $p = 0.9553$). Per-video results and aggregate statistics are available in [`results/tvsum_50videos_f1_benchmark.csv`](results/tvsum_50videos_f1_benchmark.csv) and [`results/tvsum_50videos_f1_summary.json`](results/tvsum_50videos_f1_summary.json).

![50-Video TVSum Benchmark Distribution](assets/benchmark_50videos_distribution.png)

*Figure 4: Box plots of held-out $W_2$ distances across all 50 TVSum videos (left) and per-video $W_2$ reduction distribution (right).*

---

## Motivation

Current diffusion-based video summarization methods assume that averaging human annotations produces a representative target. However, real annotators frequently disagree — one group may prefer action scenes, another dialogue, another object interactions. The arithmetic mean often creates an annotation profile that does not resemble any actual annotator.

This project instead represents annotations as probability distributions on the simplex $\Delta^{N-1}$ and computes their **Wasserstein barycenter**, preserving the geometry of human disagreement, with every operation kept on the probability simplex via simplex projection $\mathrm{Proj}_{\Delta}$.

---

## Pipeline

```mermaid
flowchart LR
    A["K Annotators x^k in Delta"] --> B["Barycenter: exact 1D W_2"]
    B --> C["Dirichlet Forward: noise + Proj_Delta"]
    C --> D["Noise Predictor: two-layer ReLU"]
    D --> E["Split Loss: Cramer train + W_2 eval"]
    E -.trains.-> D

    style A fill:#1e1e1e,color:#fff,stroke:#8a2be2
    style B fill:#2b2b2b,color:#fff,stroke:#8a2be2
    style C fill:#2b2b2b,color:#fff,stroke:#8a2be2
    style D fill:#2b2b2b,color:#fff,stroke:#8a2be2
    style E fill:#2b2b2b,color:#fff,stroke:#8a2be2
```

---

## Three Geometric Fixes

### Fix 1 — Wasserstein Barycenter (`barycenter.py`)

Instead of the arithmetic mean $\bar{x} = \frac{1}{K}\sum_{k=1}^K x^{(k)}$, we compute the exact **1D Wasserstein-2 barycenter**:

$$x^* = \arg\min_{x \in \Delta^{N-1}} \sum_{k=1}^K W_2^2\!\bigl(x,\, x^{(k)}\bigr)$$

In 1D this has the closed-form solution via quantile averaging:

$$F_{x^*}^{-1}(u) = \frac{1}{K} \sum_{k=1}^K F_{x^{(k)}}^{-1}(u), \quad u \in [0,1]$$

This preserves multimodal annotation structure, respects Wasserstein transport geometry, and minimizes average $W_2$ distance simultaneously.

### Fix 2 — Dirichlet Forward Diffusion (`forward.py`)

The standard Gaussian forward process can leave the probability simplex (negative values, sums $\neq 1$). Instead:

$$x_t \sim \mathrm{Dir}\!\bigl(\alpha_t\, x_0\bigr)$$

with a simplex-projection step $\mathrm{Proj}_{\Delta}$ applied afterward as a safety net against floating-point drift.

### Fix 3 — Split Cramér Regularizer (`losses.py`)

Training uses the Cramér distance (analytically differentiable in 1D):

$$\mathcal{L}_{\mathrm{Cram\acute{e}r}}(p, q) = \sum_{i=1}^N \bigl(F_p(i) - F_q(i)\bigr)^2, \qquad \frac{\partial \mathcal{L}}{\partial p_j} = 2\sum_{i \ge j}\bigl(F_p(i) - F_q(i)\bigr)$$

Evaluation uses the proper $W_2$ quantile metric on genuinely held-out annotators, keeping training and evaluation metrics cleanly separated.

---

## Held-Out $W_2$ and Training Loss Curves

<table>
<tr>
<td width="50%">

**Held-Out $W_2$ Comparison (50 Videos)**
![W2 Comparison](assets/w2_comparison.png)

</td>
<td width="50%">

**Training Loss Curves**
![Loss Curves](assets/loss_curve.png)

</td>
</tr>
</table>

---

## Mathematical Derivation — Whiteboard Notes

<table>
<tr>
<td width="50%">

**Board 1 — MSPDM Pipeline and $W_2$ Regularization**

![MSPDM pipeline whiteboard](https://itsmilindsahu.github.io/IIITK_Wasserstien-Annotation/assets/whiteboard1.jpg)

The 6-step pipeline: annotator probability distributions on $\Delta^{N-1}$ to $W_2$ barycenter to Dirichlet forward process to $\mathrm{Proj}_{\Delta}$ drift correction to denoiser training to loss $\mathcal{L} = \mathcal{L}_{\mathrm{MSE}} + \lambda\, W_2^2(p, \hat{p})$.

</td>
<td width="50%">

**Board 2 — Sinkhorn Algorithm and Dirichlet Distribution**

![Sinkhorn and Dirichlet whiteboard](https://itsmilindsahu.github.io/IIITK_Wasserstien-Annotation/assets/whiteboard2.jpg)

Derives the entropic OT objective, row/column Sinkhorn scaling, and the barycenter $b = \arg\min \sum_i W_2^2(\eta, p_i)$, contrasted against the arithmetically averaged baseline.

</td>
</tr>
</table>

---

## Repository Structure

```
code/
|
+-- dataset.py                # Parses TVSum TSV; linear, softmax, zscore_exp transforms
+-- download_tvsum.py         # Downloads TVSum; synthetic fallback on failure
+-- split.py                  # Leak-free annotator splitting (15 train / 5 held-out)
+-- sample.py                 # Full reverse diffusion sampling from pure noise on Delta
+-- sinkhorn.py               # Proj_Delta simplex projector (clip + renormalize)
+-- barycenter.py             # Fix 1: exact 1D W_2 barycenter + arithmetic mean baseline
+-- forward.py                # Fix 2: Dirichlet forward diffusion + Gaussian baseline
+-- losses.py                 # Fix 3: Cramer loss & analytic gradient + exact W_2 evaluation
+-- model.py                  # Two-layer ReLU noise predictor
+-- train.py                  # Training pipeline with leak-free evaluation
+-- experiment_sharpening.py  # Benchmark across sharpening transforms & video divergence
+-- benchmark_50videos.py     # Full 50-video TVSum benchmark with paired t-test & Wilcoxon
+-- generate_publication_figures.py  # Publication-grade figure generator

native_disagreement_eval.py       # Extension 1: Real-data native disagreement tercile evaluation
sharpening_50video_benchmark.py   # Extension 2: Full 50-video sharpening ablation with p-values
transformer_denoiser.py           # Extension 3: Temporal Transformer denoiser with multi-modal conditioning

paper/
+-- main.tex                  # Complete LaTeX research paper draft
+-- references.bib            # BibTeX literature citations

results/
+-- tvsum_50videos_benchmark.csv
+-- tvsum_50videos_summary.json
+-- sharpening_50video_summary.csv
+-- sharpening_50video_summary_per_video.csv
+-- native_disagreement_benchmark.csv
+-- sharpening_experiment.csv
+-- comparison.csv

assets/
+-- mathematical_consensus_and_diffusion.png    # Figure 1: Math framework (3-panel)
+-- barycenter_vs_mean_50videos_improvement.png  # Figure 2: 50-video error breakdown
+-- sharpening_gap_comparison.png               # Figure 3: Sharpening divergence analysis
+-- benchmark_50videos_distribution.png         # Figure 4: Box plots (MSPDM benchmark)
```

---

## Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

No PyTorch, no GPU, no internet required after the first dataset download.

### 2. Train (single video)

```bash
cd code
python train.py
```

### 3. Reproduce full TVSum-50 benchmark

```bash
python code/benchmark_50videos.py
```

### 4. Run full 50-video sharpening ablation benchmark

```bash
python sharpening_50video_benchmark.py
```

### 5. Run native annotator disagreement evaluation

```bash
python native_disagreement_eval.py
```

### 6. Run Temporal Transformer denoiser self-test

```bash
python transformer_denoiser.py --self-test
```

### 7. Reproduce publication figures

```bash
python code/generate_publication_figures.py
```

---

## Architectural Extensions

### Temporal Multi-Modal Transformer Denoiser (`transformer_denoiser.py`)
Phase 1 prototype utilizes a two-layer ReLU network in `model.py` for diffusion noise prediction. `transformer_denoiser.py` provides the architectural upgrade to a **Temporal Transformer** with multi-modal feature conditioning:
- **Temporal Attention**: Segment tokens across $N$ video intervals attend to one another via multi-head self-attention, capturing video-level temporal dynamics.
- **Multi-Modal Conditioning**: Conditioned on per-segment visual embeddings (e.g. CLIP ViT or Video-LLaVA). Features are precomputed offline and injected via learned feature projection and additive fusion.
- **Calling Convention**: Drop-in compatible forward interface `net(x_t, t, cond=features)`.

---

## Phase 1 Simplifications

| Component | Prototype | Full Model |
| --------- | --------- | ---------- |
| Backbone | Two-layer ReLU net / Temporal Transformer (`transformer_denoiser.py`) | Video-LLaVA conditioned Transformer |
| Dataset | Real TVSum (auto-downloaded, synthetic fallback) | TVSum / SumMe / FPVSum |
| Framework | NumPy (Core) / PyTorch (`transformer_denoiser.py`) | PyTorch |
| OT Solver | Exact 1D quantile barycenter | Optimized general Sinkhorn |

The objective is validating the mathematical pipeline on real data rather than achieving state-of-the-art performance.

---

## Open Question

Whether the closed-form 1D quantile method used in `barycenter.py` and the CDF-based proxy used for evaluation in `losses.py` are measuring exactly the same quantity is not yet formally verified — intentionally left as an open item to accurately document the prototype's current state.

---

## Future Work

- SumMe and FPVSum (beyond TVSum)
- Full training with `TemporalTransformerDenoiser` end-to-end on GPU
- Multi-modal Video-LLaVA joint conditioning
- General (non-1D) Sinkhorn OT solver
- Large-scale cross-dataset evaluation and ablation studies

---

## Requirements

```
numpy
matplotlib
```

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
