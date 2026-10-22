# Synthetics — Fully Self-Contained MSPDM Reproduction

This folder reproduces **all key results** of the MSPDM paper using **purely synthetic data**.  
No TVSum TSV file, no internet connection, and no GPU are required.

## Structure

```
synthetics/
├── core/                   # Shared mathematical primitives (barycenter, losses, model, …)
│   ├── barycenter.py
│   ├── forward.py
│   ├── losses.py
│   ├── model.py
│   ├── sample.py
│   ├── sinkhorn.py
│   └── split.py
├── data/
│   └── synthetic_dataset.py   # Generates synthetic multi-annotator video data
├── stages/
│   ├── stage1_consensus.py    # W2-barycenter vs arithmetic mean oracle comparison
│   ├── stage2_disagreement.py # Sharpening & native disagreement sensitivity
│   ├── stage3_diagnosis.py    # Gradient norm analysis & LR-collapse invariance
│   ├── stage4_benchmark.py    # 50-video end-to-end diffusion benchmark (F1, rho, tau, W2)
│   ├── stage5_sota.py         # SOTA F-score comparison table generation
│   └── stage6_transformer.py  # Temporal Transformer denoiser self-test
├── results/                   # Output CSVs, JSONs, and figures (auto-created)
├── run_all.py                 # Run every stage sequentially
└── requirements.txt
```

## Quick Start

```bash
cd synthetics
pip install -r requirements.txt

# Run every stage end-to-end
python run_all.py

# Or run individual stages
python stages/stage1_consensus.py
python stages/stage2_disagreement.py
python stages/stage3_diagnosis.py
python stages/stage4_benchmark.py
python stages/stage5_sota.py
python stages/stage6_transformer.py
```

All outputs (CSVs, JSONs, PNGs) are written to `synthetics/results/`.
