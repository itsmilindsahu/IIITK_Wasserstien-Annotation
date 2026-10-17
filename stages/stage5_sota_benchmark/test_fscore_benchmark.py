"""
test_fscore_benchmark.py

Unit and regression test suite for TVSum F-score and correlation comparison benchmark:
1. Validates that all models from the user's screenshot are accounted for.
2. Validates that 2025_LLMVS (Shang et al., AAAI-2025) numbers are exact.
3. Tests evaluate_f1_and_correlations calculation on TVSum distribution data.
4. Tests that tvsum_fscore_sota_benchmark.py runs smoothly in fast mode.
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from tvsum_fscore_sota_benchmark import (
    SOTA_TVSUM_MODELS,
    SOTA_FPVSUM_MODELS,
    SOTA_CORRELATION_MODELS,
    PAPER_3000STEPS_BENCHMARK,
    run_tvsum_eval
)
from compare_downstream_metrics import evaluate_f1_and_correlations


def test_screenshot_models_presence():
    """Verify that all 24 baseline models + 2025_LLMVS from the screenshot Table 1 are included."""
    model_names = [m[1] for m in SOTA_TVSUM_MODELS]
    expected_models = [
        "dppLSTM", "VASNet", "SUM-FCN", "A-AVS", "M-AVS",
        "DSNet_ab", "DSNet_af", "RSGN", "3DST-UNet_sup", "RR-STG",
        "CFT-GIB_sup", "VSS-Net", "LMVS", "MPFN",
        "SUM-GAN", "DR-DSN", "ACGAN", "SUM-GAN-AAE", "3DST-UNet_unsup",
        "CFT-GIB_unsup", "SSPVS", "AMFM", "PRLVS", "DMFF",
        "2025_LLMVS (DDPM)"
    ]
    for exp in expected_models:
        assert exp in model_names, f"Model {exp} from screenshot Table 1 is missing!"


def test_2025_llmvs_metrics():
    """Verify that 2025_LLMVS (Shang et al. AAAI-2025) matches the screenshot values."""
    llmvs_entry = next(m for m in SOTA_TVSUM_MODELS if "2025_LLMVS" in m[1])
    # Can: 64.8, Aug: 65.0, Tran: 60.9, SumMe Can: 58.7
    assert llmvs_entry[3] == 64.8
    assert llmvs_entry[4] == 65.0
    assert llmvs_entry[5] == 60.9
    assert llmvs_entry[6] == 58.7


def test_correlation_models():
    """Verify Table 3 correlation models from screenshot."""
    corr_dict = {m[0]: (m[2], m[3]) for m in SOTA_CORRELATION_MODELS}
    assert "dppLSTM" in corr_dict and corr_dict["dppLSTM"] == (0.042, 0.055)
    assert "DSNet_ab" in corr_dict and corr_dict["DSNet_ab"] == (0.108, 0.129)
    assert "SSPVS" in corr_dict and corr_dict["SSPVS"] == (0.177, 0.233)
    assert "2025_LLMVS (DDPM)" in corr_dict and corr_dict["2025_LLMVS (DDPM)"] == (0.179, 0.238)


def test_f1_calculation_logic():
    """Verify F1 score and rank correlation calculation."""
    # Synthetic 10-frame distribution
    pred = np.array([0.05, 0.05, 0.30, 0.35, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0])
    gt1 = np.array([0.05, 0.05, 0.35, 0.30, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0])
    gt2 = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.40, 0.40])
    
    # 20% budget = 2 frames
    res = evaluate_f1_and_correlations(pred, [gt1, gt2], budget_pct=0.20)
    assert "mean_f1" in res
    assert "mean_spearman_rho" in res
    assert "mean_kendall_tau" in res
    assert res["mean_f1"] == 50.0  # Perfect overlap (100%) on gt1, 0% on gt2 -> mean = 50%


def test_benchmark_fast_run():
    """Run a fast test on 2 TVSum videos to ensure the full pipeline executes without exceptions."""
    summary, records = run_tvsum_eval(n_videos=2, frames=100, steps=20, lr=1.0, fast_eval=True)
    assert summary["num_videos"] == 2
    assert len(records) == 2
    assert "mine_f1_mean" in summary
    assert "base_f1_mean" in summary
