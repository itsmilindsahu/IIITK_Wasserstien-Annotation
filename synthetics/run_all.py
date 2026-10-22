"""
run_all.py  —  Synthetics master runner
========================================

Runs all 6 research stages sequentially and prints a final summary table.
All outputs are written to  synthetics/results/.

Usage
-----
    # From inside the synthetics/ directory:
    python run_all.py

    # With fewer diffusion steps for a faster smoke test:
    python run_all.py --fast

    # Skip slow Stage 4 (50-video diffusion benchmark):
    python run_all.py --skip-stage4
"""
import argparse
import os
import sys
import time

# ── ensure imports resolve relative to this file ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def _hline(char="=", n=70):
    print(char * n)


def main():
    parser = argparse.ArgumentParser(description="Run all MSPDM synthetic stages")
    parser.add_argument("--fast",         action="store_true",
                        help="Use fewer steps / videos for a quick smoke test")
    parser.add_argument("--skip-stage4",  action="store_true",
                        help="Skip the slow Stage 4 (50-video diffusion benchmark)")
    args = parser.parse_args()

    fast       = args.fast
    n_videos   = 10  if fast else 50
    steps_s4   = 30  if fast else 100   # diffusion training steps per video in Stage 4
    n_s3_vids  = 3   if fast else 5     # videos used in Stage 3

    timings = {}

    _hline()
    print("  MSPDM Synthetics — All-Stages Runner")
    print("  Dataset: fully synthetic (no TVSum TSV required)")
    print(f"  Mode:    {'FAST (smoke test)' if fast else 'FULL'}")
    _hline()

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\n[Stage 1]  Oracle W2 Barycenter vs Arithmetic Mean …")
    from stages.stage1_consensus import run_stage1
    t0 = time.time()
    s1 = run_stage1(n_videos=n_videos)
    timings["Stage 1"] = time.time() - t0

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\n[Stage 2]  Disagreement Sensitivity & Sharpening …")
    from stages.stage2_disagreement import run_stage2
    t0 = time.time()
    s2 = run_stage2(n_videos=n_videos)
    timings["Stage 2"] = time.time() - t0

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print("\n[Stage 3]  Gradient Norm Diagnosis & LR-Collapse …")
    from stages.stage3_diagnosis import run_stage3
    t0 = time.time()
    s3 = run_stage3(n_videos=n_s3_vids)
    timings["Stage 3"] = time.time() - t0

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    if not args.skip_stage4:
        print(f"\n[Stage 4]  Full Diffusion Benchmark ({n_videos} videos, {steps_s4} steps) …")
        from stages.stage4_benchmark import run_stage4
        t0 = time.time()
        s4, _ = run_stage4(n_videos=n_videos, steps=steps_s4)
        timings["Stage 4"] = time.time() - t0
    else:
        print("\n[Stage 4]  SKIPPED (--skip-stage4 flag set)")
        s4 = None

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    print("\n[Stage 5]  SOTA Literature Comparison Table …")
    from stages.stage5_sota import run_stage5
    t0 = time.time()
    s5 = run_stage5()
    timings["Stage 5"] = time.time() - t0

    # ── Stage 6 ──────────────────────────────────────────────────────────────
    print("\n[Stage 6]  Temporal Transformer Denoiser Self-Test …")
    from stages.stage6_transformer import run_stage6
    t0 = time.time()
    s6 = run_stage6()
    timings["Stage 6"] = time.time() - t0

    # ── Final summary ─────────────────────────────────────────────────────────
    _hline()
    print("\n  ALL STAGES COMPLETE — Summary")
    _hline("-")

    s1_sum = s1[0] if isinstance(s1, tuple) else s1
    print(f"  Stage 1 | Barycenter W2 wins: {s1_sum['barycenter_wins']}/{s1_sum['num_videos']} "
          f"({s1_sum['win_rate_pct']:.1f}%)  "
          f"avg improvement: {s1_sum['avg_improvement_pct']:+.2f}%")

    if s4 is not None:
        print(f"  Stage 4 | F1  Mine: {s4['f1']['mine_mean']:.2f}%  Base: {s4['f1']['base_mean']:.2f}%  "
              f"wins: {s4['f1']['wins']}/{s4['f1']['total']}")
        print(f"  Stage 4 | rho   Mine: {s4['spearman']['mine_mean']:.4f}  "
              f"Base: {s4['spearman']['base_mean']:.4f}")
        print(f"  Stage 4 | W2  Mine: {s4['w2']['mine_mean']:.4f}  "
              f"Base: {s4['w2']['base_mean']:.4f}")

    g_norm = s3.get("gradient_norms", {})
    ratio = g_norm.get("ratio_cramer_over_mse", float("nan"))
    print(f"  Stage 3 | Cramér / MSE gradient ratio: {ratio:.0f}x")

    print(f"  Stage 6 | Transformer self-test: "
          f"{'ALL PASS' if s6.get('all_checks_pass') else 'FAIL — check log'}")

    _hline("-")
    print("\n  Timing:")
    for stage, elapsed in timings.items():
        print(f"    {stage}: {elapsed:.1f}s")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    print(f"\n  All outputs in: {os.path.abspath(results_dir)}/")
    _hline()


if __name__ == "__main__":
    main()
