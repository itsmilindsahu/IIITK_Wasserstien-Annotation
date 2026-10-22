"""
run_stage.py

Unified Stage-Wise Pipeline Runner for Wasserstein Diffusion Video Summarization.

Usage:
  python run_stage.py --stage 1    # Run Stage 1: Mathematical Consensus & Oracle Barycenter
  python run_stage.py --stage 2    # Run Stage 2: Stress-Testing & Disagreement Sensitivity
  python run_stage.py --stage 3    # Run Stage 3: Baseline Diagnostics & Gradient Verification
  python run_stage.py --stage 4    # Run Stage 4: Legacy annotation-only diagnostics
  python run_stage.py --stage 5    # Run Stage 5: Real TVSum five-split ablation benchmark
  python run_stage.py --all        # Run all stages sequentially
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

STAGES = {
    1: {
        "name": "Stage 1: Mathematical Consensus & Optimal Transport Barycenter",
        "script": os.path.join(PROJECT_ROOT, "stages", "stage1_consensus", "test_barycenter.py"),
        "description": "Validates exact 1D W2 barycenter computation via quantile-function averaging vs. arithmetic mean."
    },
    2: {
        "name": "Stage 2: Stress-Testing & Annotator Disagreement Sensitivity",
        "script": os.path.join(PROJECT_ROOT, "stages", "stage2_disagreement", "sharpening_50video_benchmark.py"),
        "description": "Evaluates barycenter gain across temperature-sharpened distributions and native disagreement terciles."
    },
    3: {
        "name": "Stage 3: Methodological Diagnosis & Euclidean Baseline Failure Analysis",
        "script": os.path.join(PROJECT_ROOT, "code", "generate_final_figures.py"),
        "description": "Verifies Cramér vs MSE gradient norm (3300x larger) and rules out learning rate artifacts."
    },
    4: {
        "name": "Stage 4: Legacy Annotation-Only Diagnostics",
        "script": os.path.join(PROJECT_ROOT, "code", "compare_downstream_metrics.py"),
        "description": "Historical annotation-only diagnostic; not the standard video-conditioned TVSum benchmark."
    },
    5: {
        "name": "Stage 5: Real TVSum Five-Split Ablation Benchmark",
        "script": os.path.join(PROJECT_ROOT, "tvsum_cv_entry.py"),
        "description": "Runs the feature-conditioned TVSum target/loss ablations with the supplied DSNet splits."
    }
}


def run_single_stage(stage_num, fast=False):
    stage_info = STAGES.get(stage_num)
    if not stage_info:
        print(f"[ERROR] Invalid stage: {stage_num}. Choose 1 to 5.")
        return False

    print("\n" + "=" * 80)
    print(f"[*] EXECUTING {stage_info['name']}")
    print(f"[*] Description: {stage_info['description']}")
    print(f"[*] Target Script: {stage_info['script']}")
    print("=" * 80 + "\n")

    env = os.environ.copy()
    code_dir = os.path.join(PROJECT_ROOT, "code")
    env["PYTHONPATH"] = f"{PROJECT_ROOT};{code_dir};" + env.get("PYTHONPATH", "")

    cmd = [PYTHON_EXE, stage_info["script"]]
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if ret.returncode == 0:
        print(f"\n[+] {stage_info['name']} COMPLETED SUCCESSFULLY.\n")
        return True
    else:
        print(f"\n[-] {stage_info['name']} FAILED with exit code {ret.returncode}.\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Unified Stage-Wise Pipeline Runner")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Stage number to run (1 to 5)")
    parser.add_argument("--all", action="store_true", help="Run all stages sequentially")
    parser.add_argument("--fast", action="store_true", help="Use fast mode where supported")
    args = parser.parse_args()

    if args.all:
        for s in range(1, 6):
            success = run_single_stage(s, fast=args.fast)
            if not success:
                print(f"[HALTED] Pipeline stopped due to failure in Stage {s}.")
                sys.exit(1)
        print("\n[ALL COMPLETE] All stages executed successfully!")
    elif args.stage:
        success = run_single_stage(args.stage, fast=args.fast)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
