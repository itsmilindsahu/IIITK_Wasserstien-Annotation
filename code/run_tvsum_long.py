"""Run longer validation-selected TVSum experiments for arms A and B only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import standard_tvsum_cv as benchmark  # noqa: E402


LONG_ARMS = ("A_mean_mse", "B_mean_cross_entropy")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "standard_tvsum_cv_long"


def _summarize_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    fold_video_values: dict[str, dict[str, list[float]]] = {}
    selected_epochs: dict[tuple[int, int], int] = {}
    for row in records:
        fold_video = f"fold_{row['fold']}::{row['video_key']}"
        if row["metric"] not in ("f1", "spearman_rho", "kendall_tau"):
            continue
        fold_video_values.setdefault(
            fold_video, {"f1": [], "spearman_rho": [], "kendall_tau": []}
        )[row["metric"]].append(float(row["value"]))
        selected_epochs[(row["fold"], row["seed"])] = row["selected_epoch"]

    metric_means = {
        metric: [
            float(np.mean(values[metric]))
            for values in fold_video_values.values()
        ]
        for metric in ("f1", "spearman_rho", "kendall_tau")
    }
    seed_std = [
        float(np.std(values["f1"], ddof=1))
        for values in fold_video_values.values()
        if len(values["f1"]) > 1
    ]
    return {
        "f1_mean": float(np.mean(metric_means["f1"])),
        "f1_mean_seed_std": float(np.mean(seed_std)),
        "f1_std_across_fold_video_instances": float(
            np.std(metric_means["f1"], ddof=1)
        ),
        "spearman_rho_mean": float(np.mean(metric_means["spearman_rho"])),
        "kendall_tau_mean": float(np.mean(metric_means["kendall_tau"])),
        "selected_epoch_counts": {
            str(epoch): int(count)
            for epoch, count in sorted(Counter(selected_epochs.values()).items())
        },
        "selected_epoch_by_fold_seed": {
            f"fold_{fold}_seed_{seed}": epoch
            for (fold, seed), epoch in sorted(selected_epochs.items())
        },
    }


def run_long_experiment(
    h5_path: Path,
    annotation_path: Path,
    info_path: Path,
    split_path: Path,
    output_dir: Path,
    epoch_budgets: list[int],
    torch_threads: int = 4,
) -> dict[str, Any]:
    if not epoch_budgets or any(epochs < 1 for epochs in epoch_budgets):
        raise ValueError("Provide positive epoch budgets.")
    if len(epoch_budgets) != len(set(epoch_budgets)):
        raise ValueError("Epoch budgets must be unique.")

    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = benchmark.load_tvsum_features(h5_path, annotation_path, info_path)
    folds = benchmark.load_dsnet_folds(split_path, set(videos))
    all_settings: dict[str, Any] = {}
    curve_data: dict[int, list[float]] = {}
    total_started = time.time()

    for epoch_budget in epoch_budgets:
        started = time.time()
        records: dict[str, list[dict[str, Any]]] = {
            arm: [] for arm in LONG_ARMS
        }
        validation_histories: dict[str, list[float]] = {}
        fold_logs = []

        for fold_index, fold in enumerate(folds, start=1):
            train_keys, validation_keys = benchmark.split_train_validation(
                fold["train"], fold_index
            )
            test_keys = fold["test"]
            partitions = [set(train_keys), set(validation_keys), set(test_keys)]
            if any(
                partitions[left] & partitions[right]
                for left in range(3)
                for right in range(left + 1, 3)
            ):
                raise AssertionError(f"Fold {fold_index} has data leakage.")
            if set.union(*partitions) != set(videos):
                raise AssertionError(f"Fold {fold_index} does not partition videos.")
            print(
                f"LONG_FOLD {fold_index} train_ids="
                f"{[videos[key].video_id for key in train_keys]}",
                flush=True,
            )
            print(
                f"LONG_FOLD {fold_index} val_ids="
                f"{[videos[key].video_id for key in validation_keys]}",
                flush=True,
            )
            print(
                f"LONG_FOLD {fold_index} test_ids="
                f"{[videos[key].video_id for key in test_keys]}",
                flush=True,
            )
            mean, std = benchmark.fit_feature_scaler(videos, train_keys)
            normalized = {
                key: benchmark.normalize_features(video, mean, std)
                for key, video in videos.items()
            }
            targets = {
                key: benchmark.mean_and_barycenter(
                    benchmark.annotation_distributions(videos[key])
                )[0]
                for key in train_keys
            }
            fold_logs.append({
                "fold": fold_index,
                "train_ids": [videos[key].video_id for key in train_keys],
                "validation_ids": [
                    videos[key].video_id for key in validation_keys
                ],
                "test_ids": [videos[key].video_id for key in test_keys],
            })

            for seed in benchmark.SEEDS:
                base_seed = benchmark.stable_seed(
                    f"tvsum-fold-{fold_index}-seed-{seed}"
                )
                for arm in LONG_ARMS:
                    loss = "mse" if arm == "A_mean_mse" else "cross_entropy"
                    validation_history: list[float] | None = None
                    if fold_index == 1 and seed == benchmark.SEEDS[0] and arm == "B_mean_cross_entropy":
                        validation_history = []
                    state, selected_epoch, selected_validation_f1 = (
                        benchmark.train_candidate(
                            videos,
                            normalized,
                            train_keys,
                            validation_keys,
                            targets,
                            loss,
                            base_seed,
                            epoch_budget,
                            validation_history=validation_history,
                        )
                    )
                    if validation_history is not None:
                        validation_histories[arm] = validation_history
                    model = benchmark.TemporalConvRegressor()
                    model.load_state_dict(state)
                    for key in test_keys:
                        metrics = benchmark.score_video(
                            benchmark.predict_distribution(model, normalized[key]),
                            videos[key],
                        )
                        for metric, value in metrics.items():
                            records[arm].append({
                                "fold": fold_index,
                                "seed": seed,
                                "arm": arm,
                                "video_key": key,
                                "video_id": videos[key].video_id,
                                "metric": metric,
                                "value": value,
                                "selected_epoch": selected_epoch,
                            })
                    print(
                        f"LONG epochs={epoch_budget} fold={fold_index} "
                        f"seed={seed} arm={arm} selected_epoch={selected_epoch} "
                        f"validation_f1={selected_validation_f1:.3f}",
                        flush=True,
                    )

        summary = {
            arm: _summarize_arm(records[arm])
            for arm in LONG_ARMS
        }
        elapsed = time.time() - started
        setting = {
            "protocol": {
                "epoch_budget": epoch_budget,
                "seeds": list(benchmark.SEEDS),
                "arms": list(LONG_ARMS),
                "split_count": len(fold_logs),
                "fold_video_test_instances": sum(
                    len(fold["test_ids"]) for fold in fold_logs
                ),
                "unique_test_videos": len({
                    video_id
                    for fold in fold_logs
                    for video_id in fold["test_ids"]
                }),
                "validation_checkpoint_selection": True,
                "no_test_fold_tuning": True,
                "backbone": (
                    "Linear(1024,64)-Conv1d(64,64,k=5)-"
                    "Conv1d(64,64,k=3)-Conv1d(64,1)"
                ),
                "losses": {
                    "A_mean_mse": "mean target + MSE",
                    "B_mean_cross_entropy": "mean target + cross-entropy",
                },
                "feature_sha256": benchmark.sha256_file(h5_path),
                "annotation_sha256": benchmark.sha256_file(annotation_path),
                "info_sha256": benchmark.sha256_file(info_path),
                "split_sha256": benchmark.sha256_file(split_path),
            },
            "summary": summary,
            "folds": fold_logs,
            "records": records,
            "validation_curve": validation_histories,
            "elapsed_seconds": elapsed,
        }
        json_path = output_dir / f"epochs_{epoch_budget}.json"
        if json_path.exists():
            raise FileExistsError(f"Refusing to overwrite {json_path}")
        json_path.write_text(
            json.dumps(setting, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        all_settings[str(epoch_budget)] = setting
        if "B_mean_cross_entropy" in validation_histories:
            curve_data[epoch_budget] = validation_histories[
                "B_mean_cross_entropy"
            ]
        print(
            f"LONG_SETTING_COMPLETE epochs={epoch_budget} "
            f"elapsed_seconds={elapsed:.3f} summary="
            f"{json.dumps(summary, sort_keys=True)}",
            flush=True,
        )

    csv_path = output_dir / "long_training_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch_budget", "arm", "f1_mean", "f1_mean_seed_std",
            "spearman_rho_mean", "kendall_tau_mean",
            "selected_epoch_counts", "elapsed_seconds",
        ])
        for epoch_budget in epoch_budgets:
            setting = all_settings[str(epoch_budget)]
            for arm in LONG_ARMS:
                row = setting["summary"][arm]
                writer.writerow([
                    epoch_budget,
                    arm,
                    row["f1_mean"],
                    row["f1_mean_seed_std"],
                    row["spearman_rho_mean"],
                    row["kendall_tau_mean"],
                    json.dumps(row["selected_epoch_counts"], sort_keys=True),
                    setting["elapsed_seconds"],
                ])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    for epoch_budget in epoch_budgets:
        values = curve_data.get(epoch_budget)
        if values is not None:
            axis.plot(
                np.arange(1, len(values) + 1),
                values,
                marker="o",
                label=f"B_mean_cross_entropy, budget={epoch_budget}",
            )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation F1 (%)")
    axis.set_title("TVSum validation learning curves (fold 1, seed 11)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    plot_path = output_dir / "validation_f1_fold1_seed11.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    total_elapsed = time.time() - total_started
    manifest = {
        "epoch_budgets": epoch_budgets,
        "arms": list(LONG_ARMS),
        "settings": {
            budget: {
                "summary": all_settings[str(budget)]["summary"],
                "elapsed_seconds": all_settings[str(budget)]["elapsed_seconds"],
            }
            for budget in epoch_budgets
        },
        "total_elapsed_seconds": total_elapsed,
        "summary_csv": csv_path.name,
        "validation_plot": plot_path.name,
    }
    manifest_path = output_dir / "long_training_summary.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"LONG_EXPERIMENT_COMPLETE elapsed_seconds={total_elapsed:.3f} "
        f"manifest={manifest_path} plot={plot_path}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=benchmark.DEFAULT_FEATURES)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            Path(os.environ[benchmark.ANNOTATION_ENV])
            if os.environ.get(benchmark.ANNOTATION_ENV)
            else None
        ),
    )
    parser.add_argument(
        "--info",
        type=Path,
        default=(
            Path(os.environ[benchmark.INFO_ENV])
            if os.environ.get(benchmark.INFO_ENV)
            else None
        ),
    )
    parser.add_argument("--splits", type=Path, default=benchmark.DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=[30, 60, 100]
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.annotations is None:
        parser.error(f"Provide --annotations or set {benchmark.ANNOTATION_ENV}.")
    if args.info is None:
        parser.error(f"Provide --info or set {benchmark.INFO_ENV}.")
    run_long_experiment(
        args.features,
        args.annotations,
        args.info,
        args.splits,
        args.output_dir,
        args.epochs,
        args.torch_threads,
    )


if __name__ == "__main__":
    main()
