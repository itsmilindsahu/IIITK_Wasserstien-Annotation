"""Compare temporal-convolution and BiLSTM backbones on TVSum arm B."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import standard_tvsum_cv as benchmark  # noqa: E402


BACKBONES = ("temporal_conv", "bilstm")
METRICS = ("f1", "spearman_rho", "kendall_tau")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_fold_video: dict[str, dict[str, list[float]]] = {
        backbone: defaultdict(lambda: {metric: [] for metric in METRICS})
        for backbone in BACKBONES
    }
    selected_epochs: dict[str, dict[tuple[int, int], int]] = {
        backbone: {} for backbone in BACKBONES
    }
    for row in records:
        backbone = row["backbone"]
        if row["metric"] not in METRICS:
            continue
        key = f"fold_{row['fold']}::{row['video_key']}"
        per_fold_video[backbone][key][row["metric"]].append(float(row["value"]))
        selected_epochs[backbone][(row["fold"], row["seed"])] = int(
            row["selected_epoch"]
        )

    per_fold_video_mean: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, Any] = {}
    for backbone in BACKBONES:
        per_fold_video_mean[backbone] = {
            key: {
                metric: float(np.mean(values[metric]))
                for metric in METRICS
            }
            for key, values in per_fold_video[backbone].items()
        }
        f1_seed_std = [
            float(np.std(values["f1"], ddof=1))
            for values in per_fold_video[backbone].values()
            if len(values["f1"]) > 1
        ]
        summary[backbone] = {
            "f1_mean": float(np.mean([
                row["f1"] for row in per_fold_video_mean[backbone].values()
            ])),
            "f1_mean_seed_std": float(np.mean(f1_seed_std)),
            "f1_std_across_fold_video_instances": float(np.std([
                row["f1"] for row in per_fold_video_mean[backbone].values()
            ], ddof=1)),
            "spearman_rho_mean": float(np.mean([
                row["spearman_rho"]
                for row in per_fold_video_mean[backbone].values()
            ])),
            "kendall_tau_mean": float(np.mean([
                row["kendall_tau"]
                for row in per_fold_video_mean[backbone].values()
            ])),
            "selected_epoch_counts": {
                str(epoch): int(count)
                for epoch, count in sorted(
                    Counter(selected_epochs[backbone].values()).items()
                )
            },
            "selected_epoch_by_fold_seed": {
                f"fold_{fold}_seed_{seed}": epoch
                for (fold, seed), epoch in sorted(selected_epochs[backbone].items())
            },
        }

    unique_video_scores: dict[str, dict[str, dict[str, float]]] = {}
    for backbone in BACKBONES:
        by_video: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {metric: [] for metric in METRICS}
        )
        for fold_video, metrics in per_fold_video_mean[backbone].items():
            video_key = fold_video.split("::", 1)[1]
            for metric in METRICS:
                by_video[video_key][metric].append(metrics[metric])
        unique_video_scores[backbone] = {
            key: {
                metric: float(np.mean(values[metric]))
                for metric in METRICS
            }
            for key, values in by_video.items()
        }
    common_videos = sorted(
        set(unique_video_scores[BACKBONES[0]])
        & set(unique_video_scores[BACKBONES[1]])
    )
    paired_results = {}
    for metric in METRICS:
        conv = [
            unique_video_scores["temporal_conv"][key][metric]
            for key in common_videos
        ]
        bilstm = [
            unique_video_scores["bilstm"][key][metric]
            for key in common_videos
        ]
        paired_results[metric] = {
            "mean_difference": float(np.mean(np.asarray(bilstm) - conv)),
            **benchmark.paired_statistics(bilstm, conv),
        }
    summary["paired_bilstm_minus_temporal_conv"] = paired_results
    return {
        "summary": summary,
        "per_fold_video_mean": per_fold_video_mean,
        "unique_test_video_scores": unique_video_scores,
    }


def run_comparison(
    h5_path: Path,
    annotation_path: Path,
    info_path: Path,
    split_path: Path,
    output_path: Path,
    epochs: int = benchmark.DEFAULT_EPOCHS,
    torch_threads: int = 4,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output_path}")

    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    videos = benchmark.load_tvsum_features(
        h5_path, annotation_path, info_path
    )
    folds = benchmark.load_dsnet_folds(split_path, set(videos))
    records: list[dict[str, Any]] = []
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
            raise AssertionError(f"Fold {fold_index} has train/validation/test overlap.")
        if set.union(*partitions) != set(videos):
            raise AssertionError(f"Fold {fold_index} does not partition all videos.")

        train_ids = [videos[key].video_id for key in train_keys]
        validation_ids = [videos[key].video_id for key in validation_keys]
        test_ids = [videos[key].video_id for key in test_keys]
        print(
            f"COMPARE_FOLD {fold_index} train_ids={train_ids}",
            flush=True,
        )
        print(
            f"COMPARE_FOLD {fold_index} validation_ids={validation_ids}",
            flush=True,
        )
        print(
            f"COMPARE_FOLD {fold_index} test_ids={test_ids}",
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
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "test_ids": test_ids,
        })

        for seed in benchmark.SEEDS:
            model_seed = benchmark.stable_seed(
                f"tvsum-fold-{fold_index}-seed-{seed}"
            )
            for backbone in BACKBONES:
                state, selected_epoch, validation_f1 = benchmark.train_candidate(
                    videos=videos,
                    normalized_features=normalized,
                    train_keys=train_keys,
                    validation_keys=validation_keys,
                    targets=targets,
                    loss_name="cross_entropy",
                    seed=model_seed,
                    epochs=epochs,
                    backbone=backbone,
                )
                model = benchmark.build_regressor(backbone)
                model.load_state_dict(state)
                for key in test_keys:
                    metrics = benchmark.score_video(
                        benchmark.predict_distribution(model, normalized[key]),
                        videos[key],
                    )
                    for metric, value in metrics.items():
                        records.append({
                            "fold": fold_index,
                            "seed": seed,
                            "arm": "B_mean_cross_entropy",
                            "backbone": backbone,
                            "video_key": key,
                            "video_id": videos[key].video_id,
                            "metric": metric,
                            "value": value,
                            "selected_epoch": selected_epoch,
                        })
                print(
                    f"COMPARE fold={fold_index} seed={seed} backbone={backbone} "
                    f"selected_epoch={selected_epoch} "
                    f"validation_f1={validation_f1:.3f}",
                    flush=True,
                )

    aggregates = summarize_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "dataset": "TVSum50",
            "arm": "B_mean_cross_entropy",
            "target": "training-video annotator mean distribution",
            "loss": "cross-entropy",
            "backbones": {
                "temporal_conv": (
                    "Linear(1024,64)-Conv1d(64,64,k=5)-"
                    "Conv1d(64,64,k=3)-Conv1d(64,1)"
                ),
                "bilstm": (
                    "BiLSTM(input=1024,hidden=64,directions=2,layers=2)-"
                    "Linear(128,1)"
                ),
            },
            "epoch_budget": epochs,
            "seeds": list(benchmark.SEEDS),
            "split_count": len(fold_logs),
            "fold_video_test_instances": sum(
                len(fold["test_ids"]) for fold in fold_logs
            ),
            "unique_test_videos": len({
                video_id
                for fold in fold_logs
                for video_id in fold["test_ids"]
            }),
            "validation_fraction_of_fold_train": 0.1,
            "checkpoint_selection": "highest validation F1; test fold not used",
            "feature_sha256": benchmark.sha256_file(h5_path),
            "annotation_sha256": benchmark.sha256_file(annotation_path),
            "info_sha256": benchmark.sha256_file(info_path),
            "split_sha256": benchmark.sha256_file(split_path),
            "optimizer": "AdamW",
            "learning_rate": benchmark.LEARNING_RATE,
            "weight_decay": benchmark.WEIGHT_DECAY,
            "batch_size": benchmark.BATCH_SIZE,
            "torch_threads": torch_threads,
        },
        **aggregates,
        "folds": fold_logs,
        "records": records,
        "elapsed_seconds": time.time() - started,
    }
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)

    table_path = output_path.with_name("backbone_comparison.csv")
    with table_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "backbone", "arm", "f1_mean", "f1_mean_seed_std",
            "spearman_rho_mean", "kendall_tau_mean",
        ])
        for backbone in BACKBONES:
            row = result["summary"][backbone]
            writer.writerow([
                backbone,
                "B_mean_cross_entropy",
                row["f1_mean"],
                row["f1_mean_seed_std"],
                row["spearman_rho_mean"],
                row["kendall_tau_mean"],
            ])
    print(
        f"COMPARISON_COMPLETE elapsed_seconds={result['elapsed_seconds']:.3f} "
        f"json={output_path} table={table_path} "
        f"summary={json.dumps(result['summary'], sort_keys=True)}",
        flush=True,
    )
    return result


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
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "standard_tvsum_cv_bilstm"
            / "backbone_comparison.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=benchmark.DEFAULT_EPOCHS)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.annotations is None:
        parser.error(f"Provide --annotations or set {benchmark.ANNOTATION_ENV}.")
    if args.info is None:
        parser.error(f"Provide --info or set {benchmark.INFO_ENV}.")
    run_comparison(
        args.features,
        args.annotations,
        args.info,
        args.splits,
        args.output,
        args.epochs,
        args.torch_threads,
    )


if __name__ == "__main__":
    main()
