"""Train the explicitly non-benchmark demo models on all TVSum videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import standard_tvsum_cv as benchmark  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "model"
DEMO_SEED = 20261019
TARGETS = ("mean", "barycenter")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train(
    target_by_video: dict[str, np.ndarray],
    normalized_features: dict[str, np.ndarray],
    video_keys: list[str],
    epochs: int,
) -> dict[str, torch.Tensor]:
    random.seed(DEMO_SEED)
    np.random.seed(DEMO_SEED)
    torch.manual_seed(DEMO_SEED)
    model = benchmark.BiLSTMRegressor()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=benchmark.LEARNING_RATE,
        weight_decay=benchmark.WEIGHT_DECAY,
    )
    rng = np.random.default_rng(DEMO_SEED)
    feature_tensors = {
        key: torch.from_numpy(normalized_features[key]) for key in video_keys
    }
    target_tensors = {
        key: torch.from_numpy(target_by_video[key]) for key in video_keys
    }
    length_sorted_keys = sorted(
        video_keys, key=lambda key: feature_tensors[key].shape[0]
    )

    for _ in range(epochs):
        model.train()
        batches = [
            length_sorted_keys[index:index + benchmark.BATCH_SIZE]
            for index in range(0, len(length_sorted_keys), benchmark.BATCH_SIZE)
        ]
        rng.shuffle(batches)
        for batch_keys in batches:
            feature_batch = pad_sequence(
                [feature_tensors[key] for key in batch_keys], batch_first=True
            )
            target_batch = pad_sequence(
                [target_tensors[key] for key in batch_keys], batch_first=True
            )
            lengths = torch.as_tensor(
                [feature_tensors[key].shape[0] for key in batch_keys]
            )
            valid_mask = (
                torch.arange(feature_batch.shape[1])[None, :] < lengths[:, None]
            )
            prediction = model(feature_batch, valid_mask)
            loss = benchmark.loss_for_arm(
                prediction, target_batch, "cross_entropy", valid_mask
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_models(
    annotation_path: Path,
    info_path: Path,
    feature_path: Path,
    model_dir: Path = MODEL_DIR,
    torch_threads: int = 4,
) -> dict[str, Any]:
    if model_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing demo model directory: {model_dir}")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    videos = benchmark.load_tvsum_features(feature_path, annotation_path, info_path)
    if len(videos) != 50:
        raise ValueError(f"Expected all 50 real TVSum videos, found {len(videos)}.")
    video_keys = sorted(videos)
    raw_results_path = (
        PROJECT_ROOT / "results" / "standard_tvsum_cv_fixed" / "results.json"
    )
    previous_results = json.loads(raw_results_path.read_text(encoding="utf-8"))
    selected_epochs = [
        int(record["selected_epoch"])
        for record in previous_results["per_video_seed_records"]
        if record["arm"] == "B_mean_cross_entropy"
    ]
    if not selected_epochs:
        raise ValueError("No saved selected epochs found for B_mean_cross_entropy.")
    epochs = int(statistics.median(selected_epochs))
    if epochs < 1:
        raise ValueError(f"Invalid median selected epoch count: {epochs}.")

    scaler_mean, scaler_std = benchmark.fit_feature_scaler(videos, video_keys)
    normalized_features = {
        key: benchmark.normalize_features(video, scaler_mean, scaler_std)
        for key, video in videos.items()
    }
    target_by_video = {target: {} for target in TARGETS}
    for key, video in videos.items():
        mean, barycenter = benchmark.mean_and_barycenter(
            benchmark.annotation_distributions(video)
        )
        target_by_video["mean"][key] = mean
        target_by_video["barycenter"][key] = barycenter

    model_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        model_dir / "feature_scaler.npz", mean=scaler_mean, std=scaler_std
    )
    model_files: dict[str, str] = {}
    for target in TARGETS:
        state = _train(
            target_by_video[target], normalized_features, video_keys, epochs
        )
        weights_path = model_dir / f"{target}_target.pt"
        torch.save(state, weights_path)
        model_files[target] = _sha256(weights_path)

    config: dict[str, Any] = {
        "label": "demo model, trained on all TVSum, never used for any reported number",
        "training_data": "50 real TVSum videos; not used for any reported benchmark number",
        "architecture": "two-layer bidirectional LSTM, hidden_size=64, dropout=0.1",
        "objective": "B_mean_cross_entropy",
        "targets": list(TARGETS),
        "epochs": epochs,
        "epochs_source": {
            "file": "results/standard_tvsum_cv_fixed/results.json",
            "arm": "B_mean_cross_entropy",
            "selected_epoch_record_count": len(selected_epochs),
            "median_selected_epoch": epochs,
        },
        "seed": DEMO_SEED,
        "learning_rate": benchmark.LEARNING_RATE,
        "weight_decay": benchmark.WEIGHT_DECAY,
        "batch_size": benchmark.BATCH_SIZE,
        "feature_dimension": 1024,
        "feature_preprocessing": (
            "L2-normalize each pool5 feature vector, then apply feature_scaler.npz"
        ),
        "scaler_file": "feature_scaler.npz",
        "scaler_sha256": _sha256(model_dir / "feature_scaler.npz"),
        "weights": model_files,
        "feature_h5_sha256": _sha256(feature_path),
        "annotation_tsv_sha256": _sha256(annotation_path),
        "info_tsv_sha256": _sha256(info_path),
        "reported_result_json_sha256": _sha256(raw_results_path),
        "torch_version": torch.__version__,
        "torch_threads": torch_threads,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=benchmark.DEFAULT_FEATURES)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    config = train_models(
        args.annotations, args.info, args.features, torch_threads=args.torch_threads
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
