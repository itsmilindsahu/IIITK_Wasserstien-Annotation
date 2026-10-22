"""Video-conditioned TVSum evaluation using the public DSNet feature/split files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "datasets" / "eccv16_dataset_tvsum_google_pool5.h5"
DEFAULT_SPLITS = PROJECT_ROOT / "splits" / "tvsum.yml"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "standard_tvsum_cv" / "results.json"
ANNOTATION_ENV = "TVSUM_ANNOTATION_TSV"
INFO_ENV = "TVSUM_INFO_TSV"
FEATURES_ENV = "TVSUM_FEATURES_H5"
BUDGET_FRACTION = 0.15
SEEDS = (11, 23, 37, 53, 71)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("A_mean_mse", "B_mean_cross_entropy", "C_mean_cramer",
        "D_barycenter_mse", "E_barycenter_cramer", "F_blended_cramer")
MODEL_WIDTH = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 10
BATCH_SIZE = 4


@dataclass(frozen=True)
class TVSumVideo:
    key: str
    video_id: str
    features: np.ndarray
    annotation_scores: np.ndarray
    user_summary: np.ndarray
    change_points: np.ndarray
    frames_per_segment: np.ndarray
    picks: np.ndarray
    n_frames: int
    gtscore: np.ndarray
    gtsummary: np.ndarray


class TemporalConvRegressor(nn.Module):
    """Small temporal 1D-convolution regressor shared by every ablation arm."""

    def __init__(self, feature_dim: int = 1024, width: int = MODEL_WIDTH) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, width)
        self.temporal_1 = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.temporal_2 = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.output_projection = nn.Conv1d(width, 1, kernel_size=1)

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = F.gelu(self.input_projection(features)).transpose(1, 2)
        residual = hidden
        hidden = self.dropout(F.gelu(self.temporal_1(hidden)))
        hidden = F.gelu(self.temporal_2(hidden) + residual)
        logits = self.output_projection(hidden).squeeze(1)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -torch.inf)
        return torch.softmax(logits, dim=-1)


class BiLSTMRegressor(nn.Module):
    """Two-layer bidirectional LSTM regressor with 64 units per direction."""

    def __init__(self, feature_dim: int = 1024, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.output_projection = nn.Linear(hidden_size * 2, 1)

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if valid_mask is None:
            lengths = torch.full(
                (features.shape[0],),
                features.shape[1],
                dtype=torch.long,
                device="cpu",
            )
        else:
            lengths = valid_mask.sum(dim=-1).to(device="cpu", dtype=torch.long)
            if torch.any(lengths <= 0):
                raise ValueError("Every BiLSTM input sequence must contain a valid frame.")
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.lstm(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=features.shape[1],
        )
        logits = self.output_projection(hidden).squeeze(-1)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -torch.inf)
        return torch.softmax(logits, dim=-1)


def build_regressor(backbone: str) -> nn.Module:
    if backbone == "temporal_conv":
        return TemporalConvRegressor()
    if backbone == "bilstm":
        return BiLSTMRegressor()
    raise ValueError(f"Unknown backbone {backbone!r}.")


def stable_seed(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_annotations(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Official TVSum annotations not found at {path}; set {ANNOTATION_ENV}."
        )
    grouped: dict[str, list[np.ndarray]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(
                    f"Expected official three-column TSV at line {line_number}; "
                    f"found {len(row)} columns."
                )
            try:
                scores = np.asarray([int(value) for value in row[2].split(",")], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"Invalid annotation score on line {line_number}.") from exc
            if scores.size == 0 or not np.isfinite(scores).all():
                raise ValueError(f"Empty or non-finite annotations on line {line_number}.")
            if np.any((scores < 1) | (scores > 5)):
                raise ValueError(f"Annotation scores outside [1, 5] on line {line_number}.")
            grouped.setdefault(row[0], []).append(scores)

    annotations: dict[str, np.ndarray] = {}
    for video_id, user_scores in grouped.items():
        lengths = {score.size for score in user_scores}
        if len(user_scores) != 20 or len(lengths) != 1:
            raise ValueError(
                f"{video_id} must have 20 equal-length annotator vectors; "
                f"found {len(user_scores)} users and lengths {sorted(lengths)}."
            )
        annotations[video_id] = np.stack(user_scores)
    if len(annotations) != 50:
        raise ValueError(f"Expected 50 real TVSum video IDs; found {len(annotations)}.")
    return annotations


def read_video_info(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Official TVSum video metadata not found at {path}; set {INFO_ENV}. "
            "This file supplies the official video_ID to video_N mapping."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if len(rows) != 50 or not {"video_id", "category"}.issubset(rows[0]):
        raise ValueError("Expected 50 rows with category and video_id in official info TSV.")
    ids = [row["video_id"] for row in rows]
    if len(set(ids)) != 50:
        raise ValueError("Official metadata contains duplicate video IDs.")
    return rows


def load_tvsum_features(
    h5_path: Path, annotation_path: Path, info_path: Path
) -> dict[str, TVSumVideo]:
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Preprocessed TVSum feature file not found at {h5_path}. "
            f"Set {FEATURES_ENV} to eccv16_dataset_tvsum_google_pool5.h5."
        )
    raw_annotations = read_raw_annotations(annotation_path)
    info_rows = read_video_info(info_path)
    info_ids = {row["video_id"] for row in info_rows}
    if info_ids != set(raw_annotations):
        raise ValueError("Official metadata and raw annotation video IDs do not match.")

    videos: dict[str, TVSumVideo] = {}
    with h5py.File(h5_path, "r") as h5_file:
        expected_keys = {f"video_{index}" for index in range(1, 51)}
        if set(h5_file.keys()) != expected_keys:
            raise ValueError("HDF5 groups must be exactly video_1 through video_50.")
        for index, info in enumerate(info_rows, start=1):
            key = f"video_{index}"
            group = h5_file[key]
            features = np.asarray(group["features"], dtype=np.float32)
            picks = np.asarray(group["picks"], dtype=np.int64).reshape(-1)
            n_frames = int(np.asarray(group["n_frames"]).reshape(()))
            user_summary = np.asarray(group["user_summary"], dtype=np.uint8)
            change_points = np.asarray(group["change_points"], dtype=np.int64)
            frames_per_segment = np.asarray(group["n_frame_per_seg"], dtype=np.int64).reshape(-1)
            gtscore = np.asarray(group["gtscore"], dtype=np.float32).reshape(-1)
            gtsummary = np.asarray(group["gtsummary"]).reshape(-1)
            video_id = info["video_id"]

            if features.ndim != 2 or features.shape[1] != 1024:
                raise ValueError(f"{key} must contain 1024-D features, got {features.shape}.")
            if (
                features.shape[0] != picks.size
                or gtscore.size != picks.size
                or gtsummary.size != picks.size
            ):
                raise ValueError(
                    f"{key} features, gtscore, gtsummary, and picks lengths disagree."
                )
            if user_summary.shape != (20, n_frames):
                raise ValueError(
                    f"{key} user_summary must be (20, {n_frames}), got {user_summary.shape}."
                )
            if not np.isin(user_summary, (0, 1)).all():
                raise ValueError(f"{key} user_summary must be binary.")
            if not np.isin(gtsummary, (0, 1)).all():
                raise ValueError(f"{key} gtsummary must be binary.")
            if not np.isfinite(features).all() or not np.isfinite(gtscore).all():
                raise ValueError(f"{key} contains non-finite features or gtscore.")
            if picks.size == 0 or picks[0] != 0 or np.any(np.diff(picks) <= 0):
                raise ValueError(f"{key} picks must be nonempty, strictly increasing, and start at 0.")
            if picks[-1] >= n_frames:
                raise ValueError(f"{key} contains a pick outside its original frame range.")
            if raw_annotations[video_id].shape[1] != n_frames:
                raise ValueError(
                    f"Raw annotations for {video_id} have "
                    f"{raw_annotations[video_id].shape[1]} frames, HDF5 has {n_frames}."
                )
            if change_points.ndim != 2 or change_points.shape[1] != 2:
                raise ValueError(f"{key} change_points must be a (segments, 2) array.")
            if change_points.shape[0] != frames_per_segment.size:
                raise ValueError(f"{key} change points and segment lengths disagree.")
            if np.any(frames_per_segment <= 0) or not np.array_equal(
                change_points[:, 1] - change_points[:, 0] + 1,
                frames_per_segment,
            ):
                raise ValueError(f"{key} shot boundaries and segment lengths disagree.")
            if int(frames_per_segment.sum()) != n_frames:
                raise ValueError(f"{key} segment lengths do not sum to n_frames.")
            if (
                change_points[0, 0] != 0
                or change_points[-1, 1] != n_frames - 1
                or np.any(change_points[:, 1] < change_points[:, 0])
                or np.any(change_points[1:, 0] != change_points[:-1, 1] + 1)
            ):
                raise ValueError(f"{key} change_points do not partition the original frames.")

            # The TSV contains ratings at original-frame resolution; picks are the
            # exact frame indices used to extract each row of 1024-D features.
            aligned_scores = raw_annotations[video_id][:, picks]
            if aligned_scores.shape != (20, features.shape[0]):
                raise ValueError(f"{key} raw per-user scores did not align to picks.")
            expected_gtscore = (aligned_scores.mean(axis=0) - 1) / 4
            if not np.allclose(gtscore, expected_gtscore, rtol=0, atol=1e-5):
                max_difference = float(np.max(np.abs(gtscore - expected_gtscore)))
                raise ValueError(
                    f"{key} gtscore does not match normalized raw annotations for "
                    f"{video_id} at picks (max absolute difference "
                    f"{max_difference:.6g}); verify the HDF5-to-video mapping."
                )
            videos[key] = TVSumVideo(
                key=key,
                video_id=video_id,
                features=features,
                annotation_scores=aligned_scores,
                user_summary=user_summary,
                change_points=change_points,
                frames_per_segment=frames_per_segment,
                picks=picks,
                n_frames=n_frames,
                gtscore=gtscore,
                gtsummary=gtsummary.astype(np.uint8),
            )
            print(
                f"DATA {key:8s} id={video_id:11s} "
                f"frames={n_frames:5d} picks={picks.size:4d} "
                f"shots={change_points.shape[0]:3d} users=20 feat_dim=1024"
            )
    if {video.video_id for video in videos.values()} != info_ids:
        raise ValueError("HDF5 group to official raw video ID mapping is incomplete.")
    return videos


def load_dsnet_folds(split_path: Path, known_keys: set[str]) -> list[dict[str, list[str]]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"DSNet split file not found: {split_path}.")
    with split_path.open("r", encoding="utf-8") as file:
        raw_folds = yaml.safe_load(file)
    if not isinstance(raw_folds, list) or len(raw_folds) != 5:
        raise ValueError("Canonical DSNet TVSum split must contain exactly five folds.")

    folds = []
    for fold_index, entry in enumerate(raw_folds, start=1):
        def decode(keys: Any, field: str) -> list[str]:
            if not isinstance(keys, list):
                raise ValueError(f"Fold {fold_index} {field} must be a list.")
            decoded = []
            for item in keys:
                match = re.search(r"(video_\d+)$", str(item))
                if match is None:
                    raise ValueError(f"Unrecognized DSNet video key {item!r}.")
                decoded.append(match.group(1))
            if len(decoded) != len(set(decoded)):
                raise ValueError(f"Fold {fold_index} {field} contains duplicate keys.")
            if not set(decoded).issubset(known_keys):
                raise ValueError(f"Fold {fold_index} references unknown TVSum videos.")
            return decoded

        train_keys = decode(entry.get("train_keys"), "train_keys")
        test_keys = decode(entry.get("test_keys"), "test_keys")
        if len(train_keys) != 40 or len(test_keys) != 10:
            raise ValueError(f"Fold {fold_index} must have 40 train and 10 test videos.")
        if set(train_keys) & set(test_keys):
            raise ValueError(f"Fold {fold_index} has train/test overlap.")
        folds.append({"train": train_keys, "test": test_keys})
        if set(train_keys) | set(test_keys) != known_keys:
            raise ValueError(f"Fold {fold_index} does not partition all TVSum videos.")
    return folds


def split_train_validation(
    train_keys: list[str], fold_index: int
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(stable_seed(f"tvsum-validation-fold-{fold_index}"))
    order = np.asarray(train_keys, dtype=object)
    rng.shuffle(order)
    validation_count = max(1, int(round(len(train_keys) * 0.1)))
    validation = [str(value) for value in order[:validation_count]]
    training = [str(value) for value in order[validation_count:]]
    if set(training) & set(validation) or set(training) | set(validation) != set(train_keys):
        raise AssertionError("Generated train/validation split is invalid.")
    return training, validation


def fit_feature_scaler(videos: dict[str, TVSumVideo], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feature_sum = np.zeros(1024, dtype=np.float64)
    square_sum = np.zeros(1024, dtype=np.float64)
    count = 0
    for key in keys:
        values = videos[key].features.astype(np.float64, copy=False)
        feature_sum += values.sum(axis=0)
        square_sum += np.square(values).sum(axis=0)
        count += values.shape[0]
    mean = feature_sum / count
    variance = np.maximum(square_sum / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def normalize_features(video: TVSumVideo, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray((video.features - mean) / std, dtype=np.float32)


def annotation_distributions(video: TVSumVideo) -> np.ndarray:
    scores = video.annotation_scores.astype(np.float64)
    totals = scores.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"{video.key} has an annotation with non-positive mass.")
    return (scores / totals).astype(np.float32)


def mean_and_barycenter(distributions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distributions = np.asarray(distributions, dtype=np.float64)
    if distributions.ndim != 2 or distributions.shape[0] < 1:
        raise ValueError("distributions must have shape (users, frames).")
    if not np.isfinite(distributions).all() or np.any(distributions < 0):
        raise ValueError("distributions must be finite and nonnegative.")
    totals = distributions.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every annotator distribution must have positive mass.")
    distributions = distributions / totals[:, None]
    mean_target = distributions.mean(axis=0)
    n_frames = distributions.shape[1]
    quantiles = (np.arange(n_frames, dtype=np.float64) + 0.5) / n_frames
    inverse_cdfs = np.stack(
        [np.searchsorted(np.cumsum(row), quantiles, side="left") for row in distributions]
    )
    barycenter_locations = inverse_cdfs.mean(axis=0)
    left = np.floor(barycenter_locations).astype(int)
    right = np.minimum(left + 1, n_frames - 1)
    fraction = barycenter_locations - left
    barycenter = np.zeros(n_frames, dtype=np.float64)
    np.add.at(barycenter, left, (1 - fraction) / n_frames)
    np.add.at(barycenter, right, fraction / n_frames)
    barycenter /= barycenter.sum()
    return mean_target.astype(np.float32), barycenter.astype(np.float32)


def canonical_summary(scores: np.ndarray, video: TVSumVideo) -> np.ndarray:
    """Reproduce vsumm-reinforce's picks expansion, shot averaging and 0/1 knapsack."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size != video.picks.size or not np.isfinite(scores).all():
        raise ValueError(f"{video.key} prediction must be finite and aligned to picks.")
    max_score = scores.max()
    scores = scores / max_score if max_score > 0 else np.ones_like(scores)
    positions = video.picks
    if positions[-1] != video.n_frames:
        positions = np.concatenate((positions, np.asarray([video.n_frames], dtype=np.int64)))
    frame_scores = np.zeros(video.n_frames, dtype=np.float64)
    for index in range(positions.size - 1):
        start, end = int(positions[index]), int(positions[index + 1])
        frame_scores[start:end] = scores[index] if index < scores.size else 0.0

    segment_scores = np.asarray(
        [
            int(1000 * frame_scores[int(start):int(end) + 1].mean())
            for start, end in video.change_points
        ],
        dtype=np.int64,
    )
    weights = video.frames_per_segment.astype(int)
    capacity = int(math.floor(video.n_frames * BUDGET_FRACTION))
    n_segments = len(weights)
    table = np.zeros((n_segments + 1, capacity + 1), dtype=np.int64)
    keep = np.zeros((n_segments + 1, capacity + 1), dtype=bool)
    for item in range(1, n_segments + 1):
        weight = int(weights[item - 1])
        previous = table[item - 1]
        current = table[item]
        current[:] = previous
        if weight <= capacity:
            candidate = segment_scores[item - 1] + previous[:-weight]
            better = candidate > previous[weight:]
            current[weight:] = np.where(better, candidate, previous[weight:])
            keep[item, weight:] = better

    selected: set[int] = set()
    remaining = capacity
    for item in range(n_segments, 0, -1):
        if keep[item, remaining]:
            selected.add(item - 1)
            remaining -= int(weights[item - 1])
    summary = np.concatenate(
        [
            np.ones(int(weight), dtype=np.uint8)
            if index in selected
            else np.zeros(int(weight), dtype=np.uint8)
            for index, weight in enumerate(weights)
        ]
    )
    if summary.size != video.n_frames:
        raise AssertionError("Canonical summary length differs from original video length.")
    return summary


def evaluate_summary(machine_summary: np.ndarray, user_summaries: np.ndarray) -> dict[str, float]:
    machine = (np.asarray(machine_summary).reshape(-1) > 0).astype(np.float64)
    users = (np.asarray(user_summaries) > 0).astype(np.float64)
    if users.ndim != 2 or users.shape[1] != machine.size or users.shape[0] == 0:
        raise ValueError("user_summaries must have shape (users, original_frames).")
    overlap = (users * machine[None, :]).sum(axis=1)
    precision = overlap / max(float(machine.sum()), 1e-8)
    recall = overlap / np.maximum(users.sum(axis=1), 1e-8)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "f1": float(f1.mean() * 100),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
    }


def score_video(prediction: np.ndarray, video: TVSumVideo) -> dict[str, float]:
    summary = canonical_summary(prediction, video)
    metrics = evaluate_summary(summary, video.user_summary)
    spearman = []
    kendall = []
    for annotator in video.annotation_scores:
        if np.ptp(prediction) == 0 or np.ptp(annotator) == 0:
            rho = tau = 0.0
        else:
            rho = scipy.stats.spearmanr(prediction, annotator).statistic
            tau = scipy.stats.kendalltau(prediction, annotator).statistic
        spearman.append(0.0 if not np.isfinite(rho) else float(rho))
        kendall.append(0.0 if not np.isfinite(tau) else float(tau))
    metrics["spearman_rho"] = float(np.mean(spearman))
    metrics["kendall_tau"] = float(np.mean(kendall))
    return metrics


def human_leave_one_out(video: TVSumVideo) -> float:
    scores = []
    for heldout_index in range(video.user_summary.shape[0]):
        other_users = np.delete(video.user_summary, heldout_index, axis=0)
        sampled = other_users[:, video.picks].mean(axis=0)
        if sampled.sum() <= 0:
            raise ValueError(f"{video.key} leave-one-out human target has zero mass.")
        predicted_summary = canonical_summary(sampled, video)
        scores.append(evaluate_summary(
            predicted_summary, video.user_summary[heldout_index:heldout_index + 1]
        )["f1"])
    return float(np.mean(scores))


def loss_for_arm(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    mask = valid_mask.to(dtype=prediction.dtype)
    counts = mask.sum(dim=-1).clamp_min(1)
    if loss_name == "mse":
        per_video = (((prediction - target) ** 2) * mask).sum(dim=-1) / counts
        return per_video.mean()
    if loss_name == "cross_entropy":
        per_video = -(target * torch.log(prediction.clamp_min(1e-8))).sum(dim=-1)
        return per_video.mean()
    if loss_name == "cramer":
        cdf_difference = torch.cumsum(prediction, dim=-1) - torch.cumsum(target, dim=-1)
        per_video = (cdf_difference ** 2 * mask).sum(dim=-1) / counts
        return per_video.mean()
    raise ValueError(f"Unknown objective {loss_name!r}.")


def predict_distribution(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features).unsqueeze(0)
        prediction = model(tensor).squeeze(0).cpu().numpy()
    return prediction.astype(np.float32)


def train_candidate(
    videos: dict[str, TVSumVideo],
    normalized_features: dict[str, np.ndarray],
    train_keys: list[str],
    validation_keys: list[str],
    targets: dict[str, np.ndarray],
    loss_name: str,
    seed: int,
    epochs: int,
    validation_history: list[float] | None = None,
    backbone: str = "temporal_conv",
) -> tuple[dict[str, torch.Tensor], int, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_regressor(backbone)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = np.random.default_rng(seed)
    feature_tensors = {
        key: torch.from_numpy(normalized_features[key]) for key in train_keys
    }
    target_tensors = {key: torch.from_numpy(targets[key]) for key in train_keys}
    length_sorted_keys = sorted(
        train_keys, key=lambda key: normalized_features[key].shape[0]
    )
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        batches = [
            length_sorted_keys[index:index + BATCH_SIZE]
            for index in range(0, len(length_sorted_keys), BATCH_SIZE)
        ]
        rng.shuffle(batches)
        for batch_keys in batches:
            feature_batch = pad_sequence(
                [feature_tensors[key] for key in batch_keys],
                batch_first=True,
            )
            target_batch = pad_sequence(
                [target_tensors[key] for key in batch_keys],
                batch_first=True,
            )
            lengths = torch.as_tensor(
                [feature_tensors[key].shape[0] for key in batch_keys]
            )
            valid_mask = torch.arange(feature_batch.shape[1])[None, :] < lengths[:, None]
            prediction = model(feature_batch, valid_mask)
            loss = loss_for_arm(prediction, target_batch, loss_name, valid_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        validation_scores = []
        for key in validation_keys:
            prediction = predict_distribution(model, normalized_features[key])
            validation_summary = canonical_summary(prediction, videos[key])
            validation_scores.append(
                evaluate_summary(
                    validation_summary, videos[key].user_summary
                )["f1"]
            )
        validation_f1 = float(np.mean(validation_scores))
        if validation_history is not None:
            validation_history.append(validation_f1)
        if validation_f1 > best_score:
            best_score = validation_f1
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    return best_state, best_epoch, best_score


def resample_feature_sequence(features: np.ndarray, length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("Feature sequence length must be positive.")
    if features.shape[0] == length:
        return features.copy()
    locations = np.linspace(0, features.shape[0] - 1, length)
    left = np.floor(locations).astype(int)
    right = np.minimum(left + 1, features.shape[0] - 1)
    fraction = (locations - left)[:, None]
    return (features[left] * (1 - fraction) + features[right] * fraction).astype(np.float32)


def paired_statistics(left: list[float], right: list[float]) -> dict[str, float]:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if differences.size == 0 or not np.isfinite(differences).all():
        raise ValueError("Paired tests require nonempty finite observations.")
    if np.all(differences == 0):
        return {"n": int(differences.size), "paired_t_p": 1.0, "wilcoxon_p": 1.0}
    if np.var(differences) == 0:
        t_p = 0.0
    else:
        t_p = float(scipy.stats.ttest_rel(left, right).pvalue)
    wilcoxon_p = float(
        scipy.stats.wilcoxon(left, right, zero_method="zsplit").pvalue
    )
    if not np.isfinite(t_p) or not np.isfinite(wilcoxon_p):
        raise ValueError("Paired test returned a non-finite p-value.")
    return {
        "n": int(differences.size),
        "paired_t_p": t_p,
        "wilcoxon_p": wilcoxon_p,
    }


def summarize_records(
    records: list[dict[str, Any]],
    baseline_records: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_per_video: dict[str, dict[str, dict[str, list[float]]]] = {
        arm: {} for arm in ARMS
    }
    for row in records:
        fold_video = f"fold_{row['fold']}::{row['video_key']}"
        arm_per_video[row["arm"]].setdefault(fold_video, {}).setdefault(
            row["metric"], []
        ).append(float(row["value"]))

    per_unique_video: dict[str, dict[str, dict[str, list[float]]]] = {
        arm: {} for arm in ARMS
    }
    for arm in ARMS:
        for fold_video, metric_values in arm_per_video[arm].items():
            video_key = fold_video.split("::", 1)[1]
            unique_metrics = per_unique_video[arm].setdefault(
                video_key, {"f1": [], "spearman_rho": [], "kendall_tau": []}
            )
            for metric in unique_metrics:
                unique_metrics[metric].append(
                    float(np.mean(metric_values[metric]))
                )
    unique_video_means = {
        arm: {
            video_key: {
                metric: float(np.mean(values))
                for metric, values in metrics.items()
            }
            for video_key, metrics in per_unique_video[arm].items()
        }
        for arm in ARMS
    }

    means: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, Any] = {"models": {}, "baselines": {}}
    for arm in ARMS:
        means[arm] = {}
        per_metric: dict[str, list[float]] = {
            "f1": [], "spearman_rho": [], "kendall_tau": []
        }
        seed_stds = []
        for key, metric_values in arm_per_video[arm].items():
            means[arm][key] = {}
            for metric in per_metric:
                values = metric_values[metric]
                means[arm][key][metric] = float(np.mean(values))
                per_metric[metric].append(float(np.mean(values)))
                if metric == "f1" and len(values) > 1:
                    seed_stds.append(float(np.std(values, ddof=1)))
        folds: dict[str, dict[str, float]] = {}
        for fold_index in range(1, 6):
            selected = {
                f"fold_{row['fold']}::{row['video_key']}" for row in records
                if row["arm"] == arm and row["fold"] == fold_index
            }
            fold_f1 = [means[arm][key]["f1"] for key in selected]
            folds[str(fold_index)] = {
                "f1_mean": float(np.mean(fold_f1)),
                "f1_std": float(np.std(fold_f1, ddof=1)),
            }
        summary["models"][arm] = {
            "f1_mean": float(np.mean(per_metric["f1"])),
            "f1_std_across_videos": float(np.std(per_metric["f1"], ddof=1)),
            "f1_mean_seed_std": float(np.mean(seed_stds)),
            "spearman_rho_mean": float(np.mean(per_metric["spearman_rho"])),
            "spearman_rho_std_across_videos": float(
                np.std(per_metric["spearman_rho"], ddof=1)
            ),
            "kendall_tau_mean": float(np.mean(per_metric["kendall_tau"])),
            "kendall_tau_std_across_videos": float(
                np.std(per_metric["kendall_tau"], ddof=1)
            ),
            "folds": folds,
            "selected_alpha_by_fold_seed": {},
        }
        for row in records:
            if row["arm"] == arm and row["alpha"] is not None:
                summary["models"][arm]["selected_alpha_by_fold_seed"][
                    f"fold_{row['fold']}_seed_{row['seed']}"
                ] = row["alpha"]

    reference_keys = sorted(unique_video_means[ARMS[0]])
    reference = [
        unique_video_means[ARMS[0]][key]["f1"] for key in reference_keys
    ]
    for arm in ARMS:
        if set(unique_video_means[arm]) != set(unique_video_means[ARMS[0]]):
            raise ValueError(f"Ablation arm {arm} is missing fold/video predictions.")
        candidate = [
            unique_video_means[arm][key]["f1"] for key in reference_keys
        ]
        summary["models"][arm]["paired_vs_A_mean_mse"] = paired_statistics(candidate, reference)
    for comparison, left_arm, right_arm in (
        ("cramer_effect_at_mean_target", "C_mean_cramer", "A_mean_mse"),
        ("barycenter_effect_under_mse", "D_barycenter_mse", "A_mean_mse"),
        ("barycenter_effect_under_cramer", "E_barycenter_cramer", "C_mean_cramer"),
        ("cramer_effect_at_barycenter", "E_barycenter_cramer", "D_barycenter_mse"),
        ("blended_vs_best_fixed_target", "F_blended_cramer", "E_barycenter_cramer"),
    ):
        common = sorted(
            set(unique_video_means[left_arm]) & set(unique_video_means[right_arm])
        )
        left = [unique_video_means[left_arm][key]["f1"] for key in common]
        right = [unique_video_means[right_arm][key]["f1"] for key in common]
        stats_result = paired_statistics(left, right)
        difference = float(np.mean(np.asarray(left) - np.asarray(right)))
        summary[comparison] = {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "mean_f1_difference_points": difference,
            **stats_result,
            "interpretation": (
                "similar within one F1 point; no detected paired difference"
                if abs(difference) <= 1.0 and stats_result["paired_t_p"] >= 0.05
                else "paired difference detected or effect exceeds one F1 point"
            ),
        }

    baseline_means: dict[str, dict[str, float]] = {}
    unique_baseline_means: dict[str, dict[str, float]] = {}
    for baseline, rows in baseline_records.items():
        metric_values: dict[str, list[float]] = {
            "f1": [], "spearman_rho": [], "kendall_tau": []
        }
        by_video: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            fold_video = f"fold_{row['fold']}::{row['video_key']}"
            by_video.setdefault(fold_video, {}).setdefault(row["metric"], []).append(
                float(row["value"])
            )
        seed_stds = [
            float(np.std(metrics["f1"], ddof=1))
            for metrics in by_video.values()
            if len(metrics["f1"]) > 1
        ]
        for fold_video, metrics in by_video.items():
            baseline_means.setdefault(baseline, {})[fold_video] = float(np.mean(metrics["f1"]))
            for metric in metric_values:
                if metric in metrics:
                    metric_values[metric].append(float(np.mean(metrics[metric])))
        summary["baselines"][baseline] = {
            "f1_mean": float(np.mean(metric_values["f1"])),
            "f1_std_across_videos": float(np.std(metric_values["f1"], ddof=1)),
            "f1_mean_seed_std": float(np.mean(seed_stds)) if seed_stds else None,
            "spearman_rho_mean": (
                float(np.mean(metric_values["spearman_rho"]))
                if metric_values["spearman_rho"] else None
            ),
            "kendall_tau_mean": (
                float(np.mean(metric_values["kendall_tau"]))
                if metric_values["kendall_tau"] else None
            ),
        }
        per_video_values: dict[str, list[float]] = {}
        for fold_video, value in baseline_means[baseline].items():
            video_key = fold_video.split("::", 1)[1]
            per_video_values.setdefault(video_key, []).append(value)
        unique_baseline_means[baseline] = {
            video_key: float(np.mean(values))
            for video_key, values in per_video_values.items()
        }
    summary["paired_comparisons"] = {}
    for baseline in ("random", "uniform", "human_leave_one_out"):
        common = sorted(
            set(unique_video_means[ARMS[0]]) & set(unique_baseline_means[baseline])
        )
        if len(common) != len(unique_video_means[ARMS[0]]):
            raise ValueError(f"Baseline {baseline} lacks unique-video observations.")
        for arm in ARMS:
            model_values = [unique_video_means[arm][key]["f1"] for key in common]
            baseline_values = [unique_baseline_means[baseline][key] for key in common]
            summary["paired_comparisons"][f"{arm}_vs_{baseline}"] = {
                "mean_f1_difference_points": float(
                    np.mean(np.asarray(model_values) - np.asarray(baseline_values))
                ),
                **paired_statistics(model_values, baseline_values),
            }
    return summary, means


def build_readme_results_section(
    results: dict[str, Any],
    results_path: str = "results/standard_tvsum_cv/results.json",
) -> str:
    summary = results["summary"]
    method_labels = {
        "A_mean_mse": "A: Mean target + MSE",
        "B_mean_cross_entropy": "B: Mean target + cross-entropy",
        "C_mean_cramer": "C: Mean target + Cramer",
        "D_barycenter_mse": "D: Barycenter target + MSE",
        "E_barycenter_cramer": "E: Barycenter target + Cramer",
        "F_blended_cramer": "F: Validation-blended target + Cramer",
    }
    rows = []
    for arm in ARMS:
        row = summary["models"][arm]
        rows.append(
            f"| {method_labels[arm]} | {row['f1_mean']:.2f} ± {row['f1_mean_seed_std']:.2f} "
            f"| {row['f1_std_across_videos']:.2f} "
            f"| {row['spearman_rho_mean']:.3f} | {row['kendall_tau_mean']:.3f} |"
        )
    for baseline, label in (
        ("random", "Random summary"),
        ("uniform", "Uniform summary"),
        ("human_leave_one_out", "Human upper bound (leave-one-out)"),
    ):
        row = summary["baselines"][baseline]
        rho = "—" if row["spearman_rho_mean"] is None else f"{row['spearman_rho_mean']:.3f}"
        tau = "—" if row["kendall_tau_mean"] is None else f"{row['kendall_tau_mean']:.3f}"
        seed_std = "—" if row["f1_mean_seed_std"] is None else f"{row['f1_mean_seed_std']:.2f}"
        rows.append(
            f"| {label} | {row['f1_mean']:.2f} ± {seed_std} "
            f"| {row['f1_std_across_videos']:.2f} "
            f"| {rho} | {tau} |"
        )
    comparison_rows = []
    comparison_labels = (
        ("cramer_effect_at_mean_target", "C vs A: Cramer loss at mean target"),
        ("barycenter_effect_under_mse", "D vs A: Barycenter target under MSE"),
        ("barycenter_effect_under_cramer", "E vs C: Barycenter effect under Cramer"),
        ("cramer_effect_at_barycenter", "E vs D: Cramer loss at barycenter target"),
        ("blended_vs_best_fixed_target", "F vs E: Validation-selected blend"),
    )
    for key, label in comparison_labels:
        comparison = summary[key]
        comparison_rows.append(
            f"| {label} | {comparison['mean_f1_difference_points']:+.2f} "
            f"| {comparison['paired_t_p']:.4g} | {comparison['wilcoxon_p']:.4g} "
            f"| {comparison['interpretation']} |"
        )
    baseline_comparison_rows = []
    best_arm = max(
        ARMS, key=lambda arm: summary["models"][arm]["f1_mean"]
    )
    best_mean = summary["models"][best_arm]["f1_mean"]
    random_mean = summary["baselines"]["random"]["f1_mean"]
    uniform_mean = summary["baselines"]["uniform"]["f1_mean"]
    if best_mean > random_mean and best_mean > uniform_mean:
        baseline_conclusion = (
            f"The highest mean-F1 learned arm is {method_labels[best_arm]} "
            f"({best_mean:.2f}); it exceeds the random and uniform baselines "
            f"({random_mean:.2f} and {uniform_mean:.2f}) on mean F1."
        )
    else:
        baseline_conclusion = (
            f"The highest mean-F1 learned arm is {method_labels[best_arm]} "
            f"({best_mean:.2f}); it does not exceed both random "
            f"({random_mean:.2f}) and uniform ({uniform_mean:.2f}) baselines "
            "on mean F1."
        )
    for arm in ARMS:
        for baseline, label in (
            ("random", "Random"),
            ("uniform", "Uniform"),
            ("human_leave_one_out", "Human leave-one-out"),
        ):
            comparison = summary["paired_comparisons"][f"{arm}_vs_{baseline}"]
            baseline_comparison_rows.append(
                f"| {method_labels[arm]} | {label} "
                f"| {comparison['mean_f1_difference_points']:+.2f} "
                f"| {comparison['paired_t_p']:.4g} "
                f"| {comparison['wilcoxon_p']:.4g} |"
            )
    sanity = results["sanity_checks"]
    status = "PASS" if sanity["passed"] else "FAIL"
    return "\n".join(
        [
            "## Results (real TVSum, 5-fold)",
            "",
            f"**Run status: {status}.** Values below are generated from "
            f"`{results_path}`; no literature values are mixed in.",
            "",
            "| Method | F-score mean ± seed SD (%) | Fold/video SD | Spearman ρ | Kendall τ |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "Paired contrasts average seeds and repeated split appearances per unique "
            "test video before testing, avoiding duplicate-video pseudoreplication; "
            "positive differences favor the first-listed arm. In particular, E vs C "
            "isolates the barycenter-target effect under the Cramer loss.",
            "",
            "| Contrast | Mean F1 difference (points) | Paired t p | Wilcoxon p | Interpretation |",
            "|---|---:|---:|---:|---|",
            *comparison_rows,
            "",
            baseline_conclusion,
            "",
            "| Learned arm | Baseline | Mean F1 difference (points) | Paired t p | Wilcoxon p |",
            "|---|---|---:|---:|---:|",
            *baseline_comparison_rows,
            "",
            "Paired-test p-values are unadjusted for multiple comparisons.",
            "",
            "All learned methods use the same 1024-D temporal-convolution backbone, "
            "optimizer, epoch budget, DSNet folds, and five fixed seeds. Alpha for "
            "arm F is selected using only the validation videos carved from that "
            "fold's training IDs. F-score follows the preprocessed TVSum protocol: "
            "expand scores at `picks`, average within supplied `change_points`, "
            "select shots by 0/1 knapsack under 15% duration, and average F1 over "
            "the 20 `user_summary` rows. Rank correlations compare sampled model "
            "scores with raw per-annotator scores aligned at `picks`.",
            "",
            f"Sanity checks: random mean F1 over all 50 videos="
            f"{sanity['random_f1_mean']:.2f} "
            f"(expected 50–55: {sanity['random_50_55_passed']}); shuffled-feature "
            f"F1={sanity['shuffled_feature_f1_mean']:.2f} versus baseline threshold "
            f"{sanity['shuffle_baseline_threshold']:.2f} "
            f"({sanity['shuffle_baseline_passed']}).",
            f"The random baseline on the supplied DSNet test instances is "
            f"{sanity['random_split_f1_mean']:.2f}; the all-video sanity estimate "
            "is separate because the supplied split file does not test every video.",
            "",
            "The supplied DSNet split file has "
            f"{results['protocol']['unique_test_videos']} unique test videos out of "
            f"{results['protocol']['video_count']}; test sets are not required to "
            "be mutually disjoint. Metrics average the fold/video test instances. "
            "Limitations: only the supplied five TVSum splits are included. Augmented and "
            "transfer settings were not run. These results describe this run and "
            "do not establish state of the art. No conditional diffusion model was "
            "evaluated, so these results do not validate DDPM against the regressor. "
            "The human leave-one-out baseline is an annotator-agreement oracle, not "
            "a deployable model. Use the JSON for fold-level scores, "
            "paired tests, split IDs, feature/data hashes, selected validation "
            "alphas, and per-video/seed metrics.",
        ]
    )


def write_result_artifacts(
    result: dict[str, Any], json_path: Path, update_readme: bool = True
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = json_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(json_path)
    saved_result = json.loads(json_path.read_text(encoding="utf-8"))
    table_path = json_path.with_name("results_table.csv")
    with table_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "method", "f1_mean", "f1_mean_seed_std", "f1_std_across_videos",
            "spearman_rho_mean", "kendall_tau_mean",
        ])
        for method in ARMS:
            row = saved_result["summary"]["models"][method]
            writer.writerow([
                method, row["f1_mean"], row["f1_mean_seed_std"],
                row["f1_std_across_videos"],
                row["spearman_rho_mean"], row["kendall_tau_mean"],
            ])
        for baseline in ("random", "uniform", "human_leave_one_out"):
            row = saved_result["summary"]["baselines"][baseline]
            writer.writerow([
                baseline, row["f1_mean"], row["f1_mean_seed_std"],
                row["f1_std_across_videos"],
                row["spearman_rho_mean"], row["kendall_tau_mean"],
            ])
    readme_path = PROJECT_ROOT / "README.md"
    if update_readme and readme_path.is_file():
        text = readme_path.read_text(encoding="utf-8")
        start = "<!-- REAL TVSUM CV RESULTS START -->"
        end = "<!-- REAL TVSUM CV RESULTS END -->"
        if start not in text or end not in text:
            raise ValueError("README result-section markers are missing.")
        section = build_readme_results_section(saved_result)
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        readme_path.write_text(
            before + start + "\n\n" + section + "\n\n" + end + after,
            encoding="utf-8",
        )


def run_experiment(
    h5_path: Path,
    annotation_path: Path,
    info_path: Path,
    split_path: Path,
    output_path: Path,
    epochs: int = DEFAULT_EPOCHS,
    torch_threads: int = 4,
    backbone: str = "temporal_conv",
    update_readme: bool = True,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    if backbone not in ("temporal_conv", "bilstm"):
        raise ValueError(f"Unknown backbone {backbone!r}.")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    started = time.time()

    videos = load_tvsum_features(h5_path, annotation_path, info_path)
    folds = load_dsnet_folds(split_path, set(videos))
    all_records: list[dict[str, Any]] = []
    baseline_records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("random", "uniform", "human_leave_one_out")
    }
    shuffle_records: list[dict[str, Any]] = []
    fold_logs = []
    for fold_index, fold in enumerate(folds, start=1):
        train_keys, validation_keys = split_train_validation(fold["train"], fold_index)
        test_keys = fold["test"]
        sets = [set(train_keys), set(validation_keys), set(test_keys)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError(f"Fold {fold_index} has train/val/test overlap.")
        if set.union(*sets) != set(videos):
            raise AssertionError(f"Fold {fold_index} does not partition all video IDs.")
        print(
            f"FOLD {fold_index} train_ids="
            f"{[videos[key].video_id for key in train_keys]}"
        )
        print(
            f"FOLD {fold_index} val_ids="
            f"{[videos[key].video_id for key in validation_keys]}"
        )
        print(
            f"FOLD {fold_index} test_ids="
            f"{[videos[key].video_id for key in test_keys]}"
        )
        mean, std = fit_feature_scaler(videos, train_keys)
        normalized = {
            key: normalize_features(video, mean, std)
            for key, video in videos.items()
        }
        distributions = {
            key: annotation_distributions(videos[key])
            for key in train_keys
        }
        targets: dict[str, tuple[np.ndarray, np.ndarray]] = {
            key: mean_and_barycenter(distributions[key])
            for key in train_keys
        }
        fold_logs.append({
            "fold": fold_index,
            "train_ids": [videos[key].video_id for key in train_keys],
            "validation_ids": [videos[key].video_id for key in validation_keys],
            "test_ids": [videos[key].video_id for key in test_keys],
            "train_keys": train_keys,
            "validation_keys": validation_keys,
            "test_keys": test_keys,
        })

        for seed in SEEDS:
            base_seed = stable_seed(f"tvsum-fold-{fold_index}-seed-{seed}")
            fixed_specs = (
                ("A_mean_mse", "mse", 0, None),
                ("B_mean_cross_entropy", "cross_entropy", 0, None),
                ("C_mean_cramer", "cramer", 0, None),
                ("D_barycenter_mse", "mse", 1, None),
                ("E_barycenter_cramer", "cramer", 1, None),
            )
            chosen_models: dict[
                str, tuple[dict[str, torch.Tensor], float | None, int]
            ] = {}
            for arm, loss_name, target_index, alpha in fixed_specs:
                arm_targets = {
                    key: targets[key][target_index] for key in train_keys
                }
                state, best_epoch, best_val = train_candidate(
                    videos, normalized, train_keys, validation_keys,
                    arm_targets, loss_name, base_seed, epochs,
                    backbone=backbone,
                )
                chosen_models[arm] = (state, alpha, best_epoch)
                print(
                    f"FOLD {fold_index} SEED {seed} ARM {arm} "
                    f"best_epoch={best_epoch} validation_f1={best_val:.3f}"
                )

            alpha_candidates = []
            for alpha in ALPHA_GRID:
                arm_targets = {
                    key: ((1 - alpha) * targets[key][0] + alpha * targets[key][1]).astype(np.float32)
                    for key in train_keys
                }
                state, best_epoch, best_val = train_candidate(
                    videos, normalized, train_keys, validation_keys,
                    arm_targets, "cramer", base_seed, epochs,
                    backbone=backbone,
                )
                alpha_candidates.append(
                    (best_val, -alpha, alpha, state, best_epoch)
                )
                print(
                    f"FOLD {fold_index} SEED {seed} ARM F alpha={alpha:.2f} "
                    f"best_epoch={best_epoch} validation_f1={best_val:.3f}"
                )
            best_val, _, selected_alpha, selected_state, selected_epoch = max(
                alpha_candidates, key=lambda candidate: (candidate[0], candidate[1])
            )
            chosen_models["F_blended_cramer"] = (
                selected_state, selected_alpha, selected_epoch
            )
            print(
                f"FOLD {fold_index} SEED {seed} SELECTED_ALPHA={selected_alpha:.2f} "
                f"validation_f1={best_val:.3f} epoch={selected_epoch}"
            )

            for arm in ARMS:
                state, alpha, best_epoch = chosen_models[arm]
                model = build_regressor(backbone)
                model.load_state_dict(state)
                for key in test_keys:
                    prediction = predict_distribution(model, normalized[key])
                    metric = score_video(prediction, videos[key])
                    for metric_name, value in metric.items():
                        all_records.append({
                            "fold": fold_index,
                            "seed": seed,
                            "arm": arm,
                            "video_key": key,
                            "video_id": videos[key].video_id,
                            "metric": metric_name,
                            "value": value,
                            "alpha": alpha,
                            "selected_epoch": best_epoch,
                        })
            # The shuffled-feature check evaluates the validation-selected F model only.
            shuffled_source = {
                target_key: test_keys[(index - 1) % len(test_keys)]
                for index, target_key in enumerate(test_keys)
            }
            model = build_regressor(backbone)
            model.load_state_dict(selected_state)
            for target_key, source_key in shuffled_source.items():
                source_features = videos[source_key].features
                source_resampled = resample_feature_sequence(
                    source_features, videos[target_key].features.shape[0]
                )
                source_scaled = (source_resampled - mean) / std
                prediction = predict_distribution(
                    model, np.asarray(source_scaled, dtype=np.float32)
                )
                metrics = score_video(prediction, videos[target_key])
                shuffle_records.append({
                    "fold": fold_index,
                    "seed": seed,
                    "video_key": target_key,
                    "video_id": videos[target_key].video_id,
                    "shuffled_from_key": source_key,
                    **metrics,
                })

        for key in test_keys:
            video = videos[key]
            uniform_metrics = score_video(
                np.ones(video.picks.size, dtype=np.float32), video
            )
            for metric_name, value in uniform_metrics.items():
                baseline_records["uniform"].append({
                    "fold": fold_index,
                    "video_key": key,
                    "metric": metric_name,
                    "value": value,
                })
            baseline_records["human_leave_one_out"].append({
                "fold": fold_index,
                "video_key": key,
                "metric": "f1",
                "value": human_leave_one_out(video),
            })
            for seed in SEEDS:
                random_scores = np.random.default_rng(
                    stable_seed(f"random-summary-{video.video_id}-{seed}")
                ).random(video.picks.size)
                metrics = score_video(random_scores, video)
                for metric_name, value in metrics.items():
                    baseline_records["random"].append({
                        "fold": fold_index,
                        "seed": seed,
                        "video_key": key,
                        "metric": metric_name,
                        "value": value,
                    })

    summary, per_video_model_means = summarize_records(
        all_records, baseline_records
    )
    per_video_records = []
    for fold in fold_logs:
        fold_index = fold["fold"]
        for key in fold["test_keys"]:
            fold_video = f"fold_{fold_index}::{key}"
            row = {
                "fold": fold_index,
                "video_key": key,
                "video_id": videos[key].video_id,
            }
            for arm in ARMS:
                row.update({
                    f"{arm}_{metric}": value
                    for metric, value in per_video_model_means[arm][fold_video].items()
                })
            per_video_records.append(row)

    random_by_fold_video = {}
    for record in baseline_records["random"]:
        if record["metric"] == "f1":
            fold_video = f"fold_{record['fold']}::{record['video_key']}"
            random_by_fold_video.setdefault(fold_video, []).append(record["value"])
    random_split_f1_mean = float(
        np.mean([np.mean(values) for values in random_by_fold_video.values()])
    )
    all_video_random_means = []
    for video in videos.values():
        seed_f1 = []
        for seed in SEEDS:
            random_scores = np.random.default_rng(
                stable_seed(f"random-summary-{video.video_id}-{seed}")
            ).random(video.picks.size)
            random_summary = canonical_summary(random_scores, video)
            seed_f1.append(evaluate_summary(random_summary, video.user_summary)["f1"])
        all_video_random_means.append(float(np.mean(seed_f1)))
    random_f1_mean = float(np.mean(all_video_random_means))
    shuffle_by_fold_video: dict[str, list[float]] = {}
    for record in shuffle_records:
        fold_video = f"fold_{record['fold']}::{record['video_key']}"
        shuffle_by_fold_video.setdefault(fold_video, []).append(float(record["f1"]))
    shuffled_mean = float(
        np.mean([np.mean(values) for values in shuffle_by_fold_video.values()])
    )
    random_baseline_mean = summary["baselines"]["random"]["f1_mean"]
    uniform_baseline_mean = summary["baselines"]["uniform"]["f1_mean"]
    shuffle_threshold = max(random_baseline_mean, uniform_baseline_mean) + 5.0
    sanity_checks = {
        "random_f1_mean": random_f1_mean,
        "random_split_f1_mean": random_split_f1_mean,
        "random_expected_interval": [50.0, 55.0],
        "random_50_55_passed": 50.0 <= random_f1_mean <= 55.0,
        "shuffled_feature_f1_mean": shuffled_mean,
        "random_baseline_f1_mean": random_baseline_mean,
        "uniform_baseline_f1_mean": uniform_baseline_mean,
        "shuffle_baseline_threshold": shuffle_threshold,
        "shuffle_baseline_passed": shuffled_mean <= shuffle_threshold,
        "passed": (
            50.0 <= random_f1_mean <= 55.0
            and shuffled_mean <= shuffle_threshold
        ),
    }
    fold_metrics = {}
    for fold in fold_logs:
        fold_id = fold["fold"]
        fold_metrics[str(fold_id)] = {}
        for arm in ARMS:
            values = [
                row[f"{arm}_f1"]
                for row in per_video_records
                if row["fold"] == fold_id
            ]
            fold_metrics[str(fold_id)][arm] = {
                "f1_mean": float(np.mean(values)),
                "f1_std": float(np.std(values, ddof=1)),
            }
    result = {
        "protocol": {
            "dataset": "TVSum50",
            "video_count": len(videos),
            "fold_video_test_instances": sum(
                len(fold["test_keys"]) for fold in fold_logs
            ),
            "unique_test_videos": len({
                key for fold in fold_logs for key in fold["test_keys"]
            }),
            "videos_never_in_test": sorted(
                {
                    video.video_id for video in videos.values()
                } - {
                    videos[key].video_id
                    for fold in fold_logs for key in fold["test_keys"]
                }
            ),
            "split_source": "li-plus/DSNet splits/tvsum.yml",
            "split_file_sha256": sha256_file(split_path),
            "feature_file": h5_path.name,
            "feature_sha256": sha256_file(h5_path),
            "raw_annotation_file": annotation_path.name,
            "raw_annotation_sha256": sha256_file(annotation_path),
            "official_info_file": info_path.name,
            "official_info_sha256": sha256_file(info_path),
            "feature_description": "1024-D GoogLeNet pool5 features from preprocessed TVSum HDF5",
            "feature_to_annotation_alignment": "raw annotator frame scores indexed at HDF5 picks",
            "undefined_rank_correlation": "0.0 when either input is constant",
            "canonical_summary": "DSNet/vsumm-reinforce shot scoring plus 0/1 knapsack at 15% duration",
            "f_score_reference": "HDF5 user_summary; mean F1 over 20 annotators",
            "correlation_reference": "raw per-annotator scores sampled at picks; mean over 20 users",
            "split_count": len(fold_logs),
            "split_protocol": (
                "five supplied 80/20 splits; test sets may overlap and may not "
                "cover all videos"
            ),
            "validation_fraction_of_fold_train": 0.1,
            "seeds": list(SEEDS),
            "training_epochs_max": epochs,
            "training_batch_size": BATCH_SIZE,
            "torch_threads": torch_threads,
            "backbone": (
                "Linear(1024,64)-Conv1d(64,64,k=5)-Conv1d(64,64,k=3)-Conv1d(64,1)"
                if backbone == "temporal_conv"
                else "BiLSTM(input=1024,hidden=64,directions=2,layers=2)-Linear(128,1)"
            ),
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "budget_fraction": BUDGET_FRACTION,
            "arms": list(ARMS),
            "alpha_grid_selected_on_validation_only": list(ALPHA_GRID),
            "augmented_transfer": "not run",
            "no_test_fold_tuning": True,
        },
        "video_data": [
            {
                "video_key": video.key,
                "video_id": video.video_id,
                "original_frames": video.n_frames,
                "feature_positions": int(video.picks.size),
                "shots": int(video.change_points.shape[0]),
                "annotators": int(video.annotation_scores.shape[0]),
            }
            for video in videos.values()
        ],
        "summary": summary,
        "fold_summary": fold_metrics,
        "folds": fold_logs,
        "per_fold_video_seed_averaged": per_video_records,
        "per_video_seed_records": all_records,
        "baseline_records": baseline_records,
        "shuffled_feature_records": shuffle_records,
        "sanity_checks": sanity_checks,
        "elapsed_seconds": time.time() - started,
    }
    write_result_artifacts(result, output_path, update_readme=update_readme)
    print(json.dumps(sanity_checks, indent=2))
    print(f"RESULTS {output_path}")
    if not sanity_checks["passed"]:
        failures = []
        if not sanity_checks["random_50_55_passed"]:
            failures.append(
                "random-score F1 outside the prespecified 50-55 sanity interval"
            )
        if not sanity_checks["shuffle_baseline_passed"]:
            failures.append("shuffled-feature F1 exceeded the baseline threshold")
        raise RuntimeError(
            "TVSum sanity checks failed after saving results: " + "; ".join(failures)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(os.environ.get(FEATURES_ENV, DEFAULT_FEATURES)),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(os.environ[ANNOTATION_ENV]) if os.environ.get(ANNOTATION_ENV) else None,
        help=f"official raw annotation TSV (or set {ANNOTATION_ENV})",
    )
    parser.add_argument(
        "--info",
        type=Path,
        default=Path(os.environ[INFO_ENV]) if os.environ.get(INFO_ENV) else None,
        help=f"official TVSum info TSV (or set {INFO_ENV})",
    )
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("TVSUM_RESULTS_JSON", DEFAULT_RESULTS)),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--backbone",
        choices=("temporal_conv", "bilstm"),
        default="temporal_conv",
    )
    parser.add_argument("--skip-readme-update", action="store_true")
    args = parser.parse_args()
    if args.annotations is None:
        parser.error(f"Provide --annotations or set {ANNOTATION_ENV}.")
    if args.info is None:
        parser.error(f"Provide --info or set {INFO_ENV}.")
    run_experiment(
        args.features,
        args.annotations,
        args.info,
        args.splits,
        args.output,
        epochs=args.epochs,
        torch_threads=args.torch_threads,
        backbone=args.backbone,
        update_readme=not args.skip_readme_update,
    )


if __name__ == "__main__":
    main()
