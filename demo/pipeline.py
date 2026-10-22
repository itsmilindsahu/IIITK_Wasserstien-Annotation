"""CPU inference pipeline for the TVSum summarization demo."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import gradio as gr
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scenedetect import ContentDetector, SceneManager, open_video
from torchvision.models import GoogLeNet_Weights, googlenet
from torchvision.models.feature_extraction import create_feature_extractor

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
PROJECT_ROOT = CODE_DIR.parent
MODEL_DIR = Path(__file__).resolve().parent / "model"
SAMPLE_PERIOD_SECONDS = 1.0
MAX_DURATION_SECONDS = 10 * 60
MAX_YOUTUBE_DURATION_SECONDS = 2 * 60
sys.path.insert(0, str(CODE_DIR))

import standard_tvsum_cv as benchmark  # noqa: E402

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class SampledVideo:
    features: np.ndarray
    timestamps: np.ndarray
    duration_seconds: float
    fps: float


@dataclass(frozen=True)
class SummarySupport:
    n_frames: int
    picks: np.ndarray
    change_points: np.ndarray
    frames_per_segment: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_metadata(video_path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError(f"Video has invalid FPS or frame count: {video_path}")
        return frame_count / fps, fps, frame_count
    finally:
        capture.release()


@lru_cache(maxsize=1)
def load_pool5_extractor() -> tuple[torch.nn.Module, Callable]:
    weights = GoogLeNet_Weights.IMAGENET1K_V1
    backbone = googlenet(weights=weights)
    extractor = create_feature_extractor(backbone, return_nodes={"avgpool": "pool5"})
    extractor.eval()
    return extractor, weights.transforms()


def extract_pool5(
    video_path: Path,
    extractor: torch.nn.Module,
    transform: Callable,
    sample_period_seconds: float = SAMPLE_PERIOD_SECONDS,
) -> tuple[SampledVideo, dict[str, float | list[float]]]:
    duration, fps, frame_count = video_metadata(video_path)
    capture = cv2.VideoCapture(str(video_path))
    timestamps = np.arange(0.0, duration, sample_period_seconds, dtype=np.float64)
    if timestamps.size == 0:
        timestamps = np.asarray([0.0], dtype=np.float64)
    images: list[Image.Image] = []
    decode_started = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        for timestamp in timestamps:
            frame_index = min(int(round(timestamp * fps)), frame_count - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Could not decode frame {frame_index} from {video_path}."
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
    finally:
        capture.release()
    decode_seconds = time.perf_counter() - decode_started

    feature_started = time.perf_counter()
    feature_rows = []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            batch = torch.stack(
                [transform(image) for image in images[start:start + 16]]
            )
            pooled = extractor(batch)["pool5"].flatten(start_dim=1).cpu().numpy()
            feature_rows.append(pooled.astype(np.float32))
    feature_seconds = time.perf_counter() - feature_started
    raw_features = np.concatenate(feature_rows, axis=0)
    if raw_features.shape != (timestamps.size, 1024):
        raise ValueError(f"GoogLeNet pool5 produced unexpected shape {raw_features.shape}.")
    l2_norms = np.linalg.norm(raw_features, axis=1)
    if np.any(~np.isfinite(l2_norms)) or np.any(l2_norms <= 0):
        raise ValueError("GoogLeNet pool5 produced invalid feature vectors.")
    normalized = raw_features / l2_norms[:, None]
    stats: dict[str, float | list[float]] = {
        "raw_mean": float(raw_features.mean()),
        "raw_std": float(raw_features.std()),
        "raw_l2_mean": float(l2_norms.mean()),
        "raw_l2_std": float(l2_norms.std()),
        "raw_l2_quantiles": np.quantile(
            l2_norms, [0.0, 0.25, 0.5, 0.75, 1.0]
        ).tolist(),
        "l2_normalized_mean": float(normalized.mean()),
        "l2_normalized_std": float(normalized.std()),
        "l2_normalized_norm_mean": float(
            np.linalg.norm(normalized, axis=1).mean()
        ),
        "decode_seconds": decode_seconds,
        "feature_seconds": feature_seconds,
    }
    return (
        SampledVideo(normalized.astype(np.float32), timestamps, duration, fps),
        stats,
    )


def detect_sampled_shots(
    video_path: Path, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    manager.detect_scenes(video=video)
    scenes = manager.get_scene_list(start_in_scene=True)
    scene_starts = [float(scene[0].get_seconds()) for scene in scenes]
    if not scene_starts or scene_starts[0] > 0:
        scene_starts.insert(0, 0.0)
    scene_starts = sorted(set(scene_starts))
    labels = np.searchsorted(
        np.asarray(scene_starts, dtype=np.float64),
        timestamps,
        side="right",
    ) - 1
    boundaries = np.flatnonzero(np.diff(labels) != 0) + 1
    segment_starts = np.concatenate(([0], boundaries)).astype(np.int64)
    segment_ends = np.concatenate((boundaries, [timestamps.size])).astype(np.int64)
    change_points = np.column_stack(
        (segment_starts, segment_ends - 1)
    ).astype(np.int64)
    frames_per_segment = (segment_ends - segment_starts).astype(np.int64)
    if np.any(frames_per_segment <= 0):
        raise AssertionError("Shot detector emitted an empty sampled segment.")
    return change_points, frames_per_segment, time.perf_counter() - started


def _support(n_samples: int, change_points: np.ndarray, lengths: np.ndarray):
    return SummarySupport(
        n_frames=n_samples,
        picks=np.arange(n_samples, dtype=np.int64),
        change_points=change_points,
        frames_per_segment=lengths,
    )


def split_oversized_scenes(
    change_points: np.ndarray, segment_lengths: np.ndarray, budget: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    if budget < 1:
        raise ValueError(
            "Video is too short for a non-empty shot under the 15% summary budget."
        )
    refined: list[tuple[int, int]] = []
    split_occurred = False
    for start, end in change_points:
        start_index, end_index = int(start), int(end)
        while end_index - start_index + 1 > budget:
            chunk_end = start_index + budget - 1
            refined.append((start_index, chunk_end))
            start_index = chunk_end + 1
            split_occurred = True
        refined.append((start_index, end_index))

    refined_points = np.asarray(refined, dtype=np.int64)
    refined_lengths = refined_points[:, 1] - refined_points[:, 0] + 1
    if int(refined_lengths.sum()) != int(segment_lengths.sum()):
        raise AssertionError("Budget-feasible scene split changed sampled frame count.")
    if np.any(refined_lengths > budget):
        raise AssertionError("Scene split left a shot larger than the summary budget.")
    return refined_points, refined_lengths, split_occurred


def summarize_scores(
    scores: np.ndarray,
    change_points: np.ndarray,
    segment_lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    support = _support(int(segment_lengths.sum()), change_points, segment_lengths)
    started = time.perf_counter()
    summary = benchmark.canonical_summary(scores, support)
    return summary, time.perf_counter() - started


def selected_intervals(
    summary: np.ndarray, change_points: np.ndarray, timestamps: np.ndarray,
    duration_seconds: float,
) -> list[tuple[float, float]]:
    selected = []
    for start, end in change_points:
        start_index, end_index = int(start), int(end) + 1
        if np.all(summary[start_index:end_index] > 0):
            start_seconds = float(timestamps[start_index])
            end_seconds = (
                float(timestamps[end_index])
                if end_index < timestamps.size
                else duration_seconds
            )
            if end_seconds > start_seconds:
                selected.append((start_seconds, end_seconds))
    return selected


def render_summary(
    input_path: Path, intervals: list[tuple[float, float]], output_path: Path
) -> None:
    if not intervals:
        raise RuntimeError(
            "Canonical shot selection returned an empty summary despite "
            "budget-feasible scene segments."
        )
    filters = []
    for index, (start, end) in enumerate(intervals):
        filters.append(
            f"[0:v]trim=start={start:.6f}:end={end:.6f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
    concat_inputs = "".join(f"[v{index}]" for index in range(len(intervals)))
    filters.append(
        f"{concat_inputs}concat=n={len(intervals)}:v=1:a=0[outv]"
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-i", str(input_path), "-filter_complex", ";".join(filters),
        "-map", "[outv]", "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-threads", "1", "-crf", "28", "-movflags", "+faststart",
        "-y", str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg returned success but did not create a summary clip.")


def write_timeline(
    timestamps: np.ndarray,
    scores: np.ndarray,
    selected: np.ndarray,
    change_points: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(timestamps, scores, color="#205493", linewidth=1.4, label="Model score")
    for start, end in change_points:
        if np.any(selected[int(start):int(end) + 1]):
            left = float(timestamps[int(start)])
            right = (
                float(timestamps[int(end) + 1])
                if int(end) + 1 < timestamps.size
                else float(timestamps[-1] + SAMPLE_PERIOD_SECONDS)
            )
            axis.axvspan(left, right, color="#f4a261", alpha=0.25)
    axis.set(xlabel="Time (seconds)", ylabel="Predicted importance",
             title="Scores and selected shots (shaded)")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def write_comparison_graphs(
    timestamps: np.ndarray,
    scores: dict[str, np.ndarray],
    summaries: dict[str, np.ndarray],
    output_dir: Path,
    timings: dict[str, float],
) -> dict[str, str]:
    colors = {"mean": "#205493", "barycenter": "#e76f51"}
    paths = {
        "overlay": output_dir / "mean_barycenter_overlay.png",
        "difference": output_dir / "barycenter_score_difference.png",
        "agreement": output_dir / "shot_selection_comparison.png",
        "timings": output_dir / "comparison_timings.png",
    }

    figure, axis = plt.subplots(figsize=(11, 4))
    for target in ("mean", "barycenter"):
        selected = summaries[target] > 0
        axis.plot(
            timestamps, scores[target], color=colors[target], linewidth=1.5,
            label=f"{target.title()} scores",
        )
        axis.scatter(
            timestamps[selected], scores[target][selected],
            color=colors[target], s=16, marker="o" if target == "mean" else "x",
            label=f"{target.title()} selected frames",
        )
    axis.set(
        xlabel="Time (seconds)",
        ylabel="Predicted importance",
        title="Mean and barycenter scores with selected frames",
    )
    axis.legend(loc="best", ncol=2)
    figure.tight_layout()
    figure.savefig(paths["overlay"], dpi=140)
    plt.close(figure)

    difference = scores["barycenter"] - scores["mean"]
    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.plot(timestamps, difference, color="#6a4c93", linewidth=1.4)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.fill_between(timestamps, difference, 0.0, where=difference >= 0,
                      color=colors["barycenter"], alpha=0.2)
    axis.fill_between(timestamps, difference, 0.0, where=difference < 0,
                      color=colors["mean"], alpha=0.2)
    axis.set(
        xlabel="Time (seconds)",
        ylabel="Barycenter - mean score",
        title="Per-frame score difference",
    )
    figure.tight_layout()
    figure.savefig(paths["difference"], dpi=140)
    plt.close(figure)

    mean_selected = summaries["mean"] > 0
    barycenter_selected = summaries["barycenter"] > 0
    mean_seconds = float(mean_selected.sum() * SAMPLE_PERIOD_SECONDS)
    barycenter_seconds = float(barycenter_selected.sum() * SAMPLE_PERIOD_SECONDS)
    shared_seconds = float(
        np.logical_and(mean_selected, barycenter_selected).sum()
        * SAMPLE_PERIOD_SECONDS
    )
    union_seconds = float(
        np.logical_or(mean_selected, barycenter_selected).sum()
        * SAMPLE_PERIOD_SECONDS
    )
    agreement = shared_seconds / union_seconds if union_seconds else 1.0
    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(
        ["Mean selected", "Barycenter selected", "Selected by both"],
        [mean_seconds, barycenter_seconds, shared_seconds],
        color=[colors["mean"], colors["barycenter"], "#7a9e7e"],
    )
    axis.bar_label(bars, fmt="%.1f s")
    axis.set_ylabel("Selected video duration (seconds)")
    axis.set_title(f"Shot selection overlap (IoU: {agreement:.0%})")
    figure.tight_layout()
    figure.savefig(paths["agreement"], dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    labels = [
        "Shared feature extraction",
        "Mean inference",
        "Barycenter inference",
        "Shared shot detection",
        "Both summary encodes",
    ]
    values = [
        timings["feature_seconds"],
        timings["mean_scoring_seconds"],
        timings["barycenter_scoring_seconds"],
        timings["shot_seconds"],
        timings["render_seconds"],
    ]
    bars = axis.barh(labels, values, color=[
        "#8d99ae", colors["mean"], colors["barycenter"], "#8d99ae", "#8d99ae"
    ])
    axis.bar_label(bars, fmt="%.3f s", padding=3)
    axis.set_xlabel("Wall-clock seconds")
    axis.set_title("Measured processing time (shared stages run once)")
    figure.tight_layout()
    figure.savefig(paths["timings"], dpi=140)
    plt.close(figure)
    return {key: str(path) for key, path in paths.items()}


def load_demo_model(
    target: str, model_dir: Path = MODEL_DIR
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict]:
    if target not in ("mean", "barycenter"):
        raise ValueError(f"Unknown demo target: {target!r}")
    config_path = model_dir / "config.json"
    scaler_path = model_dir / "feature_scaler.npz"
    weights_path = model_dir / f"{target}_target.pt"
    if not all(path.is_file() for path in (config_path, scaler_path, weights_path)):
        raise FileNotFoundError(f"Incomplete demo model artifacts in {model_dir}.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scaler_hash = file_sha256(scaler_path)
    weights_hash = file_sha256(weights_path)
    if scaler_hash != config["scaler_sha256"] or weights_hash != config["weights"][target]:
        raise ValueError("Demo model artifact hash does not match config.json.")
    with np.load(scaler_path) as scaler:
        mean = scaler["mean"].astype(np.float32)
        std = scaler["std"].astype(np.float32)
    model = benchmark.BiLSTMRegressor()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, mean, std, config


def process_video_comparison(
    input_path: Path,
    output_dir: Path,
    progress: Progress | None = None,
    extractor: torch.nn.Module | None = None,
    transform: Callable | None = None,
) -> dict:
    input_path = input_path.resolve()
    duration, _, _ = video_metadata(input_path)
    if duration > MAX_DURATION_SECONDS:
        raise ValueError("Video length exceeds the 10-minute limit.")
    if extractor is None or transform is None:
        extractor, transform = load_pool5_extractor()
    output_dir.mkdir(parents=True, exist_ok=False)
    if progress:
        progress(0.1, "Extracting shared video features")
    sampled, feature_stats = extract_pool5(input_path, extractor, transform)
    if progress:
        progress(0.4, "Detecting shots once for both targets")
    change_points, segment_lengths, shot_seconds = detect_sampled_shots(
        input_path, sampled.timestamps
    )
    summary_budget = int(
        math.floor(sampled.timestamps.size * benchmark.BUDGET_FRACTION)
    )
    change_points, segment_lengths, oversized_scenes_split = split_oversized_scenes(
        change_points, segment_lengths, summary_budget
    )

    target_results: dict[str, dict] = {}
    scores_by_target: dict[str, np.ndarray] = {}
    summaries_by_target: dict[str, np.ndarray] = {}
    scoring_seconds: dict[str, float] = {}
    render_seconds = 0.0
    for target in ("mean", "barycenter"):
        model, scaler_mean, scaler_std, config = load_demo_model(target)
        scaled = (sampled.features - scaler_mean) / scaler_std
        if not np.isfinite(scaled).all():
            raise ValueError("Feature scaler produced non-finite demo inputs.")
        score_started = time.perf_counter()
        scores = benchmark.predict_distribution(model, scaled)
        scoring_seconds[target] = time.perf_counter() - score_started
        summary, _ = summarize_scores(scores, change_points, segment_lengths)
        intervals = selected_intervals(
            summary, change_points, sampled.timestamps, sampled.duration_seconds
        )
        if not intervals:
            raise RuntimeError(
                f"Canonical shot selection returned an empty {target} summary."
            )
        summary_path = output_dir / f"{target}_summary.mp4"
        render_started = time.perf_counter()
        render_summary(input_path, intervals, summary_path)
        render_seconds += time.perf_counter() - render_started
        target_results[target] = {
            "summary_clip": str(summary_path),
            "selected_duration_seconds": sum(
                end - start for start, end in intervals
            ),
            "selected_intervals": intervals,
            "model_label": config["label"],
        }
        scores_by_target[target] = scores
        summaries_by_target[target] = summary
        if progress:
            progress(
                0.6 if target == "mean" else 0.78,
                f"Scoring and rendering {target} summary",
            )
    timings = {
        "feature_seconds": float(feature_stats["feature_seconds"]),
        "shot_seconds": shot_seconds,
        "mean_scoring_seconds": scoring_seconds["mean"],
        "barycenter_scoring_seconds": scoring_seconds["barycenter"],
        "render_seconds": render_seconds,
    }
    graph_paths = write_comparison_graphs(
        sampled.timestamps,
        scores_by_target,
        summaries_by_target,
        output_dir,
        timings,
    )
    if progress:
        progress(0.95, "Preparing comparison graphs")
    return {
        "targets": target_results,
        "graphs": graph_paths,
        "duration_seconds": sampled.duration_seconds,
        "sample_count": int(sampled.timestamps.size),
        "feature_stats": feature_stats,
        "timings": timings,
        "oversized_scenes_split_to_fit_budget": oversized_scenes_split,
        "budget_fraction": benchmark.BUDGET_FRACTION,
    }


def process_video(
    input_path: Path,
    target: str,
    output_dir: Path,
    progress: Progress | None = None,
    extractor: torch.nn.Module | None = None,
    transform: Callable | None = None,
) -> dict:
    input_path = input_path.resolve()
    duration, _, _ = video_metadata(input_path)
    if duration > MAX_DURATION_SECONDS:
        raise ValueError("Video length exceeds the 10-minute limit.")
    if extractor is None or transform is None:
        extractor, transform = load_pool5_extractor()
    output_dir.mkdir(parents=True, exist_ok=False)
    if progress:
        progress(0.1, "Decoding frames and extracting GoogLeNet features")
    sampled, feature_stats = extract_pool5(input_path, extractor, transform)
    if progress:
        progress(0.4, "Detecting shots")
    change_points, segment_lengths, shot_seconds = detect_sampled_shots(
        input_path, sampled.timestamps
    )
    summary_budget = int(
        math.floor(sampled.timestamps.size * benchmark.BUDGET_FRACTION)
    )
    change_points, segment_lengths, oversized_scenes_split = split_oversized_scenes(
        change_points, segment_lengths, summary_budget
    )
    model, scaler_mean, scaler_std, config = load_demo_model(target)
    scaled = (sampled.features - scaler_mean) / scaler_std
    if not np.isfinite(scaled).all():
        raise ValueError("Feature scaler produced non-finite demo inputs.")
    if progress:
        progress(0.55, "Scoring sampled frames")
    score_started = time.perf_counter()
    scores = benchmark.predict_distribution(model, scaled)
    score_seconds = time.perf_counter() - score_started
    if progress:
        progress(0.65, "Selecting shots under the 15% duration budget")
    summary, knapsack_seconds = summarize_scores(
        scores, change_points, segment_lengths
    )
    uniform_summary, uniform_knapsack_seconds = summarize_scores(
        np.ones_like(scores), change_points, segment_lengths
    )
    learned_intervals = selected_intervals(
        summary, change_points, sampled.timestamps, sampled.duration_seconds
    )
    uniform_intervals = selected_intervals(
        uniform_summary, change_points, sampled.timestamps,
        sampled.duration_seconds,
    )
    if not learned_intervals or not uniform_intervals:
        raise RuntimeError(
            "Canonical shot selection returned an empty summary despite "
            "budget-feasible scene segments."
        )
    if progress:
        progress(0.75, "Rendering summary clips with ffmpeg")
    ffmpeg_started = time.perf_counter()
    summary_path = output_dir / f"{target}_summary.mp4"
    uniform_path = output_dir / "uniform_summary.mp4"
    render_summary(input_path, learned_intervals, summary_path)
    render_summary(input_path, uniform_intervals, uniform_path)
    ffmpeg_seconds = time.perf_counter() - ffmpeg_started
    timeline_path = output_dir / "score_timeline.png"
    write_timeline(
        sampled.timestamps, scores, summary, change_points, timeline_path
    )
    timings = {
        "download_decode_seconds": float(feature_stats["decode_seconds"]),
        "feature_seconds": float(feature_stats["feature_seconds"]),
        "shot_seconds": shot_seconds,
        "scoring_seconds": score_seconds,
        "knapsack_seconds": knapsack_seconds + uniform_knapsack_seconds,
        "ffmpeg_seconds": ffmpeg_seconds,
    }
    selected_duration = sum(end - start for start, end in learned_intervals)
    uniform_duration = sum(end - start for start, end in uniform_intervals)
    return {
        "summary_clip": str(summary_path),
        "uniform_clip": str(uniform_path),
        "timeline": str(timeline_path),
        "target": target,
        "duration_seconds": sampled.duration_seconds,
        "selected_duration_seconds": selected_duration,
        "uniform_duration_seconds": uniform_duration,
        "budget_fraction": benchmark.BUDGET_FRACTION,
        "oversized_scenes_split_to_fit_budget": oversized_scenes_split,
        "selected_intervals": learned_intervals,
        "uniform_intervals": uniform_intervals,
        "sample_count": int(sampled.timestamps.size),
        "feature_stats": feature_stats,
        "timings": timings,
        "model_label": config["label"],
    }
