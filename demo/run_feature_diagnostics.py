"""Compare torchvision GoogLeNet pool5 statistics with the saved TVSum HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pipeline import extract_pool5, load_pool5_extractor, video_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "feature_diagnostics.json"
LICENSE_URL = "https://peach.blender.org/about/"
SOURCE_URL = (
    "https://download.blender.org/peach/bigbuckbunny_movies/"
    "BigBuckBunny_320x180.mp4.zip"
)


def _stats(vectors: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "feature_mean": float(vectors.mean()),
        "feature_std": float(vectors.std()),
        "l2_mean": float(norms.mean()),
        "l2_std": float(norms.std()),
        "l2_quantiles_0_25_50_75_100": np.quantile(
            norms, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "vector_count": int(vectors.shape[0]),
        "feature_dimension": int(vectors.shape[1]),
    }


def run_diagnostics(
    feature_file: Path, clips: list[Path], output: Path = DEFAULT_OUTPUT
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostics: {output}")
    if len(clips) < 2:
        raise ValueError("At least two licensed test clips are required.")
    with h5py.File(feature_file, "r") as source:
        hdf5_vectors = np.concatenate(
            [
                np.asarray(group["features"], dtype=np.float32)
                for group in source.values()
            ],
            axis=0,
        )
    hdf5_stats = _stats(hdf5_vectors)
    extractor, transform = load_pool5_extractor()
    clip_stats = []
    normalized_vectors = []
    for clip in clips:
        sampled, timing = extract_pool5(clip, extractor, transform)
        duration, _, _ = video_metadata(clip)
        clip_stats.append({
            "file": clip.name,
            "duration_seconds": duration,
            "feature_count": int(sampled.features.shape[0]),
            "l2_normalized_feature_mean": timing["l2_normalized_mean"],
            "l2_normalized_feature_std": timing["l2_normalized_std"],
            "l2_normalized_norm_mean": timing["l2_normalized_norm_mean"],
            "raw_feature_mean": timing["raw_mean"],
            "raw_feature_std": timing["raw_std"],
            "raw_l2_mean": timing["raw_l2_mean"],
            "raw_l2_std": timing["raw_l2_std"],
            "raw_l2_quantiles_0_25_50_75_100": timing["raw_l2_quantiles"],
            "decode_seconds": timing["decode_seconds"],
            "feature_seconds": timing["feature_seconds"],
        })
        normalized_vectors.append(sampled.features)

    normalized_stats = _stats(np.concatenate(normalized_vectors, axis=0))
    raw_l2_mean = float(np.mean([row["raw_l2_mean"] for row in clip_stats]))
    normalized_hdf5_difference = {
        "mean_abs_difference": abs(
            normalized_stats["feature_mean"] - hdf5_stats["feature_mean"]
        ),
        "std_relative_difference": abs(
            normalized_stats["feature_std"] - hdf5_stats["feature_std"]
        ) / hdf5_stats["feature_std"],
        "l2_mean_abs_difference": abs(
            normalized_stats["l2_mean"] - hdf5_stats["l2_mean"]
        ),
    }
    result = {
        "purpose": (
            "Extractor diagnostic only; these CC-licensed clips are not TVSum "
            "benchmark data and no benchmark metric is computed from them."
        ),
        "feature_file": str(feature_file.name),
        "source": {
            "title": "Big Buck Bunny, Blender Foundation",
            "download_url": SOURCE_URL,
            "license": "Creative Commons Attribution 3.0",
            "license_url": LICENSE_URL,
            "credit": "Blender Foundation / www.bigbuckbunny.org",
            "note": "Three time-window clips are sampled from this licensed source.",
        },
        "torchvision_weights": "GoogLeNet_Weights.IMAGENET1K_V1",
        "hdf5_features": hdf5_stats,
        "torchvision_raw_pool5": {
            "mean_across_clip_means": float(
                np.mean([row["raw_feature_mean"] for row in clip_stats])
            ),
            "std_across_clip_stds": float(
                np.mean([row["raw_feature_std"] for row in clip_stats])
            ),
            "l2_mean_across_clips": raw_l2_mean,
        },
        "torchvision_l2_normalized_pool5": normalized_stats,
        "normalized_comparison": normalized_hdf5_difference,
        "preprocessing_assessment": (
            "Raw pool5 norms differ from the HDF5's unit-norm vectors; per-frame "
            "L2 normalization is applied before the saved training scaler. "
            "Remaining channel-statistic differences are disclosed; the scaler "
            "is applied exactly as trained."
        ),
        "clips": clip_stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "datasets" / "eccv16_dataset_tvsum_google_pool5.h5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("clips", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_diagnostics(args.features, args.clips, args.output), indent=2))


if __name__ == "__main__":
    main()
