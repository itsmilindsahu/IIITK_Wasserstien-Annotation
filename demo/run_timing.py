"""Run CPU timing for three local Creative Commons test clips."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

from pipeline import load_pool5_extractor, process_video, video_metadata

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "timing.json"


def collect_timings(
    clips: list[Path], output: Path = DEFAULT_OUTPUT
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing timing file: {output}")
    if len(clips) != 3:
        raise ValueError("Exactly three test clips of different lengths are required.")
    durations = [video_metadata(path)[0] for path in clips]
    if len(set(durations)) != 3:
        raise ValueError("Timing clips must have three distinct durations.")
    extractor, transform = load_pool5_extractor()
    rows = []
    for clip in clips:
        duration, _, _ = video_metadata(clip)
        with tempfile.TemporaryDirectory(prefix="tvsum-demo-timing-") as temp_dir:
            result = process_video(
                clip,
                "mean",
                Path(temp_dir) / "outputs",
                extractor=extractor,
                transform=transform,
            )
        rows.append({
            "clip": clip.name,
            "duration_seconds": duration,
            "stages_seconds": result["timings"],
        })
    stage_names = list(rows[0]["stages_seconds"])
    stage_means = {
        stage: float(
            statistics.mean(row["stages_seconds"][stage] for row in rows)
        )
        for stage in stage_names
    }
    result = {
        "purpose": (
            "Demo-system timing only. Clips are licensed CC BY 3.0 test extracts; "
            "no TVSum benchmark scores are computed."
        ),
        "device": "CPU",
        "sampling_rate_fps": 1,
        "stage_definition": {
            "download_decode_seconds": "Upload-mode local decode of sampled frames; no network download",
            "feature_seconds": "GoogLeNet preprocessing and pool5 forward passes",
            "shot_seconds": "PySceneDetect ContentDetector",
            "scoring_seconds": "BiLSTM inference",
            "knapsack_seconds": "Canonical 0/1 shot knapsack, model and uniform",
            "ffmpeg_seconds": "Encoding model and uniform summary clips",
        },
        "clips": rows,
        "mean_stage_seconds": stage_means,
        "dominant_stage_by_mean_seconds": max(
            stage_means, key=stage_means.get
        ),
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("clips", nargs=3, type=Path)
    args = parser.parse_args()
    print(json.dumps(collect_timings(args.clips, args.output), indent=2))


if __name__ == "__main__":
    main()
