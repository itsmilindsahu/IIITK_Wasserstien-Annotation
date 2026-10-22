"""Prepare blinded study stimuli from a folder of source MP4 videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yt_dlp

if __package__ and "." in __package__:
    from ..app import run_comparison
    from ..pipeline import MAX_DURATION_SECONDS, file_sha256
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app import run_comparison
    from pipeline import MAX_DURATION_SECONDS, file_sha256

STUDY_DIR = Path(__file__).resolve().parent
STIMULI_DIR = Path(
    os.environ.get("STUDY_STIMULI_DIR", STUDY_DIR / "stimuli")
)
VIDEO_EXTENSIONS = {".mp4"}
FIXED_VIDEO_URLS = (
    "https://youtu.be/rz1B4B2yBZY",
    "https://youtu.be/ClpGjUQ7prU",
    "https://youtu.be/E85MXIbqfAc",
)


def _video_id(path: Path) -> str:
    digest = file_sha256(path)
    return f"video_{digest[:16]}"


def download_study_url(
    url: str,
    destination: Path,
    progress: Any = None,
) -> Path:
    parsed_url = urlsplit(url.strip())
    hostname = (parsed_url.hostname or "").lower()
    if (
        parsed_url.scheme not in ("http", "https")
        or not (
            hostname == "youtube.com"
            or hostname.endswith(".youtube.com")
            or hostname == "youtu.be"
        )
    ):
        raise ValueError("Study links must be public YouTube URLs.")

    duration_exceeded = False

    def reject_long_videos(info: dict[str, Any], *, incomplete: bool) -> str | None:
        nonlocal duration_exceeded
        duration = info.get("duration")
        if duration is not None and duration > MAX_DURATION_SECONDS:
            duration_exceeded = True
            return "Study videos must be 10 minutes or shorter."
        return None

    def report_progress(status: dict[str, Any]) -> None:
        if progress is None or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        downloaded = status.get("downloaded_bytes", 0)
        fraction = min(downloaded / total, 1.0) if total else 0.0
        progress(0.08 * fraction, desc="Downloading video")

    destination.mkdir(parents=True, exist_ok=True)
    options = {
        "format": (
            "bestvideo[height<=720]+bestaudio/"
            "best[height<=720]/bestvideo+bestaudio/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(destination / "study_video.%(ext)s"),
        "match_filter": reject_long_videos,
        "noplaylist": True,
        "progress_hooks": [report_progress],
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url.strip(), download=True)
            path = Path(downloader.prepare_filename(info))
    except yt_dlp.utils.DownloadError as exc:
        if duration_exceeded:
            raise ValueError("Study videos must be 10 minutes or shorter.") from exc
        raise RuntimeError(f"Could not download the YouTube video: {exc}") from exc
    if path.is_file():
        return path
    merged_path = path.with_suffix(".mp4")
    if merged_path.is_file():
        return merged_path
    outputs = [
        candidate
        for candidate in destination.glob("study_video.*")
        if candidate.suffix.lower() in {".mp4", ".mkv", ".webm"}
        and candidate.is_file()
    ]
    if len(outputs) == 1:
        return outputs[0]
    raise RuntimeError("YouTube download completed without a local video file.")


def _encode_blinded(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required and must be available on PATH.")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-vf",
        (
            "fps=30,scale=640:360:force_original_aspect_ratio=decrease,"
            "pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        "2000k",
        "-minrate",
        "2000k",
        "-maxrate",
        "2000k",
        "-bufsize",
        "4000k",
        "-x264-params",
        "nal-hrd=cbr",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not encode blinded stimulus {destination.name}: "
            f"{result.stderr.strip()}"
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create {destination}.")


def _encode_source(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required and must be available on PATH.")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not encode the source video: {result.stderr.strip()}"
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create {destination}.")


def _relative_difference(left: float, right: float) -> float:
    denominator = (abs(left) + abs(right)) / 2
    return abs(left - right) / denominator if denominator else 0.0


def _measurements(result: dict[str, Any], target: str) -> dict[str, Any]:
    target_result = result["targets"][target]
    intervals = target_result["selected_intervals"]
    shot_count = len(intervals)
    summary_duration = float(target_result["selected_duration_seconds"])
    return {
        "summary_duration_seconds": round(summary_duration, 3),
        "selected_shots": shot_count,
        "mean_shot_length_seconds": round(
            summary_duration / shot_count if shot_count else 0.0, 3
        ),
    }


def prepare_video(
    source: Path,
    output_dir: Path,
    progress: Any = None,
) -> dict[str, Any]:
    """Use the existing comparison cache, then create neutral A/B files."""
    source = source.resolve()
    video_id = _video_id(source)
    comparison_dir = (
        Path(__file__).resolve().parents[1]
        / "cache"
        / f"{file_sha256(source)}_comparison"
    )
    if (comparison_dir / "result.json").is_file():
        result = json.loads(
            (comparison_dir / "result.json").read_text(encoding="utf-8")
        )
        mean_clip = Path(result["targets"]["mean"]["summary_clip"])
        barycenter_clip = Path(result["targets"]["barycenter"]["summary_clip"])
    else:
        outputs = run_comparison(str(source), progress=progress)
        mean_clip, barycenter_clip = Path(outputs[0] or ""), Path(outputs[1] or "")
        if not mean_clip.is_file() or not barycenter_clip.is_file():
            raise RuntimeError(f"Summary generation failed for {source.name}: {outputs[6]}")
        if not comparison_dir.joinpath("result.json").is_file():
            raise RuntimeError(
                "The comparison finished without its expected cached result."
            )
        result = json.loads(
            comparison_dir.joinpath("result.json").read_text(encoding="utf-8")
        )
    if not mean_clip.is_file() or not barycenter_clip.is_file():
        raise RuntimeError(f"Cached summary clips are missing for {source.name}.")

    destination = output_dir / video_id
    destination.mkdir(parents=True, exist_ok=True)
    _encode_source(source, destination / "source.mp4")
    _encode_blinded(mean_clip, destination / "A.mp4")
    _encode_blinded(barycenter_clip, destination / "B.mp4")
    mean_stats = _measurements(result, "mean")
    barycenter_stats = _measurements(result, "barycenter")
    checks = (
        _relative_difference(
            mean_stats["summary_duration_seconds"],
            barycenter_stats["summary_duration_seconds"],
        ),
        _relative_difference(
            mean_stats["selected_shots"], barycenter_stats["selected_shots"]
        ),
        _relative_difference(
            mean_stats["mean_shot_length_seconds"],
            barycenter_stats["mean_shot_length_seconds"],
        ),
    )
    return {
        "video_id": video_id,
        "mean_summary_duration_seconds": mean_stats["summary_duration_seconds"],
        "mean_selected_shots": mean_stats["selected_shots"],
        "mean_mean_shot_length_seconds": mean_stats["mean_shot_length_seconds"],
        "barycenter_summary_duration_seconds": barycenter_stats[
            "summary_duration_seconds"
        ],
        "barycenter_selected_shots": barycenter_stats["selected_shots"],
        "barycenter_mean_shot_length_seconds": barycenter_stats[
            "mean_shot_length_seconds"
        ],
        "max_relative_difference": round(max(checks), 4),
        "pacing_length_confound_over_10_percent": max(checks) > 0.10,
    }


def append_confound_report(
    row: dict[str, Any],
    output_dir: Path = STIMULI_DIR,
) -> None:
    report_path = output_dir / "confound_check.csv"
    existing = []
    if report_path.is_file():
        with report_path.open(encoding="utf-8", newline="") as csv_file:
            existing = list(csv.DictReader(csv_file))
    by_id = {item["video_id"]: item for item in [*existing, row]}
    with report_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(by_id.values())


def prepare_folder(source_dir: Path, output_dir: Path = STIMULI_DIR) -> list[dict[str, Any]]:
    sources = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not sources:
        raise ValueError(f"No MP4 videos found in {source_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for source in sources:
        print(f"Preparing {source.name}", flush=True)
        reports.append(prepare_video(source, output_dir))
    report_path = output_dir / "confound_check.csv"
    with report_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    flagged = sum(
        1 for row in reports if row["pacing_length_confound_over_10_percent"]
    )
    print(
        f"Prepared {len(reports)} videos; {flagged} flagged for >10% "
        f"duration, shot-count, or mean-shot-length difference. "
        f"Details: {report_path}"
    )
    return reports


def prepare_fixed_links(
    output_dir: Path = STIMULI_DIR,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Download the three preset links and prepare neutral study stimuli."""
    manifest_path = output_dir / "fixed_video_ids.json"
    if not refresh and manifest_path.is_file():
        video_ids = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            len(video_ids) == len(FIXED_VIDEO_URLS)
            and all(
                all(
                    (output_dir / video_id / name).is_file()
                    for name in ("source.mp4", "A.mp4", "B.mp4")
                )
                for video_id in video_ids
            )
        ):
            return []

    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    video_ids = []
    with tempfile.TemporaryDirectory(prefix="tvsum-study-links-") as temporary:
        for index, url in enumerate(FIXED_VIDEO_URLS, start=1):
            download_dir = Path(temporary) / f"preset_{index}"
            download_dir.mkdir()
            source = download_study_url(url, download_dir)
            report = prepare_video(source, output_dir)
            reports.append(report)
            video_ids.append(report["video_id"])
    manifest_path.write_text(
        json.dumps(video_ids, indent=2) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "confound_check.csv"
    for report in reports:
        append_confound_report(report, output_dir)
    flagged = sum(
        1 for row in reports if row["pacing_length_confound_over_10_percent"]
    )
    print(
        f"Prepared {len(reports)} fixed videos; {flagged} flagged for >10% "
        f"duration, shot-count, or mean-shot-length difference. "
        f"Details: {report_path}"
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        type=Path,
        nargs="?",
        help="Optional MP4 folder; omitted to prepare the three preset links.",
    )
    parser.add_argument("--output-dir", type=Path, default=STIMULI_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download and prepare preset links again.",
    )
    args = parser.parse_args()
    if args.source_folder is None:
        prepare_fixed_links(args.output_dir, refresh=args.refresh)
    elif not args.source_folder.is_dir():
        parser.error(f"Not a folder: {args.source_folder}")
    else:
        prepare_folder(args.source_folder, args.output_dir)


if __name__ == "__main__":
    main()
