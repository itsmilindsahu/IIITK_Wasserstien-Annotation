"""Gradio upload-only demo; URL downloads require an explicit opt-in flag."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr
import yt_dlp

if __package__:
    from .pipeline import (
        MAX_DURATION_SECONDS,
        MAX_YOUTUBE_DURATION_SECONDS,
        file_sha256,
        process_video_comparison,
        video_metadata,
    )
else:
    from pipeline import (
        MAX_DURATION_SECONDS,
        MAX_YOUTUBE_DURATION_SECONDS,
        file_sha256,
        process_video_comparison,
        video_metadata,
    )

DEMO_DIR = Path(__file__).resolve().parent
CACHE_DIR = DEMO_DIR / "cache"
NOTICE = (
    "Demo only. Model trained on 50 TVSum videos; quality on other video types "
    "is not validated."
)


def _download_url(
    url: str, destination: Path, progress: gr.Progress | None = None
) -> Path:
    if not url.strip():
        raise gr.Error("Enter a video URL after enabling URL input.")
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
        raise gr.Error("URL input accepts YouTube URLs only.")
    duration_exceeded = False

    def reject_long_videos(info: dict, *, incomplete: bool) -> str | None:
        nonlocal duration_exceeded
        duration = info.get("duration")
        if duration is not None and duration > MAX_YOUTUBE_DURATION_SECONDS:
            duration_exceeded = True
            return "YouTube videos must be 2 minutes or shorter."
        return None

    def report_download_progress(status: dict) -> None:
        if progress is None or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        downloaded = status.get("downloaded_bytes", 0)
        fraction = min(downloaded / total, 1.0) if total else 0.0
        progress(0.08 * fraction, desc="Downloading YouTube video")

    download_options = {
        "format": (
            "bestvideo[ext=mp4][height<=480]/"
            "bestvideo[height<=480]/"
            "bestvideo[ext=mp4][height<=720]/"
            "bestvideo[height<=720]/bestvideo"
        ),
        "outtmpl": str(destination / "url_video.%(ext)s"),
        "match_filter": reject_long_videos,
        "noplaylist": True,
        "progress_hooks": [report_download_progress],
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(download_options) as downloader:
            info = downloader.extract_info(url, download=True)
            if duration_exceeded:
                raise gr.Error("YouTube videos must be 2 minutes or shorter.")
            path = Path(downloader.prepare_filename(info))
    except gr.Error:
        raise
    except Exception as exc:
        if duration_exceeded:
            raise gr.Error("YouTube videos must be 2 minutes or shorter.") from exc
        raise gr.Error(f"Could not download the URL: {exc}") from exc
    if not path.is_file():
        raise gr.Error("URL download completed without a local video file.")
    return path


def run_comparison(
    uploaded_file: str | None,
    url: str = "",
    progress: gr.Progress | None = None,
) -> tuple[
    str | None, str | None, str | None, str | None, str | None, str | None, str
]:
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    try:
        input_path = Path(uploaded_file) if uploaded_file else None
        from_youtube = bool(url.strip())
        if from_youtube:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="tvsum-demo-download-"
            )
            if progress:
                progress(0.02, desc="Checking YouTube URL")
            input_path = _download_url(
                url, Path(temporary_directory.name), progress=progress
            )
        if input_path is None:
            raise gr.Error("Upload a video file to begin.")
        if not input_path.is_file():
            raise gr.Error("The uploaded video file is no longer available.")
        duration, _, _ = video_metadata(input_path)
        if duration > MAX_DURATION_SECONDS:
            raise gr.Error("Video length exceeds the 10-minute limit.")
        if from_youtube and duration > MAX_YOUTUBE_DURATION_SECONDS:
            raise gr.Error("YouTube videos must be 2 minutes or shorter.")
        video_hash = file_sha256(input_path)
        output_dir = CACHE_DIR / f"{video_hash}_comparison"
        if (output_dir / "result.json").is_file():
            result = json.loads(
                (output_dir / "result.json").read_text(encoding="utf-8")
            )
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{video_hash[:12]}-", dir=CACHE_DIR
            ) as staging_directory:
                staging_output = Path(staging_directory) / "outputs"
                result = process_video_comparison(
                    input_path,
                    staging_output,
                    progress=(
                        lambda fraction, description: progress(
                            fraction, desc=description
                        )
                        if progress
                        else None
                    ),
                )
                for target_result in result["targets"].values():
                    target_result["summary_clip"] = str(
                        output_dir
                        / Path(target_result["summary_clip"]).name
                    )
                result["graphs"] = {
                    graph_name: str(output_dir / Path(graph_path).name)
                    for graph_name, graph_path in result["graphs"].items()
                }
                (staging_output / "result.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                try:
                    os.replace(staging_output, output_dir)
                except OSError:
                    if not (output_dir / "result.json").is_file():
                        raise
                    result = json.loads(
                        (output_dir / "result.json").read_text(encoding="utf-8")
                    )
        status = (
            f"Input SHA-256: {video_hash}\n"
            f"Duration: {result['duration_seconds']:.1f}s; both summaries "
            f"use the same input and 15% duration budget.\n"
            f"Feature extraction: {result['timings']['feature_seconds']:.3f}s "
            f"(shared once); shot detection: "
            f"{result['timings']['shot_seconds']:.3f}s (shared once).\n"
            f"Mean scoring: {result['timings']['mean_scoring_seconds']:.4f}s; "
            f"barycenter scoring: "
            f"{result['timings']['barycenter_scoring_seconds']:.4f}s.\n"
            "Both checkpoints use the same BiLSTM architecture; the "
            "barycenter changes the training target, not the inference steps."
        )
        if result.get("oversized_scenes_split_to_fit_budget"):
            status += (
                "\nSome detected scenes exceeded the budget and were divided "
                "into contiguous segments for selection."
            )
        return (
            result["targets"]["mean"]["summary_clip"],
            result["targets"]["barycenter"]["summary_clip"],
            result["graphs"]["overlay"],
            result["graphs"]["difference"],
            result["graphs"]["agreement"],
            result["graphs"]["timings"],
            status,
        )
    except gr.Error as exc:
        return None, None, None, None, None, None, str(exc)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, None, None, None, None, None, f"Processing failed: {exc}"
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def build_demo_app(enable_url: bool = False) -> gr.Blocks:
    with gr.Blocks(title="TVSum Summary Demo") as app:
        gr.Markdown("# TVSum Summarization Demo")
        gr.Markdown(NOTICE)
        gr.Markdown(
            "Upload a video up to 10 minutes or paste a YouTube URL up to "
            "2 minutes. Mean and barycenter summaries are generated together "
            "from one shared feature-extraction pass."
        )
        upload = gr.Video(
            label="Video file (upload)",
            sources=["upload"],
            format="mp4",
            height=240,
        )
        url_input = None
        if enable_url:
            gr.Markdown(
                "**URL download enabled:** YouTube videos must be 2 minutes "
                "or shorter. Downloading may violate YouTube's terms; ensure "
                "your use is permitted."
            )
            url_input = gr.Textbox(
                label="Optional video URL",
                placeholder="Only available when started with --enable-url",
            )

        run = gr.Button("Generate summaries", variant="primary")
        with gr.Row():
            mean_output = gr.Video(label="Mean-target summary")
            barycenter_output = gr.Video(label="Barycenter-target summary")
        overlay_output = gr.Image(label="Overlaid mean and barycenter scores")
        difference_output = gr.Image(label="Per-frame score difference")
        agreement_output = gr.Image(label="Selected-shot overlap")
        timing_output = gr.Image(label="Measured timing comparison")
        status_output = gr.Textbox(label="Status", lines=3)
        outputs = [
            mean_output, barycenter_output, overlay_output, difference_output,
            agreement_output, timing_output, status_output,
        ]
        if enable_url:
            run.click(
                fn=run_comparison,
                inputs=[upload, url_input],
                outputs=outputs,
            )
        else:
            run.click(
                fn=run_comparison,
                inputs=[upload],
                outputs=outputs,
            )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-url",
        action="store_true",
        help="Enable optional URL downloads; disabled by default.",
    )
    parser.add_argument(
        "--study",
        action="store_true",
        help="Launch the separate blinded preference-study app.",
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    if args.study:
        if __package__:
            from .study.app import build_study_app
        else:
            from study.app import build_study_app

        app = build_study_app()
    else:
        app = build_demo_app(enable_url=args.enable_url)
    app.launch(
        server_name=args.server_name, server_port=args.server_port
    )


if __name__ == "__main__":
    main()
