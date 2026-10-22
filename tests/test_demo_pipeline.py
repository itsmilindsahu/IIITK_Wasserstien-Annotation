import hashlib
import socket
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")
pytest.importorskip("gradio")

from demo import app as demo_app
from demo import pipeline


def _make_short_clip(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (96, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 encoding is unavailable.")
    for frame_index in range(300):
        scene_index = frame_index // 20
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :] = (
            (scene_index * 47) % 255,
            (scene_index * 83) % 255,
            (scene_index * 113) % 255,
        )
        x = (frame_index * 3) % 80
        cv2.rectangle(frame, (x, 12), (x + 12, 52), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _make_static_clip(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (96, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 encoding is unavailable.")
    frame = np.full((64, 96, 3), 96, dtype=np.uint8)
    for _ in range(300):
        writer.write(frame)
    writer.release()


def test_demo_pipeline_is_deterministic_and_respects_summary_budget(tmp_path):
    torch.set_num_threads(2)
    clip = tmp_path / "short_test.mp4"
    _make_short_clip(clip)
    extractor, transform = pipeline.load_pool5_extractor()
    first = pipeline.process_video(
        clip,
        "mean",
        tmp_path / "first",
        extractor=extractor,
        transform=transform,
    )
    second = pipeline.process_video(
        clip,
        "mean",
        tmp_path / "second",
        extractor=extractor,
        transform=transform,
    )

    assert first["selected_intervals"] == second["selected_intervals"]
    assert first["selected_duration_seconds"] == second["selected_duration_seconds"]
    assert first["selected_duration_seconds"] <= (
        first["duration_seconds"] * 0.15 + 1.0
    )
    for output_key in ("summary_clip", "uniform_clip", "timeline"):
        assert Path(first[output_key]).is_file()
        assert Path(second[output_key]).is_file()
    first_hash = hashlib.sha256(Path(first["summary_clip"]).read_bytes()).hexdigest()
    second_hash = hashlib.sha256(Path(second["summary_clip"]).read_bytes()).hexdigest()
    assert first_hash == second_hash


def test_no_cut_video_gets_budget_feasible_subsegments(tmp_path):
    torch.set_num_threads(2)
    clip = tmp_path / "static_test.mp4"
    _make_static_clip(clip)
    detected_cuts, detected_lengths, _ = pipeline.detect_sampled_shots(
        clip, np.arange(30, dtype=np.float64)
    )
    assert len(detected_lengths) == 1
    assert detected_lengths[0] == 30

    extractor, transform = pipeline.load_pool5_extractor()
    result = pipeline.process_video(
        clip,
        "mean",
        tmp_path / "static_result",
        extractor=extractor,
        transform=transform,
    )

    assert result["oversized_scenes_split_to_fit_budget"] is True
    assert result["selected_intervals"]
    assert result["selected_duration_seconds"] <= 30 * 0.15 + 1.0
    assert Path(result["summary_clip"]).is_file()
    assert Path(result["uniform_clip"]).is_file()


def test_comparison_pipeline_shares_extraction_and_shows_both_targets(
    tmp_path, monkeypatch
):
    clip = tmp_path / "comparison.mp4"
    clip.write_bytes(b"input")
    timestamps = np.arange(30, dtype=np.float64)
    sampled = pipeline.SampledVideo(
        features=np.zeros((30, 1024), dtype=np.float32),
        timestamps=timestamps,
        duration_seconds=30.0,
        fps=1.0,
    )
    calls = {"extract": 0, "shots": 0, "predict": 0, "render": 0}
    monkeypatch.setattr(pipeline, "video_metadata", lambda _: (30.0, 1.0, 30))
    monkeypatch.setattr(pipeline, "load_pool5_extractor", lambda: (object(), object()))

    def fake_extract(*args):
        calls["extract"] += 1
        return sampled, {"feature_seconds": 0.5, "decode_seconds": 0.1}

    def fake_detect(*args):
        calls["shots"] += 1
        return np.asarray([[0, 29]]), np.asarray([30]), 0.05

    monkeypatch.setattr(pipeline, "extract_pool5", fake_extract)
    monkeypatch.setattr(pipeline, "detect_sampled_shots", fake_detect)
    monkeypatch.setattr(
        pipeline, "load_demo_model",
        lambda target: (target, np.zeros(1024), np.ones(1024), {"label": target}),
    )

    def fake_predict(model, features):
        calls["predict"] += 1
        return np.linspace(0.0, 1.0, len(features), dtype=np.float32)

    monkeypatch.setattr(pipeline.benchmark, "predict_distribution", fake_predict)

    def fake_summarize(scores, change_points, segment_lengths):
        selected = np.zeros(len(scores), dtype=np.int64)
        selected[:4] = 1
        return selected, 0.001

    monkeypatch.setattr(pipeline, "summarize_scores", fake_summarize)

    def fake_render(input_path, intervals, output_path):
        calls["render"] += 1
        output_path.write_bytes(b"summary")

    monkeypatch.setattr(pipeline, "render_summary", fake_render)

    result = pipeline.process_video_comparison(
        clip, tmp_path / "comparison_output"
    )

    assert calls == {"extract": 1, "shots": 1, "predict": 2, "render": 2}
    assert set(result["targets"]) == {"mean", "barycenter"}
    assert set(result["graphs"]) == {
        "overlay", "difference", "agreement", "timings"
    }
    assert all(
        Path(target_result["summary_clip"]).is_file()
        for target_result in result["targets"].values()
    )
    assert all(Path(graph).is_file() for graph in result["graphs"].values())


def test_demo_app_is_upload_only_unless_url_flag_is_enabled():
    app = demo_app.build_demo_app(enable_url=False)
    component_labels = {
        item.get("props", {}).get("label")
        for item in app.config["components"]
    }
    assert "Video file (upload)" in component_labels
    assert "Optional video URL" not in component_labels

    url_app = demo_app.build_demo_app(enable_url=True)
    url_labels = {
        item.get("props", {}).get("label")
        for item in url_app.config["components"]
    }
    assert "Optional video URL" in url_labels
    assert "Mean-target summary" in url_labels
    assert "Barycenter-target summary" in url_labels
    assert "Overlaid mean and barycenter scores" in url_labels


def test_url_download_rejects_non_youtube_hosts():
    with pytest.raises(demo_app.gr.Error, match="YouTube URLs only"):
        demo_app._download_url("https://example.com/video.mp4", Path("unused"))


def test_url_download_uses_single_video_stream_and_no_playlist(tmp_path, monkeypatch):
    options_seen = []
    extraction_calls = []
    downloaded_path = tmp_path / "url_video.mp4"

    class FakeYoutubeDL:
        def __init__(self, options):
            options_seen.append(options)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def extract_info(self, url, download=False):
            assert url == "https://youtu.be/example"
            extraction_calls.append(download)
            if download:
                assert options_seen[0]["match_filter"](
                    {"duration": 120}, incomplete=False
                ) is None
                downloaded_path.write_bytes(b"video")
            return {"duration": 120}

        def prepare_filename(self, info):
            return str(downloaded_path)

    monkeypatch.setattr(demo_app.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    result = demo_app._download_url(
        "https://youtu.be/example", tmp_path
    )

    assert result == downloaded_path
    assert len(options_seen) == 1
    assert options_seen[0]["format"] == (
        "bestvideo[ext=mp4][height<=480]/"
        "bestvideo[height<=480]/"
        "bestvideo[ext=mp4][height<=720]/"
        "bestvideo[height<=720]/bestvideo"
    )
    assert options_seen[0]["noplaylist"] is True
    assert extraction_calls == [True]


def test_url_download_rejects_long_video_before_downloading(tmp_path, monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def extract_info(self, url, download=False):
            reason = self.options["match_filter"](
                {"duration": 121}, incomplete=False
            )
            assert reason == "YouTube videos must be 2 minutes or shorter."
            raise RuntimeError("download was rejected")

    monkeypatch.setattr(demo_app.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    with pytest.raises(demo_app.gr.Error, match="2 minutes or shorter"):
        demo_app._download_url("https://youtu.be/example", tmp_path)


def test_upload_only_app_starts_without_url_flag():
    app = demo_app.build_demo_app(enable_url=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        _, local_url, _ = app.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            quiet=True,
        )
        with urllib.request.urlopen(local_url, timeout=10) as response:
            assert response.status == 200
    finally:
        app.close(verbose=False)


def test_app_reports_ten_minute_limit(tmp_path, monkeypatch):
    clip = tmp_path / "oversize.mp4"
    clip.write_bytes(b"placeholder")
    monkeypatch.setattr(demo_app, "video_metadata", lambda _: (601.0, 30.0, 18030))

    outputs = demo_app.run_comparison(str(clip))

    assert outputs[:6] == (None, None, None, None, None, None)
    assert "10-minute limit" in outputs[6]


def test_comparison_cache_uses_file_hash(tmp_path, monkeypatch):
    input_path = tmp_path / "uploaded.mp4"
    input_path.write_bytes(b"test video payload")
    monkeypatch.setattr(demo_app, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        demo_app, "video_metadata", lambda _: (30.0, 30.0, 900)
    )
    calls = []

    def fake_process(path, output_dir, progress=None):
        output_dir.mkdir(parents=True)
        outputs = {
            "targets": {
                target: {
                    "summary_clip": output_dir / f"{target}.mp4",
                    "selected_duration_seconds": 4.0,
                }
                for target in ("mean", "barycenter")
            },
            "graphs": {
                graph: output_dir / f"{graph}.png"
                for graph in ("overlay", "difference", "agreement", "timings")
            },
            "timings": {
                "feature_seconds": 1.0,
                "shot_seconds": 0.1,
                "mean_scoring_seconds": 0.01,
                "barycenter_scoring_seconds": 0.01,
            },
            "duration_seconds": 30.0,
        }
        for target_result in outputs["targets"].values():
            target_result["summary_clip"] = str(target_result["summary_clip"])
            Path(target_result["summary_clip"]).write_bytes(b"summary")
        for graph_path in outputs["graphs"].values():
            graph_path.write_bytes(b"graph")
        outputs["graphs"] = {
            name: str(path) for name, path in outputs["graphs"].items()
        }
        calls.append("comparison")
        return outputs

    monkeypatch.setattr(demo_app, "process_video_comparison", fake_process)
    first = demo_app.run_comparison(str(input_path))
    cached = demo_app.run_comparison(str(input_path))

    assert first == cached
    assert calls == ["comparison"]
    assert Path(first[0]).is_file()
    assert Path(first[1]).is_file()
    assert all(Path(path).is_file() for path in first[2:6])
