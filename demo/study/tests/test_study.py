from __future__ import annotations

import random
import sqlite3
import json
from contextlib import closing
from pathlib import Path

import gradio as gr
import pytest

from demo.study.analyze import analyze, analyze_records
from demo.study import app as study_app
from demo.study import prepare_study
from demo.study.app import TESTER_UI_TEXT
from demo.study.core import (
    balanced_model_for_A,
    connect_database,
    derive_chosen_model,
    displayed_summary_paths,
    export_responses,
    record_response,
)


def test_randomization_counterbalances_assignments():
    rng = random.Random(17)
    counts = {"mean": 0, "barycenter": 0}
    for _ in range(1001):
        assigned = balanced_model_for_A(counts, rng)
        counts[assigned] += 1
    assert abs(counts["mean"] - counts["barycenter"]) == 1


def test_tester_session_selects_three_to_five_unique_videos(monkeypatch, tmp_path):
    stimuli = [
        {
            "video_id": f"video-{index}",
            "source": Path(f"source-{index}.mp4"),
            "A": Path(f"A-{index}.mp4"),
            "B": Path(f"B-{index}.mp4"),
        }
        for index in range(6)
    ]
    monkeypatch.setattr(study_app, "list_stimuli", lambda: stimuli)
    monkeypatch.setattr(study_app, "STIMULI_DIR", tmp_path)
    monkeypatch.setattr(
        study_app, "_new_assignment", lambda video_id, tester_id: "mean"
    )
    outputs = study_app.start_study(True, None)
    state = outputs[0]
    selected_ids = [video["video_id"] for video in state["videos"]]
    assert len(outputs) == 13
    assert 3 <= len(selected_ids) <= 5
    assert len(selected_ids) == len(set(selected_ids))
    comparison = study_app.continue_to_summaries(state, "yes", "No")
    assert len(comparison) == 6
    assert comparison[4] == str(state["videos"][0]["A"])
    assert comparison[5] == str(state["videos"][0]["B"])


def test_fixed_manifest_presents_exactly_three_preset_videos(
    monkeypatch, tmp_path
):
    stimuli = [
        {
            "video_id": f"preset-{index}",
            "source": Path(f"source-{index}.mp4"),
            "A": Path(f"A-{index}.mp4"),
            "B": Path(f"B-{index}.mp4"),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(study_app, "list_stimuli", lambda: stimuli)
    monkeypatch.setattr(study_app, "STIMULI_DIR", tmp_path)
    (tmp_path / "fixed_video_ids.json").write_text(
        json.dumps(["preset-0", "preset-1", "preset-2"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        study_app, "_new_assignment", lambda video_id, tester_id: "mean"
    )

    state = study_app.start_study(True, None)[0]

    assert len(state["videos"]) == 3
    assert {video["video_id"] for video in state["videos"]} == {
        "preset-0",
        "preset-1",
        "preset-2",
    }


def test_participant_can_choose_the_first_video(monkeypatch, tmp_path):
    stimuli = [
        {
            "video_id": f"preset-{index}",
            "source": Path(f"source-{index}.mp4"),
            "A": Path(f"A-{index}.mp4"),
            "B": Path(f"B-{index}.mp4"),
        }
        for index in range(3)
    ]
    monkeypatch.setattr(study_app, "list_stimuli", lambda: stimuli)
    monkeypatch.setattr(study_app, "STIMULI_DIR", tmp_path)
    (tmp_path / "fixed_video_ids.json").write_text(
        json.dumps(["preset-0", "preset-1", "preset-2"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        study_app, "_new_assignment", lambda video_id, tester_id: "mean"
    )

    state = study_app.start_study(True, None, "Video 2")[0]

    assert state["videos"][0]["video_id"] == "preset-1"
    assert state["preferred_first_video"] == "Video 2"
    assert len({video["video_id"] for video in state["videos"]}) == 3


def test_source_can_be_skipped_and_skip_is_recorded(monkeypatch):
    videos = [
        {
            "video_id": "video-1",
            "source": Path("source.mp4"),
            "A": Path("A.mp4"),
            "B": Path("B.mp4"),
        }
    ]
    state = {
        "tester_id": "tester-uuid",
        "videos": videos,
        "index": 0,
        "assignments": {},
        "age_range": "",
    }
    monkeypatch.setattr(
        study_app, "_new_assignment", lambda video_id, tester_id: "mean"
    )

    outputs = study_app.skip_source_video(state, "No")
    assert len(outputs) == 6
    assert outputs[2]["visible"] is True
    assert state["source_skipped"] is True
    assert state["current_seen_before"] == "No"


def test_watch_gate_still_requires_watching_or_explicit_skip():
    state = {
        "videos": [
            {
                "video_id": "video-1",
                "source": Path("source.mp4"),
                "A": Path("A.mp4"),
                "B": Path("B.mp4"),
            }
        ],
        "index": 0,
        "assignments": {"video-1": "mean"},
    }
    with pytest.raises(gr.Error, match="80%|Skip this video"):
        study_app.continue_to_summaries(state, "", "No")


def test_preset_urls_are_the_three_links_requested():
    assert prepare_study.FIXED_VIDEO_URLS == (
        "https://youtu.be/rz1B4B2yBZY",
        "https://youtu.be/ClpGjUQ7prU",
        "https://youtu.be/E85MXIbqfAc",
    )


def test_custom_response_database_exports_csv_beside_database(tmp_path):
    database_path = tmp_path / "persistent" / "responses.sqlite"

    exported_path = export_responses(database_path)

    assert exported_path == database_path.with_name("responses.csv")
    assert exported_path.is_file()


def test_optional_other_video_can_be_skipped():
    state = {"videos": [], "index": 0}
    outputs = study_app.add_optional_video(state, "No", None)
    assert len(outputs) == 12
    assert outputs[0] is state


def test_optional_other_video_is_processed_without_saving_its_url(
    monkeypatch, tmp_path
):
    source = tmp_path / "downloaded.mp4"
    source.touch()
    added = {
        "video_id": "hashed-added-video",
        "source": tmp_path / "hashed-added-video" / "source.mp4",
        "A": tmp_path / "hashed-added-video" / "A.mp4",
        "B": tmp_path / "hashed-added-video" / "B.mp4",
    }
    state = {
        "tester_id": "tester-uuid",
        "videos": [
            {"video_id": f"fixed-{index}"} for index in range(3)
        ],
        "index": 3,
        "assignments": {},
        "age_range": "",
    }
    received_urls = []
    monkeypatch.setattr(
        prepare_study,
        "download_study_url",
        lambda url, destination, progress=None: (
            received_urls.append(url) or source
        ),
    )
    monkeypatch.setattr(study_app, "video_metadata", lambda path: (60, 30, 60))
    monkeypatch.setattr(study_app, "STIMULI_DIR", tmp_path)
    monkeypatch.setattr(
        prepare_study,
        "prepare_video",
        lambda path, destination, progress=None: {
            "video_id": "hashed-added-video",
            "pacing_length_confound_over_10_percent": False,
        },
    )
    monkeypatch.setattr(
        prepare_study,
        "append_confound_report",
        lambda row, output_dir=tmp_path: None,
    )
    monkeypatch.setattr(study_app, "list_stimuli", lambda: [added])
    monkeypatch.setattr(
        study_app, "_new_assignment", lambda video_id, tester_id: "mean"
    )

    outputs = study_app.add_optional_video(
        state, "Yes", "https://youtu.be/example"
    )

    assert len(outputs) == 12
    assert received_urls == ["https://youtu.be/example"]
    assert len(state["videos"]) == 4
    assert state["videos"][-1] == added
    assert "url" not in state


def test_tester_copy_and_stimulus_filenames_are_blinded():
    forbidden = ("mean", "barycenter")
    filenames = ("source.mp4", "A.mp4", "B.mp4")
    for visible_string in (*TESTER_UI_TEXT, *filenames):
        assert not any(model in visible_string.casefold() for model in forbidden)
    assert "Skip this video and continue" in TESTER_UI_TEXT
    assert any(
        value.startswith("Which video would you like to start with?")
        for value in TESTER_UI_TEXT
    )


def test_derived_chosen_model_maps_letters_and_ties():
    assert derive_chosen_model("A", "mean") == "mean"
    assert derive_chosen_model("B", "mean") == "barycenter"
    assert derive_chosen_model("A", "barycenter") == "barycenter"
    assert derive_chosen_model("B", "barycenter") == "mean"
    assert derive_chosen_model("none", "barycenter") is None


def test_per_tester_assignment_swaps_the_actual_displayed_clips():
    stimulus = {"A": "internal-mean.mp4", "B": "internal-barycenter.mp4"}
    assert displayed_summary_paths(stimulus, "mean") == (
        Path("internal-mean.mp4"),
        Path("internal-barycenter.mp4"),
    )
    assert displayed_summary_paths(stimulus, "barycenter") == (
        Path("internal-barycenter.mp4"),
        Path("internal-mean.mp4"),
    )


def test_submission_logs_mapping_and_returns_all_ui_updates(tmp_path, monkeypatch):
    database_path = tmp_path / "responses.sqlite"
    monkeypatch.setattr(
        study_app,
        "connect_database",
        lambda: connect_database(database_path),
    )
    state = {
        "tester_id": "tester-uuid",
        "videos": [{"video_id": "video-1"}],
        "index": 0,
        "assignments": {"video-1": "barycenter"},
        "age_range": "",
        "current_seen_before": "No",
        "source_skipped": True,
        "preferred_first_video": "Video 2",
    }
    outputs = study_app.submit_answer(
        state,
        "A",
        ["Mood"],
        "",
        5,
        4,
        "2",
        "1",
        "B",
        "Yes",
    )
    assert len(outputs) == 17
    with closing(sqlite3.connect(database_path)) as connection:
        stored = connection.execute(
            "SELECT which_model_is_A, choice, chosen_model, seen_before, "
            "source_skipped, video_order, first_video_selection "
            "FROM responses"
        ).fetchone()
    assert stored == (
        "barycenter",
        "A",
        "barycenter",
        "No",
        1,
        1,
        "Video 2",
    )


def test_completed_optional_video_goes_to_thank_you_not_offer_again(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        study_app,
        "connect_database",
        lambda: connect_database(tmp_path / "responses.sqlite"),
    )
    state = {
        "tester_id": "tester-uuid",
        "videos": [{"video_id": "video-1"}],
        "index": 0,
        "assignments": {"video-1": "mean"},
        "age_range": "",
        "additional_video_added": True,
    }
    outputs = study_app.submit_answer(
        state, "B", [], "", 3, 4, "1", "1", "A", "No"
    )
    assert outputs[3]["visible"] is True
    assert outputs[4]["visible"] is False


def test_analysis_detects_known_synthetic_preference():
    records = []
    for index in range(20):
        barycenter_is_a = index % 2 == 0
        barycenter_wins = index < 18
        if barycenter_is_a:
            choice = "A" if barycenter_wins else "B"
            likert_a, likert_b = (5, 3) if barycenter_wins else (3, 5)
        else:
            choice = "B" if barycenter_wins else "A"
            likert_a, likert_b = (3, 5) if barycenter_wins else (5, 3)
        records.append(
            {
                "tester_id": f"tester-{index % 10}",
                "video_id": f"video-{index % 4}",
                "which_model_is_A": "barycenter" if barycenter_is_a else "mean",
                "choice": choice,
                "tags": "[]",
                "likert_A": likert_a,
                "likert_B": likert_b,
                "first_watched": "A" if index % 2 == 0 else "B",
            }
        )
    results = analyze_records(records, bootstrap_replicates=100)
    assert results["barycenter_wins"] == 18
    assert results["mean_wins"] == 2
    assert results["barycenter_win_rate"] == 0.9
    assert results["primary_p_value"] < 0.01
    assert results["n_testers"] == 10


def test_analysis_cli_outputs_report_and_figures(tmp_path):
    database_path = tmp_path / "responses.sqlite"
    connection = connect_database(database_path)
    for index in range(8):
        model_is_a = "mean" if index % 2 == 0 else "barycenter"
        choice = (
            ("B" if model_is_a == "mean" else "A")
            if index < 7
            else ("A" if model_is_a == "mean" else "B")
        )
        record_response(
            connection,
            {
                "tester_id": f"tester-{index}",
                "video_id": f"video-{index % 3}",
                "which_model_is_A": model_is_a,
                "choice": choice,
                "tags": ["Mood"],
                "free_text": "",
                "likert_A": 4,
                "likert_B": 3,
                "played_A_count": 1,
                "played_B_count": 1,
                "first_watched": "A",
                "seen_before": "",
                "age_range": "",
            },
        )
    connection.close()

    output_dir = tmp_path / "report"
    results = analyze(
        database_path,
        output_dir,
        confound_path=tmp_path / "missing-confounds.csv",
    )
    assert results["barycenter_win_rate"] == 7 / 8
    assert (output_dir / "analysis_report.md").is_file()
    assert "Source video skipped: 0 of 8 responses." in (
        output_dir / "analysis_report.md"
    ).read_text(encoding="utf-8")
    assert (output_dir / "per_video_forest.png").is_file()
    assert (output_dir / "likert_differences.png").is_file()
