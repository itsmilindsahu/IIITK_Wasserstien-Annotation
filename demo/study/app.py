"""Separate Gradio application for blinded A/B preference collection."""

from __future__ import annotations

import json
import os
import random
import secrets
import threading
import tempfile
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

import gradio as gr

if __package__ and "." in __package__:
    from ..pipeline import MAX_DURATION_SECONDS, video_metadata
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline import MAX_DURATION_SECONDS, video_metadata
from .core import (
    STIMULI_DIR,
    assignment_counts,
    balanced_model_for_A,
    connect_database,
    displayed_summary_paths,
    export_responses,
    list_stimuli,
    record_response,
)

TESTER_UI_TEXT = (
    "Blind Study",
    "This anonymous study compares ways of summarizing video. "
    "We do not request personal details. Please do not include names, contact "
    "details, or other identifying information in optional comments. "
    "Participation is voluntary; you can stop at any time.",
    "Age range (optional)",
    "I consent to participate",
    "Start",
    "Watch the original first if you can. You may continue after watching at "
    "least 80%, or choose to skip this video.",
    "Have you seen this video before? (optional)",
    "Continue to summaries",
    "Summary A",
    "Summary B",
    "Which summary captures the feeling of the original video better? "
    "What moments or qualities made you choose it?",
    "A",
    "B",
    "No real difference",
    "Reasons (optional)",
    "Mood",
    "Story continuity",
    "Important moments",
    "Pacing",
    "Visual quality",
    "Other",
    "How well does this represent the original? (Summary A)",
    "How well does this represent the original? (Summary B)",
    "Optional comment",
    "Submit answer",
    "Thank you for participating.",
    "Stop study",
    "Skip this video and continue",
    "Which video would you like to start with? (optional)",
    "Random order (recommended)",
    "Video numbers follow the order of the three preset study links.",
)

_ASSIGNMENT_LOCK = threading.Lock()
_PENDING_ASSIGNMENTS: dict[str, dict[str, tuple[str, float]]] = {}
_ASSIGNMENT_TTL_SECONDS = 30 * 60


def _new_assignment(video_id: str, tester_id: str) -> str:
    with _ASSIGNMENT_LOCK, closing(connect_database()) as connection:
        counts = assignment_counts(connection, video_id)
        now = time.monotonic()
        pending = _PENDING_ASSIGNMENTS.setdefault(video_id, {})
        expired = [
            reserved_tester
            for reserved_tester, (_, expires_at) in pending.items()
            if expires_at <= now
        ]
        for reserved_tester in expired:
            del pending[reserved_tester]
        effective_counts = {
            model: counts[model]
            + sum(
                reserved_model == model
                for reserved_model, _ in pending.values()
            )
            for model in ("mean", "barycenter")
        }
        model_is_a = balanced_model_for_A(effective_counts, secrets.SystemRandom())
        pending[tester_id] = (
            model_is_a,
            now + _ASSIGNMENT_TTL_SECONDS,
        )
        return model_is_a


def _complete_assignment(video_id: str, tester_id: str) -> None:
    with _ASSIGNMENT_LOCK:
        pending = _PENDING_ASSIGNMENTS.get(video_id)
        if pending is not None:
            pending.pop(tester_id, None)
            if not pending:
                _PENDING_ASSIGNMENTS.pop(video_id, None)


def _progress(state: dict[str, Any]) -> str:
    return f"**STUDY PROGRESS** &nbsp; · &nbsp; Video {state['index'] + 1} of {len(state['videos'])}"


def _video_page(
    state: dict[str, Any],
    *,
    show_source: bool,
) -> tuple[Any, ...]:
    stimulus = state["videos"][state["index"]]
    if show_source:
        state["source_skipped"] = False
    if stimulus["video_id"] not in state["assignments"]:
        state["assignments"][stimulus["video_id"]] = _new_assignment(
            stimulus["video_id"], state["tester_id"]
        )
    source_group = gr.update(visible=show_source)
    compare_group = gr.update(visible=not show_source)
    if show_source:
        return (
            state,
            source_group,
            compare_group,
            _progress(state),
            str(stimulus["source"]),
            None,
            None,
        )
    summary_a_path, summary_b_path = displayed_summary_paths(
        stimulus, state["assignments"][stimulus["video_id"]]
    )
    return (
        state,
        source_group,
        compare_group,
        _progress(state),
        str(stimulus["source"]),
        str(summary_a_path),
        str(summary_b_path),
    )


def start_study(
    consent: bool,
    age_range: str | None,
    first_video: str | None = None,
) -> tuple[Any, ...]:
    if not consent:
        raise gr.Error("Please confirm consent before starting.")
    rng = random.SystemRandom()
    selected_first = first_video or TESTER_UI_TEXT[29]
    preferred_index: int | None = None
    if selected_first != TESTER_UI_TEXT[29]:
        if not selected_first.startswith("Video "):
            raise gr.Error("Choose a listed video or random order.")
        try:
            preferred_index = int(selected_first.removeprefix("Video ")) - 1
        except ValueError as exc:
            raise gr.Error("Choose a listed video or random order.") from exc
        if preferred_index < 0:
            raise gr.Error("Choose a listed video or random order.")
    stimuli = list_stimuli()
    manifest_path = STIMULI_DIR / "fixed_video_ids.json"
    if manifest_path.is_file():
        fixed_ids = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_id = {item["video_id"]: item for item in stimuli}
        if len(fixed_ids) != 3 or any(item not in by_id for item in fixed_ids):
            raise gr.Error(
                "The three prepared study videos are incomplete. Ask the "
                "administrator to run the preparation script again."
            )
        videos = [by_id[item] for item in fixed_ids]
    elif len(stimuli) >= 3:
        if preferred_index is not None and preferred_index >= min(5, len(stimuli)):
            raise gr.Error("Choose a video available in this study session.")
        min_count = max(3, (preferred_index or 0) + 1)
        count = rng.randint(min_count, min(5, len(stimuli)))
        videos = rng.sample(stimuli, count)
    else:
        raise gr.Error(
            "The study videos are not prepared. Ask the administrator to "
            "run demo\\study\\prepare_study.py first."
        )
    if selected_first == TESTER_UI_TEXT[29]:
        rng.shuffle(videos)
    elif preferred_index is not None:
        if preferred_index >= len(videos):
            raise gr.Error("Choose a video available in this study session.")
        preferred_video = videos.pop(preferred_index)
        rng.shuffle(videos)
        videos.insert(0, preferred_video)
    else:
        raise gr.Error("Choose a listed video or random order.")
    state = {
        "tester_id": str(uuid.uuid4()),
        "videos": videos,
        "index": 0,
        "assignments": {},
        "age_range": age_range or "",
        "additional_video_added": False,
        "preferred_first_video": selected_first,
    }
    page = _video_page(state, show_source=True)
    return (
        page[0],
        gr.update(visible=False),
        page[1],
        page[2],
        gr.update(visible=False),
        gr.update(visible=False),
        page[3],
        gr.update(visible=True),
        page[4],
        "",
        None,
        None,
        None,
    )


def continue_to_summaries(
    state: dict[str, Any],
    source_watched: str | None,
    seen_before: str | None,
) -> tuple[Any, ...]:
    if source_watched != "yes":
        raise gr.Error("Watch at least 80% or choose “Skip this video and continue.”")
    state["current_seen_before"] = seen_before or ""
    state["source_skipped"] = False
    page = _video_page(state, show_source=False)
    return (
        page[0],
        page[1],
        page[2],
        page[3],
        page[5],
        page[6],
    )


def skip_source_video(
    state: dict[str, Any],
    seen_before: str | None,
) -> tuple[Any, ...]:
    if state is None or state["index"] >= len(state["videos"]):
        raise gr.Error("There is no current video to skip.")
    state["current_seen_before"] = seen_before or ""
    state["source_skipped"] = True
    page = _video_page(state, show_source=False)
    return (
        page[0],
        page[1],
        page[2],
        page[3],
        page[5],
        page[6],
    )


def stop_study(state: dict[str, Any] | None) -> tuple[Any, ...]:
    if state is not None and state["index"] < len(state["videos"]):
        stimulus = state["videos"][state["index"]]
        _complete_assignment(stimulus["video_id"], state["tester_id"])
    return (
        None,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        gr.update(visible=False),
    )


def submit_answer(
    state: dict[str, Any],
    choice: str | None,
    tags: list[str] | None,
    free_text: str | None,
    likert_a: int | None,
    likert_b: int | None,
    played_a: str | None,
    played_b: str | None,
    first_watched: str | None,
    seen_before: str | None,
) -> tuple[Any, ...]:
    if choice not in ("A", "B", "No real difference"):
        raise gr.Error("Choose A, B, or No real difference.")
    if likert_a not in range(1, 6) or likert_b not in range(1, 6):
        raise gr.Error("Rate both summaries from 1 to 5.")
    stimulus = state["videos"][state["index"]]
    assignment = state["assignments"][stimulus["video_id"]]
    stored_choice = {"A": "A", "B": "B", "No real difference": "none"}[choice]
    try:
        played_a_count = max(0, int(played_a or 0))
        played_b_count = max(0, int(played_b or 0))
    except ValueError as exc:
        raise gr.Error("The playback count could not be recorded.") from exc
    first = first_watched if first_watched in ("A", "B") else "none"
    with closing(connect_database()) as connection:
        record_response(
            connection,
            {
                "tester_id": state["tester_id"],
                "video_id": stimulus["video_id"],
                "which_model_is_A": assignment,
                "choice": stored_choice,
                "tags": tags or [],
                "free_text": free_text or "",
                "likert_A": int(likert_a),
                "likert_B": int(likert_b),
                "played_A_count": played_a_count,
                "played_B_count": played_b_count,
                "first_watched": first,
                "seen_before": state.get("current_seen_before", seen_before or ""),
                "age_range": state["age_range"],
                "source_skipped": state.get("source_skipped", False),
                "video_order": state["index"] + 1,
                "first_video_selection": state.get(
                    "preferred_first_video", TESTER_UI_TEXT[29]
                ),
            },
        )
    _complete_assignment(stimulus["video_id"], state["tester_id"])
    state["index"] += 1
    if state["index"] >= len(state["videos"]):
        additional_video_added = state.get("additional_video_added", False)
        return (
            state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=additional_video_added),
            gr.update(visible=not additional_video_added),
            "",
            gr.update(visible=False),
            None,
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "",
        )
    page = _video_page(state, show_source=True)
    return (
        page[0],
        page[1],
        page[2],
        gr.update(visible=False),
        gr.update(visible=False),
        page[3],
        gr.update(visible=True),
        page[4],
        "",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "",
    )


def add_optional_video(
    state: dict[str, Any],
    wants_other: str | None,
    url: str | None,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    if wants_other != "Yes":
        return (
            state,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            None,
            "",
            None,
            None,
            None,
        )
    if not url or not url.strip():
        raise gr.Error("Enter a public YouTube link or choose No.")
    if len(state["videos"]) != state["index"]:
        raise gr.Error("Finish the current video before adding another.")

    try:
        with tempfile.TemporaryDirectory(prefix="study-extra-video-") as temp:
            if __package__ and "." in __package__:
                from .prepare_study import (
                    append_confound_report,
                    download_study_url,
                    prepare_video,
                )
            else:
                from study.prepare_study import (
                    append_confound_report,
                    download_study_url,
                    prepare_video,
                )

            downloaded = download_study_url(
                url.strip(), Path(temp), progress=progress
            )
            duration, _, _ = video_metadata(downloaded)
            if duration > MAX_DURATION_SECONDS:
                raise gr.Error("Additional videos must be 10 minutes or shorter.")

            confound = prepare_video(downloaded, STIMULI_DIR, progress=progress)
    except gr.Error:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(f"Could not prepare the additional video: {exc}") from exc

    video_id = confound["video_id"]
    if any(item["video_id"] == video_id for item in state["videos"]):
        raise gr.Error("That video is already included in this session.")
    append_confound_report(confound)
    stimulus = next(
        (item for item in list_stimuli() if item["video_id"] == video_id),
        None,
    )
    if stimulus is None:
        raise gr.Error("The additional video was prepared but is unavailable.")
    state["videos"].append(stimulus)
    state["additional_video_added"] = True
    state["current_seen_before"] = ""
    page = _video_page(state, show_source=True)
    return (
        state,
        gr.update(visible=False),
        gr.update(visible=False),
        page[1],
        page[2],
        page[3],
        gr.update(visible=True),
        page[4],
        "",
        None,
        None,
        None,
    )


def _admin_counts(
    password: str | None,
) -> tuple[str, list[dict[str, Any]], Any]:
    expected = os.environ.get("STUDY_ADMIN_PASSWORD")
    if not expected:
        raise gr.Error(
            "Set STUDY_ADMIN_PASSWORD in the environment before using the "
            "admin panel."
        )
    if not password or not secrets.compare_digest(password, expected):
        raise gr.Error("Admin password is incorrect.")
    with closing(connect_database()) as connection:
        rows = connection.execute(
            """
            SELECT video_id, COUNT(*) AS responses,
                SUM(which_model_is_A = 'mean') AS mean_A,
                SUM(which_model_is_A = 'barycenter') AS barycenter_A
            FROM responses
            GROUP BY video_id
            ORDER BY video_id
            """
        ).fetchall()
    table = [
        {
            "video_id": row["video_id"],
            "completed_responses": row["responses"],
            "mean_target_as_A": row["mean_A"],
            "barycenter_target_as_A": row["barycenter_A"],
        }
        for row in rows
    ]
    total = sum(row["completed_responses"] for row in table)
    return f"{total} completed responses", table, gr.update(visible=True)


def _export_admin(password: str | None) -> str:
    _admin_counts(password)
    return str(export_responses())


SOURCE_TRACKING_JS = r"""() => {
  const watched = new Set();
  const stats = {key: "", a: 0, b: 0, first: "none"};
  const setValue = (id, value) => {
    const field = document.querySelector(`#${id} input, #${id} textarea`);
    if (!field || String(field.value) === String(value)) return;
    field.value = value;
    field.dispatchEvent(new Event("input", {bubbles: true}));
    field.dispatchEvent(new Event("change", {bubbles: true}));
  };
  const bindSource = () => {
    const video = document.querySelector("#source-player video");
    if (!video || watched.has(video)) return;
    watched.add(video);
    video.addEventListener("timeupdate", () => {
      if (!video.duration || !Number.isFinite(video.duration)) return;
      let covered = 0;
      for (let i = 0; i < video.played.length; i++) {
        covered += video.played.end(i) - video.played.start(i);
      }
      if (covered / video.duration >= 0.8) setValue("source-watched", "yes");
    });
    video.preload = "auto";
  };
  const bindSummary = (id, side) => {
    const video = document.querySelector(`#${id} video`);
    if (!video || watched.has(video)) return;
    watched.add(video);
    video.preload = "auto";
    video.addEventListener("play", () => {
      if (stats.first === "none") stats.first = side;
      if (side === "A") stats.a += 1; else stats.b += 1;
      setValue("played-a", stats.a);
      setValue("played-b", stats.b);
      setValue("first-watched", stats.first);
    });
  };
  const refresh = () => {
    bindSource();
    bindSummary("summary-a", "A");
    bindSummary("summary-b", "B");
    const a = document.querySelector("#summary-a video");
    const b = document.querySelector("#summary-b video");
    const key = `${a ? a.currentSrc : ""}|${b ? b.currentSrc : ""}`;
    if (a && b && key !== stats.key) {
      stats.key = key;
      stats.a = 0; stats.b = 0; stats.first = "none";
      setValue("played-a", 0); setValue("played-b", 0);
      setValue("first-watched", "none");
      setValue("source-watched", "");
    }
  };
  new MutationObserver(refresh).observe(document.body, {childList: true, subtree: true});
  window.setInterval(refresh, 700);
  refresh();
}"""


def build_study_app() -> gr.Blocks:
    manifest_path = STIMULI_DIR / "fixed_video_ids.json"
    if manifest_path.is_file():
        video_count = len(json.loads(manifest_path.read_text(encoding="utf-8")))
    else:
        video_count = min(5, len(list_stimuli()))
    if video_count < 3:
        video_count = 3
    first_video_choices = [
        TESTER_UI_TEXT[29],
        *(f"Video {index}" for index in range(1, video_count + 1)),
    ]

    with gr.Blocks(title="Blind Study", css="""
        :root { color-scheme: light; }
        body { background: #f2f5f1; color: #172b27; font-size: 1.04rem; }
        body.dark {
          background: #f2f5f1 !important; color: #172b27 !important;
          --body-background-fill: #f2f5f1;
          --body-text-color: #172b27;
          --block-background-fill: #fff;
          --block-label-text-color: #294a40;
        }
        body.dark main, body.dark .gradio-container {
          background: #f2f5f1 !important; color: #172b27 !important;
        }
        .gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
        #study-hero {
          color: #172b27 !important; margin: 1.5rem 0 1.2rem;
        }
        #study-hero .study-kicker {
          color: #486960; font-size: .78rem; font-weight: 750;
          letter-spacing: .14em; text-transform: uppercase;
        }
        #study-hero h1 {
          color: #183b33; font-family: Georgia, 'Times New Roman', serif;
          font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.1;
          margin: .45rem 0;
        }
        #study-hero .study-subtitle {
          color: #52645f; font-size: 1.08rem; max-width: 720px;
          line-height: 1.65;
        }
        .study-card {
          background: #fff; border: 1px solid #dce5df; border-radius: 16px;
          box-shadow: 0 8px 28px rgba(24, 59, 51, .06);
          margin: .8rem 0 1.25rem; padding: clamp(1rem, 3vw, 1.8rem);
        }
        .study-card .styler {
          background: #fff !important; color: #172b27 !important;
        }
        .study-card .styler .block {
          background: #fff !important; color: #172b27 !important;
        }
        .study-card .styler .wrap {
          background: #f7faf7 !important; color: #172b27 !important;
        }
        .study-card .styler input, .study-card .styler textarea,
        .study-card .styler [role="listbox"] {
          background: transparent !important; color: #172b27 !important;
        }
        .study-card .styler :is(p, label, span, legend, h2, h3, strong) {
          color: #172b27 !important;
        }
        .study-card .float {
          background: #f8faf8 !important; color: #294a40 !important;
        }
        .study-card h2, .study-card h3 {
          color: #183b33; font-family: Georgia, 'Times New Roman', serif;
        }
        button { min-height: 48px; font-size: 1rem !important; }
        button.primary {
          background: #1d5b4d !important; border-color: #1d5b4d !important;
        }
        button.secondary {
          background: #eef3ef !important; border-color: #cbd8cf !important;
          color: #234b40 !important;
        }
        button:focus-visible, input:focus-visible, textarea:focus-visible,
        [role="radio"]:focus-visible, [role="checkbox"]:focus-visible {
          outline: 3px solid #d29235 !important; outline-offset: 3px;
        }
        #study-progress {
          background: #e6eee8; border-radius: 999px; color: #234b40;
          font-size: .95rem; margin: .6rem 0 1rem; padding: .45rem 1rem;
        }
        #summary-a video, #summary-b video {
          width: 100%; aspect-ratio: 16/9; background: #101a17;
          border-radius: 10px; object-fit: contain;
        }
        #summary-row { gap: 1rem !important; }
        #summary-a, #summary-b {
          background: #f8faf8; border: 1px solid #e0e8e2;
          border-radius: 12px; padding: .7rem;
        }
        #answer-choice .wrap {
          display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .65rem;
        }
        #answer-choice label, #reason-tags label, #likert-a label, #likert-b label {
          align-items: center; background: #f7faf7; border: 1px solid #d6e1da;
          border-radius: 10px; cursor: pointer; min-height: 48px;
          padding: .65rem .8rem; transition: background .15s, border-color .15s;
        }
        #answer-choice label:hover, #reason-tags label:hover,
        #likert-a label:hover, #likert-b label:hover {
          background: #edf4ef; border-color: #729388;
        }
        #reason-tags .wrap {
          display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .45rem;
        }
        #likert-a .wrap, #likert-b .wrap {
          display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: .4rem;
        }
        #source-skip button {
          background: #fff !important; border: 1px solid #83958e !important;
          color: #294a40 !important;
        }
        #source-watched, #played-a, #played-b, #first-watched {
          display: none !important;
        }
        @media (max-width: 700px) {
          .gradio-container { padding: 0 .7rem !important; }
          #summary-row { flex-direction: column !important; }
          #summary-a, #summary-b { width: 100%; min-width: 0 !important; }
          #answer-choice .wrap { grid-template-columns: 1fr; }
          #reason-tags .wrap { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
    """) as app:
        state = gr.State()
        gr.Markdown(
            "<div class='study-kicker'>Research participation · anonymous</div>"
            "<h1>Video summary study</h1>"
            "<div class='study-subtitle'>Help us understand how people "
            "experience short video summaries. Your responses are anonymous "
            "and there are no right or wrong answers.</div>",
            elem_id="study-hero",
        )
        with gr.Group(visible=True, elem_classes=["study-card"]) as consent_group:
            gr.Markdown("## Before you begin")
            gr.Markdown(TESTER_UI_TEXT[1])
            first_video = gr.Dropdown(
                choices=first_video_choices,
                label=TESTER_UI_TEXT[28],
                value=TESTER_UI_TEXT[29],
            )
            gr.Markdown(TESTER_UI_TEXT[30])
            age_range = gr.Dropdown(
                choices=[
                    "Under 18",
                    "18–24",
                    "25–34",
                    "35–44",
                    "45–54",
                    "55+",
                    "Prefer not to say",
                ],
                label=TESTER_UI_TEXT[2],
                value=None,
            )
            consent = gr.Checkbox(label=TESTER_UI_TEXT[3], value=False)
            start = gr.Button(TESTER_UI_TEXT[4], variant="primary")
        progress = gr.Markdown(elem_id="study-progress", visible=False)
        stop_button = gr.Button(TESTER_UI_TEXT[26], visible=False)
        with gr.Group(visible=False, elem_classes=["study-card"]) as source_group:
            gr.Markdown("## 01 · Watch the original")
            gr.Markdown(TESTER_UI_TEXT[5])
            source_video = gr.Video(
                label="Original video", elem_id="source-player", height=300
            )
            seen_before = gr.Radio(
                choices=["Yes", "No"],
                label=TESTER_UI_TEXT[6],
                value=None,
            )
            source_watched = gr.Textbox(
                value="", elem_id="source-watched"
            )
            gr.Markdown(
                "Choose **Yes** or **No** if you remember seeing this video. "
                "You may leave it unanswered."
            )
            with gr.Row():
                continue_button = gr.Button(
                    TESTER_UI_TEXT[7], variant="primary"
                )
                skip_button = gr.Button(
                    TESTER_UI_TEXT[27],
                    elem_id="source-skip",
                )
        with gr.Group(visible=False, elem_classes=["study-card"]) as compare_group:
            gr.Markdown("## 02 · Compare the summaries")
            with gr.Row(equal_height=True, elem_id="summary-row"):
                summary_a = gr.Video(
                    label=TESTER_UI_TEXT[8], elem_id="summary-a", height=300
                )
                summary_b = gr.Video(
                    label=TESTER_UI_TEXT[9], elem_id="summary-b", height=300
                )
            gr.Markdown(f"## {TESTER_UI_TEXT[10]}")
            gr.Markdown("Choose the response that best reflects your view.")
            choice = gr.Radio(
                choices=[TESTER_UI_TEXT[11], TESTER_UI_TEXT[12], TESTER_UI_TEXT[13]],
                label="Your choice",
                value=None,
                elem_id="answer-choice",
            )
            tags = gr.CheckboxGroup(
                choices=list(TESTER_UI_TEXT[15:21]),
                label=f"{TESTER_UI_TEXT[14]} — select all that apply",
                elem_id="reason-tags",
            )
            gr.Markdown(
                "Rate each summary from **1 · Not at all** to "
                "**5 · Very well**."
            )
            likert_a = gr.Radio(
                choices=[1, 2, 3, 4, 5],
                label=TESTER_UI_TEXT[21],
                value=None,
                elem_id="likert-a",
            )
            likert_b = gr.Radio(
                choices=[1, 2, 3, 4, 5],
                label=TESTER_UI_TEXT[22],
                value=None,
                elem_id="likert-b",
            )
            free_text = gr.Textbox(
                label=TESTER_UI_TEXT[23],
                lines=2,
                max_lines=4,
            )
            played_a = gr.Textbox(value="0", elem_id="played-a")
            played_b = gr.Textbox(value="0", elem_id="played-b")
            first_watched = gr.Textbox(
                value="none", elem_id="first-watched"
            )
            submit = gr.Button(TESTER_UI_TEXT[24], variant="primary")
        with gr.Group(visible=False, elem_classes=["study-card"]) as thanks_group:
            gr.Markdown(f"## {TESTER_UI_TEXT[25]}")
        with gr.Group(visible=False, elem_classes=["study-card"]) as extra_video_group:
            gr.Markdown(
                "Would you like to compare one additional YouTube video? "
                "It must be public and 10 minutes or shorter. If submitted, "
                "the video is downloaded to generate summaries and saved "
                "with the study stimuli; the link itself is not retained."
            )
            wants_other = gr.Radio(
                choices=["Yes", "No"],
                label="Add one other video? (optional)",
                value=None,
            )
            other_video_url = gr.Textbox(
                label="Public YouTube link (optional)",
                placeholder="https://youtu.be/...",
            )
            add_video_button = gr.Button("Continue", variant="primary")

        start.click(
            start_study,
            inputs=[consent, age_range, first_video],
            outputs=[
                state,
                consent_group,
                source_group,
                compare_group,
                thanks_group,
                extra_video_group,
                progress,
                stop_button,
                source_video,
                source_watched,
                seen_before,
                summary_a,
                summary_b,
            ],
        )
        continue_button.click(
            continue_to_summaries,
            inputs=[state, source_watched, seen_before],
            outputs=[
                state,
                source_group,
                compare_group,
                progress,
                summary_a,
                summary_b,
            ],
        )
        skip_button.click(
            skip_source_video,
            inputs=[state, seen_before],
            outputs=[
                state,
                source_group,
                compare_group,
                progress,
                summary_a,
                summary_b,
            ],
        )
        stop_button.click(
            stop_study,
            inputs=[state],
            outputs=[
                state,
                source_group,
                compare_group,
                thanks_group,
                extra_video_group,
                progress,
                stop_button,
            ],
        )
        add_video_button.click(
            add_optional_video,
            inputs=[state, wants_other, other_video_url],
            outputs=[
                state,
                extra_video_group,
                thanks_group,
                source_group,
                compare_group,
                progress,
                stop_button,
                source_video,
                source_watched,
                seen_before,
                summary_a,
                summary_b,
            ],
        )
        submit.click(
            submit_answer,
            inputs=[
                state,
                choice,
                tags,
                free_text,
                likert_a,
                likert_b,
                played_a,
                played_b,
                first_watched,
                seen_before,
            ],
            outputs=[
                state,
                source_group,
                compare_group,
                thanks_group,
                extra_video_group,
                progress,
                stop_button,
                source_video,
                source_watched,
                seen_before,
                summary_a,
                summary_b,
                choice,
                tags,
                likert_a,
                likert_b,
                free_text,
            ],
        )

        with gr.Accordion("Admin", open=False):
            admin_password = gr.Textbox(
                label="Admin password", type="password"
            )
            login_admin = gr.Button("Admin access")
            with gr.Group(visible=False) as admin_group:
                with gr.Row():
                    refresh_admin = gr.Button("Refresh counts")
                    export_admin = gr.Button("Export responses")
                admin_count_output = gr.Textbox(label="Completed responses")
                admin_data_output = gr.JSON(
                    label="Per-video assignment balance"
                )
                export_file = gr.File(label="Responses CSV")
            login_admin.click(
                _admin_counts,
                inputs=[admin_password],
                outputs=[admin_count_output, admin_data_output, admin_group],
            )
            refresh_admin.click(
                _admin_counts,
                inputs=[admin_password],
                outputs=[admin_count_output, admin_data_output, admin_group],
            )
            export_admin.click(
                _export_admin,
                inputs=[admin_password],
                outputs=[export_file],
            )
        app.load(fn=None, js=SOURCE_TRACKING_JS)
    return app
