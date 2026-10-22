"""Storage and randomization primitives for the blinded preference study."""

from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
STIMULI_DIR = Path(
    os.environ.get("STUDY_STIMULI_DIR", STUDY_DIR / "stimuli")
)
DATABASE_PATH = Path(
    os.environ.get("STUDY_DATABASE_PATH", STUDY_DIR / "responses.sqlite")
)
RESPONSE_EXPORT_PATH = Path(
    os.environ.get(
        "STUDY_EXPORT_PATH", DATABASE_PATH.with_name("responses.csv")
    )
)
RESPONSE_FIELDS = (
    "tester_id",
    "video_id",
    "timestamp",
    "which_model_is_A",
    "choice",
    "chosen_model",
    "tags",
    "free_text",
    "likert_A",
    "likert_B",
    "played_A_count",
    "played_B_count",
    "first_watched",
    "seen_before",
    "age_range",
    "source_skipped",
    "video_order",
    "first_video_selection",
)


def connect_database(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            tester_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            which_model_is_A TEXT NOT NULL
                CHECK (which_model_is_A IN ('mean', 'barycenter')),
            choice TEXT NOT NULL CHECK (choice IN ('A', 'B', 'none')),
            chosen_model TEXT CHECK (
                chosen_model IN ('mean', 'barycenter') OR chosen_model IS NULL
            ),
            tags TEXT NOT NULL,
            free_text TEXT NOT NULL,
            likert_A INTEGER NOT NULL CHECK (likert_A BETWEEN 1 AND 5),
            likert_B INTEGER NOT NULL CHECK (likert_B BETWEEN 1 AND 5),
            played_A_count INTEGER NOT NULL CHECK (played_A_count >= 0),
            played_B_count INTEGER NOT NULL CHECK (played_B_count >= 0),
            first_watched TEXT NOT NULL CHECK (
                first_watched IN ('A', 'B', 'none')
            ),
            seen_before TEXT NOT NULL,
            age_range TEXT NOT NULL,
            source_skipped INTEGER NOT NULL DEFAULT 0
                CHECK (source_skipped IN (0, 1)),
            video_order INTEGER NOT NULL DEFAULT 1
                CHECK (video_order >= 1),
            first_video_selection TEXT NOT NULL
                DEFAULT 'Random order (recommended)',
            PRIMARY KEY (tester_id, video_id)
        )
        """
    )
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(responses)")
    }
    if "source_skipped" not in columns:
        connection.execute(
            "ALTER TABLE responses ADD COLUMN source_skipped INTEGER NOT NULL DEFAULT 0"
        )
    if "video_order" not in columns:
        connection.execute(
            "ALTER TABLE responses ADD COLUMN video_order INTEGER NOT NULL DEFAULT 1"
        )
    if "first_video_selection" not in columns:
        connection.execute(
            "ALTER TABLE responses ADD COLUMN first_video_selection TEXT NOT NULL "
            "DEFAULT 'Random order (recommended)'"
        )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS responses_no_update
        BEFORE UPDATE ON responses
        BEGIN
            SELECT RAISE(ABORT, 'study responses are append-only');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS responses_no_delete
        BEFORE DELETE ON responses
        BEGIN
            SELECT RAISE(ABORT, 'study responses are append-only');
        END
        """
    )
    connection.commit()
    return connection


def derive_chosen_model(choice: str, which_model_is_A: str) -> str | None:
    if which_model_is_A not in ("mean", "barycenter"):
        raise ValueError("which_model_is_A must be 'mean' or 'barycenter'.")
    if choice == "none":
        return None
    if choice not in ("A", "B"):
        raise ValueError("choice must be 'A', 'B', or 'none'.")
    if choice == "A":
        return which_model_is_A
    return "barycenter" if which_model_is_A == "mean" else "mean"


def balanced_model_for_A(
    assigned_counts: Mapping[str, int],
    rng: Any = random,
) -> str:
    """Return the less-used model for A, breaking ties randomly."""
    mean_count = assigned_counts.get("mean", 0)
    barycenter_count = assigned_counts.get("barycenter", 0)
    if mean_count < barycenter_count:
        return "mean"
    if barycenter_count < mean_count:
        return "barycenter"
    return rng.choice(("mean", "barycenter"))


def displayed_summary_paths(
    stimulus: Mapping[str, Any],
    which_model_is_A: str,
) -> tuple[Path, Path]:
    if which_model_is_A == "mean":
        return Path(stimulus["A"]), Path(stimulus["B"])
    if which_model_is_A == "barycenter":
        return Path(stimulus["B"]), Path(stimulus["A"])
    raise ValueError("which_model_is_A must be 'mean' or 'barycenter'.")


def assignment_counts(
    connection: sqlite3.Connection,
    video_id: str,
) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT which_model_is_A, COUNT(*) AS count
        FROM responses
        WHERE video_id = ?
        GROUP BY which_model_is_A
        """,
        (video_id,),
    )
    counts = {"mean": 0, "barycenter": 0}
    for row in rows:
        counts[row["which_model_is_A"]] = row["count"]
    return counts


def record_response(
    connection: sqlite3.Connection,
    response: Mapping[str, Any],
) -> str | None:
    choice = response["choice"]
    model_is_a = response["which_model_is_A"]
    chosen_model = derive_chosen_model(choice, model_is_a)
    values = {
        **response,
        "timestamp": response.get("timestamp")
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chosen_model": chosen_model,
        "tags": json.dumps(list(response.get("tags", ())), ensure_ascii=True),
        "source_skipped": int(bool(response.get("source_skipped", False))),
        "video_order": int(response.get("video_order", 1)),
        "first_video_selection": response.get(
            "first_video_selection", "Random order (recommended)"
        ),
    }
    connection.execute(
        """
        INSERT INTO responses (
            tester_id, video_id, timestamp, which_model_is_A, choice,
            chosen_model, tags, free_text, likert_A, likert_B,
            played_A_count, played_B_count, first_watched, seen_before,
            age_range, source_skipped, video_order, first_video_selection
        ) VALUES (
            :tester_id, :video_id, :timestamp, :which_model_is_A, :choice,
            :chosen_model, :tags, :free_text, :likert_A, :likert_B,
            :played_A_count, :played_B_count, :first_watched, :seen_before,
            :age_range, :source_skipped, :video_order, :first_video_selection
        )
        """,
        values,
    )
    connection.commit()
    return chosen_model


def list_stimuli(stimuli_dir: Path = STIMULI_DIR) -> list[dict[str, Any]]:
    stimuli = []
    for source in sorted(stimuli_dir.glob("*/source.mp4")):
        video_id = source.parent.name
        path_a = source.parent / "A.mp4"
        path_b = source.parent / "B.mp4"
        if path_a.is_file() and path_b.is_file():
            stimuli.append(
                {
                    "video_id": video_id,
                    "source": source,
                    "A": path_a,
                    "B": path_b,
                }
            )
    return stimuli


def export_responses(
    database_path: Path | None = None,
    csv_path: Path | None = None,
) -> Path:
    selected_database = database_path or DATABASE_PATH
    destination = csv_path or (
        RESPONSE_EXPORT_PATH
        if selected_database == DATABASE_PATH
        else selected_database.with_name("responses.csv")
    )
    with closing(connect_database(selected_database)) as connection:
        rows = connection.execute(
            "SELECT * FROM responses ORDER BY timestamp, tester_id, video_id"
        ).fetchall()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESPONSE_FIELDS)
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["tags"] = ";".join(json.loads(value["tags"]))
            writer.writerow(value)
    return destination
