"""Run the fixed, full-sample analysis for the blinded preference study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

if __package__:
    from .core import DATABASE_PATH, STUDY_DIR, connect_database
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from study.core import DATABASE_PATH, STUDY_DIR, connect_database

OUTPUT_DIR = STUDY_DIR / "report"
BOOTSTRAP_REPLICATES = 5000
CONFIDENCE_Z = 1.959963984540054


def exact_binomial_p_value(wins: int, losses: int) -> float:
    count = wins + losses
    if not count:
        return 1.0
    lower_tail = sum(math.comb(count, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * lower_tail / (2**count))


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    proportion = successes / trials
    z_squared = CONFIDENCE_Z**2
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        CONFIDENCE_Z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _wilcoxon_signed_rank(differences: list[float]) -> float:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0
    order = sorted(range(len(nonzero)), key=lambda index: abs(nonzero[index]))
    doubled_ranks = [0] * len(nonzero)
    position = 0
    while position < len(order):
        end = position + 1
        magnitude = abs(nonzero[order[position]])
        while end < len(order) and abs(nonzero[order[end]]) == magnitude:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for index in order[position:end]:
            doubled_ranks[index] = round(2 * average_rank)
        position = end

    observed = sum(
        rank for value, rank in zip(nonzero, doubled_ranks) if value > 0
    )
    total = sum(doubled_ranks)
    distribution = Counter({0: 1})
    for rank in doubled_ranks:
        next_distribution = distribution.copy()
        for subtotal, frequency in distribution.items():
            next_distribution[subtotal + rank] += frequency
        distribution = next_distribution
    denominator = 2 ** len(nonzero)
    lower = sum(
        frequency
        for subtotal, frequency in distribution.items()
        if subtotal <= observed
    )
    upper = sum(
        frequency
        for subtotal, frequency in distribution.items()
        if subtotal >= observed
    )
    return min(1.0, 2 * min(lower, upper) / denominator)


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        location = (len(ordered) - 1) * fraction
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return ordered[lower]
        weight = location - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return percentile(0.025), percentile(0.975)


def _barycenter_wins(row: dict[str, Any]) -> bool | None:
    if row["choice"] == "none":
        return None
    chosen = row["which_model_is_A"] if row["choice"] == "A" else (
        "barycenter" if row["which_model_is_A"] == "mean" else "mean"
    )
    return chosen == "barycenter"


def _tags(row: dict[str, Any]) -> list[str]:
    value = row.get("tags", "[]")
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return [tag for tag in str(value).split(";") if tag]
    return decoded if isinstance(decoded, list) else []


def _cluster_bootstrap_interval(
    rows: list[dict[str, Any]],
    rng: random.Random,
    replicates: int,
) -> tuple[float, float]:
    by_tester: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        result = _barycenter_wins(row)
        if result is not None:
            by_tester[row["tester_id"]].append(result)
    tester_ids = list(by_tester)
    if not tester_ids:
        return 0.0, 1.0
    values: list[float] = []
    for _ in range(replicates):
        wins = votes = 0
        for _ in tester_ids:
            sample = by_tester[rng.choice(tester_ids)]
            wins += sum(sample)
            votes += len(sample)
        if votes:
            values.append(wins / votes)
    return _percentile_interval(values)


def _likert_difference(row: dict[str, Any]) -> int:
    if row["which_model_is_A"] == "barycenter":
        return int(row["likert_A"]) - int(row["likert_B"])
    return int(row["likert_B"]) - int(row["likert_A"])


def _load_confound_checks(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def analyze_records(
    records: list[dict[str, Any]],
    *,
    confound_checks: list[dict[str, Any]] | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    forced = [row for row in records if row["choice"] in ("A", "B")]
    ties = [row for row in records if row["choice"] == "none"]
    bary_wins = sum(_barycenter_wins(row) is True for row in forced)
    mean_wins = len(forced) - bary_wins
    forced_count = len(forced)
    tie_count = len(ties)
    primary_rate = bary_wins / forced_count if forced_count else 0.0
    sensitivity_rate = (
        (bary_wins + 0.5 * tie_count) / len(records) if records else 0.0
    )
    primary_ci = wilson_interval(bary_wins, forced_count)
    rng = random.Random(20261019)
    cluster_ci = _cluster_bootstrap_interval(
        records, rng, bootstrap_replicates
    )

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forced:
        by_video[row["video_id"]].append(row)
    per_video = {}
    for video_id in sorted({row["video_id"] for row in records}):
        rows = by_video[video_id]
        wins = sum(_barycenter_wins(row) is True for row in rows)
        losses = len(rows) - wins
        per_video[video_id] = {
            "wins": wins,
            "losses": losses,
            "rate": wins / len(rows) if rows else None,
            "ci": wilson_interval(wins, len(rows)),
        }
    leave_one_video_out = {}
    for omitted_video in sorted(per_video):
        remaining = [
            row for row in forced if row["video_id"] != omitted_video
        ]
        wins = sum(_barycenter_wins(row) is True for row in remaining)
        leave_one_video_out[omitted_video] = {
            "wins": wins,
            "votes": len(remaining),
            "rate": wins / len(remaining) if remaining else None,
        }

    differences = [_likert_difference(row) for row in records]
    nonzero = [value for value in differences if value != 0]
    likert_ci = _percentile_interval(
        [
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(bootstrap_replicates)
        ]
        if differences
        else []
    )
    likert_mean = sum(differences) / len(differences) if differences else 0.0

    side_votes = [row for row in forced]
    choice_a_wins = sum(row["choice"] == "A" for row in side_votes)
    first_watched_rates = {}
    for side in ("A", "B"):
        rows = [
            row for row in forced if row.get("first_watched", "none") == side
        ]
        wins = sum(_barycenter_wins(row) is True for row in rows)
        first_watched_rates[side] = {
            "wins": wins,
            "votes": len(rows),
            "rate": wins / len(rows) if rows else None,
        }
    tags_by_winner: dict[str, Counter[str]] = {
        "mean": Counter(),
        "barycenter": Counter(),
    }
    for row in forced:
        winner = "barycenter" if _barycenter_wins(row) else "mean"
        tags_by_winner[winner].update(_tags(row))

    checks = confound_checks or []
    flagged = [
        row
        for row in checks
        if str(row.get("pacing_length_confound_over_10_percent", "")).lower()
        in ("true", "1", "yes")
    ]
    video_order_counts = Counter(
        int(row.get("video_order", 1)) for row in records
    )
    first_video_selections = Counter(
        row.get("first_video_selection", "Random order (recommended)")
        for row in records
    )
    return {
        "n_testers": len({row["tester_id"] for row in records}),
        "n_votes": len(records),
        "n_forced_choices": forced_count,
        "no_difference_count": tie_count,
        "barycenter_wins": bary_wins,
        "mean_wins": mean_wins,
        "barycenter_win_rate": primary_rate,
        "barycenter_win_rate_ci": primary_ci,
        "primary_p_value": exact_binomial_p_value(bary_wins, mean_wins),
        "sensitivity_win_rate_with_ties_half": sensitivity_rate,
        "tester_cluster_bootstrap_ci": cluster_ci,
        "per_video": per_video,
        "leave_one_video_out": leave_one_video_out,
        "likert_differences": differences,
        "likert_mean_difference": likert_mean,
        "likert_bootstrap_ci": likert_ci,
        "likert_wilcoxon_p_value": _wilcoxon_signed_rank(differences),
        "likert_nonzero_pairs": len(nonzero),
        "A_choice_rate": (
            choice_a_wins / len(side_votes) if side_votes else 0.0
        ),
        "A_choice_count": choice_a_wins,
        "B_choice_count": len(side_votes) - choice_a_wins,
        "first_watched_rates": first_watched_rates,
        "tags_by_winner": {
            model: dict(counts)
            for model, counts in tags_by_winner.items()
        },
        "confound_checks_count": len(checks),
        "confound_flagged_count": len(flagged),
        "confound_flagged_video_ids": [
            row.get("video_id", "unknown") for row in flagged
        ],
        "source_skipped_count": sum(
            row.get("source_skipped", 0) in (True, 1, "1")
            for row in records
        ),
        "video_order_counts": dict(sorted(video_order_counts.items())),
        "first_video_selections": dict(sorted(first_video_selections.items())),
    }


def read_records(database_path: Path) -> list[dict[str, Any]]:
    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            "SELECT * FROM responses ORDER BY timestamp, tester_id, video_id"
        ).fetchall()
    return [dict(row) for row in rows]


def write_figures(results: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = [
        (video_id, values)
        for video_id, values in results["per_video"].items()
        if values["rate"] is not None
    ]
    figure, axis = plt.subplots(figsize=(7, max(2.5, 0.38 * len(videos) + 1)))
    if videos:
        for position, (video_id, values) in enumerate(videos):
            low, high = values["ci"]
            rate = values["rate"]
            axis.errorbar(
                rate,
                position,
                xerr=[[rate - low], [high - rate]],
                fmt="o",
                color="#174a7e",
                capsize=3,
            )
        axis.set_yticks(range(len(videos)), [item[0] for item in videos])
        axis.set_ylim(-0.7, len(videos) - 0.3)
    else:
        axis.text(0.5, 0.5, "No forced choices yet", ha="center", va="center")
    axis.axvline(0.5, color="#666666", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Barycenter win rate (95% Wilson CI)")
    axis.set_title("Per-video forced-choice rates")
    figure.tight_layout()
    figure.savefig(output_dir / "per_video_forest.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 3.5))
    differences = results["likert_differences"]
    if differences:
        axis.hist(differences, bins=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], color="#3976a8")
    else:
        axis.text(0.5, 0.5, "No paired ratings yet", ha="center", va="center")
    axis.axvline(0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xticks([-2, -1, 0, 1, 2])
    axis.set_xlabel("Paired Likert difference (barycenter minus mean)")
    axis.set_ylabel("Responses")
    axis.set_title("Paired ratings")
    figure.tight_layout()
    figure.savefig(output_dir / "likert_differences.png", dpi=150)
    plt.close(figure)


def write_report(results: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rate = results["barycenter_win_rate"]
    low, high = results["barycenter_win_rate_ci"]
    cluster_low, cluster_high = results["tester_cluster_bootstrap_ci"]
    likert_low, likert_high = results["likert_bootstrap_ci"]
    p_value = results["primary_p_value"]
    if results["n_forced_choices"] == 0:
        conclusion = "No forced-choice votes are available; no preference conclusion can be drawn."
    elif p_value < 0.05 and rate > 0.5:
        conclusion = (
            "The observed forced choices favor barycenter at the preregistered "
            "two-sided 0.05 threshold."
        )
    elif p_value < 0.05 and rate < 0.5:
        conclusion = (
            "The observed forced choices favor mean at the preregistered "
            "two-sided 0.05 threshold."
        )
    else:
        conclusion = (
            "The forced choices do not show a statistically significant "
            "departure from a 50% preference at the preregistered threshold."
        )
    videos = "\n".join(
        f"- {video_id}: {entry['wins']}/{entry['wins'] + entry['losses']} "
        f"barycenter wins; rate "
        f"{entry['rate']:.3f}" if entry["rate"] is not None
        else f"- {video_id}: no forced choices"
        for video_id, entry in results["per_video"].items()
    ) or "- No video responses."
    loo = "\n".join(
        f"- Excluding {video_id}: {entry['wins']}/{entry['votes']} "
        f"({entry['rate']:.3f})" if entry["rate"] is not None
        else f"- Excluding {video_id}: no remaining forced choices"
        for video_id, entry in results["leave_one_video_out"].items()
    ) or "- No leave-one-video-out estimates."
    tags = "\n".join(
        f"- {model} wins: "
        + (", ".join(f"{tag} {count}" for tag, count in sorted(counts.items())) or "no tags")
        for model, counts in results["tags_by_winner"].items()
    )
    report = f"""# Blinded preference study analysis

## Primary result

{conclusion}

- Testers: **{results['n_testers']}**; submitted votes: **{results['n_votes']}**.
- Forced choices: **{results['n_forced_choices']}** ({results['barycenter_wins']} barycenter, {results['mean_wins']} mean).
- No real difference: **{results['no_difference_count']}**.
- Barycenter forced-choice win rate: **{rate:.3f}** (95% Wilson CI {low:.3f}–{high:.3f}); two-sided exact binomial p = **{p_value:.6g}**.
- Ties counted as 0.5: win rate **{results['sensitivity_win_rate_with_ties_half']:.3f}**.
- Tester-cluster bootstrap 95% interval for the forced-choice rate: {cluster_low:.3f}–{cluster_high:.3f}.

## Robustness and secondary outcomes

- Likert paired difference (barycenter minus mean): mean **{results['likert_mean_difference']:.3f}**, bootstrap 95% CI {likert_low:.3f}–{likert_high:.3f}; two-sided exact Wilcoxon signed-rank p = {results['likert_wilcoxon_p_value']:.6g} (nonzero pairs: {results['likert_nonzero_pairs']}).
- A-choice rate irrespective of model: {results['A_choice_rate']:.3f} ({results['A_choice_count']} A, {results['B_choice_count']} B forced choices).
- Barycenter win rate when A was watched first: {_format_first(results, 'A')}; when B was watched first: {_format_first(results, 'B')}.
- Source video skipped: {results['source_skipped_count']} of {results['n_votes']} responses. Video positions: {', '.join(f'{position}: {count}' for position, count in results['video_order_counts'].items()) or 'none'}.
- Participant-selected first video: {', '.join(f'{selection}: {count}' for selection, count in results['first_video_selections'].items()) or 'none'}.
- Confound check: {results['confound_flagged_count']} of {results['confound_checks_count']} prepared videos flagged for a >10% duration, selected-shot-count, or mean-shot-length difference. Flagged IDs: {', '.join(results['confound_flagged_video_ids']) or 'none / not supplied'}.

### Per-video rates

{videos}

### Leave-one-video-out rates

{loo}

### Reason tags on winning votes

{tags}

## Figures

![Per-video forest plot](per_video_forest.png)

![Paired Likert differences](likert_differences.png)

All submitted testers and prepared videos are included. This is a subjective preference study, not an objective quality or F-score evaluation. Results do not establish general performance beyond these participants and videos.
"""
    report_path = output_dir / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _format_first(results: dict[str, Any], side: str) -> str:
    value = results["first_watched_rates"][side]
    if value["rate"] is None:
        return f"no votes ({value['votes']})"
    return f"{value['rate']:.3f} ({value['wins']}/{value['votes']})"


def analyze(
    database_path: Path = DATABASE_PATH,
    output_dir: Path = OUTPUT_DIR,
    confound_path: Path | None = None,
) -> dict[str, Any]:
    records = read_records(database_path)
    confound_path = (
        confound_path
        if confound_path is not None
        else STUDY_DIR / "stimuli" / "confound_check.csv"
    )
    results = analyze_records(
        records, confound_checks=_load_confound_checks(confound_path)
    )
    write_figures(results, output_dir)
    report_path = write_report(results, output_dir)
    print(f"Wrote {report_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--confound-check", type=Path)
    args = parser.parse_args()
    analyze(args.database, args.output_dir, args.confound_check)


if __name__ == "__main__":
    main()
