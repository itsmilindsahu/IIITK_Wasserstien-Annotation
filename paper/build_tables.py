"""Generate paper tables and provenance from saved benchmark JSON files."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
TABLES = Path(__file__).resolve().parent / "tables"
SOURCES = {
    "learned": ROOT / "results" / "standard_tvsum_cv_fixed" / "results.json",
    "backbone": ROOT / "results" / "standard_tvsum_cv_bilstm" / "backbone_comparison.json",
    "oracle": ROOT / "results" / "consensus_oracle_8b" / "results.json",
    "secondary": ROOT / "results" / "consensus_oracle_8d" / "results.json",
    "mechanism": ROOT / "results" / "mechanism_8e" / "results.json",
    "simulation": ROOT / "results" / "simulation_jitter" / "SIMULATION_results.json",
}
ARM_LABELS = {
    "A_mean_mse": "A: Mean target + MSE",
    "B_mean_cross_entropy": "B: Mean target + cross-entropy",
    "C_mean_cramer": "C: Mean target + Cramer loss",
    "D_barycenter_mse": "D: Barycenter target + MSE",
    "E_barycenter_cramer": "E: Barycenter target + Cramer loss",
    "F_blended_cramer": "F: Validation-selected blend + Cramer loss",
}
ORACLE_METRICS = (
    ("f1", r"F1 (\%)"),
    ("spearman_rho", r"Spearman $\rho$"),
    ("kendall_tau", r"Kendall $\tau$"),
)
def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved result JSON is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, digits: int = 3) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Refusing to render non-finite paper value: {value!r}")
    return f"{number:.{digits}f}"


def _pvalue(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Refusing to render invalid p-value: {value!r}")
    return f"{number:.3g}"


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _paired_tests(left: list[float], right: list[float]) -> dict[str, float | int]:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("Paired tests require equal, nonempty per-video observations.")
    differences = first - second
    if not np.isfinite(differences).all():
        raise ValueError("Paired tests require finite per-video values.")
    if np.all(differences == 0):
        return {"n": int(first.size), "paired_t_p": 1.0, "wilcoxon_p": 1.0}
    if np.var(differences) == 0:
        t_p = 0.0
    else:
        t_p = float(stats.ttest_rel(first, second).pvalue)
    wilcoxon_p = float(stats.wilcoxon(first, second, zero_method="zsplit").pvalue)
    if not math.isfinite(t_p) or not math.isfinite(wilcoxon_p):
        raise ValueError("Paired test returned a non-finite p-value.")
    return {"n": int(first.size), "paired_t_p": t_p, "wilcoxon_p": wilcoxon_p}


def _validate_sources(data: dict[str, dict[str, Any]]) -> dict[str, str]:
    learned = data["learned"]["protocol"]
    backbone = data["backbone"]["protocol"]
    oracle = data["oracle"]["protocol"]
    secondary = data["secondary"]["protocol"]
    mechanism = data["mechanism"]["protocol"]
    simulation = data["simulation"]["protocol"]
    fingerprint_keys = ("feature_sha256", "annotation_sha256", "info_sha256")
    expected = {key: oracle[key] for key in fingerprint_keys}
    for name, protocol, key_map in (
        ("learned", learned, {
            "feature_sha256": "feature_sha256",
            "annotation_sha256": "raw_annotation_sha256",
            "info_sha256": "official_info_sha256",
        }),
        ("backbone", backbone, {
            "feature_sha256": "feature_sha256",
            "annotation_sha256": "annotation_sha256",
            "info_sha256": "info_sha256",
        }),
        ("secondary", secondary, {key: key for key in fingerprint_keys}),
        ("mechanism", mechanism, {key: key for key in fingerprint_keys}),
        ("simulation", simulation, {key: key for key in fingerprint_keys}),
    ):
        for key, value in expected.items():
            if protocol.get(key_map[key]) != value:
                raise ValueError(f"{name} {key} does not match the saved oracle protocol.")
    if learned.get("unique_test_videos") != backbone.get("unique_test_videos"):
        raise ValueError("Learned-run and backbone comparison unique test coverage differs.")
    if len(data["oracle"]["per_video"]) != oracle["video_count"]:
        raise ValueError("Oracle video rows do not match its saved protocol count.")
    if len(data["mechanism"]["per_video"]) != oracle["video_count"]:
        raise ValueError("Mechanism video rows do not match the saved oracle count.")
    if len(data["simulation"]["shift_summary"]) == 0:
        raise ValueError("Saved jitter simulation has no shift summary.")
    return expected


def _write_learned_metrics(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    learned = data["learned"]
    backbone = data["backbone"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Learned-model metrics generated from the saved real-TVSum JSON. F1 is a percentage; rank correlations are unitless.}",
        r"\label{tab:learned-metrics}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & F1 mean & Seed SD & Spearman $\rho$ & Kendall $\tau$ \\",
        r"\midrule",
    ]
    for arm, label in ARM_LABELS.items():
        row = learned["summary"]["models"][arm]
        output.append(
            f"{_tex_escape(label)} & {_number(row['f1_mean'])} & "
            f"{_number(row['f1_mean_seed_std'])} & "
            f"{_number(row['spearman_rho_mean'], 4)} & "
            f"{_number(row['kendall_tau_mean'], 4)} \\\\"
        )
    bilstm = backbone["summary"]["bilstm"]
    output.append(
        f"BiLSTM, mean target + cross-entropy & {_number(bilstm['f1_mean'])} & "
        f"{_number(bilstm['f1_mean_seed_std'])} & "
        f"{_number(bilstm['spearman_rho_mean'], 4)} & "
        f"{_number(bilstm['kendall_tau_mean'], 4)} \\\\"
    )
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "learned_metrics.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "learned_metrics.tex": [
            "standard_tvsum_cv_fixed/results.json:summary.models[arm].{f1_mean,f1_mean_seed_std,spearman_rho_mean,kendall_tau_mean}",
            "standard_tvsum_cv_bilstm/backbone_comparison.json:summary.bilstm.{f1_mean,f1_mean_seed_std,spearman_rho_mean,kendall_tau_mean}",
        ]
    }


def _write_uniform_comparisons(
    data: dict[str, dict[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    learned = data["learned"]
    backbone = data["backbone"]
    uniform_records = learned["baseline_records"]["uniform"]
    uniform_by_video: dict[str, list[float]] = {}
    for row in uniform_records:
        if row["metric"] == "f1":
            uniform_by_video.setdefault(row["video_key"], []).append(float(row["value"]))
    uniform_unique = {
        key: float(np.mean(values)) for key, values in uniform_by_video.items()
    }
    bilstm_scores = backbone["unique_test_video_scores"]["bilstm"]
    common = sorted(set(uniform_unique) & set(bilstm_scores))
    if len(common) != learned["protocol"]["unique_test_videos"]:
        raise ValueError("BiLSTM and uniform unique-video records do not align.")
    bilstm_uniform = _paired_tests(
        [float(bilstm_scores[key]["f1"]) for key in common],
        [uniform_unique[key] for key in common],
    )
    bilstm_uniform["mean_f1_difference_points"] = float(np.mean([
        float(bilstm_scores[key]["f1"]) - uniform_unique[key] for key in common
    ]))

    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Paired learned-model versus uniform-summary F1 comparisons over the saved unique test videos. Raw paired-test p-values are shown.}",
        r"\label{tab:learned-uniform}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Learned model & $\Delta$ F1 & $n$ & Paired $t$ $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for arm, label in ARM_LABELS.items():
        row = learned["summary"]["paired_comparisons"][f"{arm}_vs_uniform"]
        output.append(
            f"{_tex_escape(label)} & {_number(row['mean_f1_difference_points'])} & "
            f"{row['n']} & {_pvalue(row['paired_t_p'])} & "
            f"{_pvalue(row['wilcoxon_p'])} \\\\"
        )
    output.append(
        f"BiLSTM, mean target + cross-entropy & "
        f"{_number(bilstm_uniform['mean_f1_difference_points'])} & "
        f"{bilstm_uniform['n']} & {_pvalue(bilstm_uniform['paired_t_p'])} & "
        f"{_pvalue(bilstm_uniform['wilcoxon_p'])} \\\\"
    )
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "learned_vs_uniform.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return (
        {
            "learned_vs_uniform.tex": [
                "standard_tvsum_cv_fixed/results.json:summary.paired_comparisons[arm_vs_uniform]",
                "standard_tvsum_cv_fixed/results.json:baseline_records.uniform (F1 grouped by video_key and averaged across saved fold appearances)",
                "standard_tvsum_cv_bilstm/backbone_comparison.json:unique_test_video_scores.bilstm (paired with grouped uniform F1; scipy.stats.ttest_rel and wilcoxon(zero_method='zsplit'))",
            ]
        },
        bilstm_uniform,
    )


def _write_bilstm_comparison(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    comparison = data["backbone"]["summary"]["paired_bilstm_minus_temporal_conv"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Saved paired BiLSTM-minus-temporal-convolution comparison for arm B.}",
        r"\label{tab:bilstm}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Difference & $n$ & Paired $t$ $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for metric, label in (
        ("f1", "F1 (points)"),
        ("spearman_rho", r"Spearman $\rho$"),
        ("kendall_tau", r"Kendall $\tau$"),
    ):
        row = comparison[metric]
        output.append(
            f"{label} & {_number(row['mean_difference'])} & {row['n']} & "
            f"{_pvalue(row['paired_t_p'])} & {_pvalue(row['wilcoxon_p'])} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "bilstm_comparison.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "bilstm_comparison.tex": [
            "standard_tvsum_cv_bilstm/backbone_comparison.json:summary.paired_bilstm_minus_temporal_conv.{f1,spearman_rho,kendall_tau}"
        ]
    }


def _write_oracle_tables(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    oracle = data["oracle"]
    secondary = data["secondary"]
    paired = oracle["overall"]["paired_vs_mean"]["barycenter"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Leave-one-annotator-out real-TVSum oracle metrics. Mean and barycenter estimates are averages over videos; paired tests compare per-video means.}",
        r"\label{tab:oracle}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Metric & Mean & Barycenter & Difference & Paired $t$ $p$ (raw/Holm) & Wilcoxon $p$ (raw/Holm) \\",
        r"\midrule",
    ]
    for key, label in ORACLE_METRICS:
        mean_value = float(np.mean([row["variants"]["mean"][key] for row in oracle["per_video"]]))
        bary_value = float(np.mean([row["variants"]["barycenter"][key] for row in oracle["per_video"]]))
        if key == "f1":
            test = paired
            difference = float(test["f1_difference_vs_mean"])
            p_t = f"{_pvalue(test['paired_t_p_raw'])}/{_pvalue(test['paired_t_p_holm'])}"
            p_w = f"{_pvalue(test['wilcoxon_p_raw'])}/{_pvalue(test['wilcoxon_p_holm'])}"
        else:
            rows = [row for row in secondary["secondary_paired_tests_vs_mean"] if row["variant"] == "barycenter" and row["metric"] == key]
            if len(rows) != 1:
                raise ValueError(f"Expected one saved secondary paired-test row for {key}.")
            test = rows[0]
            difference = float(test["difference_variant_minus_mean"])
            p_t = f"{_pvalue(test['paired_t_p_raw'])}/{_pvalue(test['paired_t_p_holm'])}"
            p_w = f"{_pvalue(test['wilcoxon_p_raw'])}/{_pvalue(test['wilcoxon_p_holm'])}"
        output.append(
            f"{label} & {_number(mean_value)} & {_number(bary_value)} & "
            f"{_number(difference)} & {p_t} & {p_w} \\\\"
        )
    output.extend([
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"\vspace{1mm}",
        r"{\footnotesize Paired test columns show raw/Holm-adjusted p-values. F1 uses the saved oracle paired comparison; rank metrics use Checkpoint secondary paired tests.}",
        r"\end{table}",
    ])
    (TABLES / "oracle_metrics.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "oracle_metrics.tex": [
            "consensus_oracle_8b/results.json:per_video[*].variants.{mean,barycenter}.{f1,spearman_rho,kendall_tau}",
            "consensus_oracle_8b/results.json:overall.paired_vs_mean.barycenter (F1 raw/Holm p-values)",
            "consensus_oracle_8d/results.json:secondary_metric_means and secondary_paired_tests_vs_mean (rank metric means and raw/Holm p-values)",
        ]
    }


def _write_secondary_table(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    secondary = data["secondary"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Secondary LOAO metrics for mean and barycenter targets. Paired differences are barycenter minus mean; p-values show raw/Holm adjustment.}",
        r"\label{tab:secondary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Metric & Mean & Barycenter & Difference & Paired $t$ $p$ (raw/Holm) & Wilcoxon $p$ (raw/Holm) \\",
        r"\midrule",
    ]
    metric_labels = {
        "spearman_rho": r"Spearman $\rho$",
        "kendall_tau": r"Kendall $\tau$",
    }
    tolerance_keys = sorted(
        (
            key for key in secondary["secondary_metric_means"]
            if key.startswith("tolerance_f1_k_")
        ),
        key=lambda key: int(key.rsplit("_", 1)[-1]),
    )
    metrics = [
        (key, metric_labels[key])
        for key in ("spearman_rho", "kendall_tau")
    ]
    metrics.extend(
        (
            key,
            f"Tolerance F1, $k={key.rsplit('_', 1)[-1]}$",
        )
        for key in tolerance_keys
    )
    for key, label in metrics:
        means = secondary["secondary_metric_means"][key]
        row = next(
            item for item in secondary["secondary_paired_tests_vs_mean"]
            if item["metric"] == key and item["variant"] == "barycenter"
        )
        p_t = f"{_pvalue(row['paired_t_p_raw'])}/{_pvalue(row['paired_t_p_holm'])}"
        p_w = f"{_pvalue(row['wilcoxon_p_raw'])}/{_pvalue(row['wilcoxon_p_holm'])}"
        output.append(
            f"{label} & {_number(means['mean'])} & {_number(means['barycenter'])} & "
            f"{_number(row['difference_variant_minus_mean'])} & {p_t} & {p_w} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (TABLES / "secondary_metrics.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "secondary_metrics.tex": [
            "consensus_oracle_8d/results.json:secondary_metric_means[metric][mean,barycenter]",
            "consensus_oracle_8d/results.json:secondary_paired_tests_vs_mean[metric,barycenter].{difference_variant_minus_mean,paired_t_p_raw,paired_t_p_holm,wilcoxon_p_raw,wilcoxon_p_holm}",
        ]
    }


def _write_w2_table(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    w2 = data["secondary"]["w2_characterization_only"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Wasserstein-distance characterization only; the barycenter minimizes W2 by construction.}",
        r"\label{tab:w2}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Mean-target W2 & Barycenter W2 & Difference & Paired $t$ $p$ raw/Holm & Wilcoxon $p$ raw/Holm & $n$ \\",
        r"\midrule",
        f"{_number(w2['mean_target_average_w2'])} & "
        f"{_number(w2['barycenter_average_w2'])} & "
        f"{_number(w2['barycenter_minus_mean_w2'])} & "
        f"{_pvalue(w2['paired_t_p_raw'])}/{_pvalue(w2['paired_t_p_holm'])} & "
        f"{_pvalue(w2['wilcoxon_p_raw'])}/{_pvalue(w2['wilcoxon_p_holm'])} & "
        f"{w2['n']} \\\\",
        r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ]
    (TABLES / "w2_characterization.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "w2_characterization.tex": [
            "consensus_oracle_8d/results.json:w2_characterization_only.{mean_target_average_w2,barycenter_average_w2,barycenter_minus_mean_w2,paired_t_p_raw,paired_t_p_holm,wilcoxon_p_raw,wilcoxon_p_holm,n}"
        ]
    }


def _write_mechanism_table(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    mechanism = data["mechanism"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Associations between predeclared per-video descriptors and barycenter-minus-mean F1. All entries are descriptive associations.}",
        r"\label{tab:mechanism}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Descriptor & Spearman $\rho$ & Bootstrap \BootstrapConfidenceLevel\% CI & Videos \\",
        r"\midrule",
    ]
    labels = {
        "mean_consensus_interior_peak_count": "Mean-curve peak count",
        "mean_pairwise_spearman_rho": "Annotator pairwise Spearman",
        "annotator_dominant_peak_location_sd_picks": "Peak-location SD (picks)",
        "weighted_overlap_fraction_of_mean_selected_mass": "Weighted selected-mass overlap",
    }
    for row in mechanism["spearman_associations"]:
        ci = (
            f"[{_number(row['bootstrap_ci_95_lower'])}, "
            f"{_number(row['bootstrap_ci_95_upper'])}]"
        )
        output.append(
            f"{labels[row['descriptor']]} & {_number(row['spearman_rho'])} & "
            f"{ci} & {row['n_videos']} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "mechanism.tex").write_text("\n".join(output) + "\n", encoding="utf-8")

    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Saved qualitative case-study measurements. These selected videos are descriptive examples.}",
        r"\label{tab:mechanism-cases}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Video & Mean F1 & Barycenter F1 & Difference & Shared pick count \\",
        r"\midrule",
    ]
    for video_id in ("uGu_10sucQo", "i3wAGJaaktw"):
        row = next(item for item in mechanism["per_video"] if item["video_id"] == video_id)
        output.append(
            f"{_tex_escape(video_id)} & {_number(row['mean_f1'])} & "
            f"{_number(row['barycenter_f1'])} & "
            f"{_number(row['f1_gap_barycenter_minus_mean'])} & "
            f"{row['shared_selected_pick_count']} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "mechanism_cases.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "mechanism.tex": [
            "mechanism_8e/results.json:spearman_associations[*].{descriptor,spearman_rho,bootstrap_ci_95_lower,bootstrap_ci_95_upper,n_videos}"
        ],
        "mechanism_cases.tex": [
            "mechanism_8e/results.json:per_video[video_id].{mean_f1,barycenter_f1,f1_gap_barycenter_minus_mean,shared_selected_pick_count}"
        ],
    }


def _write_exploratory_table(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    exploratory = data["secondary"]["exploratory_high_disagreement_mean_smoothing"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Exploratory high-disagreement smoothing gains; no smoothing value was used for tuning or selection.}",
        r"\label{tab:exploratory-smoothing}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & Mean F1 gain & Bootstrap \BootstrapConfidenceLevel\% CI & Improved videos / total & Videos \\",
        r"\midrule",
    ]
    variants = sorted(
        (
            variant for variant in exploratory
            if variant.startswith("mean_gaussian_sigma_")
        ),
        key=lambda variant: int(variant.rsplit("_", 1)[-1]),
    )
    for variant in variants:
        row = exploratory[variant]
        sigma = variant.rsplit("_", 1)[-1]
        output.append(
            rf"Mean smoothing, $\sigma={sigma}$ & "
            f"{_number(row['mean_gain_vs_mean_f1'])} & "
            f"[{_number(row['bootstrap_ci_95_lower'])}, "
            f"{_number(row['bootstrap_ci_95_upper'])}] & "
            f"{row['videos_improved']} / {row['videos_total']} & "
            f"{row['video_count']} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "exploratory_smoothing.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "exploratory_smoothing.tex": [
            "consensus_oracle_8d/results.json:exploratory_high_disagreement_mean_smoothing[mean_gaussian_sigma_*].{mean_gain_vs_mean_f1,bootstrap_ci_95_lower,bootstrap_ci_95_upper,videos_improved,videos_total,video_count}"
        ]
    }


def _write_simulation_table(data: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    simulation = data["simulation"]
    output = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{SIMULATION ONLY: labelled jitter experiment on real-video source curves; these results are not real annotator measurements.}",
        r"\label{tab:simulation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Shift half-width (picks) & Mean F1 & Barycenter F1 & Gain & Bootstrap \BootstrapConfidenceLevel\% CI & Improved videos \\",
        r"\midrule",
    ]
    for row in simulation["shift_summary"]:
        output.append(
            f"{row['shift_width_picks']} & {_number(row['mean_target_f1'])} & "
            f"{_number(row['barycenter_target_f1'])} & "
            f"{_number(row['mean_gain_barycenter_minus_mean'])} & "
            f"[{_number(row['bootstrap_ci_95_lower'])}, "
            f"{_number(row['bootstrap_ci_95_upper'])}] & "
            f"{row['videos_barycenter_improved']} / {row['n_videos']} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (TABLES / "simulation.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "simulation.tex": [
            "simulation_jitter/SIMULATION_results.json:shift_summary[*].{shift_width_picks,mean_target_f1,barycenter_target_f1,mean_gain_barycenter_minus_mean,bootstrap_ci_95_lower,bootstrap_ci_95_upper,videos_barycenter_improved,n_videos}"
        ]
    }


def _write_facts(
    data: dict[str, dict[str, Any]], bilstm_uniform: dict[str, Any]
) -> dict[str, list[str]]:
    oracle_protocol = data["oracle"]["protocol"]
    learned_protocol = data["learned"]["protocol"]
    secondary_protocol = data["secondary"]["protocol"]
    mechanism_protocol = data["mechanism"]["protocol"]
    simulation_protocol = data["simulation"]["protocol"]
    crossover = data["simulation"]["shift_summary"][0][
        "first_positive_mean_gain_shift_width_picks"
    ]
    confidence_text = (
        secondary_protocol["bootstrap_ci"]
        + " "
        + mechanism_protocol["bootstrap_ci"]
        + " "
        + simulation_protocol["bootstrap"]
    )
    confidence_match = re.search(r"(\d+)%\s*(?:percentile )?CI", confidence_text, re.IGNORECASE)
    if confidence_match is None:
        raise ValueError("Saved bootstrap protocol does not declare a confidence interval level.")
    barycenter_method = oracle_protocol["methods"]["barycenter"]
    barycenter_dimension = (
        "one-dimensional" if "1D" in barycenter_method else None
    )
    if barycenter_dimension is None:
        raise ValueError("Saved oracle method does not declare the expected 1D barycenter.")
    paired_uniform = data["learned"]["summary"]["paired_comparisons"]
    ablation_uniform_not_significant = all(
        float(paired_uniform[f"{arm}_vs_uniform"][test]) >= 0.05
        for arm in ARM_LABELS
        for test in ("paired_t_p", "wilcoxon_p")
    )
    ablation_uniform_finding = (
        "None of the A--F ablation arms has a paired test against uniform "
        "below the conventional significance threshold."
        if ablation_uniform_not_significant
        else "At least one A--F ablation test against uniform is below the conventional significance threshold."
    )
    oracle_rows = data["oracle"]["per_video"]
    barycenter_lower = all(
        float(np.mean([row["variants"]["barycenter"][metric] for row in oracle_rows]))
        < float(np.mean([row["variants"]["mean"][metric] for row in oracle_rows]))
        for metric in ("f1", "spearman_rho", "kendall_tau")
    )
    oracle_direction = (
        "is lower than the arithmetic mean on F1 and both rank-correlation averages"
        if barycenter_lower
        else "does not have lower averages on every reported oracle metric"
    )
    facts = {
        "TVSumVideos": oracle_protocol["video_count"],
        "OracleAnnotators": oracle_protocol["annotators_per_video"],
        "DSNetUniqueTestVideos": learned_protocol["unique_test_videos"],
        "DSNetSplitCount": data["backbone"]["protocol"]["split_count"],
        "AblationArmCount": len(learned_protocol["arms"]),
        "BarycenterDimension": barycenter_dimension,
        "BootstrapConfidenceLevel": confidence_match.group(1),
        "OracleDirection": oracle_direction,
        "AblationUniformFinding": ablation_uniform_finding,
        "BudgetPercent": f"{100 * learned_protocol['budget_fraction']:g}",
        "BootstrapReplicates": secondary_protocol["bootstrap_samples"],
        "SimulationAnnotators": simulation_protocol["simulated_annotators_per_video_per_width"],
        "SimulationShiftWidths": ", ".join(
            str(value) for value in simulation_protocol["shift_widths_picks"]
        ),
        "SimulationNoisePercent": 100 * simulation_protocol["noise"]["standard_deviation_relative_to_source_curve_max"],
        "SimulationFirstPositiveCrossover": (
            "not observed" if crossover is None else str(crossover)
        ),
        "BarycenterW2Label": data["secondary"]["w2_characterization_only"]["label"],
        "ExploratorySigma": next(
            key.rsplit("_", 1)[-1]
            for key in data["secondary"]["exploratory_high_disagreement_mean_smoothing"]
            if key.endswith("_4")
        ),
        "BilstmUniformDifference": _number(bilstm_uniform["mean_f1_difference_points"]),
        "BilstmUniformTTestP": _pvalue(bilstm_uniform["paired_t_p"]),
        "BilstmUniformWilcoxonP": _pvalue(bilstm_uniform["wilcoxon_p"]),
    }
    definitions = {
        "TVSumVideos": "consensus_oracle_8b/results.json:protocol.video_count",
        "OracleAnnotators": "consensus_oracle_8b/results.json:protocol.annotators_per_video",
        "DSNetUniqueTestVideos": "standard_tvsum_cv_fixed/results.json:protocol.unique_test_videos",
        "DSNetSplitCount": "standard_tvsum_cv_bilstm/backbone_comparison.json:protocol.split_count",
        "AblationArmCount": "standard_tvsum_cv_fixed/results.json:protocol.arms (list length)",
        "BarycenterDimension": "consensus_oracle_8b/results.json:protocol.methods.barycenter",
        "BootstrapConfidenceLevel": "consensus_oracle_8d/results.json:protocol.bootstrap_ci; mechanism_8e/results.json:protocol.bootstrap_ci; simulation_jitter/SIMULATION_results.json:protocol.bootstrap",
        "OracleDirection": "consensus_oracle_8b/results.json:per_video[*].variants.{mean,barycenter}.{f1,spearman_rho,kendall_tau}",
        "AblationUniformFinding": "standard_tvsum_cv_fixed/results.json:summary.paired_comparisons[arm_vs_uniform].{paired_t_p,wilcoxon_p}",
        "BudgetPercent": "standard_tvsum_cv_fixed/results.json:protocol.budget_fraction multiplied by 100",
        "BootstrapReplicates": "consensus_oracle_8d/results.json:protocol.bootstrap_samples",
        "SimulationAnnotators": "simulation_jitter/SIMULATION_results.json:protocol.simulated_annotators_per_video_per_width",
        "SimulationShiftWidths": "simulation_jitter/SIMULATION_results.json:protocol.shift_widths_picks",
        "SimulationNoisePercent": "simulation_jitter/SIMULATION_results.json:protocol.noise.standard_deviation_relative_to_source_curve_max multiplied by 100",
        "SimulationFirstPositiveCrossover": "simulation_jitter/SIMULATION_results.json:shift_summary[0].first_positive_mean_gain_shift_width_picks",
        "BarycenterW2Label": "consensus_oracle_8d/results.json:w2_characterization_only.label",
        "ExploratorySigma": "consensus_oracle_8d/results.json:exploratory_high_disagreement_mean_smoothing keys",
        "BilstmUniformDifference": "derived paired comparison from saved backbone unique_test_video_scores and standard benchmark uniform baseline records",
        "BilstmUniformTTestP": "derived scipy paired t-test on the saved paired unique-video BiLSTM/uniform values",
        "BilstmUniformWilcoxonP": "derived scipy Wilcoxon zsplit test on the saved paired unique-video BiLSTM/uniform values",
    }
    output = []
    for key, value in facts.items():
        output.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    (TABLES / "facts.tex").write_text("\n".join(output) + "\n", encoding="utf-8")
    return {"facts.tex": [definitions[key] for key in facts]}


def build_tables() -> dict[str, Any]:
    data = {name: _read_json(path) for name, path in SOURCES.items()}
    fingerprints = _validate_sources(data)
    TABLES.mkdir(parents=True, exist_ok=True)
    table_map: dict[str, list[str]] = {}
    table_map.update(_write_learned_metrics(data))
    uniform_map, bilstm_uniform = _write_uniform_comparisons(data)
    table_map.update(uniform_map)
    table_map.update(_write_bilstm_comparison(data))
    table_map.update(_write_oracle_tables(data))
    table_map.update(_write_secondary_table(data))
    table_map.update(_write_w2_table(data))
    table_map.update(_write_mechanism_table(data))
    table_map.update(_write_exploratory_table(data))
    table_map.update(_write_simulation_table(data))
    table_map.update(_write_facts(data, bilstm_uniform))
    manifest = {
        "description": (
            "Every numeric table cell and generated paper fact is read from a saved JSON "
            "or derived from saved per-video JSON values using the recorded paired tests."
        ),
        "source_files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for name, path in SOURCES.items()
        },
        "input_protocol_fingerprints": fingerprints,
        "generated_files": table_map,
        "derived_statistics": {
            "bilstm_vs_uniform": {
                "unit": "unique test video",
                "aggregation": (
                    "mean saved uniform fold/video F1 records within video_key; "
                    "pair with saved BiLSTM unique_test_video_scores"
                ),
                "test": "scipy.stats.ttest_rel and scipy.stats.wilcoxon(zero_method='zsplit')",
                "n": bilstm_uniform["n"],
                "mean_f1_difference_points": bilstm_uniform["mean_f1_difference_points"],
                "paired_t_p_raw": bilstm_uniform["paired_t_p"],
                "wilcoxon_p_raw": bilstm_uniform["wilcoxon_p"],
            },
        },
    }
    manifest_path = TABLES / "table_sources.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    result = build_tables()
    print(json.dumps({
        "generated_files": sorted(result["generated_files"]),
        "source_files": result["source_files"],
        "derived_statistics": result["derived_statistics"],
    }, indent=2, sort_keys=True))
