import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "consensus_oracle_summary"

VARIANT_LABELS = {
    "mean": "Mean",
    "barycenter": "Barycenter",
    "barycenter_fine4": "Barycenter 4x grid",
    "barycenter_gaussian_sigma_1": "Barycenter sigma=1",
    "barycenter_gaussian_sigma_2": "Barycenter sigma=2",
    "barycenter_gaussian_sigma_4": "Barycenter sigma=4",
    "mean_gaussian_sigma_1": "Mean sigma=1",
    "mean_gaussian_sigma_2": "Mean sigma=2",
    "mean_gaussian_sigma_4": "Mean sigma=4",
}
VARIANT_ORDER = list(VARIANT_LABELS)
METRICS = [
    ("spearman_rho", "Spearman"),
    ("kendall_tau", "Kendall"),
    ("tolerance_f1_k_2", "Tolerance F1 k=2"),
    ("tolerance_f1_k_5", "Tolerance F1 k=5"),
    ("tolerance_f1_k_10", "Tolerance F1 k=10"),
]


def load_json(name):
    path = RESULTS / name / "results.json"
    with path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


def copy_artifacts():
    files = [
        ("consensus_oracle", "results_table.csv", "oracle_8_results_table.csv"),
        ("consensus_oracle_8b", "variant_metrics.csv", "oracle_8b_variant_metrics.csv"),
        ("consensus_oracle_8b", "paired_tests.csv", "oracle_8b_paired_tests.csv"),
        (
            "consensus_oracle_8d",
            "secondary_paired_tests.csv",
            "oracle_8d_secondary_paired_tests.csv",
        ),
        (
            "consensus_oracle_8d",
            "continuous_and_exploratory_cis.csv",
            "oracle_8d_disagreement_and_exploratory.csv",
        ),
        (
            "consensus_oracle_8d",
            "w2_characterization.csv",
            "oracle_8d_w2_characterization.csv",
        ),
    ]
    for source_dir, source_name, output_name in files:
        shutil.copy2(RESULTS / source_dir / source_name, OUTPUT / output_name)


def save_metric_table(oracle_8b):
    metrics = oracle_8b["overall"]["variants"]
    fields = [
        "variant",
        "f1",
        "spearman_rho",
        "kendall_tau",
        "tolerance_f1_k_2",
        "tolerance_f1_k_5",
        "tolerance_f1_k_10",
    ]
    with (OUTPUT / "all_variant_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANT_ORDER:
            writer.writerow({"variant": VARIANT_LABELS[variant], **metrics[variant]})


def create_dashboard(oracle_8, oracle_8b, oracle_8d):
    variants = oracle_8b["overall"]["variants"]
    f1_tests = oracle_8b["overall"]["paired_vs_mean"]
    disagreement = oracle_8d["continuous_disagreement"]
    exploration = oracle_8d["exploratory_high_disagreement_mean_smoothing"]
    w2 = oracle_8d["w2_characterization_only"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "TVSum50 leave-one-annotator-out consensus oracle",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.94,
        "50 videos | 20 held-out annotators/video | canonical 15% summary",
        ha="center",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.22, hspace=0.43, wspace=0.34)

    ax = axes[0, 0]
    colors = [
        "#2878B5"
        if variant == "mean"
        else "#D97706"
        if variant.startswith("barycenter")
        else "#3A9D5D"
        for variant in VARIANT_ORDER
    ]
    values = [variants[variant]["f1"] for variant in VARIANT_ORDER]
    y = np.arange(len(VARIANT_ORDER))
    ax.barh(y, values, color=colors, alpha=0.9)
    ax.axvline(variants["mean"]["f1"], color="#17324D", linestyle="--", linewidth=1)
    ax.set_yticks(y, [VARIANT_LABELS[variant] for variant in VARIANT_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(55, 64.5)
    ax.set_xlabel("Mean F1 (%)")
    ax.set_title("Consensus-target F1 across variants")
    ax.grid(axis="x", alpha=0.2)
    for index, variant in enumerate(VARIANT_ORDER):
        label = f"{values[index]:.2f}"
        if variant != "mean":
            test = f1_tests[variant]
            if test["paired_t_p_holm"] < 0.05 and test["wilcoxon_p_holm"] < 0.05:
                label += "  *"
        ax.text(values[index] + 0.08, index, label, va="center", fontsize=9)
    ax.text(
        0.01,
        -0.18,
        "* Both paired tests vs mean have Holm-adjusted p < .05.",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax = axes[0, 1]
    groups = [
        ("All videos", oracle_8["overall"]),
        ("High disagreement", oracle_8["tertiles"]["high_disagreement"]),
        ("Middle disagreement", oracle_8["tertiles"]["middle_disagreement"]),
        ("Low disagreement", oracle_8["tertiles"]["low_disagreement"]),
    ]
    positions = np.arange(len(groups))
    width = 0.36
    mean_values = [group["mean_target_f1"] for _, group in groups]
    bary_values = [group["barycenter_target_f1"] for _, group in groups]
    ax.bar(positions - width / 2, mean_values, width, label="Mean target", color="#2878B5")
    ax.bar(
        positions + width / 2,
        bary_values,
        width,
        label="Barycenter target",
        color="#D97706",
    )
    ax.set_xticks(positions, [name for name, _ in groups], rotation=15, ha="right")
    ax.set_ylabel("Mean F1 (%)")
    ax.set_title("F1 by annotator-disagreement tertile")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    secondary = oracle_8d["secondary_metric_means"]
    corr_names = ["Spearman rho", "Kendall tau"]
    mean_corr = [secondary["spearman_rho"]["mean"], secondary["kendall_tau"]["mean"]]
    bary_corr = [
        secondary["spearman_rho"]["barycenter"],
        secondary["kendall_tau"]["barycenter"],
    ]
    x = np.arange(2)
    ax.plot(x, mean_corr, marker="o", linewidth=2.5, label="Mean target", color="#2878B5")
    ax.plot(
        x,
        bary_corr,
        marker="o",
        linewidth=2.5,
        label="Barycenter target",
        color="#D97706",
    )
    ax.set_xticks(x, corr_names)
    ax.set_ylabel("Correlation with held-out annotator")
    ax.set_title("Rank agreement")
    ax.set_ylim(0.17, 0.42)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for i, value in enumerate(mean_corr):
        ax.annotate(f"{value:.3f}", (x[i], value), xytext=(0, 8), textcoords="offset points", ha="center")
    for i, value in enumerate(bary_corr):
        ax.annotate(f"{value:.3f}", (x[i], value), xytext=(0, -15), textcoords="offset points", ha="center")

    ax = axes[1, 1]
    tolerances = [2, 5, 10]
    mean_tol = [secondary[f"tolerance_f1_k_{k}"]["mean"] for k in tolerances]
    bary_tol = [secondary[f"tolerance_f1_k_{k}"]["barycenter"] for k in tolerances]
    ax.plot(tolerances, mean_tol, marker="o", linewidth=2.5, label="Mean target", color="#2878B5")
    ax.plot(
        tolerances,
        bary_tol,
        marker="o",
        linewidth=2.5,
        label="Barycenter target",
        color="#D97706",
    )
    ax.set_xticks(tolerances)
    ax.set_xlabel("Matching tolerance (picks)")
    ax.set_ylabel("Tolerance F1 (%)")
    ax.set_title("Tolerance-based summary overlap")
    ax.set_ylim(66, 84)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    for x_value, value in zip(tolerances, mean_tol):
        ax.annotate(f"{value:.2f}", (x_value, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    for x_value, value in zip(tolerances, bary_tol):
        ax.annotate(f"{value:.2f}", (x_value, value), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8)

    gap = disagreement["barycenter_minus_mean_f1_gap"]
    smooth_gap = disagreement["mean_sigma4_minus_mean_f1_gap"]
    high_sigma4 = exploration["mean_gaussian_sigma_4"]
    fig.text(
        0.5,
        0.135,
        (
            "F1 delta (barycenter minus mean): all "
            f"{overall_delta(oracle_8):+.2f}; high disagreement "
            f"{oracle_8['tertiles']['high_disagreement']['barycenter_minus_mean_f1_points']:+.2f}; "
            f"middle {oracle_8['tertiles']['middle_disagreement']['barycenter_minus_mean_f1_points']:+.2f}; "
            f"low {oracle_8['tertiles']['low_disagreement']['barycenter_minus_mean_f1_points']:+.2f}. "
            "All five barycenter F1 variants are significant after Holm correction."
        ),
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.075,
        (
            f"Disagreement association with F1 gap: barycenter rho={gap['spearman_rho']:.3f} "
            f"(95% CI {gap['bootstrap_ci_95_lower']:.3f}, {gap['bootstrap_ci_95_upper']:.3f}); "
            f"mean sigma=4 rho={smooth_gap['spearman_rho']:.3f} "
            f"(95% CI {smooth_gap['bootstrap_ci_95_lower']:.3f}, "
            f"{smooth_gap['bootstrap_ci_95_upper']:.3f}).  "
            f"W2: mean {w2['mean_target_average_w2']:.3f}, barycenter "
            f"{w2['barycenter_average_w2']:.3f} (characterization only; barycenter minimizes W2 by construction). "
            f"Exploratory high-disagreement mean sigma=4 gain: "
            f"{high_sigma4['mean_gain_vs_mean_f1']:+.2f} "
            f"(95% CI {high_sigma4['bootstrap_ci_95_lower']:.2f}, "
            f"{high_sigma4['bootstrap_ci_95_upper']:.2f}; "
            f"{high_sigma4['videos_improved']}/{high_sigma4['videos_total']} improved)."
        ),
        ha="center",
        fontsize=8,
        wrap=True,
    )
    fig.savefig(OUTPUT / "consensus_oracle_findings.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def overall_delta(oracle_8):
    return oracle_8["overall"]["barycenter_minus_mean_f1_points"]


def create_metric_table(oracle_8b, oracle_8d):
    variants = oracle_8b["overall"]["variants"]
    f1_tests = oracle_8b["overall"]["paired_vs_mean"]
    secondary_tests = {
        (row["metric"], row["variant"]): row
        for row in oracle_8d["secondary_paired_tests_vs_mean"]
    }

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.axis("off")
    ax.set_title(
        "Complete overall variant metrics (n=50 videos)",
        fontsize=20,
        fontweight="bold",
        pad=22,
    )
    columns = [
        "Variant",
        "F1 (%)",
        "Spearman",
        "Kendall",
        "Tolerance F1\nk=2",
        "Tolerance F1\nk=5",
        "Tolerance F1\nk=10",
        "F1 vs mean",
    ]
    rows = []
    for variant in VARIANT_ORDER:
        metrics = variants[variant]
        if variant == "mean":
            f1_mark = "Reference"
        else:
            test = f1_tests[variant]
            f1_mark = (
                "Both Holm p<.05"
                if test["paired_t_p_holm"] < 0.05 and test["wilcoxon_p_holm"] < 0.05
                else "Not supported"
            )
        rows.append(
            [
                VARIANT_LABELS[variant],
                f"{metrics['f1']:.3f}",
                f"{metrics['spearman_rho']:.4f}",
                f"{metrics['kendall_tau']:.4f}",
                f"{metrics['tolerance_f1_k_2']:.3f}",
                f"{metrics['tolerance_f1_k_5']:.3f}",
                f"{metrics['tolerance_f1_k_10']:.3f}",
                f1_mark,
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=[0.22, 0.09, 0.11, 0.10, 0.12, 0.12, 0.12, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#17324D")
        table[(0, column)].get_text().set_color("white")
        table[(0, column)].get_text().set_weight("bold")
    for row_index, variant in enumerate(VARIANT_ORDER, start=1):
        color = "#E8F1F8" if variant == "mean" else "#FFF3E0" if variant.startswith("barycenter") else "#EAF5EC"
        for column in range(len(columns)):
            table[(row_index, column)].set_facecolor(color)

    ax.text(
        0.02,
        0.24,
        "Secondary paired comparisons vs unsmoothed mean (difference = variant − mean)",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
    )
    matrix_columns = ["Variant", "Spearman", "Kendall", "Tol F1 k=2", "Tol F1 k=5", "Tol F1 k=10"]
    matrix_rows = []
    for variant in VARIANT_ORDER:
        if variant == "mean":
            continue
        row = [VARIANT_LABELS[variant]]
        for metric, _ in METRICS:
            result = secondary_tests[(metric, variant)]
            significant = (
                result["paired_t_p_holm"] < 0.05
                and result["wilcoxon_p_holm"] < 0.05
            )
            marker = "*" if significant else ""
            row.append(f"{result['difference_variant_minus_mean']:+.4f}{marker}")
        matrix_rows.append(row)
    test_table = ax.table(
        cellText=matrix_rows,
        colLabels=matrix_columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.005, 0.96, 0.22],
        colWidths=[0.32, 0.13, 0.13, 0.14, 0.14, 0.14],
    )
    test_table.auto_set_font_size(False)
    test_table.set_fontsize(9)
    for column in range(len(matrix_columns)):
        test_table[(0, column)].set_facecolor("#17324D")
        test_table[(0, column)].get_text().set_color("white")
        test_table[(0, column)].get_text().set_weight("bold")
    fig.text(
        0.02,
        0.005,
        "* Both paired t and Wilcoxon tests have Holm-adjusted p<.05 within each metric. "
        "Differences are not adjusted across metric families.",
        fontsize=9,
    )
    fig.savefig(OUTPUT / "all_variant_metrics_and_tests.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary(oracle_8, oracle_8b, oracle_8d):
    overall = oracle_8["overall"]
    paired = oracle_8b["overall"]["paired_vs_mean"]
    w2 = oracle_8d["w2_characterization_only"]
    disagreement = oracle_8d["continuous_disagreement"]
    exploratory = oracle_8d["exploratory_high_disagreement_mean_smoothing"]
    lines = [
        "TVSum50 leave-one-annotator-out consensus oracle summary",
        "",
        "Scope: Checkpoints 8, 8b, and 8d only; this is an annotation-target oracle diagnostic,",
        "not learned model performance or a deployable baseline. No model training was done.",
        "",
        "Protocol: 50 real TVSum videos, 20 annotators/video. For each held-out annotator,",
        "the other 19 annotators form the target. F1 is computed against only the held-out",
        "annotator's original-frame user_summary with the canonical 15% summary. The paired",
        "unit is a per-video mean over the 20 held-out users (n=50).",
        "",
        "Main result:",
        f"Mean-target F1={overall['mean_target_f1']:.3f}; barycenter F1={overall['barycenter_target_f1']:.3f};",
        f"barycenter-minus-mean={overall['barycenter_minus_mean_f1_points']:+.3f} points.",
        f"Checkpoint 8 raw paired t p={overall['paired_t_p']:.8g}; Wilcoxon p={overall['wilcoxon_p']:.8g}.",
        "Checkpoint 8b Holm correction across eight variants: all five barycenter-target",
        "variants have lower overall F1 than mean, with both adjusted paired tests <.05.",
        "The three smoothed-mean F1 controls are not significantly different from mean after Holm correction.",
        "",
        "Disagreement tertile F1 (descriptive means; high disagreement means lower annotator-rank agreement):",
    ]
    for label, key in [
        ("High", "high_disagreement"),
        ("Middle", "middle_disagreement"),
        ("Low", "low_disagreement"),
    ]:
        group = oracle_8["tertiles"][key]
        lines.append(
            f"{label} disagreement n={group['n']}: mean={group['mean_target_f1']:.3f}, "
            f"barycenter={group['barycenter_target_f1']:.3f}, "
            f"delta={group['barycenter_minus_mean_f1_points']:+.3f}; "
            f"raw paired p(t/Wilcoxon)={group['paired_t_p']:.6g}/{group['wilcoxon_p']:.6g}."
        )
    lines.extend(
        [
            "",
            "Secondary results:",
            "See all_variant_metrics.csv and all_variant_metrics_and_tests.png for all nine",
            "variants' F1, rank correlation, and tolerance-F1 metrics. See the copied paired-test",
            "CSVs for raw and Holm-adjusted p-values.",
            f"Continuous disagreement vs barycenter-minus-mean F1 gap: Spearman rho="
            f"{disagreement['barycenter_minus_mean_f1_gap']['spearman_rho']:.4f}, "
            f"bootstrap 95% CI "
            f"[{disagreement['barycenter_minus_mean_f1_gap']['bootstrap_ci_95_lower']:.4f}, "
            f"{disagreement['barycenter_minus_mean_f1_gap']['bootstrap_ci_95_upper']:.4f}].",
            f"Continuous disagreement vs mean sigma=4-minus-mean F1 gap: Spearman rho="
            f"{disagreement['mean_sigma4_minus_mean_f1_gap']['spearman_rho']:.4f}, "
            f"bootstrap 95% CI "
            f"[{disagreement['mean_sigma4_minus_mean_f1_gap']['bootstrap_ci_95_lower']:.4f}, "
            f"{disagreement['mean_sigma4_minus_mean_f1_gap']['bootstrap_ci_95_upper']:.4f}].",
            "These are associations, not causal effects.",
            f"W2 characterization only (barycenter minimizes W2 by construction): mean="
            f"{w2['mean_target_average_w2']:.6f}, barycenter={w2['barycenter_average_w2']:.6f}, "
            f"difference={w2['barycenter_minus_mean_w2']:+.6f}; "
            f"paired t raw/Holm={w2['paired_t_p_raw']:.6g}/{w2['paired_t_p_holm']:.6g}; "
            f"Wilcoxon raw/Holm={w2['wilcoxon_p_raw']:.6g}/{w2['wilcoxon_p_holm']:.6g}.",
            "Exploratory high-disagreement mean-smoothing gains (not tuned or inferential claims):",
        ]
    )
    for sigma in (1, 2, 4):
        result = exploratory[f"mean_gaussian_sigma_{sigma}"]
        lines.append(
            f"sigma={sigma}: gain={result['mean_gain_vs_mean_f1']:+.3f}, "
            f"bootstrap 95% CI [{result['bootstrap_ci_95_lower']:+.3f}, "
            f"{result['bootstrap_ci_95_upper']:+.3f}], "
            f"improved videos={result['videos_improved']}/{result['videos_total']}."
        )
    lines.extend(
        [
            "",
            "Files:",
            "- consensus_oracle_findings.png: graphical overview.",
            "- all_variant_metrics_and_tests.png: complete metrics and secondary test matrix.",
            "- all_variant_metrics.csv: all variant means.",
            "- oracle_8_results_table.csv, oracle_8b_variant_metrics.csv,",
            "  oracle_8b_paired_tests.csv, oracle_8d_secondary_paired_tests.csv,",
            "  oracle_8d_disagreement_and_exploratory.csv, oracle_8d_w2_characterization.csv:",
            "  detailed copied tables from the checkpoint result directories.",
            "",
            "Source JSON files: results/consensus_oracle/results.json,",
            "results/consensus_oracle_8b/results.json, results/consensus_oracle_8d/results.json.",
        ]
    )
    (OUTPUT / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary directory: {OUTPUT}")
    oracle_8 = load_json("consensus_oracle")
    oracle_8b = load_json("consensus_oracle_8b")
    oracle_8d = load_json("consensus_oracle_8d")
    OUTPUT.mkdir(parents=True)
    copy_artifacts()
    save_metric_table(oracle_8b)
    create_dashboard(oracle_8, oracle_8b, oracle_8d)
    create_metric_table(oracle_8b, oracle_8d)
    save_summary(oracle_8, oracle_8b, oracle_8d)
    print(f"Created {OUTPUT}")
    for path in sorted(OUTPUT.iterdir()):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
