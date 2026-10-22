import json

from paper import build_tables


def test_table_builder_traces_each_generated_file_to_saved_json(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tables, "TABLES", tmp_path)

    manifest = build_tables.build_tables()

    generated = manifest["generated_files"]
    tex_files = {path.name for path in tmp_path.glob("*.tex")}
    assert tex_files == set(generated)
    assert all(sources for sources in generated.values())
    assert all(source["sha256"] for source in manifest["source_files"].values())
    assert json.loads((tmp_path / "table_sources.json").read_text()) == manifest


def test_uniform_table_includes_bilstm_raw_p_values_and_ablation_values(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tables, "TABLES", tmp_path)
    manifest = build_tables.build_tables()
    table = (tmp_path / "learned_vs_uniform.tex").read_text(encoding="utf-8")
    learned = build_tables._read_json(build_tables.SOURCES["learned"])
    arm = learned["summary"]["paired_comparisons"]["A_mean_mse_vs_uniform"]
    expected_arm = (
        f"A: Mean target + MSE & {build_tables._number(arm['mean_f1_difference_points'])} "
        f"& {arm['n']} & {build_tables._pvalue(arm['paired_t_p'])} "
        f"& {build_tables._pvalue(arm['wilcoxon_p'])}"
    )
    bilstm = manifest["derived_statistics"]["bilstm_vs_uniform"]
    expected_bilstm = (
        f"BiLSTM, mean target + cross-entropy & "
        f"{build_tables._number(bilstm['mean_f1_difference_points'])} & "
        f"{bilstm['n']} & {build_tables._pvalue(bilstm['paired_t_p_raw'])} "
        f"& {build_tables._pvalue(bilstm['wilcoxon_p_raw'])}"
    )

    assert expected_arm in table
    assert expected_bilstm in table
