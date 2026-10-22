import os
import sys

import numpy as np
import pytest

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
sys.path.insert(0, CODE_DIR)

from dataset import load_tvsum_data


def test_loads_official_three_column_tsv_and_uses_environment_path(tmp_path, monkeypatch):
    annotation_path = tmp_path / "official.tsv"
    annotation_path.write_text(
        "real_video\tVT\t1,2,3\n"
        "real_video\tVT\t3,2,1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TVSUM_ANNOTATION_TSV", str(annotation_path))

    videos = load_tvsum_data("missing-default.tsv")

    np.testing.assert_array_equal(videos["real_video"], [[1, 2, 3], [3, 2, 1]])


def test_rejects_legacy_four_column_tsv(tmp_path, monkeypatch):
    annotation_path = tmp_path / "legacy.tsv"
    annotation_path.write_text("real_video\tVT\t1\t1,2,3\n", encoding="utf-8")
    monkeypatch.setenv("TVSUM_ANNOTATION_TSV", str(annotation_path))

    with pytest.raises(ValueError, match="three TSV columns"):
        load_tvsum_data()


def test_rejects_synthetic_proxy_even_with_legacy_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("TVSUM_ANNOTATION_TSV", raising=False)
    monkeypatch.setenv("TVSUM_ALLOW_SYNTHETIC", "1")
    synthetic_path = tmp_path / "synthetic.tsv"
    with synthetic_path.open("w", encoding="utf-8") as handle:
        for video_index in range(1, 51):
            for annotator_index in range(20):
                scores = ",".join(["1"] * 150)
                handle.write(f"video_{video_index:02d}\tcategory\t{scores}\n")

    with pytest.raises(ValueError, match="synthetic proxy"):
        load_tvsum_data(str(synthetic_path))
