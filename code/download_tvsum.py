import os


def require_tvsum_annotations(path="ydata-tvsum50-anno.tsv"):
    """Require an authorized local annotation file; never generate a fallback."""
    annotation_path = os.environ.get("TVSUM_ANNOTATION_TSV", path)
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(
            f"TVSum annotations not found at {annotation_path}. TVSum v1.1 is "
            "distributed under Yahoo! Webscope terms; obtain authorized access, "
            "extract data/ydata-tvsum50-anno.tsv, and set TVSUM_ANNOTATION_TSV."
        )
    return annotation_path

if __name__ == "__main__":
    require_tvsum_annotations()
