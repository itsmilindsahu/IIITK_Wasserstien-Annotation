import numpy as np
import os
import hashlib


def stable_seed(vid):
    """
    Deterministic per-video seed that is stable across process restarts.

    Python's built-in hash() on strings is randomly salted per-process
    (PYTHONHASHSEED) as a security feature, so `abs(hash(vid)) % M` gives a
    different seed every run even though everything downstream is seeded
    "correctly" within that run. Use a fixed-digest hash (md5) instead, so
    the same video id always maps to the same seed, forever.
    """
    digest = hashlib.md5(str(vid).encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31 - 1)

def load_tvsum_data(tsv_path="ydata-tvsum50-anno.tsv"):
    """
    Parses the TVSum annotation TSV and returns a dictionary of videos.

    Reads only the official three-column TVSum v1.1 TSV format.
    Returns:
        videos: dict mapping video_id to an array of shape
            (num_annotators, num_annotation_positions)
    """
    tsv_path = os.environ.get("TVSUM_ANNOTATION_TSV", tsv_path)
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"{tsv_path} not found. Set TVSUM_ANNOTATION_TSV to an authorized "
            "TVSum annotation TSV; synthetic data is not generated automatically."
        )
        
    videos = {}
    with open(tsv_path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected three TSV columns on line {line_number} of {tsv_path}"
                )
            vid = parts[0]
            scores_str = parts[2]
            try:
                scores = np.array([int(s) for s in scores_str.split(",")])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid annotation score on line {line_number} of {tsv_path}"
                ) from exc
            if scores.size == 0:
                raise ValueError(f"Empty annotation on line {line_number} of {tsv_path}")
            
            if vid not in videos:
                videos[vid] = []
            videos[vid].append(scores)
            
    # Convert lists to numpy arrays
    for vid in videos:
        lengths = {len(scores) for scores in videos[vid]}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent annotation lengths for video {vid!r}")
        videos[vid] = np.array(videos[vid])

    if not videos:
        raise ValueError(f"No TVSum annotations found in {tsv_path}")

    proxy_ids = {f"video_{index:02d}" for index in range(1, 51)}
    if set(videos) == proxy_ids and all(
        scores.shape == (20, 150) for scores in videos.values()
    ):
        raise ValueError(
            "The selected TSV is the repository's synthetic proxy, not TVSum. "
            "Set TVSUM_ANNOTATION_TSV to the authorized official annotations."
        )
        
    return videos


def get_video_distributions(video_scores, method="normalize", temperature=1.0):
    """
    Converts 1-5 importance scores into probability distributions over frames.

    video_scores: (num_annotators, num_frames)
    method:
        "normalize"  -- original behaviour: p_i = s_i / sum(s). Linear, so
                        it barely sharpens the raw 1-5 annotator scale;
                        annotators end up looking close to each other and
                        close to uniform, which is part of why the
                        barycenter-vs-mean gap was small on video_01.
        "softmax"    -- temperature-scaled softmax over raw scores:
                        p_i = exp(s_i / T) / sum(exp(s_j / T)). Lower T
                        pushes mass toward the annotator's peak frames,
                        i.e. sharper, more separated distributions.
        "zscore_exp" -- z-score each annotator's raw scores, then
                        exponentiate and normalize:
                        z_i = (s_i - mean(s)) / std(s); p_i = exp(z_i/T) / sum.
                        This first removes each annotator's mean/scale
                        (so annotators who use the 1-5 scale differently
                        are put on comparable footing) before sharpening,
                        which "softmax" alone does not do.
    temperature: only used by "softmax" and "zscore_exp"; smaller = sharper.

    Returns:
        distributions: (num_annotators, num_frames) summing to 1 over frames
    """
    video_scores = np.asarray(video_scores, dtype=float)

    if method == "normalize":
        dist = video_scores / np.sum(video_scores, axis=1, keepdims=True)

    elif method == "softmax":
        s = video_scores / temperature
        s = s - np.max(s, axis=1, keepdims=True)  # numerical stability
        e = np.exp(s)
        dist = e / np.sum(e, axis=1, keepdims=True)

    elif method == "zscore_exp":
        mean = video_scores.mean(axis=1, keepdims=True)
        std = video_scores.std(axis=1, keepdims=True) + 1e-8
        z = (video_scores - mean) / std
        z = z / temperature
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        dist = e / np.sum(e, axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    return dist
