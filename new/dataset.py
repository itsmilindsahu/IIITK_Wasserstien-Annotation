import numpy as np
import os

def load_tvsum_data(tsv_path="ydata-tvsum50-anno.tsv"):
    """
    Parses the TVSum TSV file and returns a dictionary of videos.
    Returns:
        videos: dict mapping video_id to a numpy array of shape (num_annotators, num_frames)
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"{tsv_path} not found. Please run download_tvsum.py first.")
        
    videos = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
                
            vid = parts[0]
            # category = parts[1]
            # annotator = parts[2]
            scores_str = parts[3]
            
            scores = np.array([int(s) for s in scores_str.split(',')])
            
            if vid not in videos:
                videos[vid] = []
            videos[vid].append(scores)
            
    # Convert lists to numpy arrays
    for vid in videos:
        videos[vid] = np.array(videos[vid])
        
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
