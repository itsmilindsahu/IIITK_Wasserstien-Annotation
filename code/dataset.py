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

def get_video_distributions(video_scores):
    """
    Converts 1-5 importance scores into probability distributions over frames.
    video_scores: (num_annotators, num_frames)
    Returns:
        distributions: (num_annotators, num_frames) summing to 1 over frames
    """
    # Softmax or simple normalization. Let's do simple normalization of scores
    dist = video_scores / np.sum(video_scores, axis=1, keepdims=True)
    return dist
