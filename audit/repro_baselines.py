import json, numpy as np, pandas as pd, hashlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_latent import B, load, stable_seed

vids = load()
base = json.load(open(REPO_ROOT / 'results' / 'standard_tvsum_cv_fixed' / 'results.json'))['baseline_records']

def stored(name):
    d = pd.DataFrame([x for x in base[name] if x['metric'] == 'f1'])
    return d.groupby('video_key').value.mean()

st_h = stored('human_leave_one_out')
st_u = stored('uniform')
st_r = stored('random')
rows = []
for k, v in vids.items():
    h = B.human_leave_one_out(v)
    u = B.evaluate_summary(B.canonical_summary(np.ones(v.picks.size), v), v.user_summary)['f1']
    # selection-frequency LOAO (same as human_LOO) and mean-of-ratings proxy via gtscore (NOT leave-one-out: includes held-out annotator)
    g = B.evaluate_summary(B.canonical_summary(np.maximum(v.gtscore[:v.picks.size], 1e-9), v), v.user_summary)['f1']
    rows.append(dict(video=k, human_loo=h, uniform=u, gt_mean_proxy=g))

df = pd.DataFrame(rows).set_index('video')
print("all 50 videos: human-LOO(binary-summary consensus) %.2f | uniform(np.ones) %.2f | mean-rating proxy via gtscore (leaky, not LOAO) %.2f" % (df.human_loo.mean(), df.uniform.mean(), df.gt_mean_proxy.mean()))
c = st_h.index
print("reproduction vs stored per-video baselines (%d tested videos): max|Delta| human_LOO=%.2e uniform=%.2e" % (len(c), (df.human_loo[c] - st_h).abs().max(), (df.uniform[c] - st_u[c]).abs().max()))
print("32 unique tested videos: human_LOO %.2f uniform %.2f ; the other 18 videos: human_LOO %.2f uniform %.2f" % (df.human_loo[c].mean(), df.uniform[c].mean(), df.human_loo.drop(c).mean(), df.uniform.drop(c).mean()))

# knapsack degenerate check: uniform = max number of shots
v = vids['video_1']
s = B.canonical_summary(np.ones(v.picks.size), v)
sel = [i for i, (a, b) in enumerate(v.change_points) if s[a] > 0]
print("uniform summary on video_1 picks %d/%d shots; mean selected shot len %.0f frames vs mean shot len %.0f" % (len(sel), len(v.change_points), np.mean(v.frames_per_segment[sel]), v.frames_per_segment.mean()))

out_csv = Path(__file__).resolve().parent / 'repro_baselines.csv'
df.to_csv(out_csv)
