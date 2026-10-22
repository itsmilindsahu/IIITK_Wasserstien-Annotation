from pathlib import Path
import sys, types, time, json, hashlib
import numpy as np, h5py

# --- stub torch so the repo's evaluator functions can be imported unchanged
torch = types.ModuleType('torch'); nn = types.ModuleType('torch.nn'); F = types.ModuleType('torch.nn.functional')
utils = types.ModuleType('torch.nn.utils'); rnn = types.ModuleType('torch.nn.utils.rnn')
nn.Module = type('Module', (), {}); nn.functional = F; rnn.pad_sequence = None; utils.rnn = rnn; nn.utils = utils
torch.nn = nn; torch.Tensor = object
for n, m in [('torch', torch), ('torch.nn', nn), ('torch.nn.functional', F), ('torch.nn.utils', utils), ('torch.nn.utils.rnn', rnn)]:
    sys.modules[n] = m

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'code'))
import standard_tvsum_cv as B

H5 = str(REPO_ROOT / 'datasets' / 'eccv16_dataset_tvsum_google_pool5.h5')

class V: pass

def load():
    f = h5py.File(H5, 'r'); out = {}
    for k in f.keys():
        g = f[k]; v = V(); v.key = k; v.picks = g['picks'][:].astype(np.int64); v.n_frames = int(g['n_frames'][()])
        v.change_points = g['change_points'][:].astype(np.int64); v.frames_per_segment = g['n_frame_per_seg'][:].astype(np.int64)
        v.user_summary = g['user_summary'][:]; v.gtscore = g['gtscore'][:].astype(np.float64); out[k] = v
    return out

def stable_seed(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

def shifted_noisy_curve(curve, w, rng, rel=0.15):   # identical to repo's run_tvsum_mechanism_8e.shifted_noisy_curve
    nrm = curve / curve.sum()
    shift = int(rng.integers(-w, w + 1))
    pos = np.arange(nrm.size, dtype=np.float64) - shift
    sh = np.interp(pos, np.arange(nrm.size, dtype=np.float64), nrm, left=0.0, right=0.0)
    pert = np.maximum(0.0, sh + rng.normal(0.0, rel * float(nrm.max()), nrm.size))
    return (pert / pert.sum())

def f1(sumry, ref):
    return B.evaluate_summary(sumry, ref.reshape(1, -1) if ref.ndim == 1 else ref)['f1']

def run(widths, seed_tag):
    vids = load(); rows = []
    t0 = time.time()
    for w in widths:
        for k, v in vids.items():
            latent = np.maximum(v.gtscore[:v.picks.size], 1e-6)
            rng = np.random.default_rng(stable_seed(f"{seed_tag}::{k}::{w}"))
            ann = np.stack([shifted_noisy_curve(latent, w, rng) for _ in range(20)])
            refs = [B.canonical_summary(a, v) for a in ann]
            latent_ref = B.canonical_summary(latent / latent.sum(), v)
            mj = []; bj = []
            for h in range(20):
                tr = np.delete(ann, h, axis=0)
                m, b = B.mean_and_barycenter(tr)
                mj.append(f1(B.canonical_summary(m, v), refs[h]))
                bj.append(f1(B.canonical_summary(b, v), refs[h]))
            m, b = B.mean_and_barycenter(ann)          # all 20 -> compare to LATENT truth
            ms = B.canonical_summary(m, v); bs = B.canonical_summary(b, v)
            rows.append(dict(w=w, video=k, mean_jit=np.mean(mj), bary_jit=np.mean(bj),
                             mean_lat=f1(ms, latent_ref), bary_lat=f1(bs, latent_ref)))
        print("done width", w, "elapsed %.0fs" % (time.time() - t0), flush=True)
    return rows

if __name__ == "__main__":
    import pandas as pd, scipy.stats as st
    rows = run((0, 1, 2, 4, 8, 16, 32), "audit-latent-v1")
    df = pd.DataFrame(rows)
    out_csv = Path(__file__).resolve().parent / 'sim_latent_rows.csv'
    df.to_csv(out_csv, index=False)
    out = []
    for w, g in df.groupby('w'):
        dj = (g.bary_jit - g.mean_jit).values
        dl = (g.bary_lat - g.mean_lat).values
        se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
        t = st.t.ppf(.975, len(g) - 1)
        out.append(dict(width=w, mean_jit=g.mean_jit.mean(), bary_jit=g.bary_jit.mean(),
                        gap_vs_jittered_ref=dj.mean(), ci_j=(dj.mean() - t * se(dj), dj.mean() + t * se(dj)),
                        mean_lat=g.mean_lat.mean(), bary_lat=g.bary_lat.mean(),
                        gap_vs_latent_truth=dl.mean(), ci_l=(dl.mean() - t * se(dl), dl.mean() + t * se(dl)),
                        bary_wins_latent=int((dl > 0).sum())))
    pd.set_option('display.width', 250)
    pd.set_option('display.float_format', lambda x: f"{x:.2f}")
    print(pd.DataFrame(out).to_string(index=False))
