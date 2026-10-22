import csv
import json
from collections import Counter, defaultdict
import numpy as np

with open('tests/Study response data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)[3:] # skip first 3 test rows

parsed = [json.loads(r[1]) for r in rows]

print("="*60)
print(f"1. TOTAL RESPONSES ANALYZED: {len(parsed)}")
print("="*60)

testers = set(d['tester_id'] for d in parsed)
print(f"Unique participants/testers: {len(testers)}")
by_tester = defaultdict(list)
for d in parsed:
    by_tester[d['tester_id']].append(d)

for t, t_rows in by_tester.items():
    age = t_rows[0].get("age_range", "unknown")
    start = t_rows[0].get("first_video_selection", "unknown")
    print(f"  Tester {t[:8]}...: {len(t_rows)} videos submitted | Age: {age} | Started with: {start}")

print("\n" + "="*60)
print("2. OVERALL PREFERENCE (CHOSEN MODEL)")
print("="*60)
chosen_counts = Counter(d['chosen_model'] for d in parsed)
total = len(parsed)
for model in ['barycenter', 'mean', '']:
    cnt = chosen_counts[model]
    label = model if model else "Neither / No preference"
    print(f"  {label:<25}: {cnt:2d} / {total} ({cnt/total*100:5.1f}%)")

print("\n" + "="*60)
print("3. BREAKDOWN BY VIDEO ID")
print("="*60)
by_video = defaultdict(list)
for d in parsed:
    by_video[d['video_id']].append(d)

for vid, v_rows in sorted(by_video.items()):
    c = Counter(d['chosen_model'] for d in v_rows)
    v_total = len(v_rows)
    b_cnt = c['barycenter']
    m_cnt = c['mean']
    n_cnt = c['']
    print(f"Video {vid} (n={v_total}):")
    print(f"  - Barycenter: {b_cnt} ({b_cnt/v_total*100:.1f}%)")
    print(f"  - Mean:       {m_cnt} ({m_cnt/v_total*100:.1f}%)")
    print(f"  - Neither:    {n_cnt} ({n_cnt/v_total*100:.1f}%)")

print("\n" + "="*60)
print("4. LIKERT RATINGS COMPARISON (1-5 SCALE)")
print("="*60)
bary_scores = []
mean_scores = []
for d in parsed:
    is_a_bary = (d['which_model_is_A'] == 'barycenter')
    bary_score = d['likert_A'] if is_a_bary else d['likert_B']
    mean_score = d['likert_B'] if is_a_bary else d['likert_A']
    bary_scores.append(bary_score)
    mean_scores.append(mean_score)

b_arr = np.array(bary_scores, dtype=float)
m_arr = np.array(mean_scores, dtype=float)
diffs = b_arr - m_arr

print(f"Barycenter Likert: Mean = {b_arr.mean():.2f} +/- {b_arr.std():.2f} (Median: {np.median(b_arr)})")
print(f"Mean Baseline Likert: Mean = {m_arr.mean():.2f} +/- {m_arr.std():.2f} (Median: {np.median(m_arr)})")
print(f"Pairwise diff (Barycenter - Mean): Mean diff = {diffs.mean():+.2f}")
print(f"  - Barycenter rated HIGHER than Mean: {(b_arr > m_arr).sum()} / {total} ({(b_arr > m_arr).sum()/total*100:.1f}%)")
print(f"  - Barycenter rated EQUAL to Mean:    {(b_arr == m_arr).sum()} / {total} ({(b_arr == m_arr).sum()/total*100:.1f}%)")
print(f"  - Barycenter rated LOWER than Mean:  {(b_arr < m_arr).sum()} / {total} ({(b_arr < m_arr).sum()/total*100:.1f}%)")

try:
    from scipy import stats
    t_stat, p_t = stats.ttest_rel(b_arr, m_arr)
    print(f"Paired t-test: t = {t_stat:.3f}, p-value = {p_t:.4f}")
    w_stat, p_w = stats.wilcoxon(b_arr, m_arr, alternative='two-sided')
    print(f"Wilcoxon signed-rank test: W = {w_stat:.3f}, p-value = {p_w:.4f}")
except ImportError:
    pass

print("\n" + "="*60)
print("5. REASON TAGS SELECTED BY PARTICIPANTS")
print("="*60)
all_tags = [t for d in parsed for t in d.get('tags', [])]
tag_counts = Counter(all_tags)
for tag, count in tag_counts.most_common():
    print(f"  {tag:<40}: {count:2d} ({count/total*100:5.1f}%)")

print("\n" + "="*60)
print("6. QUALITATIVE COMMENTS")
print("="*60)
has_comments = 0
for idx, d in enumerate(parsed, start=4):
    txt = d.get('free_text', '').strip()
    if txt:
        has_comments += 1
        is_a_bary = (d['which_model_is_A'] == 'barycenter')
        b_score = d['likert_A'] if is_a_bary else d['likert_B']
        m_score = d['likert_B'] if is_a_bary else d['likert_A']
        print(f"Row {idx} (Video {d['video_id'][:12]}... | Chosen: {d['chosen_model']} | Bary: {b_score}/5, Mean: {m_score}/5):")
        print(f"   \"{txt}\"\n")
if not has_comments:
    print("  (No free-text comments provided in these 19 rows)")

print("="*60)
print("7. PARTICIPANT BEHAVIOR & INTEGRITY SIGNALS")
print("="*60)
skips = sum(1 for d in parsed if d.get('source_skipped'))
print(f"Source video skipped: {skips}/{total} ({skips/total*100:.1f}%)")
seen = Counter(d.get('seen_before', '') for d in parsed)
print(f"Seen before: {dict(seen)}")
first_w = Counter(d.get('first_watched', '') for d in parsed)
print(f"First watched (Side A vs B): {dict(first_w)}")
pA = [d.get('played_A_count', 0) for d in parsed]
pB = [d.get('played_B_count', 0) for d in parsed]
print(f"Average video replays: Summary A = {np.mean(pA):.2f}, Summary B = {np.mean(pB):.2f}")
