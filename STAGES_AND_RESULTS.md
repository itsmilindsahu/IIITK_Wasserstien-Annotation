# Project stages and result provenance

## Result correction

Earlier project stages used a generated annotation proxy. The proxy file has
been removed, its fallback generation path is disabled, and proxy-derived
numerical claims are withdrawn. Later annotation-only experiments used
authorized TVSum annotations, but did not evaluate video-conditioned
summarization with the canonical TVSum shot/knapsack protocol; their scores are
not comparable to standard video-summarization literature.

The JSON and figures under `results/real_tvsum/` are retained as historical
diagnostic artifacts only. They are not results from the current
video-conditioned benchmark and must not be used as its baseline or
state-of-the-art comparison.

## Current benchmark

The active real-data experiment is documented in the
[`README`](README.md) and implemented by `code/standard_tvsum_cv.py`. It
requires the official raw annotations and metadata, preprocessed 1024-D
features, and uses the supplied DSNet split file. It writes its own JSON and
CSV artifacts to `results/standard_tvsum_cv/`; the README results section is
generated from the saved JSON. No result is reportable until that run
completes and its sanity-check status is reviewed.

The split file is validated fold by fold for disjoint train, validation, and
test IDs. Its test sets are not presumed to be mutually disjoint; actual test
coverage is recorded in the JSON and must be disclosed when interpreting the
aggregate.

## Remaining scope

- Canonical TVSum protocol only; augmented and transfer settings are not run.
- The current ablation grid evaluates non-diffusion regressors. A
  feature-conditioned diffusion model and a valid comparison against it are
  future work.
- Results are descriptive for this run and do not establish state of the art.
