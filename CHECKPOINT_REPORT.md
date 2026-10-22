# TVSum score-scale fix checkpoints

## Checkpoint 0 — Environment and baseline reproduction

**Status: PASS.**

- The first search missed the HDF5 even though it is at the documented project default. A follow-up exact recursive search found it at `datasets/eccv16_dataset_tvsum_google_pool5.h5`.
- Feature file size: 125,126,256 bytes. SHA-256 `ec9cbcc715887d30ba2ad46764cb810f07eb6d6241e96c0eac5fca51d398228b`; matches `protocol.feature_sha256` in the prior results JSON.
- Official TSVs are available in the persistent session data directory:
  - `official_annotations.tsv`: SHA-256 `13972d16f2f05016dcbb4a62563cc74db98e829facbd8151cd0276cf47b4bd98`; matches `protocol.raw_annotation_sha256`.
  - `official_info.tsv`: SHA-256 `162e37e5d027d293d5e7877aa0b7fdac417a2e7c077b95a159334717202ea59d`; matches `protocol.official_info_sha256`.
- Environment variables were unset; explicit paths were used for the existing TSVs and project-default HDF5.
- Tests:
  - `python -m pytest -q` under the default Python 3.14 failed during collection because that interpreter has no `torch`.
  - `py -3.12 -m pytest -q`: **7 passed in 3.17s**. Python 3.12 has the required dependencies.
- Reproduction: B_mean_cross_entropy, fold 1, seed 11, 10 epochs, matching the existing run protocol. The HDF5 loader validated all 50 real video IDs and printed frame/pick/shot counts.
  - Mean across the 10 test videos: F1 `51.190645342`, Spearman `0.186325543`, Kendall `0.141849993`.
  - Saved and reproduced selected epoch: `7`.
  - Maximum absolute per-video difference against `per_video_seed_records`: `0` for each of F1, Spearman, and Kendall.
- Commands/checks: `Get-FileHash -Algorithm SHA256`; `py -3.12 -m pytest -q`; direct Python 3.12 invocation of `load_tvsum_features`, DSNet fold/validation selection, and `train_candidate` for the single arm/fold/seed.
- Elapsed time: pytest reported 3.17 seconds. The single-arm reproduction command was not timed; its exact results are recorded above.

**Checkpoint 0 result: PASS.** Data hashes match, tests pass under Python 3.12, and the requested saved-run reproduction matches exactly. No score-scale changes have been made.

## Checkpoint 1 — Prove the score-scale bug

**Status: PASS (the invariance test fails on the unfixed evaluator as intended).**

- Added `tests/test_score_scale.py`. It loads the actual `video_1` using the official HDF5 and TSVs, constructs a deterministic softmax prediction, reports the DSNet-style integer shot-score statistics, and checks summary invariance for score multipliers `1`, `10`, and `1000`.
- Command:
  - Set `TVSUM_ANNOTATION_TSV` and `TVSUM_INFO_TSV` to the official TSV files in session data.
  - `py -3.12 -m pytest -q -s tests\test_score_scale.py`
- Observed softmax prediction sum: `1.000000000`.
- For the 71 shots in `video_1`: maximum integer shot score `3`; zero-score fraction `0.296` (21/71); only `4` distinct integer scores.
- Test result: **1 passed, 1 failed**. The collapse-statistics check passed. The scale-invariance check failed as expected at multiplier `10`; this confirms that `canonical_summary` changes its chosen summary solely when the positive input scores are rescaled. The factor-`1000` check is present in the test but the loop halts on the earlier factor-`10` assertion.
- No training, production code changes, or evaluator changes were made.

**Checkpoint 1 result: PASS.** The current evaluator demonstrably violates positive-scale invariance and quantizes the softmax prediction into a tiny set of integer shot scores. The expected failing regression test must turn green only after Checkpoint 2's fix.

## Checkpoint 2 — Normalize score scale before shot expansion

**Status: PASS.**

- In `canonical_summary` only, scores are divided by their maximum when it is positive; otherwise an all-ones score vector is used. Model softmax and downstream knapsack/F1 logic are unchanged.
- The existing knapsack test had asserted which of two nearly equal segments wins based on their absolute unnormalized integer scores. Since the fix intentionally removes absolute scale, updated that test to assert the meaningful relative ranking instead.
- Scale tests:
  - `py -3.12 -m pytest -q -s tests\test_score_scale.py`: **2 passed** after the fix, including equality of summaries at scales `1`, `10`, and `1000`.
  - `py -3.12 -m pytest -q`: **7 passed in 2.79s**.
  - Note: `pytest.ini` restricts default discovery to `test_fscore_benchmark.py`; the new score-scale test was therefore run explicitly as shown above.
- Recomputed baselines through `score_video`, `human_leave_one_out`, the same five DSNet splits, and the same deterministic random seeds. Aggregation is over the same 50 fold/video test instances as the saved baseline summary:

  | Baseline | Previous F1 | Fixed evaluator F1 | Change | Within ±0.5 |
  |---|---:|---:|---:|---|
  | Uniform | 59.365274 | 59.365274 | +0.000000 | PASS |
  | Random | 56.344722 | 56.312210 | -0.032512 | PASS |
  | Human leave-one-out | 65.826036 | 65.826036 | +0.000000 | PASS |

- The prior test run that verified the scale failure printed `video_1` shot statistics; after the fix, the dedicated test passes. The three baseline checks all remain within the required tolerance.
- No result artifacts under `results/standard_tvsum_cv/` were overwritten, and no model, split, or training behavior changed.
- Elapsed time: full pytest suite reported 2.79 seconds; focused scale test reported 4.92 seconds. The baseline recomputation was not separately timed.

**Checkpoint 2 result: PASS.** Scale invariance now holds and all three baseline regressions remain within tolerance. Ready for approval before starting Checkpoint 3.

## Checkpoint 3 — Full 10-epoch ablation grid with scale fix

**Status: PASS.**

- Ran the unchanged experiment grid: six arms, five fixed seeds, five supplied DSNet folds, and 10 maximum epochs. Training and checkpoint selection used only training/validation videos; tests were not used for selection.
- Wrote only to `results/standard_tvsum_cv_fixed/results.json` and `results/standard_tvsum_cv_fixed/results_table.csv`. Used the same JSON-derived table schema. The standard results directory was not overwritten; README generation was bypassed to avoid changing the already-dirty README while running this isolated output.
- Elapsed experiment time from the saved JSON: `434.982` seconds (about 7m15s).
- Sanity checks: **PASS**. Random all-video F1 `54.482` (expected interval 50–55); shuffled-feature F1 `58.883`, below the recorded threshold `64.365`.
- Paired comparisons are computed by the runner on per-unique-test-video means, avoiding duplicate split appearances: `n=32`.
- Old vs fixed metrics (F1 in %, rank correlations unitless):

  | Arm | Old F1 | Fixed F1 | Δ F1 | Fixed seed SD | Old ρ | Fixed ρ | Old τ | Fixed τ |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | A_mean_mse | 57.163 | 60.766 | +3.602 | 2.513 | 0.155 | 0.148 | 0.118 | 0.113 |
  | B_mean_cross_entropy | 57.350 | 61.216 | +3.866 | 2.483 | 0.184 | 0.168 | 0.141 | 0.128 |
  | C_mean_cramer | 56.696 | 60.283 | +3.587 | 2.949 | 0.143 | 0.135 | 0.109 | 0.103 |
  | D_barycenter_mse | 57.032 | 60.371 | +3.339 | 2.694 | 0.122 | 0.116 | 0.093 | 0.088 |
  | E_barycenter_cramer | 56.839 | 60.600 | +3.760 | 2.923 | 0.142 | 0.144 | 0.108 | 0.109 |
  | F_blended_cramer | 56.646 | 60.549 | +3.904 | 2.866 | 0.141 | 0.144 | 0.107 | 0.109 |

- Every arm's F1 change exceeds that arm's fixed-run mean seed SD. Spearman and Kendall changes are each less than `0.03` in absolute value for every arm.
- Selected-epoch counts, one count per fold × seed (25 candidates per arm):

  | Arm | Old epoch counts | Fixed epoch counts |
  |---|---|---|
  | A_mean_mse | 1:7, 2:5, 3:1, 4:4, 5:1, 6:3, 7:2, 8:1, 10:1 | 1:3, 2:7, 3:4, 4:3, 5:4, 6:1, 8:3 |
  | B_mean_cross_entropy | 2:4, 3:1, 4:4, 5:3, 7:3, 8:3, 9:2, 10:5 | 1:8, 3:4, 4:3, 5:3, 6:2, 7:2, 8:1, 9:1, 10:1 |
  | C_mean_cramer | 1:1, 2:4, 3:4, 4:4, 5:4, 6:2, 7:1, 8:1, 10:4 | 1:4, 2:5, 3:6, 5:3, 6:3, 7:1, 8:2, 9:1 |
  | D_barycenter_mse | 1:6, 2:2, 3:4, 4:4, 5:2, 6:2, 7:2, 8:1, 9:1, 10:1 | 1:6, 2:5, 3:2, 5:4, 6:3, 8:1, 9:3, 10:1 |
  | E_barycenter_cramer | 1:1, 2:5, 3:2, 4:4, 5:1, 6:3, 7:3, 8:3, 9:2, 10:1 | 1:2, 2:3, 3:3, 4:4, 5:6, 6:2, 7:2, 8:1, 10:2 |
  | F_blended_cramer | 1:1, 2:5, 3:2, 4:6, 5:2, 6:2, 7:4, 9:1, 10:2 | 1:3, 2:3, 3:5, 4:1, 5:4, 6:2, 7:2, 8:2, 9:2, 10:1 |

- Best fixed arm is B_mean_cross_entropy at F1 `61.216`.
  - Versus uniform: Δ `+2.015` F1 points, paired t `p=0.08136`, Wilcoxon `p=0.07413`, `n=32`; **not statistically significant at 0.05**.
  - Versus random: Δ `+4.607` F1 points, paired t `p=9.67e-08`, Wilcoxon `p=2.00e-06`, `n=32`.
  - Therefore the corrected model is clearly above random, but this run does **not** establish that it beats uniform with paired statistical support.

**Checkpoint 3 result: PASS.** Full corrected run completed and artifacts were written separately. No errors or sanity-check failures. Do not infer superiority over uniform from the mean difference alone.

## Checkpoint 4 — Longer training for arms A and B

**Status: PASS.**

- Added `code/run_tvsum_long.py`, a dedicated runner for A_mean_mse and B_mean_cross_entropy, with `--epochs` accepting one or more budgets (default: `30 60 100`). It reuses the same feature scaler, official train/validation/test folds, deterministic seeds, model, optimizer, losses, validation F1 checkpoint selection, and test scoring. Only validation videos choose the checkpoint; test videos are scored after selection.
- `train_candidate` accepts an optional validation-history list so one curve can be recorded without changing its checkpoint selection or training behavior.
- Validation before the run: `py -3.12 -m py_compile code\standard_tvsum_cv.py code\run_tvsum_long.py`; `py -3.12 -m pytest -q` (**7 passed in 3.26s**).
- Run command: `py -3.12 code\run_tvsum_long.py --epochs 30 60 100 --output-dir results\standard_tvsum_cv_long --torch-threads 4` with `TVSUM_ANNOTATION_TSV` and `TVSUM_INFO_TSV` set to the verified official TSVs.
- Output files:
  - `results/standard_tvsum_cv_long/epochs_30.json`
  - `results/standard_tvsum_cv_long/epochs_60.json`
  - `results/standard_tvsum_cv_long/epochs_100.json`
  - `results/standard_tvsum_cv_long/long_training_summary.csv`
  - `results/standard_tvsum_cv_long/long_training_summary.json`
  - `results/standard_tvsum_cv_long/validation_f1_fold1_seed11.png`
- The PNG was read back as a valid 1280×800 PNG. It plots B_mean_cross_entropy validation F1 for fold 1, seed 11 at all three budgets.
- Total elapsed time: `1288.561` seconds. Per setting: 30 epochs `211.644s`; 60 epochs `419.111s`; 100 epochs `656.485s`.
- Each row is mean F1 over supplied fold/video test instances; seed SD is the mean within-fold/video sample SD across the five seeds. Correlations are means over test fold/video instances.

  | Budget | Arm | F1 mean ± seed SD | Spearman ρ | Kendall τ | Δ F1 vs fixed-10 |
  |---:|---|---:|---:|---:|---:|
  | 10 (fixed run) | A_mean_mse | 60.766 ± 2.513 | 0.148 | 0.113 | — |
  | 10 (fixed run) | B_mean_cross_entropy | 61.216 ± 2.483 | 0.168 | 0.128 | — |
  | 30 | A_mean_mse | 60.832 ± 2.723 | 0.150 | 0.114 | +0.067 |
  | 30 | B_mean_cross_entropy | 61.206 ± 2.416 | 0.166 | 0.127 | -0.010 |
  | 60 | A_mean_mse | 60.869 ± 2.824 | 0.153 | 0.116 | +0.104 |
  | 60 | B_mean_cross_entropy | 61.011 ± 2.504 | 0.164 | 0.125 | -0.205 |
  | 100 | A_mean_mse | 60.949 ± 2.715 | 0.155 | 0.118 | +0.183 |
  | 100 | B_mean_cross_entropy | 60.949 ± 2.662 | 0.164 | 0.125 | -0.266 |

- Paired differences from fixed-10 F1 were computed after averaging repeated split/seed observations per unique test video (`n=32`):

  | Budget | Arm | Mean paired Δ F1 vs fixed-10 | Paired t p | Wilcoxon p |
  |---:|---|---:|---:|---:|
  | 30 | A_mean_mse | +0.123 | 0.6075 | 0.4213 |
  | 30 | B_mean_cross_entropy | -0.068 | 0.7625 | 0.8661 |
  | 60 | A_mean_mse | +0.150 | 0.5255 | 0.6536 |
  | 60 | B_mean_cross_entropy | -0.253 | 0.1876 | 0.3589 |
  | 100 | A_mean_mse | +0.203 | 0.4268 | 0.5188 |
  | 100 | B_mean_cross_entropy | -0.375 | 0.2039 | 0.5617 |

- Longer training does **not** improve beyond observed noise in this run: every paired comparison vs fixed-10 has p>0.05, and all mean-F1 changes are much smaller than the per-arm seed SD. At 100 epochs, A is only `+0.183` points and B is `-0.266` points versus 10 epochs.
- Selected epoch distributions (count out of 25 fold × seed checkpoints):

  | Budget | Arm | Selected epoch counts |
  |---:|---|---|
  | 30 | A_mean_mse | 1:2, 2:5, 5:1, 6:1, 11:1, 16:2, 17:2, 19:1, 20:2, 21:1, 22:2, 26:3, 27:1, 29:1 |
  | 30 | B_mean_cross_entropy | 1:7, 3:1, 4:2, 6:2, 7:2, 9:1, 13:1, 14:1, 16:3, 19:1, 20:1, 22:1, 23:1, 27:1 |
  | 60 | A_mean_mse | 1:2, 2:4, 5:1, 11:1, 16:2, 17:2, 19:1, 20:1, 22:1, 26:1, 27:1, 33:1, 37:1, 38:1, 39:1, 40:1, 42:1, 50:1, 51:1 |
  | 60 | B_mean_cross_entropy | 1:7, 3:1, 4:1, 6:2, 7:1, 9:1, 13:1, 14:1, 16:2, 19:1, 31:1, 33:1, 37:1, 39:1, 46:1, 52:1, 59:1 |
  | 100 | A_mean_mse | 1:2, 2:3, 5:1, 11:1, 16:1, 17:2, 19:1, 22:1, 26:1, 27:1, 38:1, 39:1, 40:1, 42:1, 66:1, 74:1, 79:1, 86:1, 90:1, 95:1, 98:1 |
  | 100 | B_mean_cross_entropy | 1:6, 4:1, 6:2, 7:1, 9:1, 13:1, 14:1, 16:2, 19:1, 33:1, 37:1, 46:1, 52:1, 59:1, 75:1, 83:2, 97:1 |

- None of the 25 checkpoints per arm selected exactly the final epoch at any budget, so selection is not almost always pinned to the cap. The 100-epoch selections are mixed: 6/25 for A and 11/25 for B are before epoch 10; several A checkpoints select late epochs (including 98). This indicates heterogeneous validation behavior, not a uniform training-length benefit.
- For the plotted B/fold-1/seed-11 curve, validation selected epoch 1 at budgets 30 and 60, but epoch 83 at budget 100; the last-epoch validation F1 at 100 was below its selected peak. The 30/60/100 curves share identical prefixes, confirming deterministic training for this sample.
- Initial execution note: a first 30-epoch attempt completed training but stopped before saving because the new summary helper encountered precision/recall records while aggregating the three requested metrics. The helper was corrected to aggregate only F1/Spearman/Kendall, and the complete final 30/60/100 run was rerun and saved successfully. The failed attempt produced no result artifact.

**Checkpoint 4 result: PASS.** All requested long runs, summaries, epoch distributions, and plot are saved. Longer training does not provide a statistically supported gain over fixed-10 in this run. Stop here before any Checkpoint 5/architecture decision.

## Checkpoint 5 — Decision gate before adding a BiLSTM

**Status: COMPLETE; awaiting explicit approval before any BiLSTM implementation or run.**

- Comparison source files:
  - Old run: `results/standard_tvsum_cv_buggy_scale/results.json`.
  - Fixed 10-epoch run: `results/standard_tvsum_cv_fixed/results.json`.
  - Best long-training candidate: `results/standard_tvsum_cv_long/epochs_30.json`, B_mean_cross_entropy (`61.206` F1). It edges the 60/100-epoch candidates but is effectively tied with fixed-10 B (`61.216`).
  - Corrected baselines: fixed-run JSON, whose baseline outputs were regression-checked against the old run in Checkpoint 2.
- F1 values below are means on the supplied split/video test instances. `Δ R/U/H` means model F1 minus random / uniform / human leave-one-out F1. Paired significance uses unique-video means (`n=32`); old p-values are from its saved paired-test results, and corrected/long p-values were computed by pairing saved per-video model values with fixed-run baseline records.

  | Stage / method | F1 | Δ random | Δ uniform | Δ human |
  |---|---:|---:|---:|---:|
  | Old buggy B_mean_cross_entropy | 57.350 | -0.619 | -3.179 | -9.493 |
  | Fixed, 10 ep B_mean_cross_entropy | 61.216 | +4.607 | +2.015 | -4.299 |
  | Best long candidate, 30 ep B_mean_cross_entropy | 61.206 | +4.539 | +1.947 | -4.367 |
  | Random reference (old / corrected) | 56.345 / 56.312 | — | — | — |
  | Uniform reference | 59.365 | — | — | — |
  | Human leave-one-out reference | 65.826 | — | — | — |

- Paired tests, model versus uniform (`p_t / p_Wilcoxon`):
  - Old buggy B: `0.1604 / 0.0628`, `n=32`.
  - Fixed-10 B: `0.0814 / 0.0741`, `n=32`.
  - Best long B at 30 epochs: `0.0951 / 0.1141`, `n=32`.
- **Conclusion:** neither the corrected 10-epoch model nor the best longer-training candidate beats uniform with statistical support at the 0.05 level. Both are significantly above random on these paired tests, but remain below the human leave-one-out reference. Longer training did not improve meaningfully over fixed-10.
- All model-vs-baseline comparison p-values used unique test-video averages to avoid treating repeated appearances in overlapping DSNet folds as independent; the supplied split file still has only 32 unique test videos of 50.
- No BiLSTM code or experiment was started.

**Decision gate:** The user approved proceeding with the BiLSTM. The matched comparison is recorded in Checkpoint 6 below.

## Checkpoint 6 — Matched BiLSTM comparison on B_mean_cross_entropy

**Status: PASS.**

- Implemented the requested 2-layer bidirectional LSTM (hidden size 64 per direction) behind the existing `--backbone` option. The 8 existing tests passed before this run.
- An initial full six-arm BiLSTM grid was stopped after approximately 23 minutes because it had not completed even the first fold/seed and had no result JSON. No partial-grid metrics are reported. To answer the architecture question within practical CPU limits, ran the established best arm, B_mean_cross_entropy, for both backbones under the same five DSNet splits, five seeds, 10-epoch budget, feature scaler, train-only target construction, and validation-only checkpoint selection.
- The comparison run used CPU-only PyTorch with one thread. For the fresh temporal-convolution run, every F1 record matched the saved fixed-10 B run exactly (`250/250`, max absolute difference `0`). All training/validation/test partitions were printed, asserted disjoint, and saved.
- Inputs verified by SHA-256 against the official benchmark inputs:
  - Feature HDF5: `ec9cbcc715887d30ba2ad46764cb810f07eb6d6241e96c0eac5fca51d398228b`.
  - Annotation TSV: `13972d16f2f05016dcbb4a62563cc74db98e829facbd8151cd0276cf47b4bd98`.
  - Metadata TSV: `162e37e5d027d293d5e7877aa0b7fdac417a2e7c077b95a159334717202ea59d`.
- Run elapsed time: `2632.664` seconds (43m 53s). Results were saved to `results/standard_tvsum_cv_bilstm/backbone_comparison.json` and `results/standard_tvsum_cv_bilstm/backbone_comparison.csv`; the original corrected and long-run result directories remain untouched.

  | Backbone | F1 mean | Mean seed SD | Spearman ρ | Kendall τ |
  |---|---:|---:|---:|---:|
  | Temporal convolution | 61.216 | 2.483 | 0.1682 | 0.1281 |
  | BiLSTM | 61.860 | 2.487 | 0.1992 | 0.1520 |

- Paired tests use per-unique-test-video means (averaged across seeds and repeated appearances in overlapping folds), `n=32`. Differences are BiLSTM minus temporal convolution:

  | Metric | Mean difference | Paired t p | Wilcoxon p |
  |---|---:|---:|---:|
  | F1 | +0.477 points | 0.1531 | 0.5068 |
  | Spearman ρ | +0.0313 | 0.0000344 | 0.0000354 |
  | Kendall τ | +0.0240 | 0.0000268 | 0.0000254 |

- **Conclusion:** The BiLSTM has higher mean F1, but the F1 difference is not statistically supported by either paired test. Its rank correlations are higher under these paired tests; these p-values are nominal (no multiple-comparison correction). This is evidence for better rank association in this run, not a supported summarization-F1 improvement. This comparison covers only arm B, not the complete ablation grid.
- Post-processing verification: saved JSON contains 2,500 per-record metric values, 5 fold partitions, all 25 candidates per backbone, and finite paired statistics for F1, Spearman, and Kendall. Script compilation and saved-result integrity checks passed. The generic `runTests` helper did not discover the provided files; explicit Python 3.12 pytest invocation passed: `10 passed in 4.20s`.

**Checkpoint 6 result: PASS.** Matched results and paired comparisons are saved. Stop here; no additional model or tuning stage has been started.

## Checkpoint 7 — Housekeeping

**Status: PASS.**

- Added `test_dataset_loader.py` and `tests/` alongside the existing benchmark test module in `pytest.ini` test paths, so the default command discovers the real-data scale test as well as the data-loader tests. Ran with the official annotation/info TSV environment paths set:
  - `py -3.12 -m pytest -q`
  - **13 passed in 3.67s.**
- Renamed `results/standard_tvsum_cv/` to `results/standard_tvsum_cv_buggy_scale/` without overwriting an existing destination. Added `results/standard_tvsum_cv_buggy_scale/README.txt` identifying the unnormalized-softmax scale issue and marking those numbers **INVALID for comparison**. Corrected and long-run results were not changed.
- Regenerated the marked README results section using `build_readme_results_section` and `results/standard_tvsum_cv_fixed/results.json`. The source JSON SHA-256 was `3f02973c5f557e786f9a9d5e7320e2df5dc7d02b3d3eac9e0a559b05d1aad123`. The regenerated section now cites that fixed-run JSON and reports its values. Removed the duplicate static heading immediately before the generated section; updated README test instructions to use the full default suite.
- Grep audit (production code; lines are the post-change line numbers):

  | File / lines | Finding | Affected by the softmax integer-quantization bug? |
  |---|---|---|
  | `code/standard_tvsum_cv.py:447` | The only production `int(1000 * ...)` conversion found; it is inside `canonical_summary` (`:430`). | **Previously yes; now fixed.** `canonical_summary` divides by its positive maximum at `:435-436` before frame expansion and integer conversion. The softmax outputs at `:83` and `:130` reach validation through `:631` and test scoring through `score_video` at `:509`; both use this corrected normalizer. Old saved results are invalid; fixed/long results use the correction. |
  | `tests/test_score_scale.py:59` | Test helper intentionally reproduces the former integer conversion to measure collapse on a softmax-style vector. | **No.** Diagnostic-only test helper; not a production evaluator. |
  | `code/compare_downstream_metrics.py:103-156, 170` | Separate, noncanonical interval-knapsack evaluator. It uses unquantized interval means and stable argsort fallback, not `int(1000 * score)`. | **No evidence of this bug.** Its selection is based on relative score ordering/value under linear knapsack; the implementation is not the canonical TVSum `change_points` evaluator. |
  | `stages/stage4_diffusion_benchmark/compare_downstream_metrics.py:43-70` | Separate top-15%-by-`argsort` frame evaluator. Its run constructs softmax annotator distributions at `:99` for training targets; evaluation ranks generated outputs. | **No.** It ranks scores directly and does not quantize them to integers. It is a different, noncanonical F1 protocol. |

- No additional production `int(1000 * ...)` occurrences or other `canonical_summary` definitions/call sites were found in `code/`, `stages/`, or root Python scripts. The standard TVSum canonical helper and the two separately described rank/knapsack evaluators are the only relevant summary-selection implementations found. Softmax uses in plotting, training, and distribution-preparation scripts do not feed another integer-quantized canonical summary path in this grep audit.
- No audit finding outside the standard evaluator was changed in this checkpoint.

**Checkpoint 7 result: PASS.** Test discovery, old-results quarantine, fixed-JSON README regeneration, and the requested summary-scale audit are complete. Stop here; Checkpoints 8–10 have not been started.

## Checkpoint 8 — Leave-one-annotator-out consensus-target oracle

**Status: PASS.**

- No model training or test-fold selection was performed. For each of the 50 real TVSum videos, normalized raw scores at HDF5 `picks` were used to form a mean distribution and a 1D Wasserstein barycenter from the 19 non-held-out annotators. Each target was passed to the fixed `canonical_summary`; F1 was evaluated only against the held-out user's original-frame `user_summary`, then averaged over the 20 held-out users to obtain one paired score per video.
- Disagreement is the mean pairwise Spearman rho across the 20 annotator score vectors aligned at `picks`. Videos were sorted by this value ascending (low rho = high disagreement), with video ID as a deterministic tie-breaker, then divided into near-equal tertiles of 17 / 17 / 16 videos.
- Saved full per-video and per-annotator scores, tertile membership, protocol and hashes to `results/consensus_oracle/results.json`; saved the overall and tertile table to `results/consensus_oracle/results_table.csv`. Existing results were not overwritten.
- Overall results (paired tests on the 50 per-video mean-F1 values; difference is barycenter minus mean):

  | Group | n videos | Mean-target F1 | Barycenter-target F1 | Difference | Paired t p | Wilcoxon p |
  |---|---:|---:|---:|---:|---:|---:|
  | All videos | 50 | 62.621 | 60.681 | -1.940 | 0.00113 | 0.000475 |
  | High disagreement (lowest mean pairwise rho) | 17 | 57.303 | 56.078 | -1.225 | 0.0550 | 0.0638 |
  | Middle disagreement | 17 | 64.040 | 63.005 | -1.035 | 0.4276 | 0.3941 |
  | Low disagreement (highest mean pairwise rho) | 16 | 66.764 | 63.102 | -3.662 | 0.000505 | 0.000427 |

- **Interpretation:** On this leave-one-out oracle protocol, the arithmetic mean target has higher average F1 than the barycenter overall, with both paired tests below 0.05. Among tertiles, a paired difference is supported only in the low-disagreement/high-rho group; neither test supports a difference in the other two tertiles at the 0.05 threshold. P-values are unadjusted for the three subgroup comparisons. These oracle results are annotation-consensus diagnostics, not learned model performance or a deployable baseline.
- The run validated all 50 distinct video IDs, 20 held-out scores per method per video, a disjoint complete 17/17/16 partition, four CSV rows, and finite paired-test values. Runtime: `10.406` seconds. Full default tests with real-data environment paths: **13 passed in 3.62s**; script compilation and `git diff --check` passed.

**Checkpoint 8 result: PASS.** Oracle JSON/table are saved. Stop here; Checkpoints 9 and 10 have not been started.

## Checkpoint 8b — Barycenter implementation-artifact checks

**Status: PASS.**

- Reused the Checkpoint 8 leave-one-out protocol: for each held-out user, the other 19 raw annotator vectors aligned to picks form the target; score against only that user's original-frame `user_summary`; average the 20 held-out F1/correlation/tolerance values per video; paired analysis over the 50 videos. No model training or fold selection.
- Tested nine variants: mean, existing barycenter, 4x-finer-quantile barycenter aggregated by linear mass assignment into the original pick bins, barycenter smoothed at Gaussian sigma 1/2/4 picks, and mean smoothed at the same sigmas as controls. Gaussian smoothing uses `mode="reflect"` and is renormalized to unit mass.
- Tolerance F1 is defined on summary values sampled at `picks`: predicted precision counts selected predicted picks having any held-out selected pick within ±k pick indices; recall applies the converse to held-out selected picks; harmonic mean is reported as percent. A toy test first checked radius 0 versus radius 1 and an unequal precision/recall case (80% expected); it passed before real-video scoring.
- Correlations compare the consensus score at each pick with that held-out annotator's raw score at the same pick. Values below average first over 20 users per video, then across videos.
- Holm adjustment covers all **eight non-mean comparisons**, separately for paired t and Wilcoxon p-values and separately within the overall set and each tertile. “Raw” is unadjusted; “Holm” is adjusted. No adjustment is made across the four group families. Full per-video/user values and each group/variant's correlation and tolerance metrics are saved, not just the aggregates.
- Results saved under the fresh `results/consensus_oracle_8b/` directory:
  - `results.json` — protocol, hashes, per-video/per-annotator metrics, groups, and raw/Holm tests.
  - `variant_metrics.csv` — nine variants' F1, Spearman, Kendall, and tolerance F1 for k=2/5/10 in overall and each tertile.
  - `paired_tests.csv` — all paired F1 contrasts, differences, and raw/Holm p-values.
- Exact reproduction check against Checkpoint 8: maximum per-video F1 difference was `0.0` for both original mean and barycenter targets. SHA-256 fingerprints for the features and both TSV inputs matched Checkpoint 8.
- Overall variant metrics (F1/tolerance F1 in percent; correlations unitless):

  | Variant | F1 | Spearman ρ | Kendall τ | Tolerance F1 k=2 | k=5 | k=10 |
  |---|---:|---:|---:|---:|---:|---:|
  | Mean | 62.621 | 0.3950 | 0.3088 | 69.262 | 76.589 | 81.905 |
  | Existing barycenter | 60.681 | 0.2437 | 0.1919 | 67.955 | 76.218 | 81.830 |
  | Barycenter, 4x quantile grid | 60.614 | 0.2646 | 0.2046 | 67.878 | 76.112 | 81.757 |
  | Barycenter, Gaussian σ=1 | 60.571 | 0.2741 | 0.2104 | 67.867 | 76.156 | 81.812 |
  | Barycenter, Gaussian σ=2 | 60.652 | 0.2821 | 0.2168 | 67.908 | 76.122 | 81.776 |
  | Barycenter, Gaussian σ=4 | 60.736 | 0.2890 | 0.2223 | 67.995 | 76.171 | 81.752 |
  | Mean, Gaussian σ=1 (smoothing control) | 62.703 | 0.3934 | 0.3062 | 69.280 | 76.602 | 81.895 |
  | Mean, Gaussian σ=2 (smoothing control) | 62.993 | 0.3889 | 0.3024 | 69.502 | 76.757 | 82.004 |
  | Mean, Gaussian σ=4 (smoothing control) | 63.013 | 0.3767 | 0.2923 | 69.772 | 77.276 | 82.517 |

- Tertile F1 means:

  | Variant | High disagreement (n=17) | Middle (n=17) | Low disagreement (n=16) |
  |---|---:|---:|---:|
  | Mean | 57.303 | 64.040 | 66.764 |
  | Existing barycenter | 56.078 | 63.005 | 63.102 |
  | Barycenter, 4x quantile grid | 55.949 | 62.906 | 63.136 |
  | Barycenter, Gaussian σ=1 | 55.895 | 62.811 | 63.160 |
  | Barycenter, Gaussian σ=2 | 55.923 | 62.914 | 63.274 |
  | Barycenter, Gaussian σ=4 | 56.295 | 63.273 | 62.760 |
  | Mean, Gaussian σ=1 | 57.421 | 64.505 | 66.400 |
  | Mean, Gaussian σ=2 | 58.069 | 65.161 | 65.920 |
  | Mean, Gaussian σ=4 | 58.801 | 64.429 | 65.982 |

- Paired F1 differences and p-values; difference is variant minus mean, so a negative value favors mean F1. In each test column, `raw → Holm` gives unadjusted then adjusted p-value:

  | Group | Variant | Δ F1 | Paired t raw → Holm | Wilcoxon raw → Holm |
  |---|---|---:|---:|---:|
  | Overall | Barycenter | -1.940 | 0.001127 → 0.006039 | 0.000475 → 0.002435 |
  | Overall | Barycenter 4x | -2.007 | 0.000713 → 0.005701 | 0.000403 → 0.002435 |
  | Overall | Barycenter σ=1 | -2.050 | 0.000733 → 0.005701 | 0.000300 → 0.002401 |
  | Overall | Barycenter σ=2 | -1.969 | 0.001006 → 0.006039 | 0.000348 → 0.002435 |
  | Overall | Barycenter σ=4 | -1.885 | 0.002096 → 0.008383 | 0.001241 → 0.004965 |
  | Overall | Mean σ=1 | +0.082 | 0.601270 → 0.836216 | 0.637410 → 0.738437 |
  | Overall | Mean σ=2 | +0.372 | 0.278739 → 0.836216 | 0.234761 → 0.704283 |
  | Overall | Mean σ=4 | +0.392 | 0.329133 → 0.836216 | 0.369219 → 0.738437 |
  | High disagreement | Barycenter | -1.225 | 0.055028 → 0.220112 | 0.063828 → 0.255310 |
  | High disagreement | Barycenter 4x | -1.354 | 0.030216 → 0.172298 | 0.026672 → 0.160034 |
  | High disagreement | Barycenter σ=1 | -1.409 | 0.028716 → 0.172298 | 0.030518 → 0.160034 |
  | High disagreement | Barycenter σ=2 | -1.380 | 0.021036 → 0.147249 | 0.015768 → 0.110375 |
  | High disagreement | Barycenter σ=4 | -1.008 | 0.073651 → 0.220952 | 0.072043 → 0.255310 |
  | High disagreement | Mean σ=1 | +0.118 | 0.724796 → 0.724796 | 0.522198 → 0.522198 |
  | High disagreement | Mean σ=2 | +0.766 | 0.156825 → 0.313651 | 0.118251 → 0.255310 |
  | High disagreement | Mean σ=4 | +1.498 | 0.004287 → 0.034298 | 0.006970 → 0.055760 |
  | Middle disagreement | Barycenter | -1.035 | 0.427616 → 1.000000 | 0.394100 → 1.000000 |
  | Middle disagreement | Barycenter 4x | -1.134 | 0.387299 → 1.000000 | 0.343692 → 1.000000 |
  | Middle disagreement | Barycenter σ=1 | -1.229 | 0.362951 → 1.000000 | 0.320106 → 1.000000 |
  | Middle disagreement | Barycenter σ=2 | -1.125 | 0.411020 → 1.000000 | 0.368360 → 1.000000 |
  | Middle disagreement | Barycenter σ=4 | -0.767 | 0.565640 → 1.000000 | 0.507442 → 1.000000 |
  | Middle disagreement | Mean σ=1 | +0.465 | 0.099037 → 0.792295 | 0.066213 → 0.529701 |
  | Middle disagreement | Mean σ=2 | +1.122 | 0.111126 → 0.792295 | 0.123403 → 0.863818 |
  | Middle disagreement | Mean σ=4 | +0.390 | 0.645101 → 1.000000 | 0.758181 → 1.000000 |
  | Low disagreement | Barycenter | -3.662 | 0.000505 → 0.003535 | 0.000427 → 0.003418 |
  | Low disagreement | Barycenter 4x | -3.628 | 0.000512 → 0.003535 | 0.001758 → 0.010546 |
  | Low disagreement | Barycenter σ=1 | -3.604 | 0.000725 → 0.003535 | 0.001472 → 0.010306 |
  | Low disagreement | Barycenter σ=2 | -3.490 | 0.000637 → 0.003535 | 0.001758 → 0.010546 |
  | Low disagreement | Barycenter σ=4 | -4.004 | 0.000317 → 0.002534 | 0.002093 → 0.010546 |
  | Low disagreement | Mean σ=1 | -0.364 | 0.019347 → 0.058041 | 0.035904 → 0.107712 |
  | Low disagreement | Mean σ=2 | -0.844 | 0.094622 → 0.189244 | 0.213835 → 0.427670 |
  | Low disagreement | Mean σ=4 | -0.782 | 0.247239 → 0.247239 | 0.313220 → 0.427670 |

- **Interpretation under the adjusted paired tests:** All five barycenter forms (existing, finer grid, and three smoothing levels) have lower F1 than the mean overall, with both Holm-adjusted test families below 0.05. Thus the tested grid/smoothing variants do not remove the observed overall gap; the gap is not explained by the original quantile-grid resolution or by the tested smoothing. The low-disagreement group also has both adjusted tests below 0.05 for all five barycenter forms. No barycenter variant has both adjusted tests below 0.05 in the high- or middle-disagreement groups. The mean-smoothed controls have no overall adjusted paired evidence of a difference from mean. In high-disagreement, mean σ=4 has adjusted t p below 0.05 but adjusted Wilcoxon p above 0.05, so the two tests do not jointly support that contrast. No claim is made from correlations or tolerance-F1 point estimates alone; no paired tests for those metrics were requested.
- Runtime: `29.940` seconds. All 50 videos and 20 held-out annotators/video were included. `test_consensus_oracle_8b.py`: **4 passed**; full default suite with official real-data paths: **17 passed in 3.82s**. Result integrity checks verified all variant/user record counts, all 32 group/variant F1 contrasts, finite raw/adjusted p-values, and CSV row counts (36 metric rows and 32 paired-test rows). Script compilation and `git diff --check` passed.

**Checkpoint 8b result: PASS.** Variant, correlation, tolerance-F1, and raw/Holm paired-test tables are saved. Stop here; Checkpoints 8c and 9 have not been started.

## Checkpoint 8d — Secondary metrics, continuous disagreement, and qualitative checks

**Status: PASS.**

- Used `results/consensus_oracle_8b/results.json` for saved per-video targets' metrics, F1 differences, and high-disagreement IDs. Reused only the already-local HDF5 and official annotation/info TSVs to reconstruct distributions/curves absent from the 8b JSON. No downloads, new dataset, or model training. Input SHA-256 values matched those recorded in 8b, and every reconstructed per-video/per-annotator F1 matched 8b exactly (maximum absolute difference `0.0`).
- **Secondary paired metrics (n=50):** paired tests compare each of the eight non-mean variants with mean, separately for Spearman, Kendall, and tolerance F1 at k=2/5/10. Holm correction is across the eight variants within each metric; no correction is applied across the five metric families. Exact per-variant differences and paired t/Wilcoxon raw and Holm-adjusted p-values are in `results/consensus_oracle_8d/secondary_paired_tests.csv`.

  | Metric | Mean estimate | Existing barycenter estimate | Variants with both Holm-adjusted paired tests <0.05 |
  |---|---:|---:|---|
  | Spearman ρ | 0.395029 | 0.243684 | Barycenter, 4x barycenter, and all three smoothed barycenters have lower values; mean smoothing σ=2/4 also differs from unsmoothed mean |
  | Kendall τ | 0.308813 | 0.191884 | All eight non-mean variants differ from mean; all differences are negative |
  | Tolerance F1, k=2 | 69.262 | 67.955 | All five barycenter variants have lower values |
  | Tolerance F1, k=5 | 76.589 | 76.218 | None |
  | Tolerance F1, k=10 | 81.905 | 81.830 | None |

- Example raw → Holm-adjusted p-values from the paired barycenter-vs-mean contrasts (t / Wilcoxon):
  - Spearman: `1.1329464e-22 → 9.063571e-22` / `1.7763568e-15 → 1.4210855e-14`.
  - Kendall: `2.9463109e-22 → 2.3570488e-21` / `1.7763568e-15 → 1.4210855e-14`.
  - Tolerance F1 k=2: `0.00638168 → 0.03190841` / `0.00182066 → 0.00910330`.
  - Tolerance F1 k=5: `0.383825 → 1.000000` / `0.137117 → 0.741673`.
  - Tolerance F1 k=10: `0.856404 → 1.000000` / `1.000000 → 1.000000`.
- **Continuous disagreement:** bootstrap percentile CIs use 10,000 resamples over 50 videos. Disagreement is the saved mean pairwise annotator Spearman ρ. The observed association of disagreement with barycenter-minus-mean F1 gap is Spearman `-0.28855`, 95% CI `[-0.51458, -0.02220]`. For mean-σ4-minus-mean F1 gap it is `-0.29624`, 95% CI `[-0.52194, -0.02609]`. Both intervals exclude zero; these are associations, not causal effects. The two scatterplots are saved as `disagreement_vs_barycenter_gap.png` and `disagreement_vs_mean_sigma4_gap.png`.
- **W2 characterization only; barycenter minimizes W2 by construction:** exact discrete 1D W2 on the shared pick-index support was calculated for each leave-one-out consensus against its held-out annotator's normalized distribution, then averaged over users per video. Across 50 videos, mean-target average W2 is `16.096935`, barycenter average W2 is `15.983935`, difference barycenter-minus-mean `-0.113000`. Paired t raw/Holm p=`1.2232146e-09 / 1.2232146e-09`; Wilcoxon raw/Holm p=`1.9540e-13 / 1.9540e-13` (single contrast, so one-test Holm adjustment is unchanged). This is characterization only and not evidence of downstream summary superiority.
- **Qualitative figures:** selected by mean-minus-barycenter F1 gap: largest, median, and smallest among 50 saved per-video gaps. They plot the mean and barycenter consensus curves, their canonical-summary selected-pick regions, and held-out annotator curves 1, 10, and 20. The selected videos/gaps from the saved artifact are:

  | Selection | Video ID | Mean F1 | Barycenter F1 | Mean − barycenter |
  |---|---|---:|---:|---:|
  | Largest barycenter loss | `i3wAGJaaktw` | 74.410 | 64.329 | +10.081 |
  | Median barycenter loss | `98MoyGZKHXc` | 64.531 | 62.516 | +2.015 |
  | Smallest barycenter loss | `uGu_10sucQo` | 41.838 | 51.800 | -9.962 |

  The corresponding PNGs are `qualitative_largest_barycenter_loss_i3wAGJaaktw.png`, `qualitative_median_barycenter_loss_98MoyGZKHXc.png`, and `qualitative_smallest_barycenter_loss_uGu_10sucQo.png`. These selected examples are descriptive, not inferential.
- **Exploratory high-disagreement smoothing controls:** gain is smoothed-mean F1 minus mean F1 over the 17 videos in the predeclared high-disagreement group; CIs are 10,000-resample percentile bootstrap CIs over videos. Mean σ=1: gain `+0.118`, CI `[-0.534, +0.708]`, improved on `8/17`; σ=2: `+0.766`, CI `[-0.241, +1.740]`, improved `11/17`; σ=4: `+1.498`, CI `[+0.657, +2.348]`, improved `12/17`. These are explicitly exploratory, use no additional sigma values, and are not used to select/tune a method.
- Saved under the new `results/consensus_oracle_8d/` directory: `results.json`, `secondary_paired_tests.csv`, `continuous_and_exploratory_cis.csv`, `w2_characterization.csv`, and five PNG figures. Runtime was `31.595` seconds. All images were opened and verified as PNGs; JSON/test counts and Holm p-value ordering were checked. Full default test suite with official data paths: **20 passed** (14 Matplotlib/Pyparsing deprecation warnings); Python compilation and `git diff --check` passed.

**Checkpoint 8d result: PASS.** Secondary tables, bootstrap CIs, W2 characterization, and qualitative/scatter figures are saved. Stop here; Checkpoints 8c and 9 have not been started.

### Checkpoint 8d continuation validation

- The `results/consensus_oracle_8d/` directory already existed, so the analysis generator was not rerun and no result files were overwritten.
- Revalidation (PowerShell): `$env:PYTHONPATH='code'; py -3.12 -m pytest -q tests\test_consensus_oracle_8d.py` — **3 passed in 3.21s** (14 Matplotlib/Pyparsing deprecation warnings).
- Read-only artifact checks passed: 40 secondary-test rows (8 variants × 5 metrics, `n=50` each), Holm-adjusted p-values recalculated from each metric's eight raw p-values for both paired tests, 50 W2 rows, five CI rows with 10,000 bootstrap samples, and five valid PNG files. The recorded high-disagreement subset contains 17 videos, and 8d's saved 8b per-annotator F1 reproduction check passes.
- Feature, annotation, and info fingerprints in the saved 8d JSON match the 8b protocol fingerprints.

**Checkpoint 8d continuation validation: PASS.** No training, new data, or result-directory writes were performed. Stop here.

## Phase A — Checkpoint 1 (STOP): Clean, mechanism, and jitter simulation

**Status: PASS with the recorded simulation outcome below.**

- **A1 cleanup:** Moved the legacy `assets/` tree, `results/real_tvsum/stage4/`, the old root `index.html`, root `update_index*.py`, the obsolete root page-rebuild script, root diffusion GIF, and paper social-preview image into `archive/withdrawn_proxy/`; the archive README contains the required withdrawal notice. The existing stage-five benchmark names were made neutral, the active entrypoint was renamed to `tvsum_cv_entry.py`, and `README.md` plus `run_stage.py` now reference it. No result directory that existed before this phase was overwritten. The deleted legacy SOTA result files that were already absent in the worktree were not restored.
- **Input provenance:** Reused only the saved `results/consensus_oracle_8b/results.json`, the existing HDF5, and the already-local official TVSum TSVs. The JSON fingerprints and independently computed hashes match: feature `ec9cbcc715887d30ba2ad46764cb810f07eb6d6241e96c0eac5fca51d398228b`, annotations `13972d16f2f05016dcbb4a62563cc74db98e829facbd8151cd0276cf47b4bd98`, info `162e37e5d027d293d5e7877aa0b7fdac417a2e7c077b95a159334717202ea59d`. All mechanism results use the real 50-video TVSum data; no training or downloads.
- **A2 mechanism associations:** Spearman correlations are with saved per-video barycenter-minus-mean F1 gaps. Bootstrap percentile CIs resample the videos; these are associations, not causal effects. Peak count uses `find_peaks` on the max-normalized mean curve at fixed prominence `0.10`; annotator peak dispersion is the population SD of dominant peak positions in picks; overlap is mass-weighted over picks selected by each canonical 15%-duration summary.

  | Descriptor | Spearman rho | 95% bootstrap CI | Videos |
  |---|---:|---:|---:|
  | Mean consensus interior peak count | -0.310438 | [-0.571962, -0.016638] | 50 |
  | Saved mean pairwise annotator Spearman rho | -0.288554 | [-0.511258, -0.028416] | 50 |
  | Annotator dominant-peak location SD (picks) | -0.209993 | [-0.460588, 0.071765] | 50 |
  | Weighted overlap fraction of mean-selected mass | 0.292825 | [0.008722, 0.542381] | 50 |

- The two requested qualitative checks each have a figure and three JSON-generated factual sentences in `results/mechanism_8e/results.json` and `results/mechanism_8e/SUMMARY.txt`:
  - `i3wAGJaaktw`: mean F1 `74.410`, barycenter F1 `64.329`, barycenter-minus-mean gap `-10.081`; `10` mean-curve interior peaks; dominant-peak location SD `61.988` picks; shared picks contain `84.40%` of mean-selected pick mass.
  - `uGu_10sucQo`: mean F1 `41.838`, barycenter F1 `51.800`, barycenter-minus-mean gap `+9.962`; `11` mean-curve interior peaks; dominant-peak location SD `66.219` picks; shared picks contain `100.00%` of mean-selected pick mass.
- **A3 labelled simulation only:** The script declares shift half-widths `[0, 1, 2, 4, 8, 16, 32]` picks and Gaussian noise SD `0.15` times the source curve maximum. It forms `20` simulated annotators per video and width, uses the oracle's LOAO mean and 1D quantile-averaged barycenter, and bootstraps paired gains over the `50` real-video source curves. All simulation filenames and the plot are explicitly prefixed `SIMULATION`; these results are not part of real-data tables.

  | Shift half-width (picks) | Mean F1 | Barycenter F1 | Barycenter − mean F1 | 95% bootstrap CI | Videos improved |
  |---:|---:|---:|---:|---:|---:|
  | 0 | 86.552858 | 85.650723 | -0.902135 | [-1.685728, -0.120698] | 10/50 |
  | 1 | 85.747056 | 84.717434 | -1.029622 | [-1.785308, -0.363208] | 7/50 |
  | 2 | 84.662560 | 83.941029 | -0.721531 | [-1.358865, -0.093618] | 11/50 |
  | 4 | 83.211428 | 82.867163 | -0.344264 | [-0.993599, 0.221510] | 12/50 |
  | 8 | 80.342013 | 79.054027 | -1.287986 | [-2.486915, -0.404445] | 10/50 |
  | 16 | 76.444760 | 75.691125 | -0.753635 | [-1.955945, 0.400393] | 19/50 |
  | 32 | 73.340409 | 72.409944 | -0.930465 | [-1.848015, -0.052405] | 13/50 |

  **Simulation finding:** No crossover was observed in the declared range: all point estimates favor the mean target, so there is no shift width at which the barycenter starts to win in these runs. This is a simulation result, not a claim about real annotators.
- **Saved artifacts:** `results/mechanism_8e/` contains the JSON, two CSVs, two qualitative PNGs, and `SUMMARY.txt`. `results/simulation_jitter/` contains only `SIMULATION_*` files: JSON, per-video and shift-summary CSVs, the gain-vs-shift PNG, and `SIMULATION_SUMMARY.txt`.
- **Commands and validation:** `py -3.12 code\run_tvsum_mechanism_8e.py --mode all` (with official TSV paths set in `TVSUM_ANNOTATION_TSV` / `TVSUM_INFO_TSV`); elapsed runtimes from the saved JSON are `30.970` seconds for mechanism analysis and `22.253` seconds for simulation. `py -3.12 -m pytest -q` with those paths and `PYTHONPATH=code`: **24 passed in 4.06s** (14 Matplotlib/Pyparsing deprecation warnings). `git diff --check` passed; all generated figures have valid PNG signatures. No `sota`-named paths remain in the active workspace.

**Checkpoint 1 result: PASS.** Phase A is complete. Stopped at this checkpoint pending approval before Phase B.

## Phase B — Checkpoint 2 (STOP): Paper skeleton and JSON-generated tables

**Status: PASS; PDF not compiled because no LaTeX compiler is installed.**

- Replaced `paper/main.tex` with the requested paper title and sections: Introduction, Related Work (explicit TODO), Consensus Targets and the LOAO Protocol, Oracle Results, Learned-Model Results, Disagreement and Mechanism Analysis, Controlled Jitter Simulation, Limitations, and Conclusion. The text states that the barycenter minimizes W2 by construction but has lower saved oracle F1 and rank-correlation averages; it labels the sigma-smoothed analysis exploratory and not used for tuning; it makes no improvement, SOTA, or diffusion claim. The barycenter is identified as the one-dimensional quantile-averaged version on pick-index support. Numerical facts in the paper use generated macros/tables, not hand-entered benchmark values.
- Replaced `paper/references.bib` with only the verified TVSum dataset citation. Related-work transport/diffusion citations remain TODO rather than asserted bibliography entries.
- Added `paper/build_tables.py`; running `py -3.12 paper\build_tables.py` generated 11 TeX table/facts files and `paper/tables/table_sources.json`. The provenance manifest traces every generated table file to specific saved JSON keys (or documents the paired test derived from saved per-video JSON records), and stores hashes for all six source JSON files. It also checks the shared real-data fingerprints before generating output. No results directories or saved result JSONs were changed.
- Generated tables cover learned A–F and BiLSTM metrics, paired learned-versus-uniform comparisons, BiLSTM comparison, oracle F1/rank metrics with raw and Holm-adjusted p-values, secondary oracle metrics, W2 characterization, mechanism associations/cases, exploratory high-disagreement smoothing, and the separately labelled jitter simulation. Every table value is traced to a saved JSON field or a documented paired calculation over saved JSON values.
- **Uniform-baseline interpretation:** the six saved A–F arms have no paired evidence against uniform in either test family; their raw paired t and Wilcoxon p-values are in `learned_vs_uniform.tex`. The saved BiLSTM comparison is mixed: from the saved unique-video BiLSTM scores paired with the saved uniform records, mean F1 difference is `+2.492` points, paired t `p=0.0492866546`, Wilcoxon `p=0.0682751531` (`n=32`). The manuscript reports both tests and does not characterize this mixed result as consistent superiority.
- **Input JSON SHA-256 values recorded in the provenance manifest:**

  | Source | SHA-256 |
  |---|---|
  | `results/standard_tvsum_cv_fixed/results.json` | `3f02973c5f557e786f9a9d5e7320e2df5dc7d02b3d3eac9e0a559b05d1aad123` |
  | `results/standard_tvsum_cv_bilstm/backbone_comparison.json` | `336d369f1874a1b0150789a6e8d7450b171b01ff0ac2fb56de52b304100628d3` |
  | `results/consensus_oracle_8b/results.json` | `a74c6d03eae9d8c264be371f4850a9af66c50230284a2c278d337fb1e980cf96` |
  | `results/consensus_oracle_8d/results.json` | `6df364e18af66d7ebccba13df445305149dce0d69cf58212e68a0b596df8b6ec` |
  | `results/mechanism_8e/results.json` | `a848615e1697fe918d0431c8656d550262f66ce5950d81120ff62c772c227153` |
  | `results/simulation_jitter/SIMULATION_results.json` | `5d5d1279ef1aa45e5d7bdef2389da824742e942025295f83192772dc4c9ece1a` |

- **Commands and validation:** table generation completed in `0.996` seconds; table-builder tests: **2 passed in 1.09s**; full suite with official TSV environment paths and `PYTHONPATH=code`: **26 passed in 4.25s** (14 Matplotlib/Pyparsing deprecation warnings). Manuscript checks verified all `\input` targets exist, every generated table has source mappings, and all citations resolve to the bibliography. `git diff --check` passed. Neither `pdflatex` nor `latexmk` is installed, so the manuscript was not compiled and no PDF was produced.

**Checkpoint 2 result: PASS.** Phase B is complete. Phase C had not started at this checkpoint; approval was received before continuing.

## Phase C — Checkpoint 10 (demo)

**Status: PASS.**

- **Demo models:** Added the separate demo trainer and trained both BiLSTM checkpoints on the 50 real TVSum videos. The epoch count is `4`, derived at runtime as the median of the `1,250` saved `B_mean_cross_entropy` selected-epoch records in `results/standard_tvsum_cv_fixed/results.json`. The two targets use the same architecture, cross-entropy objective, seed, batch size, optimizer, learning rate, and weight decay. Config label: “demo model, trained on all TVSum, never used for any reported number”. These weights are demo-only and were not used to produce or modify any benchmark result.
- **Feature extractor diagnostic:** `demo/feature_diagnostics.json` records torchvision GoogLeNet pool5 results from three distinct-duration test excerpts of Big Buck Bunny, licensed CC BY 3.0; the clips are diagnostic/timing inputs only and are not TVSum data. The HDF5 reference has mean `0.01715755`, standard deviation `0.02611858`, and mean L2 norm `1.0`. Raw torchvision pool5 has mean L2 norm `16.14522`, so the raw scale differs substantially. After framewise L2 normalization, mean L2 norm is `1.0`; normalized channel standard deviation is `0.02057811`, a relative difference of `21.21%` from HDF5. This discrepancy is documented, and the app applies the trained feature scaler unchanged after per-frame L2 normalization.
- **Pipeline/app:** `demo/pipeline.py` extracts 1024-D GoogLeNet pool5 features at 1 fps, detects shots with PySceneDetect, uses the saved target model, and imports `canonical_summary` for shot knapsack and 15% budget selection. It generates the model and uniform comparison clips plus a score timeline. `demo/app.py` defaults to uploads only, enforces the 10-minute cap, displays the required limitation notice, reports errors, shows progress, and caches results by input SHA-256 and target. URL input and its terms notice appear only with `--enable-url`. The app was launched without that flag and returned HTTP 200.
- **CPU timing:** `demo/timing.json` contains stage timings for the three CC BY 3.0 clips at distinct durations. GoogLeNet feature extraction is the largest mean stage; the JSON records decode, features, shot detection, scoring, knapsack, and ffmpeg timings. Upload-mode timing has no network-download stage, as recorded in the JSON.
- **Tests and runtime:** focused demo tests passed (**6 passed in 13.55s**); final full suite `py -3.12 -m pytest -q` with official TSV environment paths and `PYTHONPATH=.;code`: **32 passed in 14.77s**. The suite includes end-to-end processing of a short test-only generated clip, duration-budget verification, deterministic output hash comparison, cache-key behavior, length-cap errors, rejection of non-YouTube URL hosts, and upload-only Gradio startup. `py_compile` and `git diff --check` passed. The final suite reported 16 Matplotlib/Pyparsing and WebSockets deprecation warnings. The model-training command's wall-clock duration was not recorded; no training runtime is inferred.
- **Input hashes** (also saved in `demo/model/config.json`): features `ec9cbcc715887d30ba2ad46764cb810f07eb6d6241e96c0eac5fca51d398228b`; official annotations `13972d16f2f05016dcbb4a62563cc74db98e829facbd8151cd0276cf47b4bd98`; official info `162e37e5d027d293d5e7877aa0b7fdac417a2e7c077b95a159334717202ea59d`; fixed-run JSON `3f02973c5f557e786f9a9d5e7320e2df5dc7d02b3d3eac9e0a559b05d1aad123`.
- **Demo artifact SHA-256:** mean weights `fc564bed70a71cef22fcee720b4688162734b091e7289d081da5c234c3d41ca3`; barycenter weights `3efa11c6e35150cc947dd6a80a557c2a787f4d4210bc2132b774661c5d6998bd`; scaler `bc0c2a4776f4ee85b876d6c7ab01dddb5d00237b35036bf252a6d57a2586e604`; feature diagnostics `3dec95a61c1f5b5036765cde3dcf22c9a2fadc3df76d825b09133c8317184da9c`; timing `acdf5ae0b65a9ff699809ca67fa679b474402042b7473be65cfba98ff2123ab9`.
- **New deliverables:** `demo/app.py`, `demo/pipeline.py`, `demo/prepare_model.py`, `demo/run_feature_diagnostics.py`, `demo/run_timing.py`, `demo/requirements.txt`, `demo/README.md`, `demo/model/` (config, scaler, both weights), `demo/feature_diagnostics.json`, `demo/timing.json`, and `tests/test_demo_pipeline.py`. Temporary source archive and pipeline-smoke outputs were removed; three small CC BY test excerpts remain locally ignored under `demo/test_clips/`. No pre-existing result directory was overwritten.

**Checkpoint 10 result: PASS.** Phase C is complete. Stop here.

## Demo live run — Checkpoint 1, Step 1 environment check (STOP: blocked)

**Status: BLOCKED pending FFmpeg installation on PATH. No launcher or live-app changes were made.**

- **Read before proceeding:** `demo/README.md`, `demo/app.py`, `demo/pipeline.py`, and `demo/model/config.json`.
- **Environment command:** `py -3.12 -c "import importlib.util, shutil; ..."` reported `torch`, `torchvision`, `gradio`, `scenedetect`, and `cv2` all importable. `ffmpeg` and `ffprobe` both returned `None` from `shutil.which`; neither is on PATH. Although `imageio-ffmpeg` is installed and the app can use its bundled executable internally, the requested PATH precondition is not met.
- **Dependency install command:** `py -3.12 -m pip install -r demo\requirements.txt` completed successfully; all declared demo dependencies were already satisfied. No package failure or missing Python module was reported.
- **Test clips:** `demo/test_clips/BBB_clip_30s.mp4`, `BBB_clip_60s.mp4`, and `BBB_clip_90s.mp4` are present. Existing demo model weight hashes remain the values in `demo/model/config.json`; no model, benchmark JSON, result directory, or paper table was changed.
- **Exact Windows install command:** `winget install --id Gyan.FFmpeg.Shared --exact --accept-source-agreements --accept-package-agreements`. After installation, open a new PowerShell and verify both `ffmpeg -version` and `ffprobe -version` work, then resume. If winget's installed location is not automatically on PATH, add the installed FFmpeg `bin` directory to PATH and open a new shell.
- **TVSum TSV environment variables:** not needed at inference time; `demo/app.py` loads only the saved model/scaler and video input, so the launcher must not set `TVSUM_ANNOTATION_TSV` or `TVSUM_INFO_TSV`.
- **Run accounting:** dependency command exit code 0; elapsed install wall-clock time was not captured. Tests not run in this environment-only step. No code files created or modified for this live run.

**Checkpoint 1 Step 1 result: BLOCKED.** Stop here until FFmpeg and ffprobe are installed and available on PATH; do not create `run_demo.ps1`, launch the app, or run the live checks before then.

## Demo live run — Checkpoint 1 (STOP)

**Status: PASS.**

- **Environment recovery:** Installed FFmpeg with `winget install --id Gyan.FFmpeg.Shared --exact --accept-source-agreements --accept-package-agreements`; installed version `9.0.2`. Winget updated the persistent user PATH, and a refreshed PowerShell process verified both `ffmpeg -version` and `ffprobe -version`. `py -3.12 -m pip install -r demo\requirements.txt` exited successfully; `torch`, `torchvision`, `gradio`, `scenedetect`, and `cv2` were importable. No package install error. The launcher refreshes the user PATH in its own process so it sees the winget installation.
- **Launcher:** Added root `run_demo.ps1`. Command: `.\run_demo.ps1`. It activates no environment, does not set TVSum TSV variables because inference does not need them, binds only to `127.0.0.1`, defaults to port `7860`, selects the next free port if needed, and keeps URL downloads disabled unless `-EnableUrl` is passed. URL mode was not enabled.
- **Live server:** `http://127.0.0.1:7860` returned HTTP `200`; the Gradio configuration included the exact demo-only notice and no URL input. To verify the occupied-port fallback, launched a second copy while `7860` was occupied: the launcher printed `http://127.0.0.1:7861`, which also returned HTTP `200`. The fallback test server was stopped by its identified PIDs; the primary app remains available on `127.0.0.1:7860`.
- **Real-video smoke checks:** Used only the three existing local Big Buck Bunny CC BY test excerpts, not TVSum benchmark data. Called the app's direct `run_demo` interface for mean and barycenter on each clip; repeated each identical input/target to verify the cached result manifest and output paths were reused. All six runs produced summary and uniform clips with positive `ffprobe` durations, plus a nonempty timeline PNG. All measured summary durations passed the requested 15% budget plus one-second sampling/encoding tolerance:

  | Source duration | Mean run / repeat | Barycenter run / repeat | Summary duration in both modes | Uniform duration in both modes | Wall-clock for both first runs |
  |---:|---:|---:|---:|---:|---:|
  | 30 s | 3.661 / 0.029 s | 3.472 / 0.021 s | 3.958 s | 3.958 s | 7.361 s |
  | 60 s | 6.596 / 0.025 s | 6.485 / 0.022 s | 8.958 s | 8.958 s | 13.264 s |
  | 90 s | 9.763 / 0.016 s | 9.682 / 0.026 s | 12.958 s | 12.958 s | 19.640 s |

- **Length cap:** Generated a local `601`-second black test video in a temporary directory. The app returned `Video length exceeds the 10-minute limit.` in `0.003907` seconds, before feature extraction; the temporary video was deleted with its temporary directory.
- **Saved observations:** `demo/localhost_check.json` contains each clip's SHA-256, both target runtimes, repeat-cache runtimes, ffprobe durations, timeline/output existence checks, and length-cap outcome. Its SHA-256 is `14f07480c152dd147359d71b365e8cb38c57400b146e1bf5611a2c7b3e240ac5`. Clip hashes: 30 s `62579e381a1435c158a038f9783cbf4b528cdf41621cd0c0adc57f12dcba3bf7`; 60 s `5c69f6ee3776fec360bcc00d8850140cdefc6cf6c2d40fcdd3476ab49e47d330`; 90 s `cb792ba00ace52fa678c4969e25b8a860f66af5e5ab7bc8b04f4902676f69ed5`. Demo model hashes were verified against `demo/model/config.json`: mean `fc564bed70a71cef22fcee720b4688162734b091e7289d081da5c234c3d41ca3`; barycenter `3efa11c6e35150cc947dd6a80a557c2a787f4d4210bc2132b774661c5d6998bd`; scaler `bc0c2a4776f4ee85b876d6c7ab01dddb5d00237b35036bf252a6d57a2586e604`.
- **Docs and validation:** Updated `demo/README.md` with the launcher command, default/fallback port behavior, PATH requirements, inference's lack of TSV requirements, and explicit limitations, including the required demo-only sentence. Full command `py -3.12 -m pytest -q` (with official TSV paths and `PYTHONPATH=.;code`) passed: **32 passed in 14.57s**, 16 Matplotlib/Pyparsing/WebSockets deprecation warnings. `git diff --check` exited 0; it emitted only the repository's existing LF/CRLF conversion warnings. No benchmark JSON, paper table, or result directory was changed; all demo outputs are separate from reported numbers.

**Checkpoint 1 result: PASS.** Stop here and wait for approval before proceeding to the user's-video quality analysis or feature-statistics alternative.

## Demo empty-summary fix

**Status: PASS.**

- **Cause addressed:** When scene detection returned one long scene, canonical whole-scene selection could not choose it under the 15% budget, so the summary was empty. The demo now divides detected scenes longer than the frame budget into contiguous budget-feasible subsegments before applying the existing canonical knapsack selection. Scene boundaries already within budget are unchanged.
- **User-visible behavior:** When this fallback is used, the app status says that oversized detected scenes were divided into contiguous segments. An empty selection after feasible segments have been supplied is now reported as an internal consistency error rather than the earlier no-scene-fits message.
- **Regression test:** `tests/test_demo_pipeline.py` constructs a 30-second static, no-cut test clip, confirms scene detection finds one 30-frame scene, then runs both model and uniform selection/rendering through the pipeline. It verifies the fallback flag, nonempty selection, summary budget, and both output files. This is an engineering regression fixture only, not benchmark data or a reported result.
- **Validation:** Focused command `py -3.12 -m pytest -q tests\test_demo_pipeline.py` passed: **7 passed in 15.75s**. Full command `py -3.12 -m pytest -q` (with `PYTHONPATH=.;code`) passed: **31 passed, 2 skipped in 15.62s**, with 16 dependency deprecation warnings. `git diff --check` exited 0 with existing LF/CRLF conversion warnings.
- **SHA-256:** `demo/pipeline.py` `D217005733986C0A6834B0E60D362F9549A766E5ABDCEC9EA4B9792657DB7A47`; `demo/app.py` `800CF135B75C0A1BEEC1A4766B793FF474CEE0A6B699607A03254BA60ADFCC05`; `tests/test_demo_pipeline.py` `FC16F817AFBE6764116BF740D3186954C09A1F1C88695BC55B59C09EBCBBA230`.
- **Scope:** No benchmark result JSON, paper table, result directory, or demo model was changed. The user's reported video was not available for direct reproduction.

**Fix validation result: PASS.** The demo server must load the updated code before the fix is usable at localhost.

## Demo YouTube URL mode — Checkpoint 2, Step 6

**Status: PASS for the supplied public URL.**

- **Opt-in and notice:** Started the URL-enabled launcher with `.\run_demo.ps1 -EnableUrl`; it selected `http://127.0.0.1:7861` because the upload-only app remained on `7860`. The URL-enabled page returned HTTP `200` and its Gradio config contained the YouTube terms notice, URL textbox, and demo-only model notice. The default server at `7860` remains upload-only.
- **Supplied-video test:** `https://youtu.be/FAqnmK6IqQE` was accessible through yt-dlp metadata as a `339`-second video, below the 10-minute cap. The first download attempt failed because the old selector requested an unavailable format. Updated the selector to prefer a single MP4 video stream up to 720p, with single-stream fallbacks; this is appropriate because the demo output omits audio. No cookies, credentials, geo-block workarounds, or other restriction bypasses were used.
- **End-to-end result:** Submitted the URL through the local Gradio API in mean-target mode. The saved `demo/cache/7f1459a39565fc26ef9c5cad8593a051827fe4b55884e97c9d2e18e261ede4ba_mean/result.json` records source duration `338.92` seconds, selected duration `50.0` seconds (`14.75%` of source), both generated outputs, and the summary timeline. `ffprobe` measured `49.96` seconds for both the mean summary and uniform comparison; all three output files existed. End-to-end wall-clock time was `304.752` seconds. The download temporary-directory set was empty before and after processing, confirming cleanup.
- **Limitations:** This verifies one accessible public video, not every YouTube URL. Private, restricted, unavailable, or unsupported videos may fail; inputs remain capped at 10 minutes. YouTube downloading may violate its terms; the user is responsible for ensuring use is permitted. URL mode remains opt-in.
- **Docs and tests:** Updated `demo/README.md` with the `.\run_demo.ps1 -EnableUrl` command, format behavior, restrictions, and temporary-file cleanup. Focused tests passed: **8 passed in 15.20s**. Full `py -3.12 -m pytest -q` (with `PYTHONPATH=.;code`) passed: **32 passed, 2 skipped in 15.31s**, with 16 dependency deprecation warnings. `git diff --check` exited 0 with existing LF/CRLF conversion warnings.
- **SHA-256:** `demo/app.py` `076CFB1ACC31BA9AF2DB94475400E409ABC398E1262EA7315FE8B413810A391B`; `demo/README.md` `8E079FD65604A9DDF759B4CAD2758D3E7F670CEE59111FABAAFD19F604B313A3`; `tests/test_demo_pipeline.py` `98CC3A1FF0D40D385A93060B996235C7ADA8076C7484D972FDD05793FF195EF7`.
- **Scope:** No benchmark JSON or paper table was changed. The user's own-video quality/feature-statistics analysis was not run; no own videos were supplied.

**Checkpoint 2 result: PASS for the supplied URL.** The URL-enabled demo is running locally at `http://127.0.0.1:7861`; URL mode remains disabled on the default server.
