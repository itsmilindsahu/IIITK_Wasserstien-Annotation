# TVSum Summarization Demo

This is an inference-only demonstration application. It is not part of the
reported benchmark:

> Demo only. Model trained on 50 TVSum videos; quality on other video types is
> not validated.

The mean-target and barycenter-target BiLSTM checkpoints were each trained for
the median validation-selected epoch count of the saved
`B_mean_cross_entropy` runs. They use all 50 real TVSum videos and are never
used for any reported number. The model configuration, feature scaler, and
weight hashes are stored under `model/`.

## Install and run

Use Python 3.12 and the project's normal requirements, then install the
demo-specific dependencies:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pip install -r demo\requirements.txt
$env:TVSUM_ANNOTATION_TSV = "C:\path\to\authorized\ydata-tvsum50-anno.tsv"
$env:TVSUM_INFO_TSV = "C:\path\to\authorized\ydata-tvsum50-info.tsv"
py -3.12 demo\prepare_model.py --annotations $env:TVSUM_ANNOTATION_TSV --info $env:TVSUM_INFO_TSV
py -3.12 demo\app.py
```

The prepared `demo/model/` is included with this workspace; model preparation
refuses to overwrite it. The feature extractor downloads torchvision's
GoogLeNet ImageNet weights on first use. The app accepts uploads up to 10
minutes; opt-in YouTube URLs are limited to 2 minutes. It samples frames at 1
fps, extracts features and detects shots once, then runs both the mean-target
and barycenter-target checkpoints on those same features. It produces the two
summary clips, an overlaid score plot, a per-frame score-difference plot, a
shot-selection overlap plot, and a measured timing plot. Both summaries use
the repository's `canonical_summary` and the same 15% shot budget.

### Local launcher

From the repository root, run:

```powershell
.\run_demo.ps1
```

The launcher binds only to `127.0.0.1`, uses port `7860` when available, and
otherwise selects the next available port at or above `7860`, printing the
local URL. Inference uses the prepared model and does not need
`TVSUM_ANNOTATION_TSV` or `TVSUM_INFO_TSV`. The launcher activates no virtual
environment. It requires `py`, `ffmpeg`, and `ffprobe` to be available on PATH.

Optional URL downloading is off by default. To enable it explicitly:

```powershell
.\run_demo.ps1 -EnableUrl
```

The UI then displays a notice that downloading may violate YouTube's terms;
the user is responsible for ensuring their use is permitted. URL input is not
shown without the flag. The opt-in accepts accessible YouTube URLs, rejects
videos longer than 2 minutes before downloading when duration metadata is
available, and downloads a single video stream (preferring MP4 up to 480p,
with a 720p fallback). The two-minute duration check runs during the single
download metadata pass, before video bytes are downloaded.
Private, restricted, unavailable, or otherwise unsupported videos may fail;
the demo does not bypass YouTube restrictions. The download is temporary and
deleted after processing. The paired comparison is cached by input SHA-256
under `demo/cache/`.

The mean and barycenter checkpoints use the same BiLSTM inference architecture.
The barycenter target changes the training labels; it does not add a slower
inference algorithm. The timing plot separates the shared video-processing
stages from each model's measured scoring time. Consequently, training-time
costs for constructing barycenter targets are not incurred when using the
prepared model.

## User test: which summary feels more like the original?

Use this question when asking someone to compare the two outputs:

> **Which summary captures the feeling of the original video better: Mean or
> Barycenter? What moments or qualities made you choose it?**

### How to test

1. Choose one video that the tester has not already seen summarized. Use the
   same source video for both outputs; YouTube links must be 2 minutes or
   shorter, while uploaded videos may be up to 10 minutes.
2. Watch the full source video first, or enough of it to understand its
   overall mood, story, and important moments.
3. Enter or upload that video in the demo and select **Generate summaries**.
   Wait for both the Mean-target and Barycenter-target summaries to finish.
4. Watch both summaries, then answer the question above. Ask the tester to
   mention what influenced their choice, such as mood, story continuity,
   important moments, or pacing. If practical, vary which summary is watched
   first between testers.
5. Repeat with several different videos and testers. Keep each person's
   preference and reason; do not treat one vote or one video as proof that
   either model is generally better.

This is a subjective side-by-side preference test, not an objective accuracy
measurement. Both outputs use the same input, feature extraction, shot
detector, and 15% summary budget; the training target is what differs.

## Blinded preference study

The separate study mode presents anonymous A/B summaries, randomizes video
order and model-to-letter assignment for each tester-video pair, and stores
completed responses in SQLite. Locally the database is
`demo\study\responses.sqlite`; production deployments should configure the
database, CSV export, and stimuli directories on persistent storage. It does
not change the normal **Generate summaries** demo. See
[`DEPLOYMENT.md`](../DEPLOYMENT.md) for online deployment instructions.

1. From the repository root, prepare the three preset YouTube videos. The
   script downloads each link, reuses the SHA-256 comparison cache, and
   prepares the source and blinded summary files:

   ```powershell
   py -3.12 demo\study\prepare_study.py
   ```

   The preset links are [video 1](https://youtu.be/rz1B4B2yBZY),
   [video 2](https://youtu.be/ClpGjUQ7prU), and
   [video 3](https://youtu.be/E85MXIbqfAc).
   Each link must be an accessible YouTube video no longer than 10 minutes.
   The fixed videos are shuffled for each tester. To use a folder of local
   MP4s instead, pass its path as the positional argument. Use `--refresh` to
   download and prepare the preset links again. Preparation creates neutral
   `source.mp4`, `A.mp4`, and `B.mp4` files, and writes duration,
   selected-shot, mean-shot-length, and pacing-confound data to
   `demo\study\stimuli\confound_check.csv`.
2. Run `py -3.12 demo\app.py --study`. Testers consent and may choose which of
   the three preset videos to see first; the rest are randomized. They may
   watch at least 80% of each original or explicitly skip that source video,
   then compare the two summaries and answer without seeing scores or model
   labels. The response log records source skips, actual video order, and the
   selected first-video preference. Questions use quick single-choice and
   multi-select options, with optional comments. After the three fixed videos,
   each tester can optionally
   submit one additional public YouTube link (10 minutes or less). That video
   is summarized during their session and added to the local stimuli; its URL
   is not saved in the response database. The video itself is retained in the
   local stimuli folder for reproducibility.
3. For the password-protected admin panel, set `STUDY_ADMIN_PASSWORD` in the
   environment before launching the study app. The panel reports completed
   counts and model-to-letter balance. Export responses from the panel or run:

   ```powershell
   py -3.12 demo\study\export_responses.py
   ```

   Responses and the resulting CSV are append-only study data and are excluded
   from version control. In a hosted deployment, the admin panel exports the
   records from the configured persistent storage.
4. Run the fixed analysis and figures:

   ```powershell
   py -3.12 demo\study\analyze.py
   ```

   The report and PNG figures are written to `demo\study\report\`. The
   preregistered primary outcome is the two-sided exact sign/binomial test on
   forced choices, excluding ties; ties are also reported separately and in a
   0.5-weight sensitivity analysis. The report includes video- and
   tester-cluster robustness checks, per-video and leave-one-video-out rates,
   paired Likert analysis, order checks, tag counts, and all preparation
   confound flags. No tester or video is selectively removed.

### Limitations

Small n may make preference estimates uncertain. The models were trained on
TVSum-only data, and subjective preference is not an F-score or an objective
measure of summary quality. Preparation confound flags identify differences
in output pacing or length; they do not eliminate every possible confound.
Tester-submitted additional videos are optional and may vary between testers;
analyze those responses separately if comparing only the same three fixed
videos is important.

## Extractor diagnostic and timing

`feature_diagnostics.json` records a diagnostic against the repository's
preprocessed HDF5 features using three different-duration excerpts from the
Creative Commons test source noted below. Raw torchvision pool5 norms were
substantially different from HDF5 norms; framewise L2 normalization brings
the norms to one, while the normalized channel standard deviation still differs
by the recorded relative amount. The trained scaler is applied unchanged after
L2 normalization. This comparison is not a benchmark and does not establish
quality on arbitrary videos.

`timing.json` contains CPU stage times for the three licensed test clips. Run
the timing script with three local clips of distinct durations:

```powershell
py -3.12 demo\run_timing.py demo\test_clips\BBB_clip_30s.mp4 demo\test_clips\BBB_clip_60s.mp4 demo\test_clips\BBB_clip_90s.mp4
```

The timings report upload-mode decode (no download), feature extraction, shot
detection, scoring, knapsack, and ffmpeg encoding. They are illustrative for
this CPU and these clips.

## Limitations

- Upload summaries are limited to 10-minute inputs; YouTube URLs are limited
  to 2 minutes. Both use a 15% sampled-duration summary budget.
- Output clips omit audio.
- There is no ground truth for arbitrary user videos and no F-score claims are
  made for them.
- Features produced by torchvision may differ from the training extractor;
  the measured discrepancy is documented in `feature_diagnostics.json`.
- TVSum categories only are represented during training.
- This demo does not validate summaries on other video types.
- Input audio is omitted from the generated summary clips.
- Very short videos with too few sampled frames may not yield a summary clip.

Demo only. Model trained on 50 TVSum videos; quality on other video types is not validated.

## Test clip attribution

The local timing/feature diagnostic clips are time excerpts from **Big Buck
Bunny** by the Blender Foundation, sourced from
`https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4.zip`.
The film is licensed under **Creative Commons Attribution 3.0**:
`https://peach.blender.org/about/`. Attribution: Blender Foundation /
www.bigbuckbunny.org. The downloaded test clips are not TVSum data and are not
used for training or reported benchmark numbers.
