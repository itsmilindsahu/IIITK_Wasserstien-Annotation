# Wasserstein Diffusion on the Annotation Simplex — Phase 1 code

Proof-of-concept code for the 11-day sprint (Jun 16-30 2026) implementing the
three fixes from the proposal on top of the Shang et al. (AAAI-25) video
summarization diffusion model. Matches the roadmap's Phase 1 scope: small,
synthetic-data prototype, not the full TVSum/SumMe/FPVSum run (that's Phase 2).

No internet access to TVSum in this environment, so `train.py` generates a
synthetic "two-camp" toy video (some annotators peak around frame 5-10, some
around frame 20-25) that mimics the disagreement case from Figure 1 of the
proposal. Swap `makeToyVideo()` for a real TVSum loader when you have the
dataset locally — that's basically the whole Phase 2 step.

## Layout

```
code/
  sinkhorn.py     - simplex projector (the Proj_delta step)
  barycenter.py   - Fix 1, entropic sinkhorn barycenter + arithmetic mean baseline
  forward.py      - Fix 2, dirichlet forward process + gaussian baseline
  losses.py       - Fix 3, wasserstein-regularized loss + plain MSE
  model.py        - tiny linear noise predictor (stand-in for the real transformer)
  train.py        - runs both configs, dumps results/
results/
  results.json    - raw numbers, feeds the dashboard
  comparison.csv  - final loss + held-out W2 per config
  loss_curve.png  - matplotlib plot, same data as the dashboard's chart
ui/
  template.html   - dashboard shell
  build_ui.py     - stuffs results.json into template.html -> index.html
  index.html      - the actual thing to open in a browser
```

## Running it

```
cd code
python3 train.py        # regenerates results/
cd ../ui
python3 build_ui.py     # regenerates index.html from the new results
```

Then just open `ui/index.html` in a browser. No server needed, data's baked
in.

## What's actually implemented vs. simplified for Phase 1

- Barycenter: real entropic Sinkhorn barycenter (numpy only, no POT
  dependency, so it's ~15 lines instead of a library call).
- Dirichlet forward process: real, uses `np.random.dirichlet` + a crude
  simplex renormalization for the Proj_delta step instead of anything fancier.
- Noise predictor: a single linear layer trained with manual numpy gradients.
  The proposal calls for a Transformer here — this is a stand-in so the
  pipeline runs end to end in Phase 1. Needs to actually be swapped in for
  Phase 2, it's not just a config flag.
- W2 loss: uses a closed-form CDF-difference formula for 1D distributions.
  Worth double-checking — see the open question below, the numbers it
  produces don't line up with what the Sinkhorn barycenter itself is
  minimizing.

## Known issue / open question (documented honestly, not swept under the rug)

On the toy two-camp video, the barycenter's average distance to the
annotators came out *higher* than the arithmetic mean's, under the `w2_1d()`
proxy used in `losses.py`. Theory says it should be lower or equal (that's
the whole non-regression guarantee). Best guess: `w2_1d()` (sum of squared
CDF differences) isn't actually the same metric the Sinkhorn barycenter
minimizes — true 1D W2 is the integral of squared quantile differences, not
squared CDF differences. Left as-is in the results rather than fudging it;
first thing to fix before trusting any Phase 2 numbers.

## Requirements

```
numpy
matplotlib
```

That's it — no torch, no POT, deliberately kept minimal for a Phase 1
prototype. See `requirements.txt`.
