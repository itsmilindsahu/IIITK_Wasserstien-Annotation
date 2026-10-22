# TVSum annotator-consensus study (draft)

> **Status:** This draft has no reportable standard TVSum result yet. The
> numerical tables and conclusions from earlier drafts have been withdrawn:
> the original annotation file was synthetic, and later annotation-only
> experiments did not use the canonical video-conditioned summarization
> evaluation protocol.

## Current study

The current benchmark implementation is described in the
[`README`](README.md) and runs from `code/standard_tvsum_cv.py`. It requires
authorized raw TVSum50 annotations, the official video metadata, and the
preprocessed GoogLeNet pool5 feature file. It trains a shared
video-conditioned temporal-convolution regressor using the DSNet split file,
with validation restricted to each training split.

The planned analysis separates annotation-target effects from loss effects:
mean and one-dimensional Wasserstein-barycenter targets are evaluated with
MSE and Cramer objectives, with cross-entropy for the mean target and a
validation-selected blend. Results include the canonical shot/knapsack F-score,
rank correlations to raw annotator ratings, baselines, paired comparisons, and
sanity checks. The run writes a results JSON; reported tables must be generated
from that saved file.

No state-of-the-art claim is made. Augmented and transfer settings and a
feature-conditioned diffusion model have not been evaluated in this benchmark.
Any future manuscript must state the exact split coverage and protocol
limitations recorded in the run JSON.
