"""
transformer_denoiser.py

Extension (3): "Replace the lightweight prototype denoiser with a temporal
Transformer backbone conditioned on multi-modal video features
(CLIP / Video-LLaVA)."

What this does
---------------
Phase 1's `model.py` is a two-layer ReLU net that predicts noise/score for
a diffusion step on the annotation simplex x_t in Delta^{N-1}, with no
awareness of *what the video actually shows* at each segment. This file
replaces it with `TemporalTransformerDenoiser`: a Transformer encoder over
the N video segments, where each segment token is
    [scalar simplex value at that segment]  +  [timestep embedding]
and is conditioned (via a learned projection + additive fusion, or
optionally cross-attention) on a precomputed per-segment multi-modal
feature vector (CLIP or Video-LLaVA embedding for that segment's frames).
Self-attention across segments gives the model the *temporal* structure
that the old per-segment two-layer net could not see.

This requires PyTorch (README's Phase 1 explicitly avoids torch; this is
the Future Work item, so the dependency is new and intentional).

Integration
------------
- `model.py` drop-in: this module's `TemporalTransformerDenoiser.forward(x_t, t, cond)`
  is written to match the (x_t, t) calling convention implied by the README's
  "Noise Predictor: two-layer ReLU" pipeline step; if your real
  `model.py`'s forward signature differs (e.g. extra args, different
  return shape), adjust `forward()` -- that's the only place you should
  need to touch to slot this in as `model.py`'s class.
- Feature extraction is NOT run inside training: CLIP/Video-LLaVA
  inference is expensive and the README's pipeline is explicitly
  "no GPU required" for Phase 1. Use `extract_and_cache_features()` once,
  offline, to precompute a `(n_segments, feat_dim)` array per video and
  cache it to disk (e.g. `features/{video_id}.npy`); `train.py` then loads
  the cached array instead of re-running CLIP every step.
- `losses.py`'s Cramer loss / W2 eval are untouched by this change --
  only the network that PRODUCES the noise prediction is replaced.

Usage
-----
    # 1) one-time, offline (requires `pip install open_clip_torch pillow`,
    #    or transformers for Video-LLaVA):
    python transformer_denoiser.py --precompute-features \\
        --frames-dir /path/to/frames --video-id video_1 \\
        --out features/video_1.npy

    # 2) in train.py, replace:
    #        from model import NoisePredictor
    #        net = NoisePredictor(...)
    #    with:
    #        from transformer_denoiser import TemporalTransformerDenoiser
    #        net = TemporalTransformerDenoiser(feat_dim=512)
    #    and pass the cached per-video feature array as `cond` at each
    #    forward call: net(x_t, t, cond=cached_features)
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


if _HAVE_TORCH:

    class SinusoidalTimestepEmbedding(nn.Module):
        """Standard diffusion timestep embedding (as in DDPM)."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            # t: (B,) integer or float timesteps
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, device=t.device).float() / half
            )
            args = t.float()[:, None] * freqs[None, :]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.dim % 2:
                emb = torch.nn.functional.pad(emb, (0, 1))
            return emb  # (B, dim)

    class TemporalTransformerDenoiser(nn.Module):
        """
        Temporal Transformer noise predictor for diffusion on the
        annotation simplex, conditioned on multi-modal (CLIP / Video-LLaVA)
        per-segment features.

        Forward contract (matches the README's "x_t, t -> predicted noise"
        pipeline step; adjust here if your model.py's NoisePredictor differs):

            x_t:  (B, N)        current noisy simplex values per segment
            t:    (B,)          diffusion timestep per batch item
            cond: (B, N, feat_dim) or (N, feat_dim), optional
                  precomputed CLIP/Video-LLaVA features per segment
                  (broadcast across batch if 2D)

            returns: (B, N) predicted noise / score, same shape as x_t
        """

        def __init__(
            self,
            feat_dim: int = 512,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            max_segments: int = 512,
        ):
            super().__init__()
            self.d_model = d_model

            # scalar simplex value -> d_model
            self.value_proj = nn.Linear(1, d_model)
            # learned positional embedding across segments (captures order,
            # i.e. the "temporal" part the old per-segment MLP lacked)
            self.pos_embed = nn.Parameter(torch.randn(1, max_segments, d_model) * 0.02)
            # diffusion timestep embedding, added to every token
            self.time_embed = SinusoidalTimestepEmbedding(d_model)
            # multi-modal conditioning feature -> d_model
            self.cond_proj = nn.Linear(feat_dim, d_model)
            # learned gate so the model can down-weight conditioning early in
            # training if features are noisy/misaligned, rather than fighting it
            self.cond_gate = nn.Parameter(torch.tensor(1.0))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.out_norm = nn.LayerNorm(d_model)
            self.out_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        def forward(
            self,
            x_t: "torch.Tensor",
            t: "torch.Tensor",
            cond: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            B, N = x_t.shape
            device = x_t.device

            tok = self.value_proj(x_t.unsqueeze(-1))          # (B, N, d)
            tok = tok + self.pos_embed[:, :N, :].to(device)    # temporal position

            t_emb = self.time_embed(t.to(device))              # (B, d)
            tok = tok + t_emb.unsqueeze(1)                      # broadcast over segments

            if cond is not None:
                if cond.dim() == 2:
                    cond = cond.unsqueeze(0).expand(B, -1, -1)  # (B, N, feat_dim)
                cond_tok = self.cond_proj(cond.to(device))      # (B, N, d)
                tok = tok + self.cond_gate * cond_tok

            h = self.encoder(tok)                                # (B, N, d) self-attn over segments
            h = self.out_norm(h)
            pred = self.out_proj(h).squeeze(-1)                  # (B, N)
            return pred

    class VideoLLaVAConditionAdapter(nn.Module):
        """
        Optional: if your per-segment features come from Video-LLaVA rather
        than CLIP, its hidden size is typically larger (e.g. 4096) and
        benefits from a small MLP down-projection before conditioning,
        rather than a single linear layer. Swap TemporalTransformerDenoiser's
        `cond_proj` for this if you see conditioning collapse (gate -> ~0)
        with raw Video-LLaVA features.
        """

        def __init__(self, in_dim: int = 4096, d_model: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, d_model * 2),
                nn.GELU(),
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


# --------------------------------------------------------------------------- #
# Offline feature extraction (run once, cached, kept out of the training loop)
# --------------------------------------------------------------------------- #
def extract_and_cache_features(
    frames_dir: str,
    n_segments: int,
    out_path: str,
    backbone: str = "clip",
) -> np.ndarray:
    """
    Precompute one feature vector per video segment and cache to disk.

    This is intentionally decoupled from the diffusion training loop:
    CLIP/Video-LLaVA inference is the expensive part, and re-running it
    every diffusion step (or every epoch) would erase the "no GPU
    required" property of the rest of the Phase-1 pipeline. Run this
    once per video, then have train.py `np.load(...)` the cached array.

    `frames_dir` is assumed to contain one representative frame image per
    segment, named so that sorting gives segment order (e.g. seg_000.jpg,
    seg_001.jpg, ...). Adjust the frame-sampling convention to match
    however your pipeline currently extracts per-segment frames.
    """
    if backbone == "clip":
        try:
            import open_clip
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Feature extraction needs `pip install open_clip_torch pillow`. "
                "This is only required for the offline precompute step, not for "
                "training/inference with already-cached features."
            ) from e

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()

        frame_files = sorted(os.listdir(frames_dir))[:n_segments]
        feats = []
        with torch.no_grad():
            for fname in frame_files:
                img = preprocess(Image.open(os.path.join(frames_dir, fname))).unsqueeze(0)
                feat = model.encode_image(img).squeeze(0).numpy()
                feats.append(feat)
        feats = np.stack(feats, axis=0)

    elif backbone == "video-llava":
        raise NotImplementedError(
            "Video-LLaVA extraction depends on the exact checkpoint/loader you "
            "want to standardize on (e.g. `transformers`' VideoLlavaForConditionalGeneration). "
            "Wire your preferred loader here; downstream code only needs the "
            "resulting (n_segments, feat_dim) array, and VideoLLaVAConditionAdapter "
            "above is already sized to accept its wider hidden dimension."
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, feats)
    return feats


def _self_test():
    """Sanity-check the module shapes without needing real video features
    or a GPU -- run `python transformer_denoiser.py --self-test`."""
    if not _HAVE_TORCH:
        print("PyTorch is not installed in this environment; "
              "install torch to use TemporalTransformerDenoiser "
              "(the rest of the Phase-1 pipeline stays torch-free).")
        return
    B, N, feat_dim = 2, 100, 512
    net = TemporalTransformerDenoiser(feat_dim=feat_dim, d_model=64, n_heads=4, n_layers=2)
    x_t = torch.rand(B, N)
    t = torch.randint(0, 1000, (B,))
    cond = torch.randn(N, feat_dim)  # shared across batch, e.g. one video's cached features
    out = net(x_t, t, cond=cond)
    assert out.shape == (B, N), out.shape
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Self-test OK. Output shape: {tuple(out.shape)}. "
          f"Parameter count: {n_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                         help="Run a shape/forward-pass sanity check with random data.")
    parser.add_argument("--precompute-features", action="store_true",
                         help="Extract and cache per-segment CLIP features for one video.")
    parser.add_argument("--frames-dir", type=str, default=None)
    parser.add_argument("--n-segments", type=int, default=100)
    parser.add_argument("--out", type=str, default="features/video.npy")
    parser.add_argument("--backbone", type=str, default="clip", choices=["clip", "video-llava"])
    args = parser.parse_args()

    if args.precompute_features:
        if not args.frames_dir:
            raise SystemExit("--frames-dir is required with --precompute-features")
        extract_and_cache_features(args.frames_dir, args.n_segments, args.out, args.backbone)
        print(f"Cached features to {args.out}")
    else:
        _self_test()
