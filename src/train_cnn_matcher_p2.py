"""
train_cnn_matcher_p2.py
--------------------------
Retrains the SAME cnn_matcher.PatchCNN architecture used in Phase 1 (no
architecture change at all -- same conv-relu-pool x2 -> dense-relu ->
dense-sigmoid network, see cnn_matcher.py, untouched) on a distribution
that matches what Phase 2 will actually show it. train_cnn_matcher.py
is untouched and still reproduces the original Phase 1 weights exactly;
this is a sibling script producing a second weights file
(weights/cnn_matcher_p2.npz), not a replacement.

Two things change from train_cnn_matcher.py, both distributional, not
architectural:

  1. Scale widened from Phase 1's ~8.5-11.5 to the addendum's disclosed
     [8, 12] exactly.

  2. Degradation robustness: a fraction of patches (both positive and
     negative) now go through pattern_render.apply_degradation
     (charging/scan-distortion/defocus, severity 0-3) and/or
     perturb_geometry ("polygon scaling") before the sensor-noise step --
     Set B conditions the original training patches never saw.

NOT changed, deliberately: rotation. Every patch here is still rendered
at theta_deg=0.0, because CNN patches are always crops of the SEARCH
image, which in this project's capture model is always axis-aligned
(theta_deg=0.0) regardless of how the addendum's disclosed +/-5deg
rotation range widened -- rotation lives entirely in how the REFERENCE
is warped to build a matching template (see localizer.py's
_make_template / pose_refine.py), never in the search image the CNN
inspects. Retraining for "wider rotation" would be solving a problem
this network doesn't have.

Usage
-----
    python train_cnn_matcher_p2.py --num-samples 1000 --epochs 18 \
        --output ../weights/cnn_matcher_p2.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pattern_render import (default_params, render_capture, add_edge_brightening,
                             add_sensor_noise, apply_degradation, perturb_geometry)
from cnn_matcher import PatchCNN

STYLES = ["dram", "dram_arcuate", "finfet"]


def make_patch_p2(style: str, params: dict, center_world, rng: np.random.Generator,
                   noisy: bool = True, degrade_prob: float = 0.35):
    scale = rng.uniform(8.0, 12.0)  # addendum's exact disclosed range
    patch_params = params
    out_size = PatchCNN.INPUT_SIZE
    degraded = noisy and rng.uniform() < degrade_prob
    if degraded:
        patch_params = perturb_geometry(style, params, rng, max_frac=rng.uniform(0.05, 0.20))

    img = render_capture(style, patch_params, out_size, center_world, scale=scale,
                          theta_deg=0.0, base_softness_px=0.9, supersample=3)
    if noisy:
        noise_mult = min(rng.uniform(0.4, 2.5) * 1.6, 4.0)
        img = add_edge_brightening(img, strength=0.20)
        if degraded:
            severity = int(rng.integers(0, 4))
            img = apply_degradation(img, rng, severity)
            noise_mult *= (1.0 + 0.15 * severity)
        img = add_sensor_noise(img, rng, shot_gain=0.075 * noise_mult, read_noise_std=0.016 * noise_mult)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def build_dataset(num_samples: int, seed: int):
    master_rng = np.random.default_rng(seed)
    X, y = [], []
    t0 = time.time()
    for i in range(num_samples):
        style = STYLES[i % len(STYLES)]
        srng = np.random.default_rng(int(master_rng.integers(0, 2 ** 31)))
        params = default_params(style, srng, defect_prob=1.0)
        period_scale = params.get("period_x", params.get("period_u", 90.0))

        pos_jitter = (srng.uniform(-0.12, 0.12) * period_scale, srng.uniform(-0.12, 0.12) * period_scale)
        pos = make_patch_p2(style, params, pos_jitter, srng)
        X.append(pos); y.append(1.0)

        ang = srng.uniform(0, 2 * np.pi)
        if srng.uniform() < 0.35:
            dist = srng.uniform(0.6, 1.3) * period_scale
        else:
            dist = srng.uniform(3.5, 9.0) * period_scale
        neg_center = (dist * np.cos(ang), dist * np.sin(ang))
        neg = make_patch_p2(style, params, neg_center, srng)
        X.append(neg); y.append(0.0)

        if (i + 1) % 150 == 0:
            print(f"  generated {i + 1}/{num_samples} pairs ({time.time() - t0:.0f}s elapsed)")

    X = np.stack(X)[:, None, :, :]
    y = np.array(y, dtype=np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--output", type=Path, default=Path("../weights/cnn_matcher_p2.npz"))
    args = ap.parse_args()

    print(f"Generating {args.num_samples * 2} labeled patches (Phase 2 distribution)...")
    X, y = build_dataset(args.num_samples, args.seed)
    n = len(y)
    idx = np.random.default_rng(0).permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    Xtr, ytr = X[train_idx], y[train_idx]
    Xval, yval = X[val_idx], y[val_idx]
    print(f"train={len(ytr)}  val={len(yval)}  positive_frac={y.mean():.2f}")

    model = PatchCNN(seed=42)
    n_batches = max(1, len(ytr) // args.batch_size)
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = np.random.default_rng(epoch).permutation(len(ytr))
        epoch_loss = 0.0
        for b in range(n_batches):
            bidx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            if len(bidx) == 0:
                continue
            model.forward(Xtr[bidx])
            epoch_loss += model.backward_and_step(ytr[bidx], lr=args.lr)
        val_p = model.forward(Xval)
        val_acc = ((val_p > 0.5).astype(np.float32) == yval).mean()
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss={epoch_loss / n_batches:.4f}  "
              f"val_acc={val_acc:.3f}  ({time.time() - t0:.0f}s elapsed)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved Phase 2 CNN matcher -> {args.output}")


if __name__ == "__main__":
    main()
