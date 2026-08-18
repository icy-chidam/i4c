"""
train_cnn_matcher.py
----------------------
Trains cnn_matcher.PatchCNN to answer "does this small search-image patch
contain the true reference-matching site (its landmark, if any) versus an
arbitrary other patch of the same periodic pattern." This directly targets
a documented failure mode (see docs/references.md): whole-template
normalized cross-correlation gives a small, deliberately-placed landmark
too little relative weight against a strongly periodic background, so a
true-site patch and a decoy periodic-repeat patch score almost identically
under plain NCC. A learned classifier that looks at the patch itself
(rather than a single scalar correlation coefficient) can pick up the
landmark's local appearance directly.

Positives: a patch centered near the true site (small jitter, so the
network tolerates the few-pixel offset a real NMS peak will have -- not
always perfectly centered).
Negatives: a mix of "easy" (patches from elsewhere in the same periodic
pattern, far from the landmark -- an ordinary periodic repeat) and "hard"
(closer near-misses, for a sharper decision boundary) negatives.

Training patches are rendered at SEARCH-image scale/noise statistics
(scale ~8.5-11.5, matching localize.py's own search range), since that is
what will actually be cropped from the real search image at inference
time -- rendering at reference scale here would be a train/inference
distribution mismatch.

Usage
-----
    python train_cnn_matcher.py --num-samples 800 --epochs 16 \
        --output ../weights/cnn_matcher.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pattern_render import default_params, render_capture, add_edge_brightening, add_sensor_noise
from cnn_matcher import PatchCNN

STYLES = ["dram", "dram_arcuate", "finfet"]


def make_patch(style: str, params: dict, center_world, rng: np.random.Generator, noisy: bool = True):
    scale = rng.uniform(8.5, 11.5)
    out_size = PatchCNN.INPUT_SIZE
    img = render_capture(style, params, out_size, center_world, scale=scale,
                          theta_deg=0.0, base_softness_px=0.9, supersample=3)
    if noisy:
        # Noise multiplier matches dataset_generator.py's per-sample range
        # (0.4x-2.5x base, search side scaled up to 1.6x that) -- without
        # this, the CNN is trained at a single fixed noise level while
        # dataset_generator.py can now generate search images up to ~4x
        # noisier, and the CNN silently under-recognises real landmarks on
        # the noisiest samples (verified: see docs/validation_report.md,
        # "Secondary finding: CNN/dataset noise-level mismatch"). The
        # ambiguity gate in localizer_v2/v3 makes that failure mode safe
        # rather than actively harmful, but closing the gap directly here
        # is the actual fix -- the CNN's one job is recognising the
        # landmark, so it should be trained on the noise range it will
        # really be asked to recognise it in.
        noise_mult = min(rng.uniform(0.4, 2.5) * 1.6, 4.0)
        img = add_edge_brightening(img, strength=0.20)
        img = add_sensor_noise(img, rng, shot_gain=0.075 * noise_mult, read_noise_std=0.016 * noise_mult)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def build_dataset(num_samples: int, seed: int):
    master_rng = np.random.default_rng(seed)
    X, y = [], []
    t0 = time.time()
    for i in range(num_samples):
        style = STYLES[i % len(STYLES)]
        srng = np.random.default_rng(int(master_rng.integers(0, 2 ** 31)))
        params = default_params(style, srng, defect_prob=1.0)  # always has a landmark to learn from
        period_scale = params.get("period_x", params.get("period_u", 90.0))

        pos_jitter = (srng.uniform(-0.12, 0.12) * period_scale, srng.uniform(-0.12, 0.12) * period_scale)
        pos = make_patch(style, params, pos_jitter, srng)
        X.append(pos); y.append(1.0)

        ang = srng.uniform(0, 2 * np.pi)
        if srng.uniform() < 0.35:
            dist = srng.uniform(0.6, 1.3) * period_scale  # hard negative: near miss
        else:
            dist = srng.uniform(3.5, 9.0) * period_scale  # easy negative: far repeat
        neg_center = (dist * np.cos(ang), dist * np.sin(ang))
        neg = make_patch(style, params, neg_center, srng)
        X.append(neg); y.append(0.0)

        if (i + 1) % 150 == 0:
            print(f"  generated {i + 1}/{num_samples} pairs ({time.time() - t0:.0f}s elapsed)")

    X = np.stack(X)[:, None, :, :]
    y = np.array(y, dtype=np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--output", type=Path, default=Path("../weights/cnn_matcher.npz"))
    args = ap.parse_args()

    print(f"Generating {args.num_samples * 2} labeled patches...")
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
    print(f"Saved CNN matcher -> {args.output}")


if __name__ == "__main__":
    main()
