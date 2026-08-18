"""
dataset_generator.py
---------------------
Standalone synthetic dataset generator for the Drift-Sense problem
statement (Applied Materials / IESA-SEMICON Hackathon 2026, Track 2).

No proprietary wafer-inspection data is available, so every training and
self-evaluation pair is generated procedurally: a Reference image (fine,
"100x" sampling of a periodic die layout) and a Search image (a 1000x1000
capture of the *same* underlying periodic pattern at ~10x lower
magnification, with the reference's true location recorded exactly).

Usage
-----
    python dataset_generator.py --style dram          --num-pairs 200 --split train --output-dir ../data
    python dataset_generator.py --style dram_arcuate  --num-pairs 40  --split val   --output-dir ../data
    python dataset_generator.py --style both          --num-pairs 36  --split test  --output-dir ../data

`--style` accepts dram, dram_arcuate, finfet, or both (rotates through
all three registered styles). Every run is fully parametrized from the
command line and requires no manual edits.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pattern_render import default_params, render_capture, add_edge_brightening, add_sensor_noise, to_uint8

SEARCH_SIZE = 1000   # fixed by the problem statement (1000x1000, both images)
REF_SIZE = 1000      # see pattern_render.py docstring: this is what makes the
                      # reference occupy ~100x100 px (~1%) inside the search image at K~10.
ALL_STYLES = ["dram", "dram_arcuate", "finfet"]


def make_pair(style: str, rng: np.random.Generator, out_size_search: int = SEARCH_SIZE,
              out_size_ref: int = REF_SIZE, margin: int = 120, defect_prob: float = 0.5,
              vary_noise: bool = True):
    """Render one (reference, search, ground_truth) sample.

    `vary_noise`: draw a per-sample noise multiplier (0.4x-2.5x the base
    shot/read noise levels below) instead of using a single fixed noise
    strength for every sample. Without this, "results across multiple
    noise levels" (an explicit validation requirement) is impossible to
    report because every generated pair would be equally noisy. The
    search image's noise is always scaled up relative to the reference's
    (search_mult = ref_mult * 1.6, clipped) so the search stays the
    noisier of the two at every level, per the problem statement.
    """
    params = default_params(style, rng, defect_prob=defect_prob)

    scale_ref = 1.0
    scale_factor = float(rng.uniform(8.5, 11.5))          # "~10x", with jitter
    scale_search = scale_ref * scale_factor
    theta_deg = float(rng.uniform(-6.0, 6.0))              # small stage rotation

    ref_center_world = (0.0, 0.0)
    # True placement: bounded, roughly-Gaussian offset from the search
    # window's center (realistic drift), not uniform anywhere -- a real
    # re-acquisition search window is centered on where the tool expects
    # the site to be.
    center = out_size_search / 2.0
    drift_std = out_size_search * 0.09
    if rng.uniform() < 0.15:
        drift_std *= 2.6  # occasional large-drift outlier
    lo, hi = margin, out_size_search - margin
    true_x = float(np.clip(rng.normal(center, drift_std), lo, hi))
    true_y = float(np.clip(rng.normal(center, drift_std), lo, hi))
    search_center_world = (
        ref_center_world[0] - scale_search * (true_x - out_size_search / 2.0),
        ref_center_world[1] - scale_search * (true_y - out_size_search / 2.0),
    )

    reference = render_capture(style, params, out_size_ref, ref_center_world,
                                scale_ref, theta_deg=theta_deg, base_softness_px=0.8)
    search = render_capture(style, params, out_size_search, search_center_world,
                             scale_search, theta_deg=0.0, base_softness_px=0.9)

    noise_mult = float(rng.uniform(0.4, 2.5)) if vary_noise else 1.0
    search_noise_mult = min(noise_mult * 1.6, 4.0)  # search stays noisier than reference

    reference = cv2.GaussianBlur(reference, (0, 0), sigmaX=0.5)
    reference = add_edge_brightening(reference, strength=0.20)
    ref_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    reference = add_sensor_noise(reference, ref_rng, shot_gain=0.045 * noise_mult,
                                  read_noise_std=0.010 * noise_mult)

    search = cv2.GaussianBlur(search, (0, 0), sigmaX=0.8)
    search = add_edge_brightening(search, strength=0.20)
    search_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    search = add_sensor_noise(search, search_rng, shot_gain=0.075 * search_noise_mult,
                               read_noise_std=0.016 * search_noise_mult)

    gt = dict(
        style=style, x=round(true_x, 3), y=round(true_y, 3),
        rotation_deg=round(theta_deg, 3), scale_factor=round(scale_factor, 4),
        period_x=round(params.get("period_x", params.get("period_u", 0.0)), 3),
        period_y=round(params.get("period_y", params.get("period_gate", params.get("period_v", 0.0))), 3),
        has_landmark=bool(params.get("defect", False)),
        noise_level=round(noise_mult, 3),
        ref_size=out_size_ref, search_size=out_size_search,
    )
    return to_uint8(reference), to_uint8(search), gt


def generate(style: str, num_pairs: int, output_dir: Path, split: str | None,
             seed: int, out_size_search: int, out_size_ref: int, defect_prob: float = 0.5):
    root = output_dir / split if split else output_dir
    ref_dir, search_dir = root / "references", root / "searches"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(seed)
    styles = ALL_STYLES if style == "both" else [style]

    rows = []
    t0 = time.time()
    for i in range(num_pairs):
        s = styles[i % len(styles)] if style == "both" else style
        sample_seed = int(master_rng.integers(0, 2 ** 31))
        rng = np.random.default_rng(sample_seed)
        ref_img, search_img, gt = make_pair(s, rng, out_size_search, out_size_ref,
                                             defect_prob=defect_prob)

        sample_id = f"{split or 'sample'}_{i:05d}"
        cv2.imwrite(str(ref_dir / f"{sample_id}.png"), ref_img)
        cv2.imwrite(str(search_dir / f"{sample_id}.png"), search_img)

        row = {"id": sample_id, "seed": sample_seed, **gt}
        rows.append(row)

    elapsed = time.time() - t0

    csv_path = root / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = root / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_landmark = sum(1 for r in rows if r["has_landmark"])
    print(f"Generated {num_pairs} pairs ({style}) -> {root}  [{elapsed:.1f}s, "
          f"{elapsed / num_pairs:.2f}s/pair]")
    print(f"  {n_landmark}/{num_pairs} have a unique local landmark "
          f"(a deliberate alignment fiducial); the rest are purely periodic.")
    print(f"  references/  searches/  metadata.csv  metadata.json")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=["dram", "dram_arcuate", "finfet", "both"], default="both",
                    help="Die architecture style to generate (default: both, rotating through all three).")
    ap.add_argument("--num-pairs", type=int, default=30,
                    help="Number of reference/search pairs to generate (default: 30, the "
                         "hackathon-required self-evaluation minimum).")
    ap.add_argument("--output-dir", type=Path, default=Path("../data"),
                    help="Root output directory (default: ../data).")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None,
                    help="If given, writes to <output-dir>/<split>/ instead of directly to "
                         "<output-dir>/ -- run once per split to build train/val/test.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Master RNG seed. Default: auto-derived from --split (42/holdover "
                         "train=42, val=142042, test=242042; 42 if no --split given) so that "
                         "running the README Quickstart's three commands as-is produces three "
                         "INDEPENDENT splits. Earlier versions of this script defaulted every "
                         "split to the same fixed seed=42, which made data/train, data/val and "
                         "data/test pixel-identical for their first overlapping N samples -- "
                         "confirmed by direct comparison of generated metadata. Pass --seed "
                         "explicitly to reproduce that old behaviour or for any other reason.")
    ap.add_argument("--search-size", type=int, default=SEARCH_SIZE,
                    help=f"Search image side length in pixels (default: {SEARCH_SIZE}).")
    ap.add_argument("--ref-size", type=int, default=REF_SIZE,
                    help=f"Reference image side length in pixels (default: {REF_SIZE}).")
    ap.add_argument("--defect-prob", type=float, default=0.5,
                    help="Fraction of samples given a unique alignment fiducial marker "
                         "at the true site, vs. left purely periodic (default: 0.5).")
    args = ap.parse_args()

    if args.seed is not None:
        resolved_seed = args.seed
    else:
        resolved_seed = {"train": 42, "val": 142042, "test": 242042}.get(args.split, 42)

    generate(args.style, args.num_pairs, args.output_dir, args.split,
              resolved_seed, args.search_size, args.ref_size, args.defect_prob)


if __name__ == "__main__":
    main()
